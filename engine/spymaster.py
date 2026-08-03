"""
Spymaster bot: decide qué pista dar (palabra + número) dado el estado
del tablero, usando embeddings para medir asociación semántica.

Algoritmo (versión acordada):

  Para cada palabra candidata `c` del vocabulario de pistas, y para cada
  tamaño de grupo N (cuántas palabras propias intenta conectar):

    objetivo_mas_flojo = similitud(c, la N-ésima palabra propia más cercana)

    margen_asesino = objetivo_mas_flojo - similitud(c, asesino)
    margen_resto   = objetivo_mas_flojo - max(similitud(c, enemigas + neutrales))

    válida si: margen_asesino > beta_asesino  Y  margen_resto > beta_resto

  (beta_asesino más alto que beta_resto, porque tocar el asesino termina
  la partida y tocar una enemiga/neutral solo cede el turno)

  Entre todas las (candidata, N) válidas, se elige la de mejor score,
  que privilegia conectar más palabras (N) por sobre el margen extra.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.embeddings import EmbeddingStore
from engine.board import Tablero, Dueno


@dataclass
class CandidatoPista:
    palabra: str
    numero: int
    palabras_objetivo: list[str]  # las N palabras propias que se busca conectar
    margen_asesino: float
    margen_resto: float
    score: float


class SpymasterBot:
    def __init__(
        self,
        embeddings: EmbeddingStore,
        vocabulario_pistas: list[str],
        beta_asesino: float = 0.15,
        beta_resto: float = 0.05,
        max_n: int = 4,
        embeddings_secundario: EmbeddingStore | None = None,
        beta_asesino_secundario: float | None = None,
        beta_resto_secundario: float | None = None,
    ):
        """
        Args:
            embeddings: store de embeddings ya cargado (modelo principal)
            vocabulario_pistas: palabras candidatas a usar como pista
                (no deberían incluir las palabras que están en el tablero)
            beta_asesino: margen mínimo exigido respecto al asesino, en
                el modelo PRINCIPAL.
                Más alto = mucho más conservador con el asesino específicamente.
            beta_resto: margen mínimo exigido respecto a enemigas/neutrales,
                en el modelo PRINCIPAL.
            max_n: número máximo de palabras que el bot intenta conectar
                   con una sola pista
            embeddings_secundario: si se pasa, un SEGUNDO modelo de
                embeddings (idealmente de arquitectura/entrenamiento
                distinto al principal, ej. sentence-transformers vs spaCy).
                La pista SOLO se acepta si TAMBIÉN pasa los umbrales de
                seguridad en este segundo modelo -- funciona como un
                "doble chequeo" para filtrar asociaciones raras que un
                solo modelo pueda alucinar.
            beta_asesino_secundario / beta_resto_secundario: umbrales
                para el modelo secundario. Si no se especifican, se
                reusan los mismos valores que el modelo principal.
        """
        self.embeddings = embeddings
        self.vocabulario_pistas = vocabulario_pistas
        self.beta_asesino = beta_asesino
        self.beta_resto = beta_resto
        self.max_n = max_n

        self.embeddings_secundario = embeddings_secundario
        self.beta_asesino_secundario = (
            beta_asesino_secundario if beta_asesino_secundario is not None else beta_asesino
        )
        self.beta_resto_secundario = (
            beta_resto_secundario if beta_resto_secundario is not None else beta_resto
        )

    def elegir_pista(self, tablero: Tablero, equipo: Dueno) -> CandidatoPista | None:
        import numpy as np

        equipo_rival = Dueno.AZUL if equipo == Dueno.ROJO else Dueno.ROJO

        propias = tablero.palabras_de(equipo)
        peligrosas_resto = (
            tablero.palabras_de(equipo_rival) + tablero.palabras_de(Dueno.NEUTRAL)
        )
        asesino = tablero.palabras_de(Dueno.ASESINO)
        palabra_asesino = asesino[0] if asesino else None

        palabras_tablero = set(w.lower() for w in tablero.palabras_visibles())

        candidatas = [
            c for c in self.vocabulario_pistas
            if c.lower() in self.embeddings
            and not self._viola_regla_substring(c, palabras_tablero)
        ]
        if not candidatas or not propias:
            return None

        # --- Cálculo vectorizado con NumPy (una sola multiplicación de
        # matrices en vez de miles de comparaciones palabra por palabra) ---
        mat_propias = self.embeddings.bulk_similarity_matrix(candidatas, propias)  # (C, P)

        columnas_peligro = list(peligrosas_resto)
        if palabra_asesino:
            columnas_peligro = columnas_peligro + [palabra_asesino]

        if columnas_peligro:
            mat_peligro = self.embeddings.bulk_similarity_matrix(candidatas, columnas_peligro)  # (C, D)
        else:
            mat_peligro = np.zeros((len(candidatas), 0))

        n_resto = len(peligrosas_resto)
        peligro_resto_arr = (
            mat_peligro[:, :n_resto].max(axis=1) if n_resto > 0
            else np.full(len(candidatas), -np.inf)
        )
        peligro_asesino_arr = (
            mat_peligro[:, -1] if palabra_asesino
            else np.full(len(candidatas), -np.inf)
        )

        # orden de las propias por candidata, de más a menos parecida
        orden_propias = np.argsort(-mat_propias, axis=1)
        propias_ordenadas_valores = np.take_along_axis(mat_propias, orden_propias, axis=1)

        mejor: CandidatoPista | None = None
        max_n_real = min(self.max_n, len(propias))

        for n in range(1, max_n_real + 1):
            objetivo_mas_flojo = propias_ordenadas_valores[:, n - 1]  # (C,)
            margen_asesino = objetivo_mas_flojo - peligro_asesino_arr
            margen_resto = objetivo_mas_flojo - peligro_resto_arr

            validas = np.nonzero(
                (margen_asesino > self.beta_asesino) & (margen_resto > self.beta_resto)
            )[0]

            for idx in validas:
                score = n * 1.0 + min(margen_asesino[idx], margen_resto[idx]) * 0.5
                if mejor is not None and score <= mejor.score:
                    continue  # no vale la pena ni chequear el modelo secundario

                candidata = candidatas[idx]
                palabras_objetivo = [propias[k] for k in orden_propias[idx, :n]]

                if self.embeddings_secundario is not None:
                    if not self._pasa_validacion_secundaria(
                        candidata=candidata,
                        palabras_objetivo=palabras_objetivo,
                        peligrosas_resto=peligrosas_resto,
                        palabra_asesino=palabra_asesino,
                    ):
                        continue

                mejor = CandidatoPista(
                    palabra=candidata,
                    numero=n,
                    palabras_objetivo=palabras_objetivo,
                    margen_asesino=float(margen_asesino[idx]),
                    margen_resto=float(margen_resto[idx]),
                    score=score,
                )

        return mejor

    def _pasa_validacion_secundaria(
        self,
        candidata: str,
        palabras_objetivo: list[str],
        peligrosas_resto: list[str],
        palabra_asesino: str | None,
    ) -> bool:
        """
        Recalcula el mismo chequeo de márgenes, pero con el modelo
        secundario. Si la candidata o alguna de las palabras del tablero
        no existe en ese modelo, se rechaza por precaución (no podemos
        confirmar que sea segura).
        """
        emb2 = self.embeddings_secundario
        assert emb2 is not None

        if candidata.lower() not in emb2:
            return False
        if not all(p.lower() in emb2 for p in palabras_objetivo):
            return False

        sims_objetivo = emb2.similarities_to(candidata, palabras_objetivo)
        objetivo_mas_flojo_2 = min(sims_objetivo.values())

        peligro_asesino_2 = (
            emb2.similarity(candidata, palabra_asesino)
            if palabra_asesino and palabra_asesino.lower() in emb2
            else float("-inf")
        )
        peligrosas_disponibles = [p for p in peligrosas_resto if p.lower() in emb2]
        peligro_resto_2 = (
            max(emb2.similarities_to(candidata, peligrosas_disponibles).values())
            if peligrosas_disponibles else float("-inf")
        )

        margen_asesino_2 = objetivo_mas_flojo_2 - peligro_asesino_2
        margen_resto_2 = objetivo_mas_flojo_2 - peligro_resto_2

        return (
            margen_asesino_2 > self.beta_asesino_secundario
            and margen_resto_2 > self.beta_resto_secundario
        )

    @staticmethod
    def _viola_regla_substring(candidata: str, palabras_tablero: set[str]) -> bool:
        from engine.lemmatizador import viola_regla_palabra_relacionada
        return viola_regla_palabra_relacionada(candidata, palabras_tablero)
