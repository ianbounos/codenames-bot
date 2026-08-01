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
    ):
        """
        Args:
            embeddings: store de embeddings ya cargado
            vocabulario_pistas: palabras candidatas a usar como pista
                (no deberían incluir las palabras que están en el tablero)
            beta_asesino: margen mínimo exigido respecto al asesino.
                Más alto = mucho más conservador con el asesino específicamente.
            beta_resto: margen mínimo exigido respecto a enemigas/neutrales.
            max_n: número máximo de palabras que el bot intenta conectar
                   con una sola pista
        """
        self.embeddings = embeddings
        self.vocabulario_pistas = vocabulario_pistas
        self.beta_asesino = beta_asesino
        self.beta_resto = beta_resto
        self.max_n = max_n

    def elegir_pista(self, tablero: Tablero, equipo: Dueno) -> CandidatoPista | None:
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

        mejor: CandidatoPista | None = None

        for c in candidatas:
            sim_propias = self.embeddings.similarities_to(c, propias)
            sim_resto = self.embeddings.similarities_to(c, peligrosas_resto)

            peligro_resto = max(sim_resto.values(), default=float("-inf"))
            peligro_asesino = (
                self.embeddings.similarity(c, palabra_asesino)
                if palabra_asesino else float("-inf")
            )

            propias_ordenadas = sorted(
                sim_propias.items(), key=lambda kv: kv[1], reverse=True
            )

            for n in range(1, min(self.max_n, len(propias_ordenadas)) + 1):
                grupo = propias_ordenadas[:n]
                objetivo_mas_flojo = grupo[-1][1]

                margen_asesino = objetivo_mas_flojo - peligro_asesino
                margen_resto = objetivo_mas_flojo - peligro_resto

                if margen_asesino <= self.beta_asesino:
                    continue
                if margen_resto <= self.beta_resto:
                    continue

                # score: privilegiamos N (más palabras conectadas), y como
                # desempate preferimos más margen (más seguro)
                score = n * 1.0 + min(margen_asesino, margen_resto) * 0.5

                if mejor is None or score > mejor.score:
                    mejor = CandidatoPista(
                        palabra=c,
                        numero=n,
                        palabras_objetivo=[w for w, _ in grupo],
                        margen_asesino=margen_asesino,
                        margen_resto=margen_resto,
                        score=score,
                    )

        return mejor

    @staticmethod
    def _viola_regla_substring(candidata: str, palabras_tablero: set[str]) -> bool:
        c = candidata.lower()
        for palabra in palabras_tablero:
            if c in palabra or palabra in c:
                return True
        return False
