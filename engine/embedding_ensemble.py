"""
EmbeddingEnsemble: combina VARIOS modelos de embeddings en uno solo,
usando un PROMEDIO PONDERADO de similitudes en vez de un veto estricto
(que era el enfoque de la v2 original, con `embeddings_secundario` en
SpymasterBot -- ver docstring de esa clase para la comparación).

Por qué el promedio ponderado es mejor que el veto estricto:
  Con veto estricto, si CUALQUIER modelo tiene la más mínima duda sobre
  una palabra, la pista entera se descarta -- aunque el resto de los
  modelos estén muy seguros. Esto hace que el sistema sea "tan
  conservador como el modelo más nervioso", lo cual en la práctica lo
  vuelve excesivamente lento (confirmado en las simulaciones: v2 con
  veto estricto tardaba el doble de turnos que v1 solo, sin ganar
  apenas nada de seguridad extra contra el asesino).

  Con promedio ponderado, un modelo muy seguro puede "compensar" a otro
  modelo apenas dudoso, y solo se descarta la pista si el promedio
  combinado realmente indica riesgo -- más parecido a cómo un humano
  sopesaría "creo que sí, aunque no estoy 100% seguro" en vez de exigir
  unanimidad total.

DISEÑADO PARA SER UN REEMPLAZO DIRECTO (drop-in) de EmbeddingStore:
implementa los mismos métodos (__contains__, similarity, similarities_to,
bulk_similarity_matrix), así que SpymasterBot y OperativeBot lo pueden
usar sin ningún cambio, simplemente pasándolo como si fuera un store
normal -- ya no hace falta el mecanismo especial de "embeddings_secundario".

LIMITACIÓN IMPORTANTE: esta clase asume que todas las palabras que le
vas a preguntar (candidatas + palabras del tablero) ya están
PRECALCULADAS en todos los modelos que forman el ensamble -- no hace
fallback dinámico "al vuelo" como EmbeddingHibrido. Para eso (pistas de
un humano con palabras raras), se sigue necesitando el enfoque híbrido
con resolución dinámica (ver engine/embedding_hibrido.py).
"""
from __future__ import annotations

import numpy as np

from engine.embeddings import EmbeddingStore


class EmbeddingEnsemble:
    def __init__(self, modelos: list[tuple[str, EmbeddingStore, float]]):
        """
        Args:
            modelos: lista de (nombre, store, peso). Los pesos no
                necesitan sumar 1 -- se normalizan automáticamente en
                cada comparación, palabra por palabra, contemplando
                solo los modelos donde ambas palabras están disponibles.
        """
        if not modelos:
            raise ValueError("Necesitás al menos un modelo en el ensamble")
        self.modelos = modelos

    def __contains__(self, word: str) -> bool:
        return any(word.lower() in store for _, store, _ in self.modelos)

    def similarity(self, word_a: str, word_b: str) -> float:
        """
        Promedio ponderado de los modelos que SÍ conocen ambas palabras,
        con "encogimiento por cobertura": si solo una fracción de los
        modelos (por peso) conoce la palabra, el resultado se atenúa
        hacia 0 proporcionalmente -- así una palabra que solo un modelo
        minoritario califica muy alto no puede, por sí sola, dominar la
        decisión como si todos estuvieran de acuerdo.
        """
        peso_total_ensamble = sum(peso for _, _, peso in self.modelos)
        peso_disponible = 0.0
        acumulado = 0.0
        for _nombre, store, peso in self.modelos:
            if word_a.lower() in store and word_b.lower() in store:
                acumulado += peso * store.similarity(word_a, word_b)
                peso_disponible += peso
        if peso_disponible == 0:
            return float("-inf")

        promedio_disponible = acumulado / peso_disponible
        factor_confianza = peso_disponible / peso_total_ensamble
        return promedio_disponible * factor_confianza

    def cobertura_de_par(self, word_a: str, word_b: str) -> float:
        """Qué fracción del peso total del ensamble pudo evaluar este par
        (1.0 = todos los modelos la conocían, 0.5 = la mitad del peso, etc.)"""
        peso_total_ensamble = sum(peso for _, _, peso in self.modelos)
        peso_disponible = sum(
            peso for _, store, peso in self.modelos
            if word_a.lower() in store and word_b.lower() in store
        )
        return peso_disponible / peso_total_ensamble if peso_total_ensamble else 0.0

    def similarities_to(self, word: str, targets) -> dict[str, float]:
        return {t: self.similarity(word, t) for t in targets}

    def bulk_similarity_matrix(self, candidate_words: list[str], board_words: list[str]) -> np.ndarray:
        """
        Versión vectorizada de `similarity`, con la misma lógica de
        encogimiento por cobertura parcial (ver docstring de esa
        función). Si NINGÚN modelo del ensamble conoce un par candidata/
        palabra-de-tablero en particular, esa celda queda en -inf
        (imposible de evaluar), igual que haría un EmbeddingStore normal
        con una palabra desconocida.
        """
        n_candidatas, n_tablero = len(candidate_words), len(board_words)
        acumulado = np.zeros((n_candidatas, n_tablero))
        peso_disponible = np.zeros((n_candidatas, n_tablero))
        peso_total_ensamble = sum(peso for _, _, peso in self.modelos)

        for _nombre, store, peso in self.modelos:
            cand_mask = np.array([w.lower() in store for w in candidate_words])
            board_mask = np.array([w.lower() in store for w in board_words])
            if not cand_mask.any() or not board_mask.any():
                continue

            idx_cand = np.nonzero(cand_mask)[0]
            idx_board = np.nonzero(board_mask)[0]
            cand_sub = [candidate_words[i] for i in idx_cand]
            board_sub = [board_words[i] for i in idx_board]

            sub_matriz = store.bulk_similarity_matrix(cand_sub, board_sub)

            acumulado[np.ix_(idx_cand, idx_board)] += peso * sub_matriz
            peso_disponible[np.ix_(idx_cand, idx_board)] += peso

        with np.errstate(invalid="ignore", divide="ignore"):
            promedio_disponible = np.where(
                peso_disponible > 0,
                acumulado / np.where(peso_disponible == 0, 1, peso_disponible),
                0.0,
            )
        factor_confianza = peso_disponible / peso_total_ensamble
        resultado = promedio_disponible * factor_confianza
        resultado = np.where(peso_disponible > 0, resultado, -np.inf)
        return resultado

    def cobertura(self, palabras: list[str]) -> dict[str, list[str]]:
        """Utilidad de diagnóstico: para cada modelo, qué palabras de la
        lista le faltan. Útil para depurar antes de correr una simulación."""
        resultado = {}
        for nombre, store, _peso in self.modelos:
            faltantes = [w for w in palabras if w.lower() not in store]
            resultado[nombre] = faltantes
        return resultado
