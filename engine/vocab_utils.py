"""
Filtra un EmbeddingStore ya cargado para obtener la lista de palabras
que sirven como CANDIDATAS PARA PISTAS: se excluyen las stopwords y las
palabras que ya están en el tablero (no tendría sentido usarlas de pista).
"""
from __future__ import annotations

from data.stopwords_es import STOPWORDS_ES


def filtrar_vocabulario_pistas(
    todas_las_palabras: list[str],
    palabras_tablero: set[str],
    longitud_minima: int = 3,
) -> list[str]:
    tablero_lower = {w.lower() for w in palabras_tablero}
    return [
        w for w in todas_las_palabras
        if w not in STOPWORDS_ES
        and w not in tablero_lower
        and len(w) >= longitud_minima
    ]
