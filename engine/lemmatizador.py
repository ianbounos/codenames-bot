"""
Lematizador liviano compartido: para decidir si una pista y una palabra
del tablero son "la misma palabra" en distintas formas (plural,
conjugación, género), en vez de comparar substrings literales.

Por qué el chequeo de substring literal no sirve bien: "mar" es substring
de "cámara" sin ninguna relación real entre las palabras -- un falso
positivo que le hace perder al bot pistas perfectamente válidas. En
cambio, comparando el LEMA (la forma base de la palabra) evitamos esos
falsos positivos, mientras seguimos bloqueando correctamente casos reales
como "gatos" vs "gato" (mismo lema: "gato").
"""
from __future__ import annotations

_nlp = None
_cache_lemas: dict[str, str] = {}


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        # Desactivamos el parser y NER -- no los necesitamos para
        # lematizar, y así carga más rápido
        _nlp = spacy.load("es_core_news_md", disable=["parser", "ner"])
    return _nlp


def lema(palabra: str) -> str:
    clave = palabra.strip().lower()
    if clave in _cache_lemas:
        return _cache_lemas[clave]

    doc = _get_nlp()(clave)
    resultado = doc[0].lemma_.lower() if len(doc) > 0 else clave
    _cache_lemas[clave] = resultado
    return resultado


def precalentar_cache(palabras: list[str], batch_size: int = 2000) -> None:
    """
    Lematiza todo un vocabulario de una sola vez, usando procesamiento
    por lotes de spaCy (nlp.pipe), que es ~7x más rápido que lematizar
    palabra por palabra. Pensado para llamarse UNA VEZ al arrancar una
    simulación grande, así el costo (segundos) queda visible y esperado
    al principio, en vez de aparecer como una demora rara a mitad de
    la primera partida.
    """
    nlp = _get_nlp()
    faltantes = [p.strip().lower() for p in palabras if p.strip().lower() not in _cache_lemas]
    if not faltantes:
        return
    print(f"  Precalentando lematizador para {len(faltantes)} palabras...")
    for palabra, doc in zip(faltantes, nlp.pipe(faltantes, batch_size=batch_size)):
        _cache_lemas[palabra] = doc[0].lemma_.lower() if len(doc) > 0 else palabra


def _sin_tildes(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def es_misma_palabra(a: str, b: str) -> bool:
    """
    True si `a` y `b` son la misma palabra literal (ignorando mayúsculas
    y tildes), o si comparten el mismo lema (ej. plural/singular,
    conjugaciones).
    """
    a_norm, b_norm = a.strip().lower(), b.strip().lower()
    if a_norm == b_norm:
        return True
    if _sin_tildes(a_norm) == _sin_tildes(b_norm):
        return True
    return lema(a_norm) == lema(b_norm)


def viola_regla_palabra_relacionada(candidata: str, palabras_tablero) -> bool:
    """Reemplazo de la vieja regla de substring: True si la candidata es
    la MISMA PALABRA (o una variante flexionada) que alguna palabra visible
    del tablero."""
    return any(es_misma_palabra(candidata, p) for p in palabras_tablero)
