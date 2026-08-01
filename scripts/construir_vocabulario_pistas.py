"""
Construye el vocabulario de CANDIDATAS PARA PISTAS a partir del archivo
real de GloVe SBWC, aplicando el filtro que acordamos:

  1. Descartar stopwords (palabras funcionales sin significado propio)
  2. Quedarnos con las primeras `max_words` palabras del archivo
     (en GloVe SBWC, el archivo está ordenado por frecuencia descendente,
     así que esto equivale a quedarnos con las más frecuentes/comunes)
  3. Filtrar por longitud mínima y que sean puramente alfabéticas
  4. Excluir cualquier palabra que ya esté en el vocabulario del tablero
     (no tendría sentido usarla como "pista" de sí misma)

Nota: NO aplicamos POS tagging (filtrar por sustantivo/adjetivo/verbo)
todavía, según lo acordado — se puede sumar más adelante como mejora.

Uso:
    python3 scripts/construir_vocabulario_pistas.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.stopwords_es import STOPWORDS_ES
from data.vocabulario_tablero import VOCABULARIO_TABLERO
from engine.embeddings import _WORD_RE  # reutilizamos la misma regex de validación


def construir_vocabulario_pistas(
    glove_path: str,
    max_words: int = 30_000,
    longitud_minima: int = 3,
) -> list[str]:
    palabras_tablero = set(w.lower() for w in VOCABULARIO_TABLERO)
    candidatas = []

    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ", 1)
            if len(parts) < 2:
                continue
            palabra = parts[0].lower()

            if palabra in palabras_tablero:
                continue
            if palabra in STOPWORDS_ES:
                continue
            if len(palabra) < longitud_minima:
                continue
            if not _WORD_RE.match(palabra):
                continue

            candidatas.append(palabra)
            if len(candidatas) >= max_words:
                break

    return candidatas


if __name__ == "__main__":
    ruta_default = Path(__file__).parent.parent / "data" / "embeddings" / "glove-sbwc.300.txt"

    if not ruta_default.exists():
        print(f"No encontré el archivo de embeddings en: {ruta_default}")
        print("Corré primero: bash scripts/download_embeddings.sh")
        sys.exit(1)

    print("Construyendo vocabulario de candidatas para pistas...")
    candidatas = construir_vocabulario_pistas(str(ruta_default))

    print(f"Vocabulario de pistas: {len(candidatas)} palabras")
    print(f"Primeras 30 (las más frecuentes): {candidatas[:30]}")

    salida = Path(__file__).parent.parent / "data" / "vocabulario_pistas.txt"
    with salida.open("w", encoding="utf-8") as f:
        f.write("\n".join(candidatas))

    print(f"\nGuardado en: {salida}")
