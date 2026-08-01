"""
Descarga los embeddings GloVe SBWC (Spanish Billion Word Corpus) y los
descomprime. Versión en Python (funciona igual en Windows, Mac o Linux,
sin depender de bash/curl/bunzip2).

Uso:
    python scripts/download_embeddings.py
"""
import bz2
import shutil
import sys
import urllib.request
from pathlib import Path

URL = "http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.txt.bz2"

DEST_DIR = Path(__file__).parent.parent / "data" / "embeddings"
COMPRESSED = DEST_DIR / "SBW-vectors-300-min5.txt.bz2"
OUTPUT = DEST_DIR / "glove-sbwc.300.txt"


def mostrar_progreso(bloques_transferidos, tamano_bloque, tamano_total):
    descargado = bloques_transferidos * tamano_bloque
    porcentaje = min(100, descargado * 100 / tamano_total) if tamano_total > 0 else 0
    mb_descargado = descargado / (1024 * 1024)
    mb_total = tamano_total / (1024 * 1024)
    sys.stdout.write(f"\r  Descargando: {porcentaje:5.1f}%  ({mb_descargado:.1f} MB / {mb_total:.1f} MB)")
    sys.stdout.flush()


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        print(f"Ya existe {OUTPUT}, no se vuelve a descargar.")
        return

    print(f"Descargando GloVe SBWC (~380MB comprimido) desde:\n  {URL}\n")
    try:
        urllib.request.urlretrieve(URL, COMPRESSED, reporthook=mostrar_progreso)
    except Exception as e:
        print(f"\n\nError al descargar: {e}")
        print("Si el problema persiste, avisale a Claude para buscar una alternativa.")
        sys.exit(1)

    print("\n\nDescomprimiendo (puede tardar un minuto)...")
    with bz2.open(COMPRESSED, "rb") as f_in, open(OUTPUT, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"\nListo. Embeddings en: {OUTPUT}")
    print(f"Podés borrar el archivo comprimido si querés liberar espacio: {COMPRESSED}")


if __name__ == "__main__":
    main()
