"""
Descarga una lista de frecuencia de palabras en español (50.000 palabras
más comunes, según subtítulos de OpenSubtitles), usada como fuente de
candidatas para las pistas del Spymaster.

Fuente: proyecto FrequencyWords de Hermit Dave (MIT license), alojado en
GitHub -- mucho más confiable que el servidor de GloVe que nos falló.

Uso:
    python scripts/download_lista_frecuencia.py
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"
DEST = Path(__file__).parent.parent / "data" / "es_50k.txt"


def main():
    if DEST.exists():
        print(f"Ya existe {DEST}, no se vuelve a descargar.")
        return

    print(f"Descargando lista de frecuencia desde:\n  {URL}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as e:
        print(f"Error al descargar: {e}")
        sys.exit(1)

    with DEST.open(encoding="utf-8") as f:
        n_lineas = sum(1 for _ in f)

    print(f"Listo. Guardado en {DEST} ({n_lineas} palabras)")


if __name__ == "__main__":
    main()
