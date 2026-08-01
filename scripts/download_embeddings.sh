#!/usr/bin/env bash
#
# Descarga los embeddings GloVe SBWC (Spanish Billion Word Corpus)
# Fuente: Cristian Cardellino - https://crscardellino.ar/SBWCE/
#
# Uso: bash scripts/download_embeddings.sh
#
set -e

DEST_DIR="$(dirname "$0")/../data/embeddings"
mkdir -p "$DEST_DIR"

URL="http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.txt.bz2"
COMPRESSED="$DEST_DIR/SBW-vectors-300-min5.txt.bz2"
OUTPUT="$DEST_DIR/glove-sbwc.300.txt"

if [ -f "$OUTPUT" ]; then
    echo "Ya existe $OUTPUT, no se vuelve a descargar."
    exit 0
fi

echo "Descargando GloVe SBWC (~380MB comprimido, puede tardar varios minutos)..."
curl -L -o "$COMPRESSED" "$URL"

echo "Descomprimiendo..."
bunzip2 -k "$COMPRESSED"
mv "$DEST_DIR/SBW-vectors-300-min5.txt" "$OUTPUT"

echo "Listo. Embeddings en: $OUTPUT"
echo "Podés borrar el .bz2 si querés liberar espacio: rm $COMPRESSED"
