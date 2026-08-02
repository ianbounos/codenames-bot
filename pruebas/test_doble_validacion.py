"""
Prueba del Spymaster con DOBLE VALIDACIÓN: exige que tanto spaCy como
sentence-transformers (dos modelos de arquitectura y entrenamiento
distintos) coincidan en que una pista es segura.

Requiere: pip install sentence-transformers
La primera vez que corras esto, va a descargar el modelo desde Hugging
Face (~470MB para paraphrase-multilingual-MiniLM-L12-v2) -- puede
tardar un par de minutos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.embeddings import EmbeddingStore
from engine.board import Tablero, Dueno
from engine.spymaster import SpymasterBot
from engine.operative import OperativeBot
from engine.vocab_utils import filtrar_vocabulario_pistas
from data.vocabulario_tablero import VOCABULARIO_TABLERO

LISTA_FRECUENCIA = Path(__file__).parent.parent / "data" / "es_50k.txt"

if not LISTA_FRECUENCIA.exists():
    print("Falta la lista de frecuencia. Corré primero:")
    print("  python scripts/download_lista_frecuencia.py")
    sys.exit(1)

print("=== Cargando modelo PRINCIPAL (spaCy) ===")
store_principal = EmbeddingStore.from_spacy_model(
    palabras_incluir=VOCABULARIO_TABLERO,
    lista_frecuencia_path=str(LISTA_FRECUENCIA),
    max_candidatas=50_000,
)
print(f"Cargadas: {len(store_principal)} palabras\n")

print("=== Cargando modelo SECUNDARIO (sentence-transformers) ===")
print("(la primera vez tarda más, porque descarga el modelo)")
store_secundario = EmbeddingStore.from_sentence_transformer(
    words=store_principal.words,  # codificamos las mismas palabras que ya tenemos
)
print(f"Cargadas: {len(store_secundario)} palabras\n")

import random
random.seed(42)
tablero = Tablero.generar(VOCABULARIO_TABLERO, n_palabras=25, equipo_inicial=Dueno.ROJO, seed=42)

print("=== Tablero ===")
print(tablero.resumen_texto(mostrar_duenos=True))
print()

palabras_tablero = set(tablero.palabras_visibles())
vocab_pistas = filtrar_vocabulario_pistas(store_principal.words, palabras_tablero)

print("=== Comparación: Spymaster CON vs SIN doble validación ===\n")

spymaster_simple = SpymasterBot(
    embeddings=store_principal,
    vocabulario_pistas=vocab_pistas,
    beta_asesino=0.05,
    beta_resto=0.02,
)
pista_simple = spymaster_simple.elegir_pista(tablero, Dueno.ROJO)
print("--- Sin doble validación (solo spaCy) ---")
if pista_simple:
    print(f"  '{pista_simple.palabra}', numero={pista_simple.numero}, objetivo={pista_simple.palabras_objetivo}")
else:
    print("  No encontró pista segura")

spymaster_doble = SpymasterBot(
    embeddings=store_principal,
    vocabulario_pistas=vocab_pistas,
    beta_asesino=0.05,
    beta_resto=0.02,
    embeddings_secundario=store_secundario,
    beta_asesino_secundario=0.05,
    beta_resto_secundario=0.02,
)
pista_doble = spymaster_doble.elegir_pista(tablero, Dueno.ROJO)
print("\n--- CON doble validación (spaCy + sentence-transformers) ---")
if pista_doble:
    print(f"  '{pista_doble.palabra}', numero={pista_doble.numero}, objetivo={pista_doble.palabras_objetivo}")
else:
    print("  No encontró pista segura")

print("\n¿Cambió la pista elegida? Si es distinta (o más conservadora en N),")
print("es evidencia de que el filtro cruzado está descartando asociaciones")
print("que solo uno de los dos modelos consideraba fuertes.")
