"""
Test integrador END-TO-END con embeddings REALES de español (spaCy +
lista de frecuencia). Genera un tablero al azar del vocabulario real,
hace que el Spymaster dé una pista, y que el Operative decida qué tocar.

Requiere haber corrido antes:
    python -m spacy download es_core_news_md
    python scripts/download_lista_frecuencia.py
"""
import sys
import random
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

print("Cargando embeddings (spaCy + lista de frecuencia)... puede tardar unos segundos")
store = EmbeddingStore.from_spacy_model(
    palabras_incluir=VOCABULARIO_TABLERO,
    lista_frecuencia_path=str(LISTA_FRECUENCIA),
    max_candidatas=50_000,
)
print(f"Total de palabras cargadas: {len(store)}\n")

# Semilla fija para que el resultado sea reproducible al correrlo de nuevo
random.seed(42)
tablero = Tablero.generar(VOCABULARIO_TABLERO, n_palabras=25, equipo_inicial=Dueno.ROJO, seed=42)

print("=== Tablero (vista de espía, para debug) ===")
print(tablero.resumen_texto(mostrar_duenos=True))
print()

palabras_tablero = set(tablero.palabras_visibles())
vocab_pistas = filtrar_vocabulario_pistas(store.words, palabras_tablero)
print(f"Vocabulario de candidatas para pistas (filtrado): {len(vocab_pistas)} palabras\n")

spymaster = SpymasterBot(
    embeddings=store,
    vocabulario_pistas=vocab_pistas,
    beta_asesino=0.05,
    beta_resto=0.02,
)

print("=== Spymaster ROJO decide una pista ===")
pista = spymaster.elegir_pista(tablero, Dueno.ROJO)

if pista is None:
    print("  No encontró ninguna pista lo bastante segura.")
    sys.exit(0)

print(f"  Pista: '{pista.palabra}', número = {pista.numero}")
print(f"  Palabras objetivo: {pista.palabras_objetivo}")
print(f"  Margen vs asesino: {pista.margen_asesino:.3f}")
print(f"  Margen vs resto: {pista.margen_resto:.3f}\n")

print(f"=== Operative ROJO recibe la pista '{pista.palabra}, {pista.numero}' ===")
operative = OperativeBot(embeddings=store, arriesgar_extra=False)
plan = operative.planear(tablero, pista.palabra, pista.numero)

aciertos = 0
for palabra, sim in zip(plan.orden, plan.similitudes):
    dueno_real = tablero.carta_de(palabra).dueno.value
    marca = "✅" if dueno_real == "rojo" else ("💀" if dueno_real == "asesino" else "❌")
    if dueno_real == "rojo":
        aciertos += 1
    print(f"  {marca} {palabra:15s} sim={sim:.3f}  (dueño real: {dueno_real})")

print()
if aciertos == len(plan.orden):
    print(f"🎉 El operative acertó las {aciertos} palabras, sin errores.")
else:
    print(f"⚠️ El operative acertó {aciertos} de {len(plan.orden)}.")
