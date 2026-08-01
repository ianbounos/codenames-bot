"""
Prueba rápida de EmbeddingStore con vectores hechos a mano (no reales),
solo para validar que la lógica de similitud funciona antes de bajar
el archivo GloVe real (380MB).

Construimos un mini "espacio semántico" de juguete: palabras relacionadas
al mar tienen vectores parecidos entre sí, y las de muebles, otro cluster.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.embeddings import EmbeddingStore

# Vectores sintéticos de 4 dimensiones, pensados a mano:
# dim0: "acuático", dim1: "grande", dim2: "mueble", dim3: "cotidiano"
word_vectors = {
    "oceano":   [0.9, 0.6, 0.0, 0.1],
    "ballena":  [0.85, 0.9, 0.0, 0.0],
    "agua":     [0.95, 0.1, 0.0, 0.3],
    "barco":    [0.8, 0.5, 0.0, 0.2],
    "submarino":[0.9, 0.4, 0.0, 0.1],
    "silla":    [0.0, 0.1, 0.9, 0.4],
    "mesa":     [0.0, 0.3, 0.95, 0.3],
    "sofa":     [0.0, 0.4, 0.9, 0.5],
    "lampara":  [0.05, 0.1, 0.7, 0.6],
    "gato":     [0.1, 0.05, 0.1, 0.9],
}

store = EmbeddingStore.from_dict(word_vectors)

print(f"Vocabulario cargado: {len(store)} palabras\n")

pruebas = [
    ("oceano", "ballena"),
    ("oceano", "silla"),
    ("agua", "barco"),
    ("silla", "mesa"),
    ("silla", "ballena"),
]

print("=== Similitudes individuales ===")
for a, b in pruebas:
    sim = store.similarity(a, b)
    print(f"  similitud({a:12s}, {b:10s}) = {sim:.3f}")

print("\n=== Matriz de similitud (candidato 'oceano' vs tablero) ===")
tablero = ["ballena", "silla", "barco", "mesa", "gato"]
matriz = store.bulk_similarity_matrix(["oceano"], tablero)
for palabra, sim in zip(tablero, matriz[0]):
    print(f"  {palabra:10s}: {sim:.3f}")

print("\n✅ Si 'oceano' está mucho más cerca de ballena/barco que de silla/mesa/gato,")
print("   la lógica de similitud funciona como se espera.")
