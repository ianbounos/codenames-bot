"""
Prueba rápida de spaCy con el modelo de español real, para confirmar
que las similitudes semánticas tienen sentido antes de integrarlo
al resto del proyecto.
"""
import spacy

print("Cargando modelo es_core_news_md...")
nlp = spacy.load("es_core_news_md")
print("Modelo cargado.\n")

pares = [
    ("océano", "ballena"),
    ("océano", "barco"),
    ("océano", "silla"),
    ("silla", "mesa"),
    ("gato", "perro"),
    ("león", "leon"),  # con y sin tilde -- para confirmar que la tilde importa
]

print("=== Similitudes ===")
for a, b in pares:
    doc_a, doc_b = nlp(a), nlp(b)
    sim = doc_a.similarity(doc_b)
    print(f"  {a:12s} vs {b:12s} = {sim:.3f}")

print("\nSi 'océano' está más cerca de 'ballena'/'barco' que de 'silla',")
print("y 'león' vs 'leon' (sin tilde) da una similitud baja,")
print("todo está funcionando como esperamos. ✅")
