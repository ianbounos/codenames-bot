"""
HERRAMIENTA INTERACTIVA v2: igual que decodificar_pista_manual.py, pero
con resolución HÍBRIDA de la pista -- si escribís una palabra que no
está en el vocabulario precalculado de spaCy, en vez de rechazarla, cae
automáticamente a sentence-transformers para codificarla al vuelo.
Así casi cualquier palabra real en español que escribas va a funcionar.

Requiere: pip install sentence-transformers
(la primera vez descarga un modelo de ~470MB)

Uso:
    python pruebas/decodificar_pista_manual_v2.py
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.embeddings import EmbeddingStore
from engine.embedding_hibrido import EmbeddingHibrido
from engine.board import Tablero, Dueno
from data.vocabulario_tablero import VOCABULARIO_TABLERO

LISTA_FRECUENCIA = Path(__file__).parent.parent / "data" / "es_50k.txt"


def cargar_hibrido() -> EmbeddingHibrido:
    print("Cargando modelo principal (spaCy)... un momento")
    kwargs = dict(palabras_incluir=VOCABULARIO_TABLERO)
    if LISTA_FRECUENCIA.exists():
        kwargs["lista_frecuencia_path"] = str(LISTA_FRECUENCIA)
        kwargs["max_candidatas"] = 50_000
    principal = EmbeddingStore.from_spacy_model(**kwargs)
    print(f"  {len(principal)} palabras cargadas\n")

    print("Cargando modelo secundario (sentence-transformers)...")
    print("(la primera vez tarda más, porque descarga el modelo)")
    from sentence_transformers import SentenceTransformer
    modelo_raw = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    secundario = EmbeddingStore.from_sentence_transformer(
        words=VOCABULARIO_TABLERO,  # alcanza con precalcular el tablero;
                                     # las pistas nuevas se codifican al vuelo
    )
    print(f"  {len(secundario)} palabras precalculadas\n")

    return EmbeddingHibrido(principal, secundario, modelo_raw)


def mostrar_tablero(tablero: Tablero) -> None:
    print("=" * 60)
    print("TABLERO (vos ves todo, como el Spymaster)")
    print("=" * 60)
    for i, c in enumerate(tablero.cartas, 1):
        estado = "  (ya revelada)" if c.revelada else ""
        print(f"  {i:2d}. {c.palabra:15s} [{c.dueno.value.upper()}]{estado}")
    print("=" * 60)
    print()


def main():
    hibrido = cargar_hibrido()

    seed_input = input("Semilla para el tablero (Enter para uno al azar): ").strip()
    seed = int(seed_input) if seed_input else random.randint(0, 999_999)

    tablero = Tablero.generar(
        VOCABULARIO_TABLERO, n_palabras=25, equipo_inicial=Dueno.ROJO, seed=seed
    )
    print(f"\n(semilla usada: {seed})\n")
    mostrar_tablero(tablero)

    while True:
        print("\nEscribí una pista para tu equipo ROJO (o 'salir' para terminar)")
        palabra = input("  Palabra de la pista: ").strip().lower()
        if palabra in ("salir", "exit", "q"):
            break

        try:
            numero = int(input("  Número: ").strip())
        except ValueError:
            print("  ⚠️ Tiene que ser un número entero.")
            continue

        visibles = [c.palabra for c in tablero.cartas if not c.revelada]
        if palabra in [v.lower() for v in visibles]:
            print("  ⚠️ Esa palabra ya está en el tablero, no se puede usar como pista.")
            continue

        sims, fuente = hibrido.similarities_to_tablero(palabra, visibles)

        if fuente == "no_resuelta":
            print(f"  ⚠️ No pude resolver un vector para '{palabra}'.")
            continue

        etiqueta_fuente = {
            "principal": "modelo principal (spaCy, precalculado)",
            "secundario": "modelo secundario (ya precalculado)",
            "secundario_al_vuelo": "modelo secundario (codificada AL VUELO, palabra no precalculada)",
        }[fuente]
        print(f"  (resuelta usando: {etiqueta_fuente})")

        ordenadas = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[:numero]

        print(f"\n  Con la pista '{palabra}, {numero}', el Operative tocaría en este orden:\n")
        for i, (p, sim) in enumerate(ordenadas, 1):
            carta = tablero.carta_de(p)
            dueno = carta.dueno.value
            marca = {"rojo": "✅", "azul": "❌ (rival)", "neutral": "➖ (neutral)", "asesino": "💀 ¡ASESINO!"}[dueno]
            print(f"    {i}. {p:15s} (similitud={sim:.3f})  →  {marca}")

        aplicar = input("\n  ¿Marcar esas cartas como reveladas en el tablero? (s/n): ").strip().lower()
        if aplicar == "s":
            for p, _ in ordenadas:
                tablero.revelar(p)
            print()
            mostrar_tablero(tablero)


if __name__ == "__main__":
    main()
