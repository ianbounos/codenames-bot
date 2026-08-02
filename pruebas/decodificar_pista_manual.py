"""
HERRAMIENTA INTERACTIVA: vos hacés de Spymaster.

Genera un tablero al azar (te muestra la vista completa, con los dueños,
como si fueras el espía), y te deja escribir vos mismo una pista
(palabra + número). El OperativeBot decodifica esa pista igual que lo
haría jugando contra otro bot, y te muestra qué habría tocado y en qué
orden -- así podés "sentir" qué tan bien (o mal) interpreta tus pistas.

Uso:
    python pruebas/decodificar_pista_manual.py
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.embeddings import EmbeddingStore
from engine.board import Tablero, Dueno
from engine.operative import OperativeBot
from data.vocabulario_tablero import VOCABULARIO_TABLERO

LISTA_FRECUENCIA = Path(__file__).parent.parent / "data" / "es_50k.txt"


def cargar_embeddings() -> EmbeddingStore:
    print("Cargando embeddings (spaCy + lista de frecuencia)... un momento\n")
    kwargs = dict(palabras_incluir=VOCABULARIO_TABLERO)
    if LISTA_FRECUENCIA.exists():
        kwargs["lista_frecuencia_path"] = str(LISTA_FRECUENCIA)
        kwargs["max_candidatas"] = 50_000
    return EmbeddingStore.from_spacy_model(**kwargs)


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
    store = cargar_embeddings()
    print(f"Vocabulario cargado: {len(store)} palabras\n")

    seed_input = input("Semilla para el tablero (Enter para uno al azar): ").strip()
    seed = int(seed_input) if seed_input else random.randint(0, 999_999)

    tablero = Tablero.generar(
        VOCABULARIO_TABLERO, n_palabras=25, equipo_inicial=Dueno.ROJO, seed=seed
    )
    print(f"\n(semilla usada: {seed})\n")
    mostrar_tablero(tablero)

    operative = OperativeBot(embeddings=store, arriesgar_extra=False)

    while True:
        print("\nEscribí una pista para tu equipo ROJO (o 'salir' para terminar)")
        palabra = input("  Palabra de la pista: ").strip().lower()
        if palabra in ("salir", "exit", "q"):
            break

        if palabra not in store:
            print(f"  ⚠️ '{palabra}' no está en el vocabulario cargado, probá con otra.")
            continue

        try:
            numero = int(input("  Número: ").strip())
        except ValueError:
            print("  ⚠️ Tiene que ser un número entero.")
            continue

        visibles = [c.palabra for c in tablero.cartas if not c.revelada]
        if palabra in [v.lower() for v in visibles]:
            print("  ⚠️ Esa palabra ya está en el tablero, no se puede usar como pista.")
            continue

        plan = operative.planear(tablero, palabra, numero)

        print(f"\n  El Operative, con la pista '{palabra}, {numero}', tocaría en este orden:\n")
        for i, (p, sim) in enumerate(zip(plan.orden, plan.similitudes), 1):
            carta = tablero.carta_de(p)
            dueno = carta.dueno.value
            marca = {"rojo": "✅", "azul": "❌ (rival)", "neutral": "➖ (neutral)", "asesino": "💀 ¡ASESINO!"}[dueno]
            print(f"    {i}. {p:15s} (similitud={sim:.3f})  →  {marca}")

        # Preguntar si querés "jugar" de verdad y revelar esas cartas
        aplicar = input("\n  ¿Marcar esas cartas como reveladas en el tablero? (s/n): ").strip().lower()
        if aplicar == "s":
            for p in plan.orden:
                tablero.revelar(p)
            print()
            mostrar_tablero(tablero)


if __name__ == "__main__":
    main()
