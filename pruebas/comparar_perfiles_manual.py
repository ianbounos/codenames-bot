"""
HERRAMIENTA INTERACTIVA: compara lado a lado los distintos perfiles de bot
(pista1_decoder1, pista2_decoder2, v3_ensamble, etc.) sobre el MISMO tablero.

Para cada perfil disponible, muestra:
  1. Qué pista sugeriría su Spymaster
  2. Cómo la decodificaría su propio Operative

Y además te deja escribir VOS una pista, para ver cómo la interpretaría
cada perfil por separado -- útil para comparar qué tan distinto "entienden"
el mismo texto los distintos modelos/ensambles.

Uso:
    python pruebas/comparar_perfiles_manual.py
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from engine.board import Tablero, Dueno
from engine.vocab_utils import filtrar_vocabulario_pistas
from data.vocabulario_tablero import VOCABULARIO_TABLERO
from simular_partidas import cargar_perfiles


def mostrar_tablero(tablero: Tablero) -> None:
    print("=" * 60)
    print("TABLERO (vos ves todo, como el Spymaster)")
    print("=" * 60)
    for i, c in enumerate(tablero.cartas, 1):
        estado = "  (ya revelada)" if c.revelada else ""
        print(f"  {i:2d}. {c.palabra:15s} [{c.dueno.value.upper()}]{estado}")
    print("=" * 60)
    print()


def mostrar_decodificacion(tablero: Tablero, operative, palabra: str, numero: int) -> None:
    plan = operative.planear(tablero, palabra, numero)
    for p, sim in zip(plan.orden, plan.similitudes):
        dueno = tablero.carta_de(p).dueno.value
        marca = {"rojo": "✅", "azul": "❌", "neutral": "➖", "asesino": "💀"}[dueno]
        print(f"      {marca} {p:15s} sim={sim:.3f}  ({dueno})")


def main():
    print("Cargando todos los perfiles disponibles (puede tardar varios minutos")
    print("la primera vez, si hace falta descargar modelos nuevos)...\n")
    perfiles = cargar_perfiles()
    print(f"\nPerfiles disponibles: {list(perfiles.keys())}\n")

    seed_input = input("Semilla para el tablero (Enter para uno al azar): ").strip()
    seed = int(seed_input) if seed_input else random.randint(0, 999_999)

    tablero = Tablero.generar(
        VOCABULARIO_TABLERO, n_palabras=25, equipo_inicial=Dueno.ROJO, seed=seed
    )
    print(f"\n(semilla usada: {seed})\n")
    mostrar_tablero(tablero)

    palabras_tablero = set(tablero.palabras_visibles())

    print("\n" + "#" * 60)
    print("# QUÉ SUGIERE Y CÓMO DECODIFICA CADA PERFIL, PARA SU PROPIA PISTA")
    print("#" * 60)

    for nombre, perfil in perfiles.items():
        print(f"\n--- {nombre} ---")
        vocab_pistas = filtrar_vocabulario_pistas(perfil.vocab_completo, palabras_tablero)
        spymaster = perfil.construir_spymaster(vocab_pistas)
        pista = spymaster.elegir_pista(tablero, Dueno.ROJO)

        if pista is None:
            print("    No encontró ninguna pista segura.")
            continue

        print(f"    Sugiere: '{pista.palabra}', número={pista.numero}")
        print(f"    Objetivo: {pista.palabras_objetivo}")
        print(f"    Margen asesino={pista.margen_asesino:.3f}  margen resto={pista.margen_resto:.3f}")
        print(f"    Cómo la decodifica su propio Operative:")
        mostrar_decodificacion(tablero, perfil.operative, pista.palabra, pista.numero)

    print("\n" + "#" * 60)
    print("# AHORA VOS: escribí una pista y mirá cómo la interpreta cada perfil")
    print("#" * 60)

    while True:
        print()
        palabra = input("Palabra de la pista (o 'salir'): ").strip().lower()
        if palabra in ("salir", "exit", "q"):
            break
        try:
            numero = int(input("Número: ").strip())
        except ValueError:
            print("  ⚠️ Tiene que ser un número entero.")
            continue

        for nombre, perfil in perfiles.items():
            print(f"\n  --- {nombre} decodifica '{palabra}, {numero}' ---")
            try:
                mostrar_decodificacion(tablero, perfil.operative, palabra, numero)
            except Exception as e:
                print(f"      (no se pudo: {e})")


if __name__ == "__main__":
    main()
