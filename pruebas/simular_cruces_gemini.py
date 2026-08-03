"""
SIMULACIÓN CRUZADA CON GEMINI: en vez de comparar "equipos completos"
(Gemini vs v1, cada uno dando Y decodificando sus propias pistas), este
script arma las combinaciones CRUZADAS -- Gemini da la pista pero v1/v3
la decodifica, y al revés -- para ver específicamente qué tan bien se
entienden entre sí un LLM y los bots locales.

Perfiles armados:
  gemini_puro       -- Gemini da y decodifica sus propias pistas
  v1_puro           -- v1 da y decodifica sus propias pistas
  v3_puro           -- v3 da y decodifica sus propias pistas
  gemini_da_v1_lee  -- Gemini da la pista, v1 la decodifica
  v1_da_gemini_lee  -- v1 da la pista, Gemini la decodifica
  gemini_da_v3_lee  -- Gemini da la pista, v3 la decodifica
  v3_da_gemini_lee  -- v3 da la pista, Gemini la decodifica

Imprime cada partida EN VIVO (pista + adivinanzas turno a turno) para
poder seguir qué está pasando mientras corre.

Uso:
    python pruebas/simular_cruces_gemini.py
"""
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from engine.board import Dueno
from engine.simulacion import PerfilBot, jugar_partida
from engine.gemini_bots import GeminiSpymasterBot, GeminiOperativeBot
from data.vocabulario_tablero import VOCABULARIO_TABLERO
from simular_partidas import cargar_perfiles
from reportes import guardar_y_reportar

RESULTADOS_DIR = Path(__file__).parent.parent / "results"

# CONTROL DE COSTO: pocas partidas por matchup, y solo unos matchups
# elegidos (no todos contra todos) -- ver más abajo cuáles.
N_JUEGOS_POR_MATCHUP = 3


def main():
    print("Cargando perfiles locales (v1 / v3, sin costo)...\n")
    perfiles_locales = cargar_perfiles()

    v1 = perfiles_locales.get("pista1_decoder1")
    v3 = perfiles_locales.get("v3_ensamble")

    print("Armando el perfil de Gemini...")
    gemini_spymaster = GeminiSpymasterBot()
    gemini_operative = GeminiOperativeBot()

    def sm_gemini(vocab_pistas):
        return gemini_spymaster

    perfiles: dict[str, PerfilBot] = {
        "gemini_puro": PerfilBot(
            nombre="gemini_puro", vocab_completo=[],
            construir_spymaster=sm_gemini, operative=gemini_operative,
        ),
    }

    if v1 is not None:
        perfiles["v1_puro"] = replace(v1, nombre="v1_puro")
        perfiles["gemini_da_v1_lee"] = PerfilBot(
            nombre="gemini_da_v1_lee", vocab_completo=[],
            construir_spymaster=sm_gemini, operative=v1.operative,
        )
        perfiles["v1_da_gemini_lee"] = PerfilBot(
            nombre="v1_da_gemini_lee", vocab_completo=v1.vocab_completo,
            construir_spymaster=v1.construir_spymaster, operative=gemini_operative,
        )

    if v3 is not None:
        perfiles["v3_puro"] = replace(v3, nombre="v3_puro")
        perfiles["gemini_da_v3_lee"] = PerfilBot(
            nombre="gemini_da_v3_lee", vocab_completo=[],
            construir_spymaster=sm_gemini, operative=v3.operative,
        )
        perfiles["v3_da_gemini_lee"] = PerfilBot(
            nombre="v3_da_gemini_lee", vocab_completo=v3.vocab_completo,
            construir_spymaster=v3.construir_spymaster, operative=gemini_operative,
        )

    print(f"\nPerfiles armados: {list(perfiles.keys())}\n")

    # Matchups ESPECÍFICOS (no todos contra todos, para controlar costo):
    # cada perfil cruzado contra su contraparte "pura", para ver si la
    # mezcla rinde mejor o peor que cada uno jugando consigo mismo.
    matchups: list[tuple[str, str]] = []
    if v1:
        matchups += [
            ("gemini_puro", "v1_puro"),
            ("gemini_da_v1_lee", "v1_puro"),
            ("v1_da_gemini_lee", "v1_puro"),
        ]
    if v3:
        matchups += [
            ("gemini_puro", "v3_puro"),
            ("gemini_da_v3_lee", "v3_puro"),
            ("v3_da_gemini_lee", "v3_puro"),
        ]

    print(f"Matchups elegidos: {len(matchups)}")
    for a, b in matchups:
        print(f"  {a} (rojo) vs {b} (azul)")
    print(f"\n~{len(matchups) * N_JUEGOS_POR_MATCHUP} partidas totales (con impresión en vivo)")

    confirmar = input("\n¿Continuar? (s/n): ").strip().lower()
    if confirmar != "s":
        print("Cancelado.")
        return

    todos_resultados = []
    seed_counter = 9000
    for nombre_rojo, nombre_azul in matchups:
        print(f"\n{'='*70}\nMATCHUP: {nombre_rojo} (rojo) vs {nombre_azul} (azul)\n{'='*70}")
        for i in range(N_JUEGOS_POR_MATCHUP):
            equipo_inicial = Dueno.ROJO if i % 2 == 0 else Dueno.AZUL
            seed_counter += 1
            resultado = jugar_partida(
                vocabulario_tablero=VOCABULARIO_TABLERO,
                perfil_rojo=perfiles[nombre_rojo],
                perfil_azul=perfiles[nombre_azul],
                seed=seed_counter,
                equipo_inicial=equipo_inicial,
                verbose=True,
            )
            todos_resultados.append(resultado)

    print(f"\nTotal de partidas simuladas: {len(todos_resultados)}\n")
    guardar_y_reportar(todos_resultados, list(perfiles.keys()), RESULTADOS_DIR)


if __name__ == "__main__":
    main()
