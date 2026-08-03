"""
SIMULACIÓN CONTROLADA CON GEMINI: a diferencia de simular_partidas.py
(que corre un todos-contra-todos completo entre perfiles locales), este
script corre solo los matchups que involucran a Gemini, con una cantidad
CHICA y explícita de partidas -- para tener control total sobre cuántas
llamadas a la API se hacen.

Con la capa gratuita de Gemini (sin facturación activada), esto no tiene
costo -- como mucho, si te pasás del límite diario gratuito, las llamadas
fallan con error (nunca te cobra sin que vos actives facturación a propósito).

Requiere la variable de entorno GEMINI_API_KEY (ver engine/gemini_bots.py).

Uso:
    python pruebas/simular_partidas_con_gemini.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from engine.simulacion import PerfilBot, simular_lote
from engine.gemini_bots import GeminiSpymasterBot, GeminiOperativeBot
from data.vocabulario_tablero import VOCABULARIO_TABLERO
from simular_partidas import cargar_perfiles
from reportes import guardar_y_reportar

RESULTADOS_DIR = Path(__file__).parent.parent / "results"

# CONTROL DE COSTO: cuántas partidas por matchup. Con este valor y los
# matchups elegidos abajo, el total de llamadas a Gemini queda acotado
# y predecible -- subilo con cuidado si querés más confiabilidad estadística.
N_JUEGOS_POR_MATCHUP = 5


def main():
    print("Cargando perfiles locales (v1 / v2 / v3, sin costo)...\n")
    perfiles_locales = cargar_perfiles()

    print("Armando el perfil de Gemini...")
    gemini_spymaster = GeminiSpymasterBot()
    gemini_operative = GeminiOperativeBot()

    perfil_gemini = PerfilBot(
        nombre="gemini",
        vocab_completo=[],  # Gemini no usa un vocabulario de candidatas precalculado
        construir_spymaster=lambda vocab_pistas: gemini_spymaster,
        operative=gemini_operative,
    )

    # Elegimos matchups ESPECÍFICOS a propósito (no todos-contra-todos)
    # para controlar cuántas llamadas a la API se hacen.
    # OJO: hay que renombrar también el campo .nombre interno del
    # PerfilBot, no solo la clave del diccionario -- si no, el nombre
    # que queda registrado en cada ResultadoPartida sigue siendo el viejo
    # y rompe el dashboard (KeyError al no coincidir con la lista de
    # nombres esperados).
    from dataclasses import replace

    perfiles = {"gemini": perfil_gemini}
    if "pista1_decoder1" in perfiles_locales:
        perfiles["v1"] = replace(perfiles_locales["pista1_decoder1"], nombre="v1")
    if "v3_ensamble" in perfiles_locales:
        perfiles["v3"] = replace(perfiles_locales["v3_ensamble"], nombre="v3")

    print(f"\nPerfiles en este torneo: {list(perfiles.keys())}")
    n = len(perfiles)
    total_partidas = n * n * N_JUEGOS_POR_MATCHUP
    # Estimamos llamadas a Gemini: cada partida usa Gemini solo si "gemini"
    # es alguno de los dos equipos (no en los matchups puramente locales)
    matchups_con_gemini = sum(
        1 for r in perfiles for a in perfiles if r == "gemini" or a == "gemini"
    )
    llamadas_aprox = matchups_con_gemini * N_JUEGOS_POR_MATCHUP * 5  # ~5 turnos promedio por partida
    print(f"Total de partidas: {total_partidas}")
    print(f"Llamadas a la API de Gemini (estimado): ~{llamadas_aprox}\n")

    confirmar = input("¿Continuar? (s/n): ").strip().lower()
    if confirmar != "s":
        print("Cancelado.")
        return

    resultados = simular_lote(
        vocabulario_tablero=VOCABULARIO_TABLERO,
        perfiles=perfiles,
        n_juegos_por_matchup=N_JUEGOS_POR_MATCHUP,
        seed_base=5000,
    )
    print(f"\nTotal de partidas simuladas: {len(resultados)}\n")

    guardar_y_reportar(resultados, list(perfiles.keys()), RESULTADOS_DIR)


if __name__ == "__main__":
    main()
