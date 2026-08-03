"""
TORNEO COMPLETO: todos-contra-todos entre los perfiles locales (v1, v3) y
todas las variantes que involucran a Gemini (puro, y las 4 combinaciones
cruzadas de dar/decodificar). A diferencia de simular_cruces_gemini.py
(que solo probaba cada variante contra su "referencia pura"), este script
corre el TODOS-CONTRA-TODOS completo entre los 7 perfiles.

Perfiles (7): v1_puro, v3_puro, gemini_puro, gemini_da_v1_lee,
              v1_da_gemini_lee, gemini_da_v3_lee, v3_da_gemini_lee

Con N_JUEGOS_POR_MATCHUP=10 son 7x7x10 = 490 partidas totales.
Costo estimado con gemini-3.1-flash-lite: bien por debajo de $1.
Tiempo estimado (con la pausa configurada): confirmalo corriendo primero
un lote chico si querés estar seguro antes de lanzar todo.

Uso:
    python pruebas/torneo_completo_gemini.py
"""
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from engine.simulacion import PerfilBot, simular_lote
from engine.gemini_bots import GeminiSpymasterBot, GeminiOperativeBot
from data.vocabulario_tablero import VOCABULARIO_TABLERO
from simular_partidas import cargar_perfiles
from reportes import guardar_y_reportar

RESULTADOS_DIR = Path(__file__).parent.parent / "results"

# CONTROL DE COSTO / TIEMPO: partidas por matchup. Con 7 perfiles, cada
# unidad acá son 49 partidas más. Empezá chico (ej. 3) para confirmar
# tiempos, después subís.
N_JUEGOS_POR_MATCHUP = 10

# Si querés imprimir cada partida en vivo (pista + tablero + adivinanzas),
# dejalo en True. Con muchas partidas, esto genera MUCHO texto en pantalla.
VERBOSE = False


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

    n = len(perfiles)
    total_partidas = n * n * N_JUEGOS_POR_MATCHUP
    print(f"\nPerfiles: {list(perfiles.keys())} ({n} perfiles)")
    print(f"Matchups: {n*n}  |  Partidas por matchup: {N_JUEGOS_POR_MATCHUP}  |  Total: {total_partidas}")

    # Estimar cuántos de esos matchups involucran a Gemini de algún lado
    matchups_con_gemini = sum(
        1 for r in perfiles for a in perfiles if "gemini" in r or "gemini" in a
    )
    llamadas_aprox = matchups_con_gemini * N_JUEGOS_POR_MATCHUP * 5  # ~5 turnos promedio
    print(f"Llamadas a Gemini (estimado): ~{llamadas_aprox}")
    print(f"Tiempo estimado (con pausa de {__import__('engine.gemini_bots', fromlist=['PAUSA_ENTRE_LLAMADAS_SEGUNDOS']).PAUSA_ENTRE_LLAMADAS_SEGUNDOS}s): "
          f"~{llamadas_aprox * __import__('engine.gemini_bots', fromlist=['PAUSA_ENTRE_LLAMADAS_SEGUNDOS']).PAUSA_ENTRE_LLAMADAS_SEGUNDOS / 60:.0f} minutos\n")

    confirmar = input("¿Continuar? (s/n): ").strip().lower()
    if confirmar != "s":
        print("Cancelado.")
        return

    resultados = simular_lote(
        vocabulario_tablero=VOCABULARIO_TABLERO,
        perfiles=perfiles,
        n_juegos_por_matchup=N_JUEGOS_POR_MATCHUP,
        seed_base=20000,
        verbose=VERBOSE,
    )
    print(f"\nTotal de partidas simuladas: {len(resultados)}\n")

    guardar_y_reportar(resultados, list(perfiles.keys()), RESULTADOS_DIR)


if __name__ == "__main__":
    main()
