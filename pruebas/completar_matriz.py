"""
COMPLETAR LA MATRIZ: agrega los 2 perfiles cruzados que faltaban
(v1 da la pista / v3 lee, y v3 da / v1 lee), y corre SOLO los matchups
nuevos que hacen falta para completar la matriz de 9 perfiles --
reutilizando los resultados que ya existen en results/ (no vuelve a
jugar los 49 matchups que ya corriste antes).

Requiere que ya hayas corrido antes `torneo_completo_gemini.py` (o que
existan results/resultados_simulacion.csv y results/historial_completo.jsonl).

Uso:
    python pruebas/completar_matriz.py
"""
import sys
import csv
import json
from pathlib import Path
from dataclasses import replace, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from engine.simulacion import PerfilBot, jugar_partida
from engine.gemini_bots import GeminiSpymasterBot, GeminiOperativeBot
from data.vocabulario_tablero import VOCABULARIO_TABLERO
from simular_partidas import cargar_perfiles

RESULTS_DIR = Path(__file__).parent.parent / "results"
CSV_VIEJO = RESULTS_DIR / "resultados_simulacion.csv"
JSONL_VIEJO = RESULTS_DIR / "historial_completo.jsonl"

N_JUEGOS_POR_MATCHUP = 10


def cargar_filas_viejas() -> list[dict]:
    if not CSV_VIEJO.exists():
        print(f"No encontré {CSV_VIEJO} -- ¿corriste torneo_completo_gemini.py antes?")
        sys.exit(1)
    with CSV_VIEJO.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    filas_viejas = cargar_filas_viejas()
    matchups_ya_hechos = {(r["perfil_rojo"], r["perfil_azul"]) for r in filas_viejas}
    print(f"Partidas ya existentes: {len(filas_viejas)}  "
          f"({len(matchups_ya_hechos)} matchups distintos)\n")

    print("Cargando perfiles (spaCy + sentence-transformers + Gemini)...\n")
    perfiles_locales = cargar_perfiles()
    v1 = perfiles_locales["pista1_decoder1"]
    v3 = perfiles_locales["v3_ensamble"]

    gemini_spymaster = GeminiSpymasterBot()
    gemini_operative = GeminiOperativeBot()

    def sm_gemini(vocab_pistas):
        return gemini_spymaster

    perfiles: dict[str, PerfilBot] = {
        "v1_puro": replace(v1, nombre="v1_puro"),
        "v3_puro": replace(v3, nombre="v3_puro"),
        "gemini_puro": PerfilBot(
            nombre="gemini_puro", vocab_completo=[],
            construir_spymaster=sm_gemini, operative=gemini_operative,
        ),
        "gemini_da_v1_lee": PerfilBot(
            nombre="gemini_da_v1_lee", vocab_completo=[],
            construir_spymaster=sm_gemini, operative=v1.operative,
        ),
        "v1_da_gemini_lee": PerfilBot(
            nombre="v1_da_gemini_lee", vocab_completo=v1.vocab_completo,
            construir_spymaster=v1.construir_spymaster, operative=gemini_operative,
        ),
        "gemini_da_v3_lee": PerfilBot(
            nombre="gemini_da_v3_lee", vocab_completo=[],
            construir_spymaster=sm_gemini, operative=v3.operative,
        ),
        "v3_da_gemini_lee": PerfilBot(
            nombre="v3_da_gemini_lee", vocab_completo=v3.vocab_completo,
            construir_spymaster=v3.construir_spymaster, operative=gemini_operative,
        ),
        # --- Los 2 perfiles NUEVOS que faltaban ---
        "v1_da_v3_lee": PerfilBot(
            nombre="v1_da_v3_lee", vocab_completo=v1.vocab_completo,
            construir_spymaster=v1.construir_spymaster, operative=v3.operative,
        ),
        "v3_da_v1_lee": PerfilBot(
            nombre="v3_da_v1_lee", vocab_completo=v3.vocab_completo,
            construir_spymaster=v3.construir_spymaster, operative=v1.operative,
        ),
    }

    nombres = list(perfiles.keys())
    matchups_faltantes = [
        (r, a) for r in nombres for a in nombres
        if (r, a) not in matchups_ya_hechos
    ]
    print(f"Perfiles totales: {len(nombres)}")
    print(f"Matchups faltantes a correr: {len(matchups_faltantes)} "
          f"(x {N_JUEGOS_POR_MATCHUP} partidas = {len(matchups_faltantes)*N_JUEGOS_POR_MATCHUP} partidas nuevas)\n")

    confirmar = input("¿Continuar? (s/n): ").strip().lower()
    if confirmar != "s":
        print("Cancelado.")
        return

    resultados_nuevos = []
    seed_counter = 50000
    for i, (nombre_rojo, nombre_azul) in enumerate(matchups_faltantes, 1):
        print(f"  Matchup {i}/{len(matchups_faltantes)}: {nombre_rojo} (rojo) vs {nombre_azul} (azul)...")
        for j in range(N_JUEGOS_POR_MATCHUP):
            from engine.board import Dueno
            equipo_inicial = Dueno.ROJO if j % 2 == 0 else Dueno.AZUL
            seed_counter += 1
            resultado = jugar_partida(
                vocabulario_tablero=VOCABULARIO_TABLERO,
                perfil_rojo=perfiles[nombre_rojo],
                perfil_azul=perfiles[nombre_azul],
                seed=seed_counter,
                equipo_inicial=equipo_inicial,
            )
            resultados_nuevos.append(resultado)

    print(f"\nPartidas nuevas jugadas: {len(resultados_nuevos)}\n")

    # --- Combinar con lo viejo y guardar ---
    csv_combinado = RESULTS_DIR / "resultados_matriz_completa.csv"
    with csv_combinado.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "perfil_rojo", "perfil_azul", "equipo_inicial", "ganador", "razon",
            "equipo_toco_asesino", "turnos_totales",
            "pistas_forzadas_rojo", "pistas_forzadas_azul", "seed",
        ])
        for r in filas_viejas:
            writer.writerow([
                r["perfil_rojo"], r["perfil_azul"], r["equipo_inicial"], r["ganador"], r["razon"],
                r["equipo_toco_asesino"], r["turnos_totales"],
                r["pistas_forzadas_rojo"], r["pistas_forzadas_azul"], r["seed"],
            ])
        for r in resultados_nuevos:
            writer.writerow([
                r.perfil_rojo, r.perfil_azul, r.equipo_inicial, r.ganador, r.razon,
                r.equipo_toco_asesino, r.turnos_totales,
                r.pistas_forzadas_rojo, r.pistas_forzadas_azul, r.seed,
            ])
    print(f"CSV combinado guardado en: {csv_combinado}")

    jsonl_combinado = RESULTS_DIR / "historial_matriz_completa.jsonl"
    with jsonl_combinado.open("w", encoding="utf-8") as f_out:
        if JSONL_VIEJO.exists():
            with JSONL_VIEJO.open(encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
        for r in resultados_nuevos:
            f_out.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"Historial combinado guardado en: {jsonl_combinado}")

    # --- Resumen: matriz completa 3x3 (v1 / v3 / Gemini como dador y lector) ---
    print("\n" + "=" * 60)
    print("MATRIZ COMPLETA (dador x lector)")
    print("=" * 60)

    todas_las_filas = filas_viejas + [
        {"perfil_rojo": r.perfil_rojo, "perfil_azul": r.perfil_azul, "ganador": r.ganador}
        for r in resultados_nuevos
    ]

    combos = {
        ("v1", "v1"): "v1_puro", ("v3", "v3"): "v3_puro", ("Gemini", "Gemini"): "gemini_puro",
        ("v1", "Gemini"): "v1_da_gemini_lee", ("Gemini", "v1"): "gemini_da_v1_lee",
        ("v3", "Gemini"): "v3_da_gemini_lee", ("Gemini", "v3"): "gemini_da_v3_lee",
        ("v1", "v3"): "v1_da_v3_lee", ("v3", "v1"): "v3_da_v1_lee",
    }

    print(f"{'Da / Lee':12s} {'v1':>8s} {'v3':>8s} {'Gemini':>8s}")
    for dador in ["v1", "v3", "Gemini"]:
        fila_txt = f"{dador:12s}"
        for lector in ["v1", "v3", "Gemini"]:
            perfil = combos[(dador, lector)]
            jugados = sum(1 for r in todas_las_filas if r["perfil_rojo"] == perfil or r["perfil_azul"] == perfil)
            ganados = sum(1 for r in todas_las_filas
                          if (r["perfil_rojo"] == perfil and r["ganador"] == "rojo")
                          or (r["perfil_azul"] == perfil and r["ganador"] == "azul"))
            pct = 100 * ganados / jugados if jugados else float("nan")
            fila_txt += f"{pct:7.1f}%"
        print(fila_txt)


if __name__ == "__main__":
    main()
