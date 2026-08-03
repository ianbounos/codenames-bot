"""
Función compartida para guardar y reportar resultados de una simulación:
CSV resumido, historial completo en JSONL, resumen en consola, y dashboard
visual. La usan tanto simular_partidas.py (torneo local) como
simular_partidas_con_gemini.py (torneo controlado con la API de Gemini).
"""
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path


def guardar_y_reportar(resultados: list, nombres_perfiles: list[str], resultados_dir: Path) -> None:
    resultados_dir.mkdir(exist_ok=True)

    csv_path = resultados_dir / "resultados_simulacion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "perfil_rojo", "perfil_azul", "equipo_inicial", "ganador", "razon",
            "equipo_toco_asesino", "turnos_totales",
            "pistas_forzadas_rojo", "pistas_forzadas_azul", "seed",
        ])
        for r in resultados:
            writer.writerow([
                r.perfil_rojo, r.perfil_azul, r.equipo_inicial, r.ganador, r.razon,
                r.equipo_toco_asesino, r.turnos_totales,
                r.pistas_forzadas_rojo, r.pistas_forzadas_azul, r.seed,
            ])
    print(f"CSV guardado en: {csv_path}\n")

    jsonl_path = resultados_dir / "historial_completo.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"Historial completo guardado en: {jsonl_path}\n")

    print("=" * 60)
    print("RESUMEN RÁPIDO")
    print("=" * 60)

    razones = Counter(r.razon for r in resultados)
    print(f"\nCómo terminaron las partidas: {dict(razones)}")

    anuladas = sum(1 for r in resultados if getattr(r, "anulada", False))
    if anuladas > 0:
        print(f"\n⚠️  Partidas ANULADAS por fallos repetidos de Gemini: {anuladas}/{len(resultados)}")
    else:
        print(f"\nPartidas anuladas: 0 (ninguna)")

    ganadores_por_asesino = Counter(
        r.equipo_toco_asesino for r in resultados if r.razon == "asesino"
    )
    print(f"Quién tocó el asesino (y por lo tanto perdió): {dict(ganadores_por_asesino)}")

    gano_iniciando = sum(1 for r in resultados if r.ganador == r.equipo_inicial)
    print(f"\nGanó el equipo que arrancó: {gano_iniciando}/{len(resultados)} "
          f"({100*gano_iniciando/len(resultados):.1f}%)")

    print(f"\nTurnos promedio por partida: "
          f"{sum(r.turnos_totales for r in resultados)/len(resultados):.1f}")

    print("\nGenerando dashboard visual...")
    from generar_dashboard import generar_dashboard
    generar_dashboard(resultados, nombres_perfiles, resultados_dir)
