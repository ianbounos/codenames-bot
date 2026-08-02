"""
Genera un dashboard visual (PNG) con las estadísticas de la simulación.
Usa una paleta inspirada en Codenames: rojo/azul para los equipos, gris
para neutral, negro para el asesino.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no necesita pantalla, solo generar archivos
import matplotlib.pyplot as plt
import numpy as np

ROJO = "#c0392b"
AZUL = "#2980b9"
GRIS = "#95a5a6"
NEGRO = "#1c1c1c"
FONDO = "#f4f1ea"  # como el color de las cartas de Codenames


def generar_dashboard(resultados: list, nombres_perfiles: list[str], output_dir: Path) -> Path:
    fig = plt.figure(figsize=(18, 11), facecolor=FONDO)
    fig.suptitle(
        "Dashboard de Simulación — Codenames Bot",
        fontsize=20, fontweight="bold", color=NEGRO, y=0.98,
    )
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    _panel_winrate_por_perfil(fig.add_subplot(gs[0, 0]), resultados, nombres_perfiles)
    _panel_razon_perdida(fig.add_subplot(gs[0, 1]), resultados, nombres_perfiles)
    _panel_ventaja_inicial(fig.add_subplot(gs[0, 2]), resultados)
    _panel_heatmap_matchups(fig.add_subplot(gs[1, 0]), resultados, nombres_perfiles)
    _panel_distribucion_turnos(fig.add_subplot(gs[1, 1]), resultados)
    _panel_pistas_forzadas(fig.add_subplot(gs[1, 2]), resultados, nombres_perfiles)

    out_path = output_dir / "dashboard_simulacion.png"
    fig.savefig(out_path, dpi=130, facecolor=FONDO, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard guardado en: {out_path}")
    return out_path


def _partidas_de_perfil(resultados, nombre_perfil):
    """Devuelve (juegos_jugados, juegos_ganados) para un perfil, contando
    tanto cuando jugó de rojo como de azul."""
    jugados, ganados = 0, 0
    for r in resultados:
        if r.perfil_rojo == nombre_perfil:
            jugados += 1
            if r.ganador == "rojo":
                ganados += 1
        if r.perfil_azul == nombre_perfil:
            jugados += 1
            if r.ganador == "azul":
                ganados += 1
    return jugados, ganados


def _panel_winrate_por_perfil(ax, resultados, nombres_perfiles):
    tasas = []
    for nombre in nombres_perfiles:
        jugados, ganados = _partidas_de_perfil(resultados, nombre)
        tasas.append(100 * ganados / jugados if jugados else 0)

    colores = plt.cm.viridis(np.linspace(0.2, 0.8, len(nombres_perfiles)))
    barras = ax.bar(nombres_perfiles, tasas, color=colores, edgecolor=NEGRO, linewidth=1.2)
    ax.axhline(50, color=NEGRO, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% de partidas ganadas")
    ax.set_title("¿Qué perfil gana más?", fontweight="bold")
    ax.set_facecolor(FONDO)
    for b, t in zip(barras, tasas):
        ax.text(b.get_x() + b.get_width()/2, t + 2, f"{t:.1f}%", ha="center", fontweight="bold")


def _panel_razon_perdida(ax, resultados, nombres_perfiles):
    conteo = {n: Counter() for n in nombres_perfiles}
    for r in resultados:
        if r.razon not in ("asesino", "completado"):
            continue
        if r.razon == "asesino":
            perdedor = r.equipo_toco_asesino  # 'rojo' o 'azul' (el que tocó, pierde)
            nombre_perdedor = r.perfil_rojo if perdedor == "rojo" else r.perfil_azul
            conteo[nombre_perdedor]["asesino"] += 1
        else:  # completado: el que NO ganó, perdió por quedarse atrás
            nombre_perdedor = r.perfil_azul if r.ganador == "rojo" else r.perfil_rojo
            conteo[nombre_perdedor]["se_quedo_atras"] += 1

    x = np.arange(len(nombres_perfiles))
    asesino_vals = [conteo[n]["asesino"] for n in nombres_perfiles]
    atras_vals = [conteo[n]["se_quedo_atras"] for n in nombres_perfiles]

    ax.bar(x, asesino_vals, color=NEGRO, label="Tocó el asesino", edgecolor=FONDO)
    ax.bar(x, atras_vals, bottom=asesino_vals, color=GRIS, label="Se quedó atrás", edgecolor=FONDO)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres_perfiles)
    ax.set_ylabel("Cantidad de derrotas")
    ax.set_title("¿Por qué pierde cada perfil?", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_facecolor(FONDO)


def _panel_ventaja_inicial(ax, resultados):
    gano_iniciando = sum(1 for r in resultados if r.ganador == r.equipo_inicial)
    gano_segundo = len(resultados) - gano_iniciando
    valores = [gano_iniciando, gano_segundo]
    etiquetas = ["Ganó quien\narrancó", "Ganó el\nsegundo equipo"]

    colores = ["#e67e22", GRIS]
    wedges, _texts, autotexts = ax.pie(
        valores, labels=etiquetas, colors=colores, autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(edgecolor=FONDO, linewidth=2),
        textprops=dict(fontweight="bold"),
    )
    ax.set_title("¿Da ventaja arrancar primero?", fontweight="bold")


def _panel_heatmap_matchups(ax, resultados, nombres_perfiles):
    n = len(nombres_perfiles)
    matriz = np.zeros((n, n))
    conteo = np.zeros((n, n))

    idx = {nombre: i for i, nombre in enumerate(nombres_perfiles)}
    for r in resultados:
        i, j = idx[r.perfil_rojo], idx[r.perfil_azul]
        conteo[i, j] += 1
        if r.ganador == "rojo":
            matriz[i, j] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        tasa = np.where(conteo > 0, 100 * matriz / conteo, np.nan)

    im = ax.imshow(tasa, cmap="RdBu_r", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(nombres_perfiles)
    ax.set_yticklabels(nombres_perfiles)
    ax.set_xlabel("Perfil AZUL")
    ax.set_ylabel("Perfil ROJO")
    ax.set_title("% de victorias de ROJO\nsegún el matchup", fontweight="bold")

    for i in range(n):
        for j in range(n):
            if not np.isnan(tasa[i, j]):
                ax.text(j, i, f"{tasa[i,j]:.0f}%", ha="center", va="center",
                         color="white" if tasa[i, j] < 30 or tasa[i, j] > 70 else "black",
                         fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _panel_distribucion_turnos(ax, resultados):
    turnos = [r.turnos_totales for r in resultados]
    ax.hist(turnos, bins=range(min(turnos), max(turnos) + 2), color="#8e44ad",
             edgecolor=FONDO, alpha=0.85)
    ax.axvline(np.mean(turnos), color=NEGRO, linestyle="--", linewidth=1.5,
               label=f"Promedio: {np.mean(turnos):.1f}")
    ax.set_xlabel("Turnos hasta terminar la partida")
    ax.set_ylabel("Cantidad de partidas")
    ax.set_title("¿Cuánto duran las partidas?", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor(FONDO)


def _panel_pistas_forzadas(ax, resultados, nombres_perfiles):
    """Promedio de veces por partida que un perfil no encontró ninguna
    pista segura y tuvo que forzar una (señal de tablero difícil)."""
    suma = {n: 0 for n in nombres_perfiles}
    juegos = {n: 0 for n in nombres_perfiles}

    for r in resultados:
        suma[r.perfil_rojo] += r.pistas_forzadas_rojo
        juegos[r.perfil_rojo] += 1
        suma[r.perfil_azul] += r.pistas_forzadas_azul
        juegos[r.perfil_azul] += 1

    promedios = [suma[n] / juegos[n] if juegos[n] else 0 for n in nombres_perfiles]

    colores = plt.cm.magma(np.linspace(0.3, 0.7, len(nombres_perfiles)))
    ax.bar(nombres_perfiles, promedios, color=colores, edgecolor=NEGRO, linewidth=1.2)
    ax.set_ylabel("Pistas forzadas promedio / partida")
    ax.set_title("¿Qué tan seguido se queda\nsin pistas seguras?", fontweight="bold")
    ax.set_facecolor(FONDO)
