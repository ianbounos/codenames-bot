"""
Motor de simulación de partidas completas, jugadas por bots de punta a
punta (sin intervención humana), para poder correr cientos o miles de
partidas y sacar estadísticas.

Conceptos clave:

  PerfilBot: una combinación de (cómo da pistas, cómo las decodifica).
      Ej: "v1" = Spymaster solo-spaCy + Operative solo-spaCy
          "v2" = Spymaster con doble validación + Operative híbrido

  jugar_partida: corre UNA partida completa entre dos perfiles (uno por
      equipo), turno a turno, hasta que termina, y devuelve un resumen
      con todo lo que nos interesa medir (ganador, por qué terminó,
      cuántos turnos, etc.)

  simular_lote: corre un torneo todos-contra-todos entre varios perfiles,
      muchas partidas por cada combinación, y arma una tabla de resultados.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

from engine.board import Tablero, Dueno
from engine.game import Partida, Resultado
from engine.spymaster import SpymasterBot
from engine.vocab_utils import filtrar_vocabulario_pistas


class OperativeLike(Protocol):
    def planear(self, tablero: Tablero, pista_palabra: str, pista_numero: int):
        ...


@dataclass
class PerfilBot:
    nombre: str
    vocab_completo: list[str]  # todas las palabras disponibles para este perfil (antes de filtrar por tablero)
    # recibe el vocabulario de candidatas YA FILTRADO para este tablero
    # específico, y devuelve un SpymasterBot listo para usar
    construir_spymaster: Callable[[list[str]], SpymasterBot]
    operative: OperativeLike


@dataclass
class ResultadoPartida:
    perfil_rojo: str
    perfil_azul: str
    equipo_inicial: str
    ganador: str
    razon: str  # "asesino" o "completado"
    equipo_toco_asesino: str | None
    turnos_totales: int
    pistas_forzadas_rojo: int
    pistas_forzadas_azul: int
    seed: int


def _elegir_pista_forzada(spymaster: SpymasterBot, tablero: Tablero, equipo: Dueno):
    """
    Si el spymaster normal no encuentra ninguna pista que pase sus propios
    márgenes de seguridad, en vez de trabar la simulación, probamos de
    nuevo con umbrales muy relajados (acepta casi cualquier cosa) y
    apuntando a una sola palabra -- la opción más conservadora posible.
    Se cuenta aparte en las estadísticas como "pista forzada", porque es
    una señal de que el bot estaba en una posición difícil.
    """
    sm_forzado = SpymasterBot(
        embeddings=spymaster.embeddings,
        vocabulario_pistas=spymaster.vocabulario_pistas,
        beta_asesino=-10.0,
        beta_resto=-10.0,
        max_n=1,
        embeddings_secundario=spymaster.embeddings_secundario,
        beta_asesino_secundario=-10.0 if spymaster.embeddings_secundario else None,
        beta_resto_secundario=-10.0 if spymaster.embeddings_secundario else None,
    )
    return sm_forzado.elegir_pista(tablero, equipo)


def jugar_partida(
    vocabulario_tablero: list[str],
    perfil_rojo: PerfilBot,
    perfil_azul: PerfilBot,
    seed: int,
    equipo_inicial: Dueno = Dueno.ROJO,
    max_turnos: int = 40,
) -> ResultadoPartida:
    tablero = Tablero.generar(
        vocabulario_tablero, n_palabras=25, equipo_inicial=equipo_inicial, seed=seed
    )
    partida = Partida(tablero, equipo_inicial=equipo_inicial)

    palabras_tablero = set(tablero.palabras_visibles())
    pistas_forzadas = {Dueno.ROJO: 0, Dueno.AZUL: 0}

    turno = 0
    while partida.en_curso() and turno < max_turnos:
        turno += 1
        equipo = partida.equipo_actual
        perfil = perfil_rojo if equipo == Dueno.ROJO else perfil_azul

        vocab_pistas = filtrar_vocabulario_pistas(
            perfil.vocab_completo, palabras_tablero
        )
        spymaster = perfil.construir_spymaster(vocab_pistas)

        pista = spymaster.elegir_pista(tablero, equipo)
        if pista is None:
            pista = _elegir_pista_forzada(spymaster, tablero, equipo)
            pistas_forzadas[equipo] += 1

        if pista is None:
            # Caso extremo: ni siquiera relajando todo encontró nada
            # (no debería pasar con un vocabulario grande). Pasamos el turno.
            partida.dar_pista("_paso_", 0)
            partida.terminar_turno_voluntariamente()
            continue

        partida.dar_pista(pista.palabra, pista.numero)
        plan = perfil.operative.planear(tablero, pista.palabra, pista.numero)

        equipo_al_iniciar = partida.equipo_actual
        for palabra in plan.orden:
            if not partida.en_curso():
                break
            if partida.equipo_actual != equipo_al_iniciar:
                break  # el turno ya se cerró solo (falló una adivinanza)
            partida.adivinar(palabra)

        if partida.en_curso() and partida.turno_abierto:
            partida.terminar_turno_voluntariamente()

    # Determinar la razón del resultado mirando el último turno jugado
    razon = "sin_terminar"
    equipo_toco_asesino = None
    if partida.historial:
        ultimo_turno = partida.historial[-1]
        if Dueno.ASESINO in ultimo_turno.resultados:
            razon = "asesino"
            equipo_toco_asesino = ultimo_turno.equipo.value
        elif partida.resultado != Resultado.EN_CURSO:
            razon = "completado"

    ganador_map = {
        Resultado.GANA_ROJO: "rojo",
        Resultado.GANA_AZUL: "azul",
        Resultado.EN_CURSO: "empate_por_limite_turnos",
    }

    return ResultadoPartida(
        perfil_rojo=perfil_rojo.nombre,
        perfil_azul=perfil_azul.nombre,
        equipo_inicial=equipo_inicial.value,
        ganador=ganador_map[partida.resultado],
        razon=razon,
        equipo_toco_asesino=equipo_toco_asesino,
        turnos_totales=len(partida.historial),
        pistas_forzadas_rojo=pistas_forzadas[Dueno.ROJO],
        pistas_forzadas_azul=pistas_forzadas[Dueno.AZUL],
        seed=seed,
    )


def simular_lote(
    vocabulario_tablero: list[str],
    perfiles: dict[str, PerfilBot],
    n_juegos_por_matchup: int = 50,
    seed_base: int = 0,
) -> list[ResultadoPartida]:
    """
    Corre un torneo todos-contra-todos: cada perfil contra cada perfil
    (incluido contra sí mismo), alternando quién arranca, con distintas
    semillas de tablero para tener variedad.
    """
    resultados: list[ResultadoPartida] = []
    nombres = list(perfiles.keys())
    seed_counter = seed_base

    total_matchups = len(nombres) * len(nombres)
    matchup_actual = 0

    for nombre_rojo in nombres:
        for nombre_azul in nombres:
            matchup_actual += 1
            print(f"  Matchup {matchup_actual}/{total_matchups}: "
                  f"{nombre_rojo} (rojo) vs {nombre_azul} (azul)...")

            for i in range(n_juegos_por_matchup):
                equipo_inicial = Dueno.ROJO if i % 2 == 0 else Dueno.AZUL
                seed_counter += 1
                resultado = jugar_partida(
                    vocabulario_tablero=vocabulario_tablero,
                    perfil_rojo=perfiles[nombre_rojo],
                    perfil_azul=perfiles[nombre_azul],
                    seed=seed_counter,
                    equipo_inicial=equipo_inicial,
                )
                resultados.append(resultado)

    return resultados
