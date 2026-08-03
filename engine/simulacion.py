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
class AdivinanzaRegistro:
    palabra: str
    dueno_real: str  # "rojo", "azul", "neutral", "asesino"


@dataclass
class TurnoRegistro:
    equipo: str
    pista_palabra: str
    pista_numero: int
    fue_forzada: bool
    adivinanzas: list[AdivinanzaRegistro] = field(default_factory=list)


@dataclass
class CartaRegistro:
    palabra: str
    dueno: str  # "rojo", "azul", "neutral", "asesino"


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
    anulada: bool = False
    tablero: list[CartaRegistro] = field(default_factory=list)
    historial: list[TurnoRegistro] = field(default_factory=list)


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
    verbose: bool = False,
) -> ResultadoPartida:
    tablero = Tablero.generar(
        vocabulario_tablero, n_palabras=25, equipo_inicial=equipo_inicial, seed=seed
    )
    partida = Partida(tablero, equipo_inicial=equipo_inicial)

    tablero_registro = [
        CartaRegistro(palabra=c.palabra, dueno=c.dueno.value) for c in tablero.cartas
    ]

    if verbose:
        print(f"\n  --- Partida seed={seed}: ROJO={perfil_rojo.nombre} vs AZUL={perfil_azul.nombre} "
              f"(arranca {equipo_inicial.value}) ---")
        print("  Tablero:")
        for c in tablero.cartas:
            print(f"      {c.palabra:15s} [{c.dueno.value.upper()}]")

    palabras_tablero = set(tablero.palabras_visibles())
    pistas_forzadas = {Dueno.ROJO: 0, Dueno.AZUL: 0}
    historial_registro: list[TurnoRegistro] = []

    from engine.gemini_bots import reset_contador_fallos_partida, PartidaAnuladaPorFallosRepetidos
    reset_contador_fallos_partida()

    turno = 0
    try:
        while partida.en_curso() and turno < max_turnos:
            turno += 1
            equipo = partida.equipo_actual
            perfil = perfil_rojo if equipo == Dueno.ROJO else perfil_azul

            vocab_pistas = filtrar_vocabulario_pistas(
                perfil.vocab_completo, palabras_tablero
            )
            spymaster = perfil.construir_spymaster(vocab_pistas)

            pista = spymaster.elegir_pista(tablero, equipo)
            fue_forzada = False
            if pista is None:
                # El mecanismo de "relajar los márgenes y reintentar" solo
                # tiene sentido para spymasters basados en embeddings locales
                # (SpymasterBot). Para bots basados en LLM (ej. Gemini), que
                # ya reintentan internamente, si devuelven None simplemente
                # no hay pista disponible ese turno -- se pasa el turno.
                if hasattr(spymaster, "embeddings"):
                    pista = _elegir_pista_forzada(spymaster, tablero, equipo)
                pistas_forzadas[equipo] += 1
                fue_forzada = True

            if pista is None:
                # Caso extremo: ni siquiera relajando todo encontró nada
                # (no debería pasar con un vocabulario grande). Pasamos el turno.
                partida.dar_pista("_paso_", 0)
                partida.terminar_turno_voluntariamente()
                historial_registro.append(TurnoRegistro(
                    equipo=equipo.value, pista_palabra="_paso_", pista_numero=0,
                    fue_forzada=fue_forzada, adivinanzas=[],
                ))
                continue

            partida.dar_pista(pista.palabra, pista.numero)
            plan = perfil.operative.planear(tablero, pista.palabra, pista.numero)

            registro_turno = TurnoRegistro(
                equipo=equipo.value, pista_palabra=pista.palabra,
                pista_numero=pista.numero, fue_forzada=fue_forzada,
            )

            equipo_al_iniciar = partida.equipo_actual
            for palabra in plan.orden:
                if not partida.en_curso():
                    break
                if partida.equipo_actual != equipo_al_iniciar:
                    break  # el turno ya se cerró solo (falló una adivinanza)
                dueno_real = tablero.carta_de(palabra).dueno.value
                partida.adivinar(palabra)
                registro_turno.adivinanzas.append(
                    AdivinanzaRegistro(palabra=palabra, dueno_real=dueno_real)
                )

            historial_registro.append(registro_turno)

            if verbose:
                forzada_txt = " (FORZADA/paso)" if fue_forzada else ""
                print(f"    [{equipo.value.upper()}] pista: '{pista.palabra}', {pista.numero}{forzada_txt}")
                for a in registro_turno.adivinanzas:
                    if a.dueno_real == equipo.value:
                        marca = "✅"  # acertó una palabra de SU PROPIO equipo
                    elif a.dueno_real == "asesino":
                        marca = "💀"
                    elif a.dueno_real == "neutral":
                        marca = "➖"
                    else:
                        marca = "❌"  # era del equipo rival
                    print(f"        {marca} {a.palabra} ({a.dueno_real})")

            if partida.en_curso() and partida.turno_abierto:
                partida.terminar_turno_voluntariamente()

    except PartidaAnuladaPorFallosRepetidos as e:
        if verbose:
            print(f"  --- Partida ANULADA (turno {turno}): {e} ---\n")
        return ResultadoPartida(
            perfil_rojo=perfil_rojo.nombre,
            perfil_azul=perfil_azul.nombre,
            equipo_inicial=equipo_inicial.value,
            ganador="anulada",
            razon="anulada_por_fallos_repetidos",
            equipo_toco_asesino=None,
            turnos_totales=turno,
            pistas_forzadas_rojo=pistas_forzadas[Dueno.ROJO],
            pistas_forzadas_azul=pistas_forzadas[Dueno.AZUL],
            seed=seed,
            anulada=True,
            tablero=tablero_registro,
            historial=historial_registro,
        )

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

    if verbose:
        print(f"  --- Resultado: gana {ganador_map[partida.resultado].upper()} "
              f"(razón: {razon}, turnos: {len(partida.historial)}) ---\n")

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
        tablero=tablero_registro,
        historial=historial_registro,
    )


def simular_lote(
    vocabulario_tablero: list[str],
    perfiles: dict[str, PerfilBot],
    n_juegos_por_matchup: int = 50,
    seed_base: int = 0,
    verbose: bool = False,
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
                    verbose=verbose,
                )
                resultados.append(resultado)

    return resultados
