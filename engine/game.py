"""
Motor de una partida de Código Secreto: maneja turnos, pistas, y
condiciones de fin de juego.

Este módulo NO decide qué pista dar ni qué palabra adivinar — eso lo
hacen los "jugadores" (humanos o bots) desde afuera. Este módulo solo
aplica las reglas: valida una pista, aplica una adivinanza, cambia el
turno cuando corresponde, y detecta cuándo termina la partida.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from engine.board import Tablero, Dueno, Carta


class Resultado(str, Enum):
    EN_CURSO = "en_curso"
    GANA_ROJO = "gana_rojo"
    GANA_AZUL = "gana_azul"


@dataclass
class Pista:
    palabra: str
    numero: int  # cuántas palabras del tablero dice conectar el spymaster


@dataclass
class Turno:
    """Registro de lo que pasó en un turno, para el historial/estadísticas."""
    equipo: Dueno
    pista: Pista
    adivinanzas: list[str] = field(default_factory=list)       # palabras tocadas, en orden
    resultados: list[Dueno] = field(default_factory=list)      # dueño real de cada una


class Partida:
    def __init__(self, tablero: Tablero, equipo_inicial: Dueno = Dueno.ROJO):
        self.tablero = tablero
        self.equipo_actual = equipo_inicial
        self.resultado = Resultado.EN_CURSO
        self.historial: list[Turno] = []
        self._turno_actual: Turno | None = None

    @property
    def equipo_rival(self) -> Dueno:
        return Dueno.AZUL if self.equipo_actual == Dueno.ROJO else Dueno.ROJO

    def dar_pista(self, palabra: str, numero: int) -> None:
        """El spymaster del equipo actual da una pista. Abre un turno nuevo."""
        if self.resultado != Resultado.EN_CURSO:
            raise RuntimeError("La partida ya terminó")
        if self._turno_actual is not None:
            raise RuntimeError("Ya hay un turno en curso, hay que cerrarlo primero")

        self._turno_actual = Turno(equipo=self.equipo_actual, pista=Pista(palabra, numero))

    def adivinar(self, palabra: str) -> Dueno:
        """
        El equipo actual toca una palabra del tablero.
        Devuelve el dueño real de esa carta (para que el llamador sepa
        qué pasó: acertó, era neutral, era del rival, o era el asesino).

        Se encarga de:
          - marcar la carta como revelada
          - cambiar el turno si corresponde (falló, o se acabaron los intentos)
          - detectar fin de partida (asesino tocado, o se completó un equipo)
        """
        if self.resultado != Resultado.EN_CURSO:
            raise RuntimeError("La partida ya terminó")
        if self._turno_actual is None:
            raise RuntimeError("Hay que dar una pista antes de adivinar")

        carta = self.tablero.revelar(palabra)
        self._turno_actual.adivinanzas.append(palabra)
        self._turno_actual.resultados.append(carta.dueno)

        # Caso 1: tocó el asesino -> pierde la partida en el acto
        if carta.dueno == Dueno.ASESINO:
            self.resultado = Resultado.GANA_AZUL if self.equipo_actual == Dueno.ROJO else Resultado.GANA_ROJO
            self._cerrar_turno()
            return carta.dueno

        # Revisar si algún equipo ya completó todas sus palabras
        if self.tablero.quedan_para_ganar(Dueno.ROJO) == 0:
            self.resultado = Resultado.GANA_ROJO
            self._cerrar_turno()
            return carta.dueno
        if self.tablero.quedan_para_ganar(Dueno.AZUL) == 0:
            self.resultado = Resultado.GANA_AZUL
            self._cerrar_turno()
            return carta.dueno

        # Caso 2: acertó una propia y todavía le quedan intentos -> sigue el turno
        intentos_permitidos = self._turno_actual.pista.numero + 1  # +1 es la regla estándar
        intentos_usados = len(self._turno_actual.adivinanzas)

        acerto_propia = carta.dueno == self.equipo_actual
        le_quedan_intentos = intentos_usados < intentos_permitidos

        if acerto_propia and le_quedan_intentos:
            return carta.dueno  # sigue jugando el mismo equipo, mismo turno

        # Caso 3: falló (neutral o rival) o se le acabaron los intentos -> pasa el turno
        self._cerrar_turno()
        return carta.dueno

    def terminar_turno_voluntariamente(self) -> None:
        """El equipo decide no seguir adivinando aunque le queden intentos."""
        if self._turno_actual is None:
            raise RuntimeError("No hay turno en curso")
        self._cerrar_turno()

    def _cerrar_turno(self) -> None:
        assert self._turno_actual is not None
        self.historial.append(self._turno_actual)
        self._turno_actual = None
        if self.resultado == Resultado.EN_CURSO:
            self.equipo_actual = self.equipo_rival

    def en_curso(self) -> bool:
        return self.resultado == Resultado.EN_CURSO

    @property
    def turno_abierto(self) -> bool:
        return self._turno_actual is not None
