"""
Representación del tablero de Código Secreto.

El tablero tiene N palabras, cada una con un "dueño" oculto:
- ROJO / AZUL: pertenecen a un equipo
- NEUTRAL: no pertenece a nadie
- ASESINO: si se toca, ese equipo pierde la partida inmediatamente

Solo los espías maestros conocen esta asignación al empezar.
Los agentes de campo solo ven la palabra y si ya fue revelada o no.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class Dueno(str, Enum):
    ROJO = "rojo"
    AZUL = "azul"
    NEUTRAL = "neutral"
    ASESINO = "asesino"


@dataclass
class Carta:
    palabra: str
    dueno: Dueno
    revelada: bool = False


@dataclass
class Tablero:
    cartas: list[Carta] = field(default_factory=list)

    @classmethod
    def generar(
        cls,
        vocabulario: list[str],
        n_palabras: int = 25,
        n_asesino: int = 1,
        equipo_inicial: Dueno = Dueno.ROJO,
        seed: int | None = None,
    ) -> "Tablero":
        """
        Genera un tablero nuevo al azar.

        La distribución estándar de Codenames para una grilla de 25 (5x5) es:
          - 9 para el equipo que empieza (tiene una palabra más de ventaja)
          - 8 para el otro equipo
          - 7 neutrales
          - 1 asesino
        Para tableros de otro tamaño, escalamos proporcionalmente.
        """
        if seed is not None:
            random.seed(seed)

        if len(vocabulario) < n_palabras:
            raise ValueError(
                f"El vocabulario tiene {len(vocabulario)} palabras, "
                f"pero se necesitan {n_palabras}."
            )

        palabras = random.sample(vocabulario, n_palabras)

        # Distribución proporcional a la estándar de 25 cartas (9/8/7/1)
        restantes = n_palabras - n_asesino
        n_equipo_inicial = round(restantes * (9 / 24))
        n_equipo_rival = round(restantes * (8 / 24))
        n_neutral = restantes - n_equipo_inicial - n_equipo_rival

        equipo_rival = Dueno.AZUL if equipo_inicial == Dueno.ROJO else Dueno.ROJO

        duenos = (
            [equipo_inicial] * n_equipo_inicial
            + [equipo_rival] * n_equipo_rival
            + [Dueno.NEUTRAL] * n_neutral
            + [Dueno.ASESINO] * n_asesino
        )
        random.shuffle(duenos)

        cartas = [Carta(palabra=p, dueno=d) for p, d in zip(palabras, duenos)]
        return cls(cartas=cartas)

    def palabras_de(self, dueno: Dueno, solo_no_reveladas: bool = True) -> list[str]:
        return [
            c.palabra for c in self.cartas
            if c.dueno == dueno and (not solo_no_reveladas or not c.revelada)
        ]

    def palabras_visibles(self) -> list[str]:
        """Palabras que todavía se pueden elegir (no reveladas)."""
        return [c.palabra for c in self.cartas if not c.revelada]

    def carta_de(self, palabra: str) -> Carta:
        for c in self.cartas:
            if c.palabra.lower() == palabra.lower():
                return c
        raise ValueError(f"'{palabra}' no está en el tablero")

    def revelar(self, palabra: str) -> Carta:
        carta = self.carta_de(palabra)
        carta.revelada = True
        return carta

    def quedan_para_ganar(self, equipo: Dueno) -> int:
        return len(self.palabras_de(equipo, solo_no_reveladas=True))

    def resumen_texto(self, mostrar_duenos: bool = False) -> str:
        """Representación en texto del tablero, útil para debug en consola."""
        lineas = []
        for c in self.cartas:
            estado = "revelada" if c.revelada else "oculta"
            dueno_txt = f" [{c.dueno.value}]" if (mostrar_duenos or c.revelada) else ""
            lineas.append(f"  {c.palabra:15s} ({estado}){dueno_txt}")
        return "\n".join(lineas)
