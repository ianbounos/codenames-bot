"""
Operative bot (agente de campo): dada una pista (palabra + número) y el
tablero visible, decide qué palabras tocar, en qué orden, y cuándo parar.

Sin memoria entre turnos (versión acordada para la v1): cada turno se
decide solo con la información de ESA pista y el tablero actual. No hay
registro de pistas anteriores, propias ni del rival.

Comportamiento por defecto: conservador. Se detiene apenas cumple el
número exacto pedido por el spymaster, sin arriesgar el intento extra
(+1) que dan las reglas, salvo que se le indique lo contrario con
`arriesgar_extra=True`.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.embeddings import EmbeddingStore
from engine.board import Tablero


@dataclass
class PlanDeJuego:
    """El orden en que el operative planea tocar palabras, con sus similitudes.
    Se calcula todo de una vez al recibir la pista; el motor de partida
    (Partida.adivinar) se va aplicando de a una."""
    orden: list[str]           # palabras del tablero, ordenadas de más a menos relacionadas
    similitudes: list[float]   # similitud de cada una con la pista, mismo orden


class OperativeBot:
    def __init__(
        self,
        embeddings: EmbeddingStore,
        arriesgar_extra: bool = False,
        caida_maxima_para_extra: float = 0.15,
    ):
        """
        Args:
            embeddings: store de embeddings ya cargado
            arriesgar_extra: si es True, además de las N palabras pedidas,
                intenta también el intento extra (+1) que dan las reglas,
                siempre que la caída de similitud no sea muy grande.
            caida_maxima_para_extra: qué tan grande puede ser la caída de
                similitud entre la última palabra "segura" (dentro de N) y
                la candidata extra, para animarse a intentarla igual.
        """
        self.embeddings = embeddings
        self.arriesgar_extra = arriesgar_extra
        self.caida_maxima_para_extra = caida_maxima_para_extra

    def planear(self, tablero: Tablero, pista_palabra: str, pista_numero: int) -> PlanDeJuego:
        visibles = tablero.palabras_visibles()

        sims = self.embeddings.similarities_to(pista_palabra, visibles)
        ordenadas = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)

        # cuántas palabras vamos a intentar tocar como máximo
        limite = pista_numero
        if self.arriesgar_extra and len(ordenadas) > pista_numero:
            ultima_segura = ordenadas[pista_numero - 1][1]
            candidata_extra = ordenadas[pista_numero][1]
            caida = ultima_segura - candidata_extra
            if caida <= self.caida_maxima_para_extra:
                limite = pista_numero + 1

        seleccion = ordenadas[:limite]
        return PlanDeJuego(
            orden=[w for w, _ in seleccion],
            similitudes=[s for _, s in seleccion],
        )
