"""
OperativeBot híbrido: misma interfaz que OperativeBot (planear -> PlanDeJuego),
pero usando un EmbeddingHibrido en vez de un solo EmbeddingStore, así puede
decodificar pistas aunque no estén precalculadas (ver engine/embedding_hibrido.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.board import Tablero
from engine.embedding_hibrido import EmbeddingHibrido
from engine.operative import PlanDeJuego


class OperativeBotHibrido:
    def __init__(
        self,
        hibrido: EmbeddingHibrido,
        arriesgar_extra: bool = False,
        caida_maxima_para_extra: float = 0.15,
    ):
        self.hibrido = hibrido
        self.arriesgar_extra = arriesgar_extra
        self.caida_maxima_para_extra = caida_maxima_para_extra

    def planear(self, tablero: Tablero, pista_palabra: str, pista_numero: int) -> PlanDeJuego:
        visibles = tablero.palabras_visibles()

        sims, _fuente = self.hibrido.similarities_to_tablero(pista_palabra, visibles)
        ordenadas = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)

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
