"""
EmbeddingHibrido: combina dos fuentes de embeddings para resolver el
vector de una palabra cualquiera.

  1. Primero intenta el modelo PRINCIPAL (rápido, precalculado, ej. spaCy)
  2. Si la palabra no está ahí, cae al modelo SECUNDARIO, codificándola
     "al vuelo" con sentence-transformers (más lento, pero prácticamente
     nunca falla -- puede generar un vector para cualquier string)

Pensado sobre todo para el DECODIFICADOR interactivo: cuando un humano
escribe una pista, no queremos rechazarla solo porque es una palabra
poco común que no estaba precalculada.
"""
from __future__ import annotations

import numpy as np

from engine.embeddings import EmbeddingStore


class EmbeddingHibrido:
    def __init__(
        self,
        principal: EmbeddingStore,
        secundario: EmbeddingStore,
        modelo_secundario_raw,  # instancia de sentence_transformers.SentenceTransformer
    ):
        """
        Args:
            principal: store rápido (ej. spaCy), ya cargado
            secundario: store con las MISMAS palabras que `principal`,
                pre-codificadas con sentence-transformers (para las
                palabras del tablero y candidatas ya conocidas)
            modelo_secundario_raw: el modelo de sentence-transformers en
                sí (no el store), para poder codificar palabras nuevas
                que no estaban precalculadas ni en `principal` ni en
                `secundario`
        """
        self.principal = principal
        self.secundario = secundario
        self.modelo_secundario_raw = modelo_secundario_raw
        self._cache_vectores_nuevos: dict[str, np.ndarray] = {}

    def resolver(self, palabra: str) -> tuple[np.ndarray, str] | None:
        """
        Devuelve (vector, fuente) para una palabra, o None si no se pudo
        resolver de ninguna forma. `fuente` es "principal", "secundario"
        (ya precalculado), o "secundario_al_vuelo" (recién codificada).
        """
        clave = palabra.lower()

        if clave in self.principal:
            return self.principal.vector(clave), "principal"

        if clave in self.secundario:
            return self.secundario.vector(clave), "secundario"

        if clave in self._cache_vectores_nuevos:
            return self._cache_vectores_nuevos[clave], "secundario_al_vuelo"

        # Última instancia: codificarla ahora mismo con el modelo crudo
        try:
            vector = self.modelo_secundario_raw.encode([clave], convert_to_numpy=True)[0]
        except Exception:
            return None

        self._cache_vectores_nuevos[clave] = vector
        return vector, "secundario_al_vuelo"

    def similarity(self, palabra_a: str, palabra_b: str) -> float | None:
        """
        Similitud entre dos palabras, resolviendo cada una por separado.

        IMPORTANTE: si las dos palabras se resuelven en fuentes distintas
        (ej. una en el modelo principal y otra en el secundario), la
        comparación puede no ser del todo consistente, porque son
        espacios vectoriales diferentes. Por eso, si la pista (palabra_a)
        cae al modelo secundario, conviene comparar TODO en el modelo
        secundario (ver `similarities_to_tablero` más abajo, que ya hace
        esto bien).
        """
        ra = self.resolver(palabra_a)
        rb = self.resolver(palabra_b)
        if ra is None or rb is None:
            return None
        va, _ = ra
        vb, _ = rb
        return float(
            np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        )

    def similarities_to_tablero(
        self, pista: str, palabras_tablero: list[str]
    ) -> tuple[dict[str, float], str]:
        """
        Calcula la similitud de una pista contra todas las palabras del
        tablero, siendo CONSISTENTE con el espacio vectorial: si la pista
        está en el modelo principal, compara todo ahí; si no, usa el
        modelo secundario para la pista Y para las palabras del tablero
        (que sabemos que siempre están precalculadas en el secundario,
        porque el tablero sale del vocabulario fijo del juego).

        Devuelve (diccionario de similitudes, qué fuente se usó).
        """
        clave = pista.lower()

        if clave in self.principal:
            sims = self.principal.similarities_to(clave, palabras_tablero)
            return sims, "principal"

        # Cae al modelo secundario para TODO (consistencia del espacio vectorial)
        resultado = self.resolver(clave)
        if resultado is None:
            return {}, "no_resuelta"
        vector_pista, fuente = resultado

        sims = {}
        for palabra in palabras_tablero:
            if palabra.lower() not in self.secundario:
                continue
            v = self.secundario.vector(palabra)
            sim = float(
                np.dot(vector_pista, v)
                / (np.linalg.norm(vector_pista) * np.linalg.norm(v) + 1e-8)
            )
            sims[palabra] = sim

        return sims, fuente
