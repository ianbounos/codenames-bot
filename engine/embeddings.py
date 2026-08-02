"""
Carga y manejo de embeddings de palabras (formato GloVe: texto plano,
una palabra por línea seguida de su vector).

Diseñado para poder cargar SOLO un subconjunto de palabras (para no
gastar RAM innecesaria), o el archivo completo si hace falta.
"""
from __future__ import annotations

import re
import numpy as np
from pathlib import Path
from typing import Iterable


_WORD_RE = re.compile(r"^[a-záéíóúüñ]+$", re.IGNORECASE)


class EmbeddingStore:
    """Guarda vectores de palabras y ofrece operaciones de similitud."""

    def __init__(self):
        self.words: list[str] = []
        self.word_to_idx: dict[str, int] = {}
        self.vectors: np.ndarray | None = None  # shape (N, dim)
        self._norms: np.ndarray | None = None

    @classmethod
    def from_glove_file(
        cls,
        path: str | Path,
        vocab_filter: set[str] | None = None,
        max_words: int | None = None,
    ) -> "EmbeddingStore":
        """
        Carga un archivo GloVe (texto plano).

        Args:
            path: ruta al archivo .txt de GloVe
            vocab_filter: si se pasa, solo se cargan palabras en este set
                          (útil para cargar rápido solo el vocabulario
                          del tablero + candidatos de pistas)
            max_words: límite opcional de cantidad de palabras a cargar
                       (respeta el orden del archivo, que en GloVe SBWC
                       suele estar ordenado por frecuencia descendente)
        """
        store = cls()
        vectors = []
        words = []

        path = Path(path)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) < 3:
                    continue  # probablemente la línea de header (n_words dim)
                word = parts[0]

                if vocab_filter is not None and word not in vocab_filter:
                    continue
                if not _WORD_RE.match(word):
                    continue

                try:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                except ValueError:
                    continue

                words.append(word)
                vectors.append(vec)

                if max_words is not None and len(words) >= max_words:
                    break

        store.words = words
        store.word_to_idx = {w: i for i, w in enumerate(words)}
        store.vectors = np.vstack(vectors) if vectors else np.zeros((0, 0))
        store._compute_norms()
        return store

    @classmethod
    def from_dict(cls, word_vectors: dict[str, list[float]]) -> "EmbeddingStore":
        """Crea un store a partir de un diccionario {palabra: vector}.
        Útil para tests con vectores sintéticos."""
        store = cls()
        store.words = list(word_vectors.keys())
        store.word_to_idx = {w: i for i, w in enumerate(store.words)}
        store.vectors = np.vstack(
            [np.asarray(v, dtype=np.float32) for v in word_vectors.values()]
        )
        store._compute_norms()
        return store

    @classmethod
    def from_spacy_model(
        cls,
        model_name: str = "es_core_news_md",
        palabras_incluir: Iterable[str] | None = None,
        lista_frecuencia_path: str | Path | None = None,
        max_candidatas: int = 30_000,
    ) -> "EmbeddingStore":
        """
        Carga embeddings usando un modelo de spaCy (por defecto, el de
        español "es_core_news_md") para obtener los VECTORES, combinado
        con una lista de frecuencia de palabras reales en español para
        saber QUÉ palabras probar como candidatas (el vocabulario interno
        de spaCy no sirve para esto -- solo contiene palabras ya "tocadas",
        no una lista completa del idioma).

        Args:
            model_name: nombre del modelo de spaCy ya instalado
            palabras_incluir: palabras que SIEMPRE se incluyen (típicamente
                el vocabulario del tablero), aunque no estén en la lista
                de frecuencia o no figuren entre las primeras `max_candidatas`
            lista_frecuencia_path: ruta a un archivo de texto con formato
                "palabra frecuencia" por línea, ordenado de más a menos
                frecuente (ej. hermitdave/FrequencyWords). Si no se pasa,
                solo se cargan las `palabras_incluir`.
            max_candidatas: cuántas palabras tomar de la lista de frecuencia
        """
        import spacy

        nlp = spacy.load(model_name)
        store = cls()

        words: list[str] = []
        vectors: list[np.ndarray] = []
        vistas: set[str] = set()

        def agregar(palabra: str) -> None:
            clave = palabra.lower()
            if clave in vistas:
                return
            lex = nlp.vocab[clave]
            if not lex.has_vector or lex.vector_norm == 0:
                return
            vistas.add(clave)
            words.append(clave)
            vectors.append(np.asarray(lex.vector, dtype=np.float32))

        if palabras_incluir:
            for p in palabras_incluir:
                agregar(p)

        if lista_frecuencia_path:
            path = Path(lista_frecuencia_path)
            candidatas_agregadas = 0
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if candidatas_agregadas >= max_candidatas:
                        break
                    parts = line.rstrip().split(" ")
                    if not parts:
                        continue
                    palabra = parts[0]
                    if not _WORD_RE.match(palabra):
                        continue
                    antes = len(words)
                    agregar(palabra)
                    if len(words) > antes:
                        candidatas_agregadas += 1

        store.words = words
        store.word_to_idx = {w: i for i, w in enumerate(words)}
        store.vectors = np.vstack(vectors) if vectors else np.zeros((0, 0))
        store._compute_norms()
        return store



    @classmethod
    def from_sentence_transformer(
        cls,
        words: list[str],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
    ) -> "EmbeddingStore":
        """
        Carga embeddings usando un modelo de sentence-transformers (una
        red Transformer, entrenada con un enfoque y datos totalmente
        distintos a spaCy/GloVe). Se usa como SEGUNDO modelo, para
        cruzar contra el primero y filtrar asociaciones "raras" que un
        solo modelo pueda alucinar (ver SpymasterBot con doble validación).

        A diferencia de GloVe/spaCy (que tienen un vector fijo por
        palabra), acá generamos el vector "al vuelo" codificando cada
        palabra como si fuera una mini-oración. Por eso recibimos la
        lista de palabras a codificar como argumento (no hay un archivo
        de vectores precomputado).

        Requiere: pip install sentence-transformers
        La primera vez que se usa un modelo, se descarga desde Hugging
        Face (puede tardar uno o dos minutos según la conexión).
        """
        from sentence_transformers import SentenceTransformer

        modelo = SentenceTransformer(model_name)
        store = cls()

        palabras_unicas = list(dict.fromkeys(w.lower() for w in words))  # sin duplicados, orden preservado
        vectores = modelo.encode(
            palabras_unicas,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        store.words = palabras_unicas
        store.word_to_idx = {w: i for i, w in enumerate(palabras_unicas)}
        store.vectors = vectores.astype(np.float32)
        store._compute_norms()
        return store


    def _compute_norms(self):
        if self.vectors is not None and len(self.vectors) > 0:
            self._norms = np.linalg.norm(self.vectors, axis=1)
            self._norms[self._norms == 0] = 1e-8
        else:
            self._norms = np.zeros(0)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.word_to_idx

    def __len__(self) -> int:
        return len(self.words)

    def vector(self, word: str) -> np.ndarray:
        idx = self.word_to_idx[word.lower()]
        return self.vectors[idx]

    def similarity(self, word_a: str, word_b: str) -> float:
        """Similitud coseno entre dos palabras."""
        if word_a.lower() not in self.word_to_idx or word_b.lower() not in self.word_to_idx:
            return float("-inf")
        i = self.word_to_idx[word_a.lower()]
        j = self.word_to_idx[word_b.lower()]
        dot = float(np.dot(self.vectors[i], self.vectors[j]))
        return dot / (self._norms[i] * self._norms[j])

    def similarities_to(self, word: str, targets: Iterable[str]) -> dict[str, float]:
        """Similitud de una palabra contra una lista de palabras objetivo."""
        return {t: self.similarity(word, t) for t in targets}

    def bulk_similarity_matrix(self, candidate_words: list[str], board_words: list[str]) -> np.ndarray:
        """
        Devuelve una matriz (n_candidatos x n_tablero) de similitudes coseno.
        Pensado para hacer el cálculo del spymaster de forma vectorizada
        y rápida, en vez de palabra por palabra.
        """
        cand_idx = [self.word_to_idx[w.lower()] for w in candidate_words]
        board_idx = [self.word_to_idx[w.lower()] for w in board_words]

        cand_vecs = self.vectors[cand_idx]  # (C, D)
        board_vecs = self.vectors[board_idx]  # (B, D)

        cand_norms = self._norms[cand_idx].reshape(-1, 1)  # (C, 1)
        board_norms = self._norms[board_idx].reshape(1, -1)  # (1, B)

        dot = cand_vecs @ board_vecs.T  # (C, B)
        return dot / (cand_norms * board_norms)
