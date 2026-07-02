"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Matching

File        : semantic.py

Description
-----------
Tier-2 matcher: catches paraphrase variants that exact
string matching misses (e.g. "Apple Inc" vs "Apple").

Embeddings are computed once per distinct normalized
string during offline cache building (see belief_cache.py)
and persisted to an embedding cache file, so online
matching never runs the embedding model - it only does
numpy dot products against the cached vectors.
---------------------------------------------------------
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


class SemanticMatcher:
    """
    Wraps a sentence-embedding model (default: MiniLM) plus
    an optional precomputed embedding cache.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model=None,
        embedding_cache: Optional[Dict[str, np.ndarray]] = None,
        device: str | None = None,
    ) -> None:

        self._model = model

        self._device = device

        self.embedding_cache: Dict[str, np.ndarray] = (
            embedding_cache or {}
        )

    # --------------------------------------------------

    def _ensure_model(self):

        if self._model is None:

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.MODEL_NAME,
                device=self._device,
            )

        return self._model

    # --------------------------------------------------

    def embed_batch(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Embed a batch of strings, running the model only
        for the model (used offline, during cache building).
        """

        model = self._ensure_model()

        vectors = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vectors.astype(np.float32)

    # --------------------------------------------------

    def build_cache(
        self,
        strings: Iterable[str],
    ) -> Dict[str, np.ndarray]:
        """
        Embed every distinct string once and store the
        result in ``self.embedding_cache``.
        """

        distinct = sorted({s for s in strings if s})

        if not distinct:
            return self.embedding_cache

        vectors = self.embed_batch(distinct)

        for string, vector in zip(distinct, vectors):
            self.embedding_cache[string] = vector

        return self.embedding_cache

    # --------------------------------------------------

    def similarity(
        self,
        a: str,
        b: str,
    ) -> float:
        """
        Cosine similarity between two strings using cached
        embeddings when available, falling back to a live
        model call for strings not seen during cache
        building (e.g. ad-hoc/interactive use).
        """

        if a == b:
            return 1.0

        vec_a = self.embedding_cache.get(a)
        vec_b = self.embedding_cache.get(b)

        if vec_a is None or vec_b is None:
            embedded = self.embed_batch([a, b])
            vec_a = vec_a if vec_a is not None else embedded[0]
            vec_b = vec_b if vec_b is not None else embedded[1]

        # Embeddings are pre-normalized, so dot product == cosine sim.
        return float(np.dot(vec_a, vec_b))

    # --------------------------------------------------

    def save(self, file_path: str | Path) -> None:

        file_path = Path(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(
                self.embedding_cache,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    # --------------------------------------------------

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        device: str | None = None,
    ) -> "SemanticMatcher":

        file_path = Path(file_path)

        with open(file_path, "rb") as f:
            embedding_cache = pickle.load(f)

        return cls(embedding_cache=embedding_cache, device=device)
