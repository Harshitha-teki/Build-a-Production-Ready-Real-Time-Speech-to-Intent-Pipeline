"""Embeddings.

The harness pins ``pipeline_a_embedder`` and ``pipeline_b_embedder`` in
``config/eval_pins.json``. The default pinned embedder, ``tfidf``, is a
deterministic, dependency-light sparse embedder that makes evaluation runs
fully reproducible offline. Any embedder exposing a ``fit``/``transform``
sklearn-style API can be dropped in by changing the pins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.corpus import Chunk


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: Sequence[str]) -> "Embedder":
        ...

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        ...


class TfidfEmbedder(Embedder):
    """L2-normalized word (1,2)-gram TF-IDF sparse embedder."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.transform called before fit")
        return self._vectorizer.transform(texts).toarray()


def build_embedder(name: str) -> Embedder:
    """Instantiate the embedder referenced by the pins file."""
    if name == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unsupported embedder '{name}'. Known embedders: tfidf")


def embed_corpus(embedder: Embedder, chunks: Sequence[Chunk]) -> np.ndarray:
    """Fit the embedder on the corpus sentences and return the chunk matrix."""
    embedder.fit([chunk.sentence for chunk in chunks])
    return embedder.transform([chunk.sentence for chunk in chunks])