"""Retrieval over the corpus chunk matrix.

Scores every chunk against the query embedding using cosine similarity and
returns the top-k chunks. Query and chunk vectors are expected to already be
L2-normalized, so the cosine similarity is a simple dot product.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.corpus import Chunk


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query row and every corpus row."""
    q = np.asarray(query_vec).reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    return (normalized @ q.T).ravel()


def retrieve_top_k(
    chunks: Sequence[Chunk],
    query_vec: np.ndarray,
    matrix: np.ndarray,
    top_k: int,
) -> List[Dict[str, object]]:
    """Return the top-k chunks with their cosine scores, high to low."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    scores = cosine_scores(query_vec, matrix)
    order = np.argsort(scores)[::-1][:top_k]
    return [
        {"chunk": chunks[idx], "score": float(scores[idx]), "rank": rank + 1}
        for rank, idx in enumerate(order)
    ]


def unique_titles(ranked: Sequence[Dict[str, object]]) -> List[str]:
    """Document titles of the retrieved chunks, deduplicated, order preserved."""
    seen: set[str] = set()
    titles: List[str] = []
    for item in ranked:
        title = item["chunk"].title
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return titles