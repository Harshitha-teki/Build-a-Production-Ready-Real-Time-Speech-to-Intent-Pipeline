"""Vector index abstraction.

Two backends are available:

* :class:`InMemoryIndex` — cosine search over the in-process chunk matrix.
  Used by default so evaluations are fully reproducible offline.
* :class:`QdrantIndex` — dense-vector search against a running Qdrant
  instance (``docker-compose.yml``). Enabled by setting ``QDRANT_URL`` /
  ``QDRANT_HOST``.

Both backends return chunks ranked high-to-low, which is the only interface
the retrieval stage depends on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.corpus import Chunk
from src.embeddings import Embedder


class Index(ABC):
    @abstractmethod
    def search_top_k(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, object]]:
        ...


class InMemoryIndex(Index):
    def __init__(self, chunks: Sequence[Chunk], matrix: np.ndarray) -> None:
        self._chunks = chunks
        self._matrix = np.asarray(matrix, dtype=float)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        self._normalized = self._matrix / np.maximum(norms, 1e-12)

    def search_top_k(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, object]]:
        q = np.asarray(query_vec).reshape(1, -1)
        scores = (self._normalized @ q.T).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk": self._chunks[idx], "score": float(scores[idx]), "rank": rank + 1}
            for rank, idx in enumerate(order)
        ]


class QdrantIndex(Index):
    """Qdrant backed index using the http JSON API."""

    def __init__(
        self,
        base_url: str,
        collection: str,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        vector_size: int,
    ) -> None:
        import requests

        self._requests = requests
        self._base_url = base_url.rstrip("/")
        self._collection = collection
        self._chunks = chunks
        self._embedder = embedder
        self._vector_size = vector_size
        self._ensure_collection()
        self._upsert()

    def _ensure_collection(self) -> None:
        url = f"{self._base_url}/collections/{self._collection}"
        existing = self._requests.get(url, timeout=30)
        if existing.status_code != 200:
            payload = {"vectors": {"size": self._vector_size, "distance": "Cosine"}}
            self._requests.put(url, json=payload, timeout=30).raise_for_status()

    def _upsert(self) -> None:
        url = f"{self._base_url}/collections/{self._collection}/points"
        vectors = self._embedder.transform([chunk.sentence for chunk in self._chunks])
        points = [
            {
                "id": chunk.chunk_id,
                "vector": vectors[i].tolist(),
                "payload": {"title": chunk.title, "sentence": chunk.sentence},
            }
            for i, chunk in enumerate(self._chunks)
        ]
        for batch_start in range(0, len(points), 64):
            batch = points[batch_start : batch_start + 64]
            self._requests.put(url, json={"points": batch}, timeout=60).raise_for_status()

    def search_top_k(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, object]]:
        url = f"{self._base_url}/collections/{self._collection}/points/search"
        payload = {"vector": query_vec.tolist(), "limit": top_k, "with_payload": True}
        response = self._requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        hits = response.json().get("result", [])
        out: List[Dict[str, object]] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.get("payload", {})
            out.append(
                {
                    "chunk": Chunk(
                        title=str(payload.get("title", "")),
                        sentence=str(payload.get("sentence", "")),
                        chunk_id=int(hit["id"]),
                    ),
                    "score": None if hit.get("score") is None else float(hit["score"]),
                    "rank": rank,
                    "vector_source": "qdrant",
                }
            )
        return out


def build_index(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    matrix: Optional[np.ndarray] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Index:
    """Build the configured index.

    When ``QDRANT_URL`` (or ``QDRANT_HOST``) is set, the harness indexes the
    knowledge base into Qdrant and answers retrieval queries against it;
    otherwise the in-process matrix index is used for offline, reproducible
    runs (this is what produced the committed ``results/`` artifacts).
    """
    settings = settings or {}
    qdrant_url = settings.get("qdrant_url") or ""
    if not qdrant_url:
        return InMemoryIndex(chunks, matrix)
    return QdrantIndex(
        base_url=qdrant_url,
        collection=settings.get("qdrant_collection", "knowledge_base"),
        chunks=chunks,
        embedder=embedder,
        vector_size=int(matrix.shape[1]) if matrix is not None else 0,
    )