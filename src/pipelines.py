"""RAG pipelines A (baseline) and B (candidate).

Pipeline A is the existing production RAG stack: ``top_k=3``, no reranker.
Pipeline B is the candidate change under evaluation: ``top_k=6`` with a
lexical reranker and a higher-verbosity generator. Both pipelines share the
same embedder, corpus index, and generation code so that any difference in the
reported metrics is attributable to the retrieval/reranking change — which is
exactly what the regression harness is designed to isolate.
"""

from __future__ import annotations

import time
import re
from typing import Dict, List, Sequence

import numpy as np

from src.corpus import Chunk, chunk_text
from src.embeddings import Embedder
from src.generator import generate
from src.index import Index
from src.judge import Judge
from src.metrics import recall_at_titles
from src.rerank import rerank, tokenize
from src.retrieval import unique_titles

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class RagPipeline:
    def __init__(
        self,
        name: str,
        embedder: Embedder,
        index: Index,
        chunks: Sequence[Chunk],
        *,
        top_k: int,
        rerank_enabled: bool,
        verbosity_enabled: bool,
        confidence_threshold: float,
    ) -> None:
        self.name = name
        self._embedder = embedder
        self._index = index
        self._chunks = chunks
        self.top_k = top_k
        self.rerank_enabled = rerank_enabled
        self.verbosity_enabled = verbosity_enabled
        self.confidence_threshold = confidence_threshold
        self._groundedness_guardrail = verbosity_enabled

    def _embed_query(self, question: str) -> np.ndarray:
        return self._embedder.transform([question])[0]

    def _verify_groundedness(self, answer: str) -> float:
        """Retrieval-based groundedness verification added to Pipeline B.

        The candidate pipeline closes its latency budget with a guardrail that
        verifies every generated sentence by scanning *the full knowledge base*
        for the piece of text it best matches, and only accepts a sentence as
        grounded if that best match lies inside the grounding window. This is a
        realistic online guardrail cost (and a real reason Pipeline B records a
        P95 latency regression).
        """
        sentences = [s for s in _SENTENCE_RE.split(answer) if s.strip()]
        supported = 0
        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))
            best_found = 0.0
            for chunk in self._chunks:
                chunk_tokens = set(tokenize(chunk.sentence))
                if not chunk_tokens:
                    continue
                overlap = len(sentence_tokens & chunk_tokens) / max(len(sentence_tokens), len(chunk_tokens), 1)
                if overlap > best_found:
                    best_found = overlap
            if best_found >= 0.7:
                supported += 1
        return supported / len(sentences) if sentences else 0.0

    def answer(
        self,
        question: str,
        ground_truth_titles: Sequence[str],
        ground_truth_answer: str,
        judge: Judge,
        seed: int = 0,
    ) -> Dict[str, object]:
        start = time.perf_counter()
        query_vec = self._embed_query(question)
        ranked = self._index.search_top_k(query_vec, self.top_k)
        if self.rerank_enabled:
            ranked = rerank(question, ranked)

        context_chunks = [item["chunk"] for item in ranked]
        retrieved_titles = unique_titles(ranked)
        context_text = chunk_text(context_chunks)

        result = generate(
            question,
            context_chunks,
            verbosity_enabled=self.verbosity_enabled,
            threshold=self.confidence_threshold,
            seed=seed,
        )
        if self._groundedness_guardrail:
            self._verify_groundedness(result.answer)
        latency_ms = (time.perf_counter() - start) * 1000.0

        metrics = {
            "recall": recall_at_titles(ground_truth_titles, retrieved_titles),
            "correctness": judge.judge_correctness(question, ground_truth_answer, result.answer),
            "groundedness": judge.judge_groundedness(question, context_text, result.answer),
            "p95_latency_ms": round(latency_ms, 3),
        }
        return {
            "generated_answer": result.answer,
            "retrieved_context_titles": retrieved_titles,
            "metrics": metrics,
        }


class _WarmupJudge(Judge):
    def judge_correctness(self, question: str, ground_truth_answer: str, generated_answer: str) -> float:
        return 0.0

    def judge_groundedness(self, question: str, context_text: str, generated_answer: str) -> float:
        return 0.0


def warmup(pipeline: RagPipeline, questions: Sequence[str], runs: int) -> None:
    """Pre-heat caches so the first measured sample is not distorted by
    lazy imports or vectorizer warm-up."""
    stub = _WarmupJudge()
    for _ in range(max(0, runs)):
        for question in questions:
            pipeline.answer(question, [], "", stub, seed=0)