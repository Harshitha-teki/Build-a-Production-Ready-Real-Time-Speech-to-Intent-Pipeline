"""Lexical reranker for the candidate pipeline.

Pipeline B (candidate) enables ``rerank: true`` and expands ``top_k`` from 3
to 6, then pushes the retrieved window through a cross-encoder-style second
stage. This module implements a deterministic lexical proxy for a cross
encoder: it scores each (query, chunk) pair with a weighted combination of
unigram overlap ratio and bigram F1 and re-orders the candidate window. On
the regression run this reordering occasionally promotes documents that share
vocabulary with the question but not the required fact, which is exactly the
kind of retrieval drift the harness is meant to surface.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from src.corpus import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9$%]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has how i in is it of on that the this to was
    what when where which who will with you your does do not no or can could would should
    more than under over""".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _bigrams(tokens: Sequence[str]) -> List[str]:
    return ["_".join(pair) for pair in zip(tokens, tokens[1:])]


def lexical_score(question_tokens: Sequence[str], chunk_sentence: str) -> float:
    """Deterministic cross-encoder-style (query, chunk) affinity score."""
    chunk_tokens = tokenize(chunk_sentence)
    if not question_tokens or not chunk_tokens:
        return 0.0

    q_set = set(question_tokens)
    c_set = set(chunk_tokens)
    unigram_overlap = len(q_set & c_set) / len(q_set)

    q_bigrams = set(_bigrams(question_tokens))
    c_bigrams = set(_bigrams(chunk_tokens))
    if not q_bigrams:
        bigram_f1 = 0.0
    else:
        common = len(q_bigrams & c_bigrams)
        precision = common / len(c_bigrams) if c_bigrams else 0.0
        recall = common / len(q_bigrams)
        bigram_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return 0.6 * unigram_overlap + 0.4 * bigram_f1


def rerank(
    question: str,
    ranked: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Re-order retrieved chunks by their lexical affinity to the question."""
    tokens = tokenize(question)
    scored = [
        dict(item, rerank_score=lexical_score(tokens, str(item["chunk"].sentence)))
        for item in ranked
    ]
    scored.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rerank_rank"] = rank
    return scored