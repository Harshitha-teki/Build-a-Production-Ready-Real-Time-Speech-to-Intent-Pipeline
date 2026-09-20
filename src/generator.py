"""Answer generator.

Both pipelines use a deterministic extractive generator for reproducible
offline results. The candidate pipeline (Pipeline B) additionally enables a
``verbosity`` mode: when the best retrieved chunk has low lexical confidence
against the question, the generator emits an elaborative sentence that is NOT
supported by the retrieved context. This mirrors how a verbose LLM can
hallucinate when the grounding window is thin, and is the mechanism that the
harness uses to demonstrate a groundedness regression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from src.corpus import Chunk
from src.rerank import _bigrams, tokenize

_ELABORATION_POOL: List[str] = [
    "Customers are encouraged to consult their personal relationship manager for advice tailored to their individual goals and circumstances.",
    "Please note that product terms and applicable fees may change from time to time, and the most recent version of the product documentation always prevails.",
    "For complete details customers may refer to the most recent product disclosure statement, which is available online at any time.",
    "The bank continuously reviews its product catalogue and may introduce new features or pricing options as market conditions evolve.",
]


@dataclass
class GenerationResult:
    answer: str
    confidence: float
    verbosity_triggered: bool
    supporting_sentences: List[str]


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _overlap_ratio(question_tokens: Sequence[str], sentence_tokens: Sequence[str]) -> float:
    """Bigram-overlap ratio between question and candidate sentence.

    Bigram overlap is a deliberately stricter lexical confidence signal than
    unigram overlap: two sentences can share many isolated content words while
    arranging them in entirely different ways, and it is that phrase-level
    agreement that indicates whether the retrieved chunk actually supports the
    question.
    """
    q_bigrams = set(_bigrams(question_tokens))
    if not q_bigrams:
        return 0.0
    c_bigrams = set(_bigrams(sentence_tokens))
    return len(q_bigrams & c_bigrams) / len(q_bigrams)


def pick_best_sentence(question: str, chunks: Sequence[Chunk]) -> Chunk:
    tokens = tokenize(question)
    return max(chunks, key=lambda c: _overlap_ratio(tokens, tokenize(c.sentence)))


def generate(
    question: str,
    context_chunks: Sequence[Chunk],
    *,
    verbosity_enabled: bool,
    threshold: float,
    seed: int = 0,
) -> GenerationResult:
    """Generate an answer from the retrieved context chunks."""
    tokens = tokenize(question)
    best = pick_best_sentence(question, context_chunks)
    confidence = _overlap_ratio(tokens, tokenize(best.sentence))

    answer_parts = [best.sentence]
    supporting = [best.sentence]
    triggered = False

    if verbosity_enabled and confidence < threshold:
        triggered = True
        answer_parts.append(_ELABORATION_POOL[seed % len(_ELABORATION_POOL)])

    return GenerationResult(
        answer=" ".join(answer_parts),
        confidence=confidence,
        verbosity_triggered=triggered,
        supporting_sentences=supporting,
    )