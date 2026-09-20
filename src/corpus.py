"""Corpus loading and chunking.

Each knowledge base document is a Markdown file under ``dataset/corpus/``.
The first line must be an H1 renderer title (``# Title``); remaining non
title lines are treated as sentences. Each sentence becomes an indexed chunk
so that retrieval operates at sentence granularity while groundedness is
reported against whole documents (titles).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

_TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    title: str
    sentence: str
    chunk_id: int


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def load_corpus(corpus_dir: Path) -> List[Chunk]:
    """Load every markdown document and split it into sentence chunks."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    chunks: List[Chunk] = []

    def flush(body_lines: List[str], title: str) -> None:
        for sentence in _split_sentences(" ".join(body_lines)):
            chunks.append(Chunk(title=title, sentence=sentence, chunk_id=len(chunks)))

    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown documents found in: {corpus_dir}")

    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        title = path.stem.replace("_", " ").title()
        body_lines: List[str] = []
        for line in lines:
            match = _TITLE_RE.match(line)
            if match:
                flush(body_lines, title)
                title = match.group(1).strip()
                body_lines = []
            elif line.strip():
                body_lines.append(line.strip())
        flush(body_lines, title)
    return chunks


def chunk_text(chunks: Sequence[Chunk]) -> str:
    """Concatenated text used as the grounding context for an answer."""
    return " ".join(chunk.sentence for chunk in chunks)