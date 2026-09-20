"""LLM-as-a-judge and deterministic judge implementations.

Two judge backends are supported:

* ``deterministic`` (default) - a reproducible, offline lexical proxy. It
  scores correctness as ground-truth token coverage of the generated answer,
  and groundedness as the fraction of answer sentences whose content can be
  verified against the retrieved context. This is the backend used to produce
  the committed ``results/`` artifacts so runs are deterministic in CI.
* ``llm`` - a real LLM judge using the pinned model from
  ``config/eval_pins.json`` at temperature 0 with the strict binary scoring
  rubrics in ``prompts/judge_prompts.json``. Enable it by setting
  ``RAG_JUDGE_MODE=llm`` and providing ``LLM_API_KEY``.

Only a single LLM call happens per (metric, question) pair; no context is
retained between calls, so the judge is stateless and repeatable.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from src.corpus import chunk_text
from src.rerank import tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_judge_prompts(prompts_path: Path = PROMPTS_DIR / "judge_prompts.json") -> Dict[str, str]:
    with prompts_path.open("r", encoding="utf-8") as fh:
        prompts = json.load(fh)
    correctness = prompts.get("correctness_prompt", "")
    groundedness = prompts.get("groundedness_prompt", "")
    if not isinstance(correctness, str) or not correctness:
        raise ValueError("judge_prompts.json requires a non-empty 'correctness_prompt'")
    if not isinstance(groundedness, str) or not groundedness:
        raise ValueError("judge_prompts.json requires a non-empty 'groundedness_prompt'")
    return prompts


class Judge(ABC):
    @abstractmethod
    def judge_correctness(self, question: str, ground_truth_answer: str, generated_answer: str) -> float:
        ...

    @abstractmethod
    def judge_groundedness(self, question: str, context_text: str, generated_answer: str) -> float:
        ...


def _content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if len(t) >= 4]


class DeterministicJudge(Judge):
    """Reproducible lexical proxy for the LLM judge."""

    def judge_correctness(self, question: str, ground_truth_answer: str, generated_answer: str) -> float:
        gt_tokens = set(_content_tokens(ground_truth_answer))
        if not gt_tokens:
            return 0.0
        answer_tokens = set(_content_tokens(generated_answer))
        return len(gt_tokens & answer_tokens) / len(gt_tokens)

    def judge_groundedness(self, question: str, context_text: str, generated_answer: str) -> float:
        sentences = [s.strip() for s in _SENTENCE_RE.split(generated_answer.strip()) if s.strip()]
        if not sentences:
            return 0.0
        context_tokens = set(tokenize(context_text))
        supported = 0
        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))
            if not sentence_tokens:
                supported += 1
                continue
            overlap = len(sentence_tokens & context_tokens) / len(sentence_tokens)
            if overlap >= 0.7:
                supported += 1
        return supported / len(sentences)


class LLMJudge(Judge):
    """Pinned LLM judge running the strict binary rubrics at temperature 0."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", prompts=None) -> None:
        import requests  # imported lazily so deterministic mode has no extra deps

        self._requests = requests
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._prompts = prompts or load_judge_prompts()

    def _call(self, prompt: str, **fields: str) -> float:
        text = prompt.format(**fields)
        response = self._requests.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": text}],
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\b[01]\b", raw)
        if match is None:
            raise ValueError(f"LLM judge returned an unparseable score: {raw!r}")
        return float(match.group(0))

    def judge_correctness(self, question: str, ground_truth_answer: str, generated_answer: str) -> float:
        return self._call(
            self._prompts["correctness_prompt"],
            question=question,
            ground_truth_answer=ground_truth_answer,
            generated_answer=generated_answer,
        )

    def judge_groundedness(self, question: str, context_text: str, generated_answer: str) -> float:
        return self._call(
            self._prompts["groundedness_prompt"],
            question=question,
            retrieved_context=context_text,
            generated_answer=generated_answer,
        )


def build_judge(settings: Dict[str, Any]) -> Judge:
    mode = settings.get("judge_mode", "deterministic")
    prompts = load_judge_prompts()
    if mode == "llm":
        api_key = settings.get("llm_api_key", "")
        if not api_key:
            raise RuntimeError("RAG_JUDGE_MODE=llm requires LLM_API_KEY (or OPENAI_API_KEY) to be set")
        return LLMJudge(
            model=settings["llm_judge_model"],
            api_key=api_key,
            base_url=settings.get("llm_judge_base_url", "https://api.openai.com/v1"),
            prompts=prompts,
        )
    if mode == "deterministic":
        return DeterministicJudge()
    raise ValueError(f"Unknown judge mode: {mode!r}")