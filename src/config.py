"""Configuration loading for the RAG evaluation harness.

Pins that govern a reproducible evaluation run live in ``config/eval_pins.json``.
Additional machine-specific settings are supplied through environment
variables (see ``.env.example``). Values that are not pinned fall back to
environment variables and finally to built-in defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
PINS_PATH = ROOT_DIR / "config" / "eval_pins.json"

DEFAULTS: Dict[str, Any] = {
    "dataset_size": 40,
    "llm_judge_model": "openai/gpt-4o-mini",
    "llm_judge_temperature": 0.0,
    "llm_judge_base_url": "https://api.openai.com/v1",
    "reciprocity_threshold": 0.40,
    "pipeline_a": {"name": "baseline", "top_k": 3, "rerank": False},
    "pipeline_b": {"name": "candidate", "top_k": 6, "rerank": True},
    "deterministic_answer_coverage_tokens_min": 4,
}

_REQUIRED_PIN_KEYS = ("dataset_size", "llm_judge_model", "pipeline_a_embedder", "pipeline_b_embedder")


def load_pins(pins_path: Path = PINS_PATH) -> Dict[str, Any]:
    """Load and validate ``config/eval_pins.json``."""
    if not pins_path.exists():
        raise FileNotFoundError(f"Missing pins file: {pins_path}")

    with pins_path.open("r", encoding="utf-8") as fh:
        pins = json.load(fh)

    for key in _REQUIRED_PIN_KEYS:
        if key not in pins:
            raise KeyError(f"Pins file is missing required key '{key}'")

    if not isinstance(pins["dataset_size"], int) or isinstance(pins["dataset_size"], bool):
        raise TypeError("'dataset_size' must be a number")
    for key in ("llm_judge_model", "pipeline_a_embedder", "pipeline_b_embedder"):
        if not isinstance(pins[key], str) or not pins[key]:
            raise TypeError(f"'{key}' must be a non-empty string")

    return {**DEFAULTS, **pins}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def load_settings() -> Dict[str, Any]:
    """Merge pins with run-time environment settings into a single settings map."""
    pins = load_pins()
    settings = dict(pins)
    settings["dataset_path"] = os.getenv("EVAL_DATASET_PATH", str(ROOT_DIR / "dataset" / "eval_set.json"))
    settings["corpus_dir"] = os.getenv("EVAL_CORPUS_DIR", str(ROOT_DIR / "dataset" / "corpus"))
    settings["results_dir"] = os.getenv("EVAL_RESULTS_DIR", str(ROOT_DIR / "results"))
    settings["judge_mode"] = os.getenv("RAG_JUDGE_MODE", "deterministic").lower()
    settings["llm_api_key"] = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    settings["qdrant_host"] = os.getenv("QDRANT_HOST", "qdrant")
    settings["qdrant_port"] = os.getenv("QDRANT_PORT", "6333")
    settings["qdrant_url"] = os.getenv(
        "QDRANT_URL", os.getenv("QDRANT_HOST") and f"http://{settings['qdrant_host']}:{settings['qdrant_port']}" or ""
    )
    settings["qdrant_collection"] = os.getenv("QDRANT_COLLECTION", "acme_knowledge_base")
    settings["reciprocity_threshold"] = float(os.getenv("RECIPROCITY_THRESHOLD", str(pins.get("reciprocity_threshold", 0.35))))
    settings["latency_warmup_runs"] = _env_int("LATENCY_WARMUP_RUNS", int(pins.get("latency_warmup_runs", 3)))
    return settings