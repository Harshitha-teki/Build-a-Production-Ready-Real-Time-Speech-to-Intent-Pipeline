"""Schema tests for the required artifacts: dataset, pins, prompts, details."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_eval_set_schema():
    rows = json.loads((ROOT / "dataset" / "eval_set.json").read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert 30 <= len(rows) <= 50, len(rows)
    for row in rows:
        for key in ("id", "question", "ground_truth_answer", "ground_truth_context_titles"):
            assert key in row, key
        assert isinstance(row["ground_truth_context_titles"], list)
    assert len({row["id"] for row in rows}) == len(rows), "ids must be unique"


def test_eval_pins_schema():
    pins = json.loads((ROOT / "config" / "eval_pins.json").read_text(encoding="utf-8"))
    assert isinstance(pins["dataset_size"], (int, float)) and not isinstance(pins["dataset_size"], bool)
    assert isinstance(pins["llm_judge_model"], str) and pins["llm_judge_model"]
    assert isinstance(pins["pipeline_a_embedder"], str) and pins["pipeline_a_embedder"]
    assert isinstance(pins["pipeline_b_embedder"], str) and pins["pipeline_b_embedder"]


def test_judge_prompts_schema():
    prompts = json.loads((ROOT / "prompts" / "judge_prompts.json").read_text(encoding="utf-8"))
    assert isinstance(prompts["correctness_prompt"], str) and prompts["correctness_prompt"].strip()
    assert isinstance(prompts["groundedness_prompt"], str) and prompts["groundedness_prompt"].strip()


def test_details_schema():
    eval_set = json.loads((ROOT / "dataset" / "eval_set.json").read_text(encoding="utf-8"))
    for path in (ROOT / "results" / "eval_A_details.json", ROOT / "results" / "eval_B_details.json"):
        details = json.loads(path.read_text(encoding="utf-8"))
        assert len(details) == len(eval_set)
        for row in details:
            for key in ("id", "generated_answer", "retrieved_context_titles", "metrics"):
                assert key in row, key
            metrics = row["metrics"]
            for key in ("recall", "correctness", "groundedness", "p95_latency_ms"):
                assert key in metrics, key
            assert len(metrics) == 4


def test_regression_csv_schema():
    with (ROOT / "results" / "regression_report.csv").open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["metric", "baseline", "candidate", "delta", "flag"]
        rows = list(reader)
    assert len(rows) == 4
    assert [r["metric"] for r in rows] == ["Recall", "Correctness", "Groundedness", "P95_Latency"]
    for r in rows:
        float(r["baseline"])
        float(r["candidate"])
        float(r["delta"])


def test_env_example_has_no_real_secrets():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "=" in line and line.split("=", 1)[1].strip().lower().startswith("sk-"):
            raise AssertionError(f"placeholder secrets must not start with sk-: {line}")