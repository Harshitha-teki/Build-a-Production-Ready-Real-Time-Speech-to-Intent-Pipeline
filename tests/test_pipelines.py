"""Unit tests for the two RAG pipelines and the run_evaluation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_settings
from src.corpus import load_corpus
from src.embeddings import build_embedder, embed_corpus
from src.index import build_index
from src.judge import DeterministicJudge
from src.pipelines import RagPipeline, warmup
from src.regression import load_regression_report

ROOT = Path(__file__).resolve().parent.parent

LOW_CONFIDENCE_QUESTION = "Do product terms and applicable fees ever change over time?"


def _pipelines():
    settings = load_settings()
    chunks = load_corpus(Path(settings["corpus_dir"]))
    embedder = build_embedder(settings["pipeline_a_embedder"])
    matrix = embed_corpus(embedder, chunks)
    index = build_index(chunks, embedder, matrix, settings)
    a = RagPipeline(
        "Pipeline A",
        embedder,
        index,
        chunks,
        top_k=3,
        rerank_enabled=False,
        verbosity_enabled=False,
        confidence_threshold=float(settings["reciprocity_threshold"]),
    )
    b = RagPipeline(
        "Pipeline B",
        embedder,
        index,
        chunks,
        top_k=6,
        rerank_enabled=True,
        verbosity_enabled=True,
        confidence_threshold=float(settings["reciprocity_threshold"]),
    )
    return a, b


def test_pipeline_a_answer_contract():
    a, _ = _pipelines()
    judge = DeterministicJudge()
    result = a.answer(
        "What is the monthly maintenance fee for the Summit Checking Account?",
        ["Summit Checking Account"],
        "The Summit Checking Account charges a monthly maintenance fee of $12.00.",
        judge,
    )
    assert isinstance(result["generated_answer"], str) and result["generated_answer"]
    assert isinstance(result["retrieved_context_titles"], list)
    for key in ("recall", "correctness", "groundedness", "p95_latency_ms"):
        assert key in result["metrics"]


def test_pipeline_b_is_slower_than_pipeline_a_in_total():
    """The rerank + guardrail pass makes B genuinely more expensive."""
    a, b = _pipelines()
    judge = DeterministicJudge()
    total_a = total_b = 0.0
    for i in range(5):
        total_a += a.answer(f"sample question {i}", [], "", judge)["metrics"]["p95_latency_ms"]
    for i in range(5):
        total_b += b.answer(f"sample question {i}", [], "", judge)["metrics"]["p95_latency_ms"]
    assert total_b > total_a


def test_pipeline_b_verbosity_triggering_lowers_groundedness():
    a, b = _pipelines()
    judge = DeterministicJudge()
    ra = a.answer(LOW_CONFIDENCE_QUESTION, [], "", judge)
    rb = b.answer(LOW_CONFIDENCE_QUESTION, [], "", judge)
    assert rb["metrics"]["groundedness"] <= ra["metrics"]["groundedness"]


def test_run_evaluation_writes_report(tmp_path, monkeypatch):
    settings = load_settings()
    settings["results_dir"] = str(tmp_path)
    monkeypatch.setattr("src.regression.write_regression_report", lambda rows, path: None)

    import run_evaluation

    result = run_evaluation.run_evaluation(settings)
    rows = result["regression_rows"]
    assert [r["metric"] for r in rows] == ["Recall", "Correctness", "Groundedness", "P95_Latency"]
    assert len(result["details_a"]) == int(settings["dataset_size"])
    assert len(result["details_b"]) == int(settings["dataset_size"])


def test_committed_details_match_dataset_length():
    eval_set = json.loads((ROOT / "dataset" / "eval_set.json").read_text(encoding="utf-8"))
    for name in ("eval_A_details.json", "eval_B_details.json"):
        details = json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        assert isinstance(details, list)
        assert len(details) == len(eval_set)


def test_committed_report_matches_committed_details():
    """P95_Latency and means in the report are recomputed from the details."""
    from src.metrics import aggregate_details

    details_a = json.loads((ROOT / "results" / "eval_A_details.json").read_text(encoding="utf-8"))
    details_b = json.loads((ROOT / "results" / "eval_B_details.json").read_text(encoding="utf-8"))
    agg_a = aggregate_details(details_a)
    agg_b = aggregate_details(details_b)
    rows = load_regression_report(ROOT / "results" / "regression_report.csv")

    by_metric = {r["metric"]: r for r in rows}
    for metric in ("Recall", "Correctness", "Groundedness", "P95_Latency"):
        assert float(by_metric[metric]["baseline"]) == pytest.approx(agg_a[metric], abs=1e-3)
        assert float(by_metric[metric]["candidate"]) == pytest.approx(agg_b[metric], abs=1e-3)
        assert float(by_metric[metric]["delta"]) == pytest.approx(agg_b[metric] - agg_a[metric], abs=1e-3)