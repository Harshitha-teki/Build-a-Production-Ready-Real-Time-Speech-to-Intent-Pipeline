"""LLM Evaluation Regression Harness for RAG Pipelines.

Runs both RAG pipelines (A = baseline, B = candidate) over the pinned
evaluation set, scores every answer with the configured judge, and writes:

    results/eval_A_details.json        per-question results, Pipeline A
    results/eval_B_details.json        per-question results, Pipeline B
    results/regression_report.csv      aggregated regression report

End-to-end usage:

    python run_evaluation.py [--tune-grounding] [--pins config/eval_pins.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.config import load_settings
from src.corpus import load_corpus
from src.embeddings import build_embedder, embed_corpus
from src.index import build_index
from src.judge import build_judge
from src.metrics import aggregate_details
from src.pipelines import RagPipeline, warmup
from src.regression import build_regression_rows, write_regression_report

PROJECT_ROOT = Path(__file__).resolve().parent


def load_eval_set(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array")
    expected = _settings_for(path)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{path} contains {len(rows)} items; eval_pins.dataset_size expects {expected}")
    return rows


def _settings_for(dataset_path: Path) -> int | None:
    try:
        from src.config import load_pins

        pins_path = PROJECT_ROOT / "config" / "eval_pins.json"
        if dataset_path.name == "eval_set.json" and pins_path.exists():
            return int(load_pins(pins_path)["dataset_size"])
    except Exception:
        return None
    return None


def _seed_for(question_id: str) -> int:
    digits = "".join(ch for ch in question_id if ch.isdigit())
    return int(digits) if digits else 0


def run_evaluation(settings: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = Path(settings["dataset_path"])
    eval_set = load_eval_set(dataset_path)
    pins = settings

    judge = build_judge(settings)

    chunks = load_corpus(Path(settings["corpus_dir"]))
    embedder = build_embedder(pins["pipeline_a_embedder"])
    matrix = embed_corpus(embedder, chunks)
    index = build_index(chunks, embedder, matrix, settings)

    a_pipeline = RagPipeline(
        "Pipeline A",
        embedder,
        index,
        chunks,
        top_k=int(pins["pipeline_a"]["top_k"]),
        rerank_enabled=bool(pins["pipeline_a"].get("rerank", False)),
        verbosity_enabled=False,
        confidence_threshold=float(settings["reciprocity_threshold"]),
    )
    b_pipeline = RagPipeline(
        "Pipeline B",
        embedder,
        index,
        chunks,
        top_k=int(pins["pipeline_b"]["top_k"]),
        rerank_enabled=bool(pins["pipeline_b"].get("rerank", True)),
        verbosity_enabled=bool(pins["pipeline_b"].get("verbosity_enabled", True)),
        confidence_threshold=float(settings["reciprocity_threshold"]),
    )

    questions = [str(row["question"]) for row in eval_set]
    warmup(a_pipeline, questions[:5], settings["latency_warmup_runs"])
    warmup(b_pipeline, questions[:5], settings["latency_warmup_runs"])

    def _evaluate(pipeline: RagPipeline) -> List[Dict[str, object]]:
        details: List[Dict[str, object]] = []
        for index, row in enumerate(eval_set):
            question_id = str(row["id"])
            seed = _seed_for(question_id) or index
            result = pipeline.answer(
                str(row["question"]),
                [str(t) for t in row["ground_truth_context_titles"]],
                str(row["ground_truth_answer"]),
                judge,
                seed=seed,
            )
            details.append(
                {
                    "id": question_id,
                    "question": str(row["question"]),
                    "ground_truth_answer": str(row["ground_truth_answer"]),
                    "ground_truth_context_titles": [str(t) for t in row["ground_truth_context_titles"]],
                    "generated_answer": result["generated_answer"],
                    "retrieved_context_titles": result["retrieved_context_titles"],
                    "metrics": result["metrics"],
                }
            )
        return details

    details_a = _evaluate(a_pipeline)
    details_b = _evaluate(b_pipeline)

    results_dir = Path(settings["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "eval_A_details.json", details_a)
    _write_json(results_dir / "eval_B_details.json", details_b)

    aggregate_a = aggregate_details(details_a)
    aggregate_b = aggregate_details(details_b)
    rows = build_regression_rows(aggregate_a, aggregate_b)
    write_regression_report(rows, results_dir / "regression_report.csv")

    return {
        "aggregate_baseline": aggregate_a,
        "aggregate_candidate": aggregate_b,
        "regression_rows": rows,
        "details_a": details_a,
        "details_b": details_b,
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", default=str(PROJECT_ROOT / "config" / "eval_pins.json"))
    args = parser.parse_args(argv)

    settings = load_settings()
    settings["pins_path"] = args.pins
    result = run_evaluation(settings)

    print(f"{'metric':<14}{'baseline':>10}{'candidate':>10}{'delta':>10}  flag")
    for row in result["regression_rows"]:
        print(
            f"{row['metric']:<14}{row['baseline']:>10.4f}{row['candidate']:>10.4f}"
            f"{row['delta']:>10.4f}  {row['flag']}"
        )
    regressions = [row["metric"] for row in result["regression_rows"] if row["flag"] == "REGRESS"]
    print(f"\nRegressions detected: {', '.join(regressions) if regressions else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))