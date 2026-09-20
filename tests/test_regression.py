"""Unit tests for the regression report contract and flag logic."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.regression import flag_for, build_regression_rows, write_regression_report, load_regression_report

ROOT = Path(__file__).resolve().parent.parent


def test_lower_is_better_metrics_flag_without_threshold():
    assert flag_for("Recall", 0.90, 0.95) == "CLEAN"
    assert flag_for("Recall", 0.90, 0.89) == "REGRESS"
    assert flag_for("Correctness", 0.50, 0.51) == "CLEAN"
    assert flag_for("Correctness", 0.50, 0.49) == "REGRESS"
    assert flag_for("Groundedness", 1.0, 0.999) == "REGRESS"


def test_latency_flag_applies_ten_percent_threshold():
    assert flag_for("P95_Latency", 10.0, 10.9) == "CLEAN"
    assert flag_for("P95_Latency", 10.0, 11.000001) == "REGRESS"
    assert flag_for("P95_Latency", 10.0, 9.0) == "CLEAN"


def test_build_rows_produce_expected_columns_and_order():
    baseline = {"Recall": 0.925, "Correctness": 0.8852, "Groundedness": 1.0, "P95_Latency": 4.5}
    candidate = {"Recall": 0.95, "Correctness": 0.9018, "Groundedness": 0.925, "P95_Latency": 7.75}
    rows = build_regression_rows(baseline, candidate)
    assert [r["metric"] for r in rows] == ["Recall", "Correctness", "Groundedness", "P95_Latency"]
    assert all(set(r) == {"metric", "baseline", "candidate", "delta", "flag"} for r in rows)
    assert rows[0]["delta"] == pytest.approx(0.025)
    assert rows[2]["delta"] == pytest.approx(-0.075)
    assert rows[2]["flag"] == "REGRESS"
    assert rows[3]["flag"] == "REGRESS"


def test_write_and_load_round_trip(tmp_path):
    baseline = {"Recall": 1.0, "Correctness": 1.0, "Groundedness": 1.0, "P95_Latency": 10.0}
    candidate = {"Recall": 1.0, "Correctness": 1.0, "Groundedness": 1.0, "P95_Latency": 10.0}
    rows = build_regression_rows(baseline, candidate)
    path = tmp_path / "regression_report.csv"
    write_regression_report(rows, path)
    loaded = load_regression_report(path)
    assert len(loaded) == len(rows)
    for original, loaded_row in zip(rows, loaded):
        assert loaded_row["metric"] == original["metric"]
        assert loaded_row["flag"] == original["flag"]
        assert float(loaded_row["baseline"]) == pytest.approx(original["baseline"], abs=1e-4)
        assert float(loaded_row["candidate"]) == pytest.approx(original["candidate"], abs=1e-4)
        assert float(loaded_row["delta"]) == pytest.approx(original["delta"], abs=1e-4)


def test_committed_report_satisfies_contract():
    """The committed results/regression_report.csv meets the full rubric."""
    path = ROOT / "results" / "regression_report.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert list(rows[0].keys()) == ["metric", "baseline", "candidate", "delta", "flag"]
    assert [r["metric"] for r in rows] == ["Recall", "Correctness", "Groundedness", "P95_Latency"]

    summary = {r["metric"]: r for r in rows}
    for metric, row in summary.items():
        baseline = float(row["baseline"])
        candidate = float(row["candidate"])
        assert float(row["delta"]) == pytest.approx(candidate - baseline, abs=2e-4)
        assert row["flag"] == flag_for(metric, baseline, candidate)

    assert float(summary["Correctness"]["delta"]) >= 0
    assert any(r["flag"] == "REGRESS" for r in rows)