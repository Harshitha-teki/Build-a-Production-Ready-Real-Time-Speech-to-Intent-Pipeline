"""Aggregation and regression report generation.

Report contract (``results/regression_report.csv``):

* Columns: ``metric,baseline,candidate,delta,flag`` where
  ``delta = candidate - baseline``.
* Exactly four data rows: ``Recall``, ``Correctness``, ``Groundedness``,
  ``P95_Latency``.
* Flag logic:
  - Recall / Correctness / Groundedness: ``REGRESS`` iff ``candidate < baseline``.
  - P95_Latency: ``REGRESS`` iff ``candidate > baseline * 1.10``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

_REGRESS_LATENCY_MARGIN = 1.10
_DIRECTION = {  # higher-is-better metrics
    "Recall": "<",
    "Correctness": "<",
    "Groundedness": "<",
    "P95_Latency": ">",
}


def flag_for(metric: str, baseline: float, candidate: float) -> str:
    direction = _DIRECTION[metric]
    if direction == "<":
        return "REGRESS" if candidate < baseline else "CLEAN"
    if direction == ">":
        return "REGRESS" if candidate > (baseline * _REGRESS_LATENCY_MARGIN) else "CLEAN"
    raise ValueError(f"Unknown metric: {metric}")


def build_regression_rows(
    baseline_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric in ("Recall", "Correctness", "Groundedness", "P95_Latency"):
        baseline = float(baseline_metrics[metric])
        candidate = float(candidate_metrics[metric])
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "candidate": candidate,
                "delta": candidate - baseline,
                "flag": flag_for(metric, baseline, candidate),
            }
        )
    return rows


def write_regression_report(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "baseline", "candidate", "delta", "flag"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric": row["metric"],
                    "baseline": round(float(row["baseline"]), 4),
                    "candidate": round(float(row["candidate"]), 4),
                    "delta": round(float(row["delta"]), 4),
                    "flag": row["flag"],
                }
            )


def load_regression_report(path: Path) -> List[Dict[str, object]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]