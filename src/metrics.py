"""Metrics: recall, latency percentiles, detail-schema helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def recall_at_titles(ground_truth_titles: Sequence[str], retrieved_titles: Sequence[str]) -> float:
    """Fraction of ground-truth context titles recovered by retrieval.

    1.0 when every title the answer depends on was retrieved, 0.0 when none
    were. ``retrieved_titles`` is expected to already be deduplicated.
    """
    gt = set(ground_truth_titles)
    if not gt:
        return 0.0
    retrieved = set(retrieved_titles)
    return len(gt & retrieved) / len(gt)


def percentile(values: Sequence[float], q: float = 95) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def aggregate_details(details: Iterable[Dict[str, object]]) -> Dict[str, float]:
    """Aggregate per-question metrics into report-level numbers."""
    rows = list(details)
    if not rows:
        return {"Recall": 0.0, "Correctness": 0.0, "Groundedness": 0.0, "P95_Latency": 0.0}
    values = {metric: [] for metric in ("recall", "correctness", "groundedness", "p95_latency_ms")}
    for row in rows:
        metrics = row["metrics"]
        for key in values:
            values[key].append(float(metrics[key]))
    return {
        "Recall": sum(values["recall"]) / len(values["recall"]),
        "Correctness": sum(values["correctness"]) / len(values["correctness"]),
        "Groundedness": sum(values["groundedness"]) / len(values["groundedness"]),
        "P95_Latency": percentile(values["p95_latency_ms"]),
    }