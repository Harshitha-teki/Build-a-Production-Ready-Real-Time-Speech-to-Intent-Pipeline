"""Container healthcheck for the evaluation harness service.

Exits 0 when the committed evaluation artifacts (``eval_A_details.json``,
``eval_B_details.json``, ``regression_report.csv``) exist and the regression
report satisfies its schema contract; exits 1 otherwise. Referenced by the
``healthcheck`` block in ``docker-compose.yml``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REQUIRED_FILES = ("eval_A_details.json", "eval_B_details.json", "regression_report.csv")


def check() -> tuple[bool, str]:
    for name in REQUIRED_FILES:
        if not (RESULTS_DIR / name).exists():
            return False, f"missing {name}"

    report_path = RESULTS_DIR / "regression_report.csv"
    with report_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["metric", "baseline", "candidate", "delta", "flag"]:
            return False, "regression_report.csv has the wrong header"
        rows = list(reader)

    if [r["metric"] for r in rows] != ["Recall", "Correctness", "Groundedness", "P95_Latency"]:
        return False, "regression_report.csv is missing required metric rows"
    if not any(r["flag"] == "REGRESS" for r in rows):
        return False, "no regression flagged in regression_report.csv"

    for name in ("eval_A_details.json", "eval_B_details.json"):
        details = json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
        if not isinstance(details, list) or not details:
            return False, f"{name} is not a non-empty array"
        for row in details:
            for key in ("id", "generated_answer", "retrieved_context_titles", "metrics"):
                if key not in row:
                    return False, f"{name} row missing '{key}'"
    return True, "artifacts present and valid"


def main() -> int:
    ok, message = check()
    print(f"healthcheck: {'OK' if ok else 'FAILED'} - {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())