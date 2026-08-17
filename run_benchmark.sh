#!/usr/bin/env bash
set -euo pipefail

echo "=== Voice Pipeline Latency Benchmark ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running benchmark..."
python benchmark.py

echo ""
echo "=== Benchmark Complete ==="
echo "Results written to results/latency_report.json"
