#!/usr/bin/env python3
"""Benchmark script for measuring pipeline latency."""

import os
import sys
import json
import time
import glob
import requests
import numpy as np

API_URL = os.environ.get("API_URL", "http://localhost:8000/process-intent")
NUM_REQUESTS = int(os.environ.get("BENCHMARK_REQUESTS", "30"))
NUM_WARMUP = int(os.environ.get("BENCHMARK_WARMUP", "3"))
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "samples")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


def collect_sample_files(samples_dir: str) -> list:
    wav_files = sorted(glob.glob(os.path.join(samples_dir, "*.wav")))
    if not wav_files:
        print(f"ERROR: No .wav files found in {samples_dir}/")
        sys.exit(1)
    return wav_files


def send_request(audio_path: str) -> dict:
    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
        resp = requests.post(API_URL, files=files, timeout=30)
    resp.raise_for_status()
    return resp.json()


def wait_for_service(timeout: int = 120, interval: int = 5):
    health_url = API_URL.replace("/process-intent", "/health")
    start = time.time()
    print(f"Waiting for service at {health_url}...")
    while time.time() - start < timeout:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print("Service is ready.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    print("ERROR: Service did not become ready within timeout.")
    sys.exit(1)


def main():
    wait_for_service()

    sample_files = collect_sample_files(SAMPLES_DIR)
    print(f"Found {len(sample_files)} sample files in {SAMPLES_DIR}/")

    asr_latencies = []
    intent_latencies = []
    tts_latencies = []
    total_latencies = []

    # Warmup
    print(f"Running {NUM_WARMUP} warmup requests...")
    for i in range(NUM_WARMUP):
        for sf in sample_files[:3]:
            try:
                result = send_request(sf)
            except Exception as e:
                print(f"  Warmup request failed: {e}")
        print(f"  Warmup round {i+1}/{NUM_WARMUP} done.")

    print(f"Running {NUM_REQUESTS} benchmark requests...")
    success_count = 0
    fail_count = 0

    idx = 0
    while success_count < NUM_REQUESTS:
        sf = sample_files[idx % len(sample_files)]
        idx += 1
        try:
            result = send_request(sf)
            lat = result["latencies_ms"]
            asr_latencies.append(lat["asr"])
            intent_latencies.append(lat["intent"])
            tts_latencies.append(lat["tts"])
            total_latencies.append(lat["total"])
            success_count += 1
            if success_count % 5 == 0:
                print(f"  Completed {success_count}/{NUM_REQUESTS} requests...")
        except Exception as e:
            fail_count += 1
            print(f"  Request failed: {e}")
            if fail_count > NUM_REQUESTS:
                print("Too many failures, aborting.")
                sys.exit(1)

    def compute_percentiles(data):
        arr = np.array(data)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        }

    report = {
        "asr_ms": compute_percentiles(asr_latencies),
        "intent_ms": compute_percentiles(intent_latencies),
        "tts_ms": compute_percentiles(tts_latencies),
        "total_ms": compute_percentiles(total_latencies),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "latency_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nLatency Report written to {report_path}:")
    print(json.dumps(report, indent=2))

    p95_total = report["total_ms"]["p95"]
    if p95_total < 2000:
        print(f"\nPASSED: p95 total latency ({p95_total:.1f}ms) < 2000ms target.")
    else:
        print(f"\nFAILED: p95 total latency ({p95_total:.1f}ms) >= 2000ms target.")
        sys.exit(1)


if __name__ == "__main__":
    main()
