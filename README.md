# Real-Time Speech-to-Intent Pipeline

A fully local, end-to-end voice assistant pipeline with a strict sub-2-second latency budget. Built with faster-whisper (ASR), scikit-learn (NLU), and espeak-ng (TTS), all orchestrated via a FastAPI REST service and containerized with Docker.

## Architecture

```
Audio Input (.wav) → ASR (faster-whisper) → NLU (Intent Classifier) → TTS (espeak-ng) → Audio Response (base64)
```

**Pipeline stages:**
1. **ASR:** faster-whisper `tiny.en` with INT8 quantization on CPU
2. **NLU:** TF-IDF + LogisticRegression classifier for 8 smart-home intents
3. **TTS:** espeak-ng text-to-speech synthesis

**Supported intents:** `TurnOn`, `TurnOff`, `SetBrightness`, `GetTemperature`, `PlayMusic`, `StopMusic`, `SetTimer`, `GetWeather`

## Setup

### Prerequisites
- Docker and Docker Compose installed
- 2GB+ free disk space for models

### Build and Run

```bash
# Clone/navigate to the project directory
cd speech-to-intent-pipeline

# Build and start the service
docker compose up --build

# The service will be available at http://localhost:8000
```

The first run downloads the ASR model (~75MB for tiny.en) and trains the NLU model (~2 seconds). All models are cached for subsequent runs.

### Health Check

```bash
curl http://localhost:8000/health
# Returns: {"status": "ok"}
```

## API Usage

### POST /process-intent

Process an audio file through the full pipeline.

**Request:**
```bash
curl -X POST http://localhost:8000/process-intent \
  -F "audio=@samples/sample_01.wav"
```

**Response (200 OK):**
```json
{
  "transcribed_text": "turn on the lights",
  "intent": "TurnOn",
  "confidence": 0.95,
  "response_audio_b64": "UklGRi...",
  "latencies_ms": {
    "asr": 320.5,
    "intent": 1.2,
    "tts": 15.3,
    "total": 337.0
  }
}
```

**Error Responses:**
- `400 Bad Request` — Missing file, wrong file type, or file too large
- `500 Internal Server Error` — Pipeline processing failure

## Generating Sample Audio

The `samples/` directory contains pre-generated WAV files for benchmarking. To regenerate them:

```bash
# Inside the container (espeak-ng is pre-installed)
docker compose exec pipeline_service python generate_samples.py

# Or locally if espeak-ng is installed
python generate_samples.py
```

## Running the Benchmark

The benchmark script runs from the host and requires `requests` and `numpy`:

```bash
# Install benchmark dependencies
pip install -r requirements-benchmark.txt

# Start the service
docker compose up --build -d

# Run the benchmark using the shell script
./run_benchmark.sh

# Or run directly
python benchmark.py
```

The benchmark sends 30 requests to the API, collects latency measurements, and writes a report to `results/latency_report.json`.

### Benchmark Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_REQUESTS` | 30 | Number of requests to measure |
| `BENCHMARK_WARMUP` | 3 | Warmup rounds (discarded) |
| `API_URL` | http://localhost:8000/process-intent | Target endpoint |
| `SAMPLES_DIR` | samples | Directory with .wav files |
| `RESULTS_DIR` | results | Output directory for report |

### Sample Output

```json
{
  "asr_ms": { "p50": 310.2, "p95": 450.8, "p99": 520.1 },
  "intent_ms": { "p50": 1.1, "p95": 2.3, "p99": 3.5 },
  "tts_ms": { "p50": 12.5, "p95": 18.2, "p99": 22.1 },
  "total_ms": { "p50": 325.0, "p95": 472.3, "p99": 545.0 }
}
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── pipeline.py           # Pipeline orchestrator
│   ├── asr_module.py         # ASR (faster-whisper)
│   ├── nlu_module.py         # NLU (TF-IDF + LogisticRegression)
│   └── tts_module.py         # TTS (espeak-ng)
├── samples/                  # Sample WAV files for testing
├── results/                  # Benchmark output
├── models/nlu/               # Trained NLU model artifacts
├── benchmark.py              # Latency benchmarking script
├── run_benchmark.sh          # Benchmark runner script
├── generate_samples.py       # Sample audio generator
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── MODEL_CHOICES.md          # Model selection justification
└── README.md                 # This file
```

## Environment Variables

See `.env.example` for all configurable options. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL_SIZE` | tiny.en | faster-whisper model size |
| `ASR_COMPUTE_TYPE` | int8 | Quantization type |
| `ASR_DEVICE` | cpu | Compute device |
| `TTS_VOICE` | en-us | espeak-ng voice |
| `TTS_RATE` | 150 | Speech rate (words/min) |
| `API_PORT` | 8000 | API server port |

## License

MIT
