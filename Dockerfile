FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY app/ app/
COPY generate_samples.py .
COPY samples/ samples/
COPY benchmark.py .
COPY run_benchmark.sh .
RUN chmod +x run_benchmark.sh

RUN mkdir -p /app/models/nlu /app/results

ARG ASR_MODEL_SIZE=tiny.en

RUN python -c "\
from faster_whisper import WhisperModel; \
import logging; logging.basicConfig(level=logging.INFO); \
print('Pre-downloading ASR model: ${ASR_MODEL_SIZE}'); \
WhisperModel('${ASR_MODEL_SIZE}', device='cpu', compute_type='int8'); \
print('ASR model cached successfully.')" || echo "WARNING: ASR model pre-download failed, will retry on startup"

RUN python -c "\
from app.nlu_module import IntentClassifier; \
import os; \
c = IntentClassifier(); \
os.makedirs('/app/models/nlu', exist_ok=True); \
c.save('/app/models/nlu/intent_classifier.joblib', '/app/models/nlu/tfidf_vectorizer.joblib'); \
print('NLU model trained and saved.')"

RUN python generate_samples.py

ENV PYTHONUNBUFFERED=1
ENV ASR_MODEL_SIZE=tiny.en
ENV ASR_COMPUTE_TYPE=int8
ENV ASR_DEVICE=cpu
ENV ASR_CPU_THREADS=4
ENV NLU_MODEL_PATH=/app/models/nlu/intent_classifier.joblib
ENV NLU_VECTORIZER_PATH=/app/models/nlu/tfidf_vectorizer.joblib
ENV TTS_VOICE=en-us
ENV TTS_RATE=150
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
