FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY dataset ./dataset
COPY config ./config
COPY prompts ./prompts
COPY run_evaluation.py .
COPY tests ./tests

RUN python -m pytest -q && python run_evaluation.py

CMD ["python", "run_evaluation.py"]