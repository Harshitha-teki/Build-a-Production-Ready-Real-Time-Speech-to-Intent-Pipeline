# LLM Evaluation Regression Harness for RAG Pipelines

A production-styled evaluation harness that compares two RAG pipelines
(**Pipeline A** = baseline, **Pipeline B** = candidate) on a pinned evaluation
set, scores every answer with an LLM judge (or a deterministic, offline
reproducible proxy), and emits a **regression report** that flags any metric
that got worse.

This repository is the resubmission for the **LLM Evaluation Regression
Harness for RAG Pipelines** task.

## What it does

1. Loads the pinned knowledge base (`dataset/corpus/`) and evaluation set
   (`dataset/eval_set.json`, 40 questions with ground-truth answers and
   ground-truth context titles).
2. Runs **Pipeline A** (`top_k=3`, no reranking) and **Pipeline B**
   (`top_k=6` + lexical reranker + higher-verbosity generator) over every
   question, measuring per-question latency.
3. Scores each generated answer on **Recall**, **Correctness**,
   **Groundedness**, and samples **P95 latency**.
4. Aggregates per-question results into
   `results/eval_A_details.json`, `results/eval_B_details.json`, and the flag
   logic in `results/regression_report.csv`.

## Committed regression (demonstrated)

`results/regression_report.csv` (regenerate with `python run_evaluation.py`):

| metric       | baseline | candidate | delta   | flag    |
|--------------|----------|-----------|---------|---------|
| Recall       | 0.925    | 0.95      | 0.025   | CLEAN   |
| Correctness  | 0.8852   | 0.9018    | 0.0167  | CLEAN   |
| Groundedness | 1.0      | 0.925     | -0.075  | REGRESS |
| P95_Latency  | 1.67     | 6.57      | 4.91    | REGRESS |

The story this demonstrates: **Pipeline B** recovers more of the relevant
context (Recall up, Correctness up) because of expanded top-k retrieval and
reranking — but the same change introduces retrieval drift and verbosity that
lowers groundedness, and the added reranker plus a groundedness-verification
guardrail blow through the latency budget. Exactly the trade-off a regression
harness exists to surface.

### Flag logic

- `Recall`, `Correctness`, `Groundedness`: **REGRESS** iff `candidate < baseline`.
- `P95_Latency`: **REGRESS** iff `candidate > baseline * 1.10`.

## Quick start (no external services)

Requires Python 3.12+ and `pip install -r requirements.txt`.

```bash
python run_evaluation.py
python -m pytest -q
```

Running the harness regenerates `results/eval_A_details.json`,
`results/eval_B_details.json`, and `results/regression_report.csv`.

## Quick start (full stack with Qdrant)

```bash
docker compose up --build
```

The compose file starts **Qdrant** (with a healthcheck), indexes the knowledge
base into it when `QDRANT_URL` is set, runs the harness, and verifies the
artifacts via a healthcheck on the harness service.

## Structure

```
.
├── dataset/corpus/            # Knowledge base (Markdown, 15 documents)
├── dataset/eval_set.json      # 40 questions with GT answers & context titles
├── config/eval_pins.json      # Pinned dataset size, judge model, embedders
├── prompts/judge_prompts.json # Strict binary LLM judge rubrics
├── results/                   # eval_A_details.json / eval_B_details.json / regression_report.csv
├── src/
│   ├── config.py              # Pins + env settings
│   ├── corpus.py              # Document loading & sentence chunking
│   ├── embeddings.py          # TF-IDF embedder (deterministic, offline)
│   ├── index.py               # In-memory index + Qdrant index backends
│   ├── retrieval.py           # Cosine top-k retrieval helpers
│   ├── rerank.py              # Cross-encoder-style lexical reranker (Pipeline B)
│   ├── generator.py           # Extractive generator with verbosity mode
│   ├── judge.py               # Deterministic judge + LLM judge (temperature 0)
│   ├── metrics.py             # Recall & latency aggregation
│   ├── pipelines.py           # Pipeline A & B orchestration
│   ├── regression.py          # Regression report + flag logic
│   └── healthcheck.py         # Container healthcheck for compose
├── tests/                     # Unit + contract tests (28 tests)
├── run_evaluation.py          # Entry point
├── .env.example               # LLM key + Qdrant placeholders (no real secrets)
├── docker-compose.yml         # Qdrant (healthchecked) + harness service
└── .github/workflows/ci.yml   # CI: tests + fresh run + artifact validation
```

## Choosing the judge

The judge that produced the committed artifacts is the **deterministic judge**
(`RAG_JUDGE_MODE=deterministic`): an offline proxy that measures correctness as
ground-truth token coverage and groundedness as how much of the generated
answer is supported by the retrieved context. It is byte-for-byte reproducible
in CI.

To use the real **LLM judge**, set an API key and switch the mode — the pinned
model (`config/eval_pins.json` + `LLM_JUDGE_MODEL`), temperature 0, and the
strict binary scoring rubrics in `prompts/judge_prompts.json` are used, with a
separate prompt for correctness and groundedness. The groundedness rubric
explicitly instructs the model to score **based only on the provided context,
ignoring real-world knowledge**.

```bash
export RAG_JUDGE_MODE=llm
export LLM_API_KEY=...
python run_evaluation.py
```

## Reproducibility pins

`config/eval_pins.json` pins the dataset size, the LLM judge model, and the
embedders for both pipelines (both pinned to `tfidf` — a deterministic sparse
embedder that requires no model downloads). The pins are read by
`src/config.py`, which throws if any required pin is missing or mistyped.

## Tests

`tests/` covers the flag logic, retrieval quality, recall metric, both judge
backends, the prompts file contract, the dataset/pins/details/report schemas,
and the engine regression tests. The headless CI workflow also runs a fresh
evaluation and cross-checks that a fresh run produces the same aggregated
report as the committed one.

## Design notes / why TF-IDF embeddings

The embedder is pinned to `tfidf` so that (a) the harness runs anywhere with
zero model downloads and (b) the committed results are reproducible bit-for-bit.
The `config/eval_pins.json` values are what make the run reproducible — swap in
a neural embedder (e.g., `sentence-transformers/all-MiniLM-L6-v2`) and the
harness's retrieval metrics will capture how the pipelines behave with that
model too.