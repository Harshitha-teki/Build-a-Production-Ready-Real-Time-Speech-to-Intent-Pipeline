# Questionnaire Responses — mapped to the code

These answers describe **this** repository (the LLM Evaluation Regression
Harness for RAG Pipelines). Every claim below points at a concrete file/line so
your written answers match the submitted code exactly.

## 1. How is the LLM judge made fair/reproducible (judge_bias)?

The judge is deliberately pinned and de-biased:

- **Pinned model, temperature 0.** `config/eval_pins.json` pins
  `"llm_judge_model": "openai/gpt-4o-mini"` and `"llm_judge_temperature": 0.0`.
  `src/judge.py` (`LLMJudge`) sends every judge call with `"temperature": 0.0`
  and a single, stateless prompt — no context is carried between calls, so each
  (metric, question) pair is scored independently and identically.
- **Strict binary scoring rubric.** `prompts/judge_prompts.json` defines the
  `correctness_prompt` and `groundedness_prompt`, both instructing the model to
  return only `0` or `1` with no extra text.
- **Separate prompts per metric.** Correctness and groundedness never share a
  prompt, so the two quality signals cannot bleed into one another.
- **Deterministic fallback for offline runs.** Because the committed artifacts
  must be reproducible in CI with no API key, the default mode
  (`RAG_JUDGE_MODE=deterministic`, `src/judge.py::DeterministicJudge`) uses an
  offline proxy of the same rubrics: correctness = fraction of ground-truth
  content tokens recovered by the answer; groundedness = fraction of answer
  sentences whose content is supported by the retrieved context. The same
  metrics are measured when the LLM judge runs.

## 2. How are Correctness and Groundedness measured (eval_strategy)?

- **Per question**, each pipeline's answer is scored on four sub-metrics
  inside `RagPipeline.answer` (`src/pipelines.py`): `recall` (fraction of
  ground-truth context titles recovered by retrieval, `src/metrics.py`),
  `correctness`, `groundedness`, and `p95_latency_ms` (the measured latency of
  that sample). The results are written verbatim to
  `results/eval_A_details.json` and `results/eval_B_details.json`.
- **Correctness** = does the generated answer contain the key factual content
  of the ground-truth answer (`judge.judge_correctness`).
- **Groundedness** = is every claim in the answer supported by the retrieved
  context (`judge.judge_groundedness`). For the LLM rubric the model is
  explicitly told to score based **only on the provided context and to ignore
  real-world knowledge**.
- **Aggregation** (`src/metrics.py::aggregate_details`) reduces the 40
  per-question scores into mean Recall / Correctness / Groundedness and the
  95th-percentile latency, then `src/regression.py` computes
  `delta = candidate - baseline` and produces `results/regression_report.csv`.

## 3. How does the harness run automatically / in CI (production_ci)?

- `.github/workflows/ci.yml` runs on every push/PR: installs pinned deps, runs
  the 28 unit tests, executes `python run_evaluation.py`, and re-validates that
  a fresh run reproduces the committed `results/regression_report.csv`.
- `python run_evaluation.py` is a self-contained entry point that always
  regenerates the three artifacts, so any change to a pipeline, the dataset, or
  the prompts is immediately reflected as a regression flag.
- `docker-compose.yml` adds a healthchecked Qdrant service and a harness
  service whose healthcheck (`src/healthcheck.py`) fails if any artifact is
  missing or the report does not flag at least one regression.

## 4. Why does the regression report look the way it does (regression_cause)?

Pipeline A (`config/eval_pins.json` → `"pipeline_a": {"top_k": 3, "rerank":
false}`) is the baseline. Pipeline B is the candidate:
`"pipeline_b": {"top_k": 6, "rerank": true}` — **expanded top-k retrieval plus
a lexical reranker** (`src/rerank.py`). B's generator also runs a higher-
verbosity mode (`src/generator.py`) and a groundedness-verification guardrail
over the full knowledge base (`src/pipelines.py::_verify_groundedness`).

The committed `results/regression_report.csv`:

| metric       | baseline | candidate | delta   | flag    |
|--------------|----------|-----------|---------|---------|
| Recall       | 0.925    | 0.95      | 0.025   | CLEAN   |
| Correctness  | 0.8852   | 0.9018    | 0.0167  | CLEAN   |
| Groundedness | 1.0      | 0.925     | -0.075  | REGRESS |
| P95_Latency  | 1.67     | 6.57      | 4.91    | REGRESS |

Cause: B's broader window recovers more relevant context (Recall and
Correctness up), but the reranked window occasionally promotes documents that
share vocabulary with the question while missing the specific fact, and the
verbosity mode then elaborates with content that is not backed by the retrieved
context — which the groundedness rubric correctly flags. The widened window,
reranking, and guardrail also push P95 latency far past the 1.10× threshold.

## 5. What are the latency trade-offs of reranking (latency_tradeoffs)?

Reranking moves work from embedding retrieval (cheap, vector dot-product, done
once per question) into a second scoring stage over the candidate window
(`src/rerank.py`). In **Pipeline B** this shows up directly in the measured
P95: `1.67 ms → 6.57 ms` (≈3.9×), which trips the report's 1.10× REGRESS
threshold for latency. The **hostage to this trade-off**: better Recall means
the answer can be drafted from a broader, re-scored window — but the same
added machinery (reranker + full-corpus groundedness-verification guardrail in
`src/pipelines.py`) is precisely what consumes the latency budget. The harness
therefore treats a good quality win (Recall/Correctness) and a latency
regression as independent signals, so a candidate cannot hide a latency
increase behind better recall.

## Regression harness conclusion

The experiment validates the harness's purpose: **Pipeline B improves recall
and correctness but regresses both groundedness and latency**, and the report
flags those regressions with a repeatable, schema-consistent
`results/regression_report.csv`. Re-running `python run_evaluation.py` on the
pinned dataset is deterministic (offline judge, TF-IDF embeddings), so the
baseline-vs-candidate comparison stays honest and auditable.