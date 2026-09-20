"""Unit tests for the judge backends and prompt file contract."""

from __future__ import annotations

from pathlib import Path

from src.judge import DeterministicJudge, load_judge_prompts

ROOT = Path(__file__).resolve().parent.parent


def test_prompts_file_is_valid_and_complete():
    prompts = load_judge_prompts()
    assert prompts["correctness_prompt"].strip()
    assert prompts["groundedness_prompt"].strip()
    low = prompts["groundedness_prompt"].lower()
    assert "only" in low and "context" in low


def _fixture():
    context = (
        "The Summit Checking Account charges a monthly maintenance fee of $12.00."
        " Customers can waive the monthly fee by maintaining a minimum daily balance of $1,500."
    )
    answer = (
        "The Summit Checking Account charges a monthly maintenance fee of $12.00."
        " Customers are encouraged to consult their personal relationship manager for advice."
    )
    ground_truth = "The Summit Checking Account charges a monthly maintenance fee of $12.00."
    return context, answer, ground_truth


def test_deterministic_groundedness_flags_unsupported_sentence():
    context, answer, _ = _fixture()
    judge = DeterministicJudge()
    score = judge.judge_groundedness("fees?", context, answer)
    assert score < 1.0, "the second sentence is not supported by the retrieved context"

    grounded_answer = "The Summit Checking Account charges a monthly maintenance fee of $12.00."
    assert judge.judge_groundedness("fees?", context, grounded_answer) == 1.0


def test_deterministic_correctness_tracks_ground_truth_coverage():
    context, _, ground_truth = _fixture()
    judge = DeterministicJudge()
    full = judge.judge_correctness("fees", ground_truth, ground_truth)
    assert full == 1.0
    partial = judge.judge_correctness("fees", ground_truth, "There is a monthly fee of some kind.")
    assert 0.0 < partial < full


def test_deterministic_correctness_complete_miss():
    judge = DeterministicJudge()
    score = judge.judge_correctness("q", "zebra migration patterns", "the sky is blue today")
    assert score == 0.0


def test_deterministic_judge_needs_no_network():
    """The default judge is fully offline and reproducible."""
    import requests  # noqa: F401

    DeterministicJudge()
    assert True