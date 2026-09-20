"""Unit tests for retrieval, recall and the corpus index."""

from __future__ import annotations

import json
from pathlib import Path

from src.corpus import load_corpus, chunk_text
from src.config import load_settings
from src.embeddings import build_embedder, embed_corpus
from src.metrics import recall_at_titles
from src.retrieval import retrieve_top_k, unique_titles

ROOT = Path(__file__).resolve().parent.parent


def _index():
    settings = load_settings()
    chunks = load_corpus(Path(settings["corpus_dir"]))
    embedder = build_embedder(settings["pipeline_a_embedder"])
    matrix = embed_corpus(embedder, chunks)
    return chunks, embedder, matrix


def test_corpus_has_all_documents_and_unique_titles():
    chunks, _, _ = _index()
    assert len(chunks) > 100, "corpus should contain 100+ sentence chunks"
    titles = {c.title for c in chunks}
    assert {"Summit Checking Account", "Atlas Credit Card", "Cloud Rewards Program"} <= titles


def test_chunk_text_joins_sentences():
    chunks, _, _ = _index()
    first = chunks[0]
    text = chunk_text(chunks[:2])
    assert first.sentence in text


def test_retrieval_recovers_ground_truth_document():
    chunks, embedder, matrix = _index()
    query = "What credit limit policies apply to the AtlaS Credit Card?".replace("AtlaS", "Atlas")
    query_vec = embedder.transform([query])[0]
    ranked = retrieve_top_k(chunks, query_vec, matrix, 6)
    titles = unique_titles(ranked)
    assert "Atlas Credit Card" in titles or "Credit Limit Increase Policy" in titles


def test_top_k_respected_and_scores_descending():
    chunks, embedder, matrix = _index()
    query_vec = embedder.transform(["What is the monthly maintenance fee for the Summit Checking Account?"])[0]
    ranked = retrieve_top_k(chunks, query_vec, matrix, 3)
    assert len(ranked) == 3
    scores = [float(item["score"]) for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_recall_metric():
    assert recall_at_titles(["A", "B"], ["A", "B", "C"]) == 1.0
    assert recall_at_titles(["A", "B"], ["A"]) == 0.5
    assert recall_at_titles(["A", "B"], ["C"]) == 0.0
    assert recall_at_titles([], ["A"]) == 0.0


def test_dataset_ground_truth_titles_exist_in_corpus():
    chunks, _, _ = _index()
    corpus_titles = {c.title for c in chunks}
    eval_set = json.loads((ROOT / "dataset" / "eval_set.json").read_text(encoding="utf-8"))
    for row in eval_set:
        for title in row["ground_truth_context_titles"]:
            assert title in corpus_titles, title