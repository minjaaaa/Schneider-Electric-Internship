"""Kvalitet konteksta: vrednosti izračunate ručno, ivični slučajevi, zamrznut kontekst."""
import json

import pytest

from context_metrics import average_precision, context_precision_at_k, context_recall, evaluate_run, score_question

CTX = ["A", "B", "C", "D", "E"]


def test_two_relevant_parts():
    """Relevantni na pozicijama 2 i 4: precision@5 = 2/5, AP = (1/2 + 2/4) / 2 = 0,5."""
    assert context_precision_at_k(CTX, ["B", "D"]) == 0.4
    assert average_precision(CTX, ["B", "D"]) == 0.5
    assert context_recall(CTX, ["B", "D"]) == 1.0


def test_relevant_on_top_gives_full_average_precision():
    assert average_precision(CTX, ["A", "B"]) == 1.0


def test_missing_source_lowers_recall_not_precision_rank():
    assert context_recall(CTX, ["B", "Z"]) == 0.5
    assert average_precision(CTX, ["B", "Z"]) == 0.5          # jedini relevantan je na 2. mestu


def test_no_relevant_part():
    assert (context_precision_at_k(CTX, ["Z"]), average_precision(CTX, ["Z"]), context_recall(CTX, ["Z"])) == (0, 0, 0)


def test_duplicate_label_counts_as_two_parts():
    """Dva FAQ dela imaju istu oznaku '14' - oba se računaju kao delovi konteksta."""
    assert context_precision_at_k(["14", "5.2", "14", "x", "y"], ["14"]) == 0.4


def test_question_without_sources_is_skipped():
    assert score_question({"id": "Q14", "category": "abstain", "sources": [], "context": CTX}) is None


def test_evaluate_run_summary_groups():
    rows = [{"id": "Q1", "category": "t", "sources": ["A"], "context": CTX},
            {"id": "Q2", "category": "t", "sources": ["B", "Z"], "context": CTX},
            {"id": "Q3", "category": "abstain", "sources": [], "context": CTX}]
    s = evaluate_run(rows)["summary"]
    assert (s["all"]["n"], s["one_source"]["n"], s["multi_source"]["n"]) == (2, 1, 1)
    assert s["all"]["context_recall"] == 0.75 and s["missing_sources"] == {"Q2": ["Z"]}


def test_frozen_context_matches_report(root):
    """Bazno pokretanje: brojevi iz izveštaja (recall 0,981, precision@5 0,289, AP 0,799)."""
    path = root / "results_generator" / "2026-10-14_1022_gen_hybrid_qwen7b" / "answers.jsonl"
    if not path.exists():
        pytest.skip("nema baznog pokretanja")
    s = evaluate_run([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()])["summary"]["all"]
    assert (s["context_recall"], s["context_precision_at_k"], s["average_precision"], s["all_sources_found"]) == \
           (0.981, 0.289, 0.799, 43)
