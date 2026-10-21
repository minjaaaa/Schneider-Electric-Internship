"""Celovitost evaluacionog skupa i sačuvanih rezultata (hvata pokvaren ili nepotpun fajl pre analize)."""
import json

import pytest


def test_every_question_has_facts_with_core(questions, facts):
    assert {q["id"] for q in questions} == set(facts)
    for qid, fs in facts.items():
        assert fs and any(f["core"] for f in fs), qid
        assert all(f["text"].strip() for f in fs), qid


def test_human_reference_matches_facts(root, facts):
    path = root / "eval" / "judge_validation" / "human_verdicts.jsonl"
    if not path.exists():
        pytest.skip("nema reference")
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        assert len(r["verdicts"]) == len(facts[r["id"]]), (r["run"], r["id"])
        assert set(r["verdicts"]) <= {"present", "missing", "contradicted"}


def test_generation_runs_cover_all_questions(root, questions):
    ids = {q["id"] for q in questions}
    runs = sorted((root / "results_generator").glob("*/answers.jsonl"))
    assert runs, "nema pokretanja u results_generator/"
    for f in runs:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert {r["id"] for r in rows} == ids, f.parent.name
        assert all(len(r["context_ids"]) == 5 for r in rows), f.parent.name
