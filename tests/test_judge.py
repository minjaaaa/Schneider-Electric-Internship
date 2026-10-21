"""Ocenjivač tačnosti: parsiranje verdikta, pravilo ocene, provera doslednosti, prosleđivanje verzije."""
import json

import pytest

from judge import (JUDGE_PROMPTS, consistency_flags, grade_from_verdicts, judge_fingerprint, judge_run,
                   numbers, parse_verdicts)

P, M, X = "present", "missing", "contradicted"
FACTS = [{"text": "Gifts up to 50 EUR per gift", "core": True},
         {"text": "At most 150 EUR per year", "core": True},
         {"text": "Change comes from memo PT-2026-02", "core": False}]


def test_judge_fingerprints_frozen():
    assert {v: judge_fingerprint(v) for v in JUDGE_PROMPTS} == {"j1": "0652b572", "j2": "cc2cb3bf"}


@pytest.mark.parametrize("text", [
    '{"note": "ok", "verdicts": {"1": "present", "2": "missing", "3": "contradicted"}}',
    '```json\n{"note": "ok", "verdicts": {"1": "Present", "2": "MISSING", "3": "contradicted"}}\n```',
    '{"note": nevazeci json, "verdicts": {"1": "present", "2": "missing", "3": "contradicted"}}',
])
def test_parse_verdicts_variants(text):
    r = parse_verdicts(text, 3)
    assert r["verdicts"] == [P, M, X] and not r["judge_error"]


def test_parse_verdicts_missing_verdict_is_error():
    assert parse_verdicts('{"verdicts": {"1": "present"}}', 3)["judge_error"]


@pytest.mark.parametrize("verdicts, grade", [
    ([P, P, P], "correct"),
    ([P, P, M], "correct"),            # dodatna činjenica fali -> i dalje tačno
    ([P, M, P], "partial"),            # fali ključna
    ([P, P, X], "partial"),            # dodatna u suprotnosti
    ([X, P, P], "wrong"),              # ključna u suprotnosti
    ([M, M, P], "wrong"),              # nijedna ključna nije prisutna
])
def test_grade_rule(verdicts, grade):
    assert grade_from_verdicts(FACTS, verdicts)["judge_grade"] == grade


def test_grade_metrics():
    g = grade_from_verdicts(FACTS, [P, M, X])
    assert (g["fact_recall"], g["core_recall"], g["n_contradicted"]) == (0.333, 0.5, 1)


def test_numbers_normalization():
    assert numbers("1,500 EUR and 2025.3") == {"1500", "2025.3"}
    assert numbers("within one hour", words=True) == {"1"} and numbers("about one") == set()


def test_flag_false_contradiction():
    """Lažna kontradikcija iz j1: '220' bez valute proglašen kao contradicted."""
    facts = [{"text": "The hotel cap in Singapore is 220 EUR per night", "core": True}]
    assert consistency_flags(facts, [X], "currency missing", "220", "Hotel cap in Singapore?") == ["numbers_in_answer:1"]


def test_flag_lenient_present_ignores_question_numbers():
    facts = [{"text": "No, a T2 technician cannot use Work from Anywhere", "core": True},
             {"text": "Carried-over days must be taken by 30 April", "core": True}]
    flags = consistency_flags(facts, [P, P], "all facts present", "No, 3 days", "Can a T2 technician ...?")
    assert flags == ["numbers_missing:2"]


def test_flag_note_vs_verdicts():
    assert "note_vs_verdicts" in consistency_flags(FACTS, [P, M, P], "All facts present.", "x", "")


def test_judge_run_passes_prompt_version(tmp_path):
    """Regresija: judge_run je ranije uvek koristio podrazumevanu verziju prompta."""
    (tmp_path / "answers.jsonl").write_text(json.dumps(
        {"id": "Q1", "category": "text", "question": "q", "answer": "a"}) + "\n", encoding="utf-8")
    seen = []

    def fake_llm(messages):
        seen.append(messages[0]["content"])
        return '{"note": "x", "verdicts": {"1": "present", "2": "present", "3": "present"}}'

    judge_run(tmp_path, {"Q1": FACTS}, fake_llm, "t", "j1")
    assert seen == [JUDGE_PROMPTS["j1"]]
