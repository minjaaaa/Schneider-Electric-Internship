"""Verodostojnost: parsiranje tvrdnji, metrika, alarmi, zbir."""
import pytest

from faithfulness import (FAITHFULNESS_PROMPTS, claim_flags, faithfulness_fingerprint, parse_claims,
                          score_claims, summarize)

S, U, X = "supported", "unsupported", "contradicted"


def test_faithfulness_fingerprint_frozen():
    assert {v: faithfulness_fingerprint(v) for v in FAITHFULNESS_PROMPTS} == {"f1": "25cfad73"}


def test_parse_claims_fenced_and_case():
    r = parse_claims('```json\n{"claims": [{"claim": "Cap is 220 EUR", "verdict": "Supported", '
                     '"source": "Source: 5.2"}]}\n```')
    assert not r["judge_error"] and r["claims"] == [{"claim": "Cap is 220 EUR", "verdict": S, "source": "5.2"}]


def test_parse_claims_empty_list_is_valid():
    assert parse_claims('{"claims": []}') == {"claims": [], "judge_error": False}


@pytest.mark.parametrize("text", ["no json", '{"claims": [{"claim": "x", "verdict": "maybe"}]}',
                                  '{"claims": [{"claim": "", "verdict": "supported"}]}'])
def test_parse_claims_errors(text):
    assert parse_claims(text)["judge_error"]


def test_score_claims():
    s = score_claims([{"verdict": S}, {"verdict": S}, {"verdict": U}, {"verdict": X}])
    assert (s["faithfulness"], s["fully_grounded"], s["n_unsupported"], s["n_contradicted"]) == (0.5, False, 1, 1)
    assert score_claims([])["faithfulness"] is None


def test_claim_flag_number_not_in_context():
    claims = [{"claim": "The fee is 18 EUR", "verdict": S}, {"claim": "The fee is 25 EUR", "verdict": S}]
    assert claim_flags(claims, "Badge fee | 18 EUR (charged in SGD)") == ["number_not_in_context:2"]


def test_summarize_excludes_answers_without_claims():
    rows = [{"id": "Q1", "judge_error": False, "claims": [{"verdict": S}], "n_claims": 1,
             "faithfulness": 1.0, "fully_grounded": True},
            {"id": "Q2", "judge_error": False, "claims": [], "n_claims": 0,
             "faithfulness": None, "fully_grounded": None},
            {"id": "Q3", "judge_error": True, "claims": []}]
    s = summarize(rows)
    assert (s["mean_faithfulness"], s["no_claims"], s["judge_errors"], s["fully_grounded_answers"]) == \
           (1.0, ["Q2"], ["Q3"], "1/1")
