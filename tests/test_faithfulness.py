"""Verodostojnost: parsiranje tvrdnji, metrika, alarmi, zbir."""
import pytest

from faithfulness import (FAITHFULNESS_PROMPTS, claim_flags, faithfulness_fingerprint, parse_claims,
                          score_claims, summarize, ungrounded_tokens)

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


def test_claim_flag_number_not_in_context_ignores_question_numbers():
    claims = [{"claim": "For 3,200 EUR the fee is 18 EUR", "verdict": S}, {"claim": "The fee is 25 EUR", "verdict": S}]
    assert claim_flags(claims, "Badge fee | 18 EUR (charged in SGD)", "Fee for 3,200 EUR?") == ["number_not_in_context:2"]


def test_invented_name_from_pattern_is_caught():
    """Regresija: ocenjivač f1 je 'NS-Guest-LIS' proglasio podržanim (postoji samo NS-Guest-PRG)."""
    ctx = "Network: NS-Guest-PRG. Each office provides a separate guest network; names differ by office."
    assert ungrounded_tokens("NS-Guest-LIS", ctx, "Guest Wi-Fi in Lisbon?") == {"numbers": [], "names": ["NS-Guest-LIS"]}
    assert claim_flags([{"claim": "NS-Guest-LIS", "verdict": S}], ctx) == ["name_not_in_context:1"]


def test_names_present_in_context_or_question_are_fine():
    assert ungrounded_tokens("Use NS-Corp and memo PT-2026-02; e-mailing is not allowed. E-mailing too.",
                             "NS-Corp ... Memo PT-2026-02", "")["names"] == []


def test_summarize_excludes_answers_without_claims():
    rows = [{"id": "Q1", "judge_error": False, "claims": [{"verdict": S}], "n_claims": 1,
             "faithfulness": 1.0, "fully_grounded": True},
            {"id": "Q2", "judge_error": False, "claims": [], "n_claims": 0,
             "faithfulness": None, "fully_grounded": None},
            {"id": "Q3", "judge_error": True, "claims": []}]
    s = summarize(rows)
    assert (s["mean_faithfulness"], s["no_claims"], s["judge_errors"], s["fully_grounded_answers"]) == \
           (1.0, ["Q2"], ["Q3"], "1/1")
