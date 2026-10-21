"""Statistika u validate_judge.py, proverena na brojevima iz izveštaja."""
import pytest

from validate_judge import cohen_kappa, holm, mcnemar_exact


def pairs_from_confusion(matrix, labels):
    return [(a, b) for a, row in zip(labels, matrix) for b, n in zip(labels, row) for _ in range(n)]


def test_kappa_matches_report():
    """Ocene j2 prema referenci: 85/98 slaganje, 44% slučajno -> kappa 0,763."""
    labels = ["correct", "partial", "wrong"]
    k = cohen_kappa(pairs_from_confusion([[52, 2, 2], [3, 27, 4], [0, 2, 6]], labels), labels)
    assert (k["agreement"], k["chance_agreement"], k["kappa"]) == (0.867, 0.44, 0.763)


def test_kappa_always_same_label_is_not_better_than_chance():
    """Ocenjivač koji uvek kaže 'correct' slaže se u 60%, ali kappa je 0."""
    pairs = [("correct", "correct")] * 6 + [("wrong", "correct")] * 4
    assert cohen_kappa(pairs, ["correct", "wrong"])["kappa"] == 0.0


def test_mcnemar_matches_report():
    a = {f"Q{i}": "correct" if i <= 22 else "wrong" for i in range(1, 50)}       # 22 tačna
    b = dict(a)
    b["Q1"] = "partial"                                                         # samo A: 1
    for i in range(23, 35):                                                     # samo B: 12
        b[f"Q{i}"] = "correct"
    r = mcnemar_exact(a, b)
    assert (len(r["only_a"]), len(r["only_b"]), r["p"]) == (1, 12, 0.0034)


def test_mcnemar_no_discordant_pairs():
    assert mcnemar_exact({"Q1": "correct"}, {"Q1": "correct"})["p"] == 1.0


@pytest.mark.parametrize("p, expected", [
    ([0.0034, 0.2188, 0.424], [True, False, False]),     # izveštaj, m = 3
    ([0.01, 0.04, 0.03], [True, False, False]),           # 0,03 > 0,05/2 -> stop
    ([0.001, 0.2, 0.012], [True, False, True]),           # redosled ulaza se čuva
])
def test_holm(p, expected):
    assert holm(p) == expected
