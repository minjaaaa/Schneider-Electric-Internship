"""Validacija LLM ocenjivača i poređenje sistema na osnovu njegovih ocena - lokalno, bez GPU-a i bez LLM-a.

Ocenjivač je merni instrument: pre nego što njegove ocene koristimo za zaključke (npr. "Gemma je bolja"),
proveravamo ga na tri načina:
  1. pouzdanost   - isti ocenjivač, ponovljen prolaz -> isti verdikti? (--compare-tag)
  2. valjanost    - slaganje sa ljudskom referencom po ISTOJ rubrici (Cohen's kappa, matrice neslaganja)
  3. pristrasnost - da li greši sistematski: broj "correct" prema referenci po pokretanju, greške po
                    kategoriji pitanja i po dužini odgovora
Zatim, ako se traži (--pairs), McNemar između pokretanja sa Holm korekcijom za višestruka poređenja.

Upotreba (iz korena repoa):
  python src/validate_judge.py --tag qwen14b_j2
  python src/validate_judge.py --tag qwen14b_j2 --compare-tag qwen14b_j2_run2
  python src/validate_judge.py --tag qwen14b_j2 --pairs v2b_qwen7b:v2b_gemma4-12b v2b_qwen7b:v2c_qwen7b
  python src/validate_judge.py --tag qwen14b_j2 --import <folder sa raspakovanim zipom iz Colab-a>

Ocene ocenjivača se čitaju iz results_generator/<pokretanje>/judge_<tag>.jsonl.
Referenca: eval/judge_validation/human_verdicts.jsonl (verdikti po činjenicama iz eval/expected_facts.json).
Izveštaj se upisuje u results_generator/judge_validation_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from math import comb
from pathlib import Path

from judge import consistency_flags, grade_from_verdicts, load_facts

VERDICTS = ["present", "missing", "contradicted"]
GRADES = ["correct", "partial", "wrong"]
SHORT_ANSWER_WORDS = 5          # odgovor do 5 reči smatra se kratkim (npr. "220", "No", "3 days")


# ------------------------------statistika---------------------------------- 

def cohen_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> dict:
    """Cohen's kappa = (p_o - p_e) / (1 - p_e).
    p_o: posmatrano slaganje; p_e: slaganje koje se očekuje slučajno iz proporcija oznaka oba ocenjivača."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    p_o = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    p_e = sum(ca[l] * cb[l] for l in labels) / n ** 2
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0
    return {"n": n, "agreement": round(p_o, 3), "chance_agreement": round(p_e, 3), "kappa": round(kappa, 3)}


def confusion(pairs: list[tuple[str, str]], labels: list[str]) -> dict:
    """Matrica neslaganja: red = referenca (čovek), kolona = ocenjivač."""
    return {a: {b: sum(1 for x, y in pairs if x == a and y == b) for b in labels} for a in labels}


def mcnemar_exact(a: dict[str, str], b: dict[str, str]) -> dict:
    """Egzaktni McNemar za "correct" prema svemu ostalom, na istim pitanjima.
    Gledaju se samo pitanja na kojima se sistemi razlikuju: only_b (tačan samo B) i only_a (tačan samo A).
    H0: svako takvo pitanje je 50:50  ->  p = 2 * P(X <= min) za X ~ Bin(only_a + only_b, 0.5)."""
    ids = sorted(set(a) & set(b))
    only_a = [q for q in ids if a[q] == "correct" and b[q] != "correct"]
    only_b = [q for q in ids if b[q] == "correct" and a[q] != "correct"]
    n, k = len(only_a) + len(only_b), min(len(only_a), len(only_b))
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n_questions": len(ids), "correct_a": sum(a[q] == "correct" for q in ids),
            "correct_b": sum(b[q] == "correct" for q in ids),
            "only_a": only_a, "only_b": only_b, "p": round(p, 4)}


def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm korekcija: p-vrednosti se sortiraju, i-ta najmanja se poredi sa alpha / (m - i).
    Čim jedna ne prođe, ne prolaze ni sve veće. Vraća listu True/False (značajno) u originalnom redosledu."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    significant, still_ok = [False] * len(pvalues), True
    for rank, i in enumerate(order):
        still_ok = still_ok and pvalues[i] <= alpha / (len(pvalues) - rank)
        significant[i] = still_ok
    return significant


# -------------------------------učitavanje--------------------------------- 

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def find_run(results_dir: Path, key: str) -> Path:
    """Pokretanje po celom imenu foldera ili po jedinstvenom kraju imena (npr. 'v2b_gemma4-12b')."""
    exact = results_dir / key
    if exact.is_dir():
        return exact
    hits = [d for d in results_dir.iterdir() if d.is_dir() and d.name.endswith(key)]
    if len(hits) != 1:
        raise SystemExit(f"'{key}' odgovara pokretanjima: {[d.name for d in hits] or 'nijednom'}")
    return hits[0]


def load_judged(results_dir: Path, tag: str) -> dict[str, dict]:
    """{ime pokretanja: {"answers": {id: red}, "judge": {id: red}}} za sva pokretanja ocenjena tagom."""
    runs = {}
    for f in sorted(results_dir.glob(f"*/judge_{tag}.jsonl")):
        runs[f.parent.name] = {"answers": {r["id"]: r for r in read_jsonl(f.parent / "answers.jsonl")},
                               "judge": {r["id"]: r for r in read_jsonl(f)}}
    return runs


def import_flat_files(src: Path, results_dir: Path) -> list[str]:
    """Fajlove iz Colab zipa ('<pokretanje>__judge_<tag>.jsonl') kopira u results_generator/<pokretanje>/."""
    copied = []
    for f in sorted(src.glob("*__judge_*.jsonl")):
        run, name = f.name.split("__", 1)
        if (results_dir / run).is_dir():
            shutil.copy(f, results_dir / run / name)
            copied.append(f"{run}/{name}")
        else:
            print(f"  preskačem {f.name}: nema foldera {run}")
    return copied


# -------------------------------delovi izveštaja---------------------------------

def overview(runs: dict, facts: dict) -> dict:
    """Po pokretanju: ocene, greške ocenjivača, alarmi (ponovo izračunati iz odgovora), prosečan fact_recall."""
    out = {}
    for name, r in runs.items():
        judged = r["judge"].values()
        flagged = [q for q, j in r["judge"].items() if not j["judge_error"] and consistency_flags(
            facts[q], j["verdicts"], j["note"], r["answers"][q]["answer"], r["answers"][q]["question"])]
        ok = [j for j in judged if not j["judge_error"]]
        out[name] = {"grades": {g: sum(j.get("judge_grade") == g for j in judged) for g in GRADES},
                     "judge_errors": sorted(j["id"] for j in judged if j["judge_error"]),
                     "flagged": flagged,
                     "fact_recall": round(sum(j["fact_recall"] for j in ok) / len(ok), 3) if ok else None}
    return out


def reproducibility(runs_a: dict, runs_b: dict) -> dict:
    """Dva prolaza istog ocenjivača: koliko je izlaza, verdikta i ocena identično."""
    out = {}
    for name in sorted(set(runs_a) & set(runs_b)):
        a, b = runs_a[name]["judge"], runs_b[name]["judge"]
        ids = sorted(set(a) & set(b))
        out[name] = {"n": len(ids),
                     "identical_raw": sum(a[q]["judge_raw"] == b[q]["judge_raw"] for q in ids),
                     "identical_verdicts": sum(a[q]["verdicts"] == b[q]["verdicts"] for q in ids),
                     "identical_grades": sum(a[q].get("judge_grade") == b[q].get("judge_grade") for q in ids),
                     "different_grades": [q for q in ids if a[q].get("judge_grade") != b[q].get("judge_grade")]}
    return out


def validity(runs: dict, reference: list[dict], facts: dict) -> dict:
    """Slaganje ocenjivača sa referencom: verdikti, ocene, pristrasnost po pokretanju/kategoriji/dužini."""
    v_pairs, g_pairs, details = [], [], []
    per_run = {}
    for ref in reference:
        run, q = ref["run"], ref["id"]
        if run not in runs or q not in runs[run]["judge"]:
            continue
        j, ans = runs[run]["judge"][q], runs[run]["answers"][q]
        ref_grade = grade_from_verdicts(facts[q], ref["verdicts"])["judge_grade"]
        pr = per_run.setdefault(run, {"n": 0, "ref_correct": 0, "judge_correct": 0, "grade_agree": 0})
        pr["n"] += 1
        pr["ref_correct"] += ref_grade == "correct"
        if j["judge_error"]:
            details.append({"run": run, "id": q, "judge_error": True})
            continue
        v_pairs += list(zip(ref["verdicts"], j["verdicts"]))
        g_pairs.append((ref_grade, j["judge_grade"]))
        pr["judge_correct"] += j["judge_grade"] == "correct"
        pr["grade_agree"] += j["judge_grade"] == ref_grade
        details.append({"run": run, "id": q, "category": ans["category"],
                        "short": len(ans["answer"].split()) <= SHORT_ANSWER_WORDS,
                        "ref_grade": ref_grade, "judge_grade": j["judge_grade"],
                        "verdict_diffs": [{"fact": i, "text": facts[q][i - 1]["text"], "ref": x, "judge": y}
                                          for i, (x, y) in enumerate(zip(ref["verdicts"], j["verdicts"]), 1) if x != y]})

    def error_rate(key):
        groups = {}
        for d in details:
            if "ref_grade" in d:
                groups.setdefault(str(d[key]), []).append(d["ref_grade"] != d["judge_grade"])
        return {g: {"n": len(v), "grade_errors": sum(v)} for g, v in sorted(groups.items())}

    return {"verdicts": {**cohen_kappa(v_pairs, VERDICTS), "confusion": confusion(v_pairs, VERDICTS)},
            "grades": {**cohen_kappa(g_pairs, GRADES), "confusion": confusion(g_pairs, GRADES)},
            "per_run": per_run,
            "grade_errors_by_category": error_rate("category"),
            "grade_errors_by_answer_length": {("short" if k == "True" else "long"): v
                                              for k, v in error_rate("short").items()},
            "judge_errors": [d for d in details if d.get("judge_error")],
            "disagreements": [d for d in details if d.get("verdict_diffs")]}


def compare_systems(runs: dict, pairs: list[str], results_dir: Path) -> list[dict]:
    """McNemar za svaki par 'A:B' (ocene ocenjivača) + Holm korekcija preko svih traženih parova."""
    out = []
    for pair in pairs:
        ka, kb = pair.split(":")
        na, nb = find_run(results_dir, ka).name, find_run(results_dir, kb).name
        if na not in runs or nb not in runs:
            raise SystemExit(f"Nema ocena ovog ocenjivača za {na if na not in runs else nb}")
        grades = lambda n: {q: j.get("judge_grade", "error") for q, j in runs[n]["judge"].items()}
        out.append({"a": na, "b": nb, **mcnemar_exact(grades(na), grades(nb))})
    for row, sig in zip(out, holm([r["p"] for r in out])):
        row["significant_holm_0.05"] = sig
    return out


# -------------------------------ispis---------------------------------

def print_matrix(title: str, m: dict) -> None:
    labels = list(m)
    print(f"  {title} (red = referenca, kolona = ocenjivač)")
    print("    " + " " * 13 + "".join(f"{l[:12]:>13}" for l in labels))
    for a in labels:
        print(f"    {a[:12]:<13}" + "".join(f"{m[a][b]:>13}" for b in labels))


def main() -> None:
    from config import EVAL_DIR, GEN_RESULTS_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="JUDGE_TAG iz Colab-a, npr. qwen14b_j2")
    ap.add_argument("--compare-tag", help="drugi prolaz istog ocenjivača, za proveru ponovljivosti")
    ap.add_argument("--reference", default=str(EVAL_DIR / "judge_validation" / "human_verdicts.jsonl"))
    ap.add_argument("--facts", default=str(EVAL_DIR / "expected_facts.json"))
    ap.add_argument("--pairs", nargs="*", default=[], help="parovi pokretanja A:B za McNemar")
    ap.add_argument("--import", dest="import_dir", help="folder sa fajlovima '<pokretanje>__judge_<tag>.jsonl' iz zipa")
    args = ap.parse_args()

    if args.import_dir:
        copied = import_flat_files(Path(args.import_dir), GEN_RESULTS_DIR)
        print(f"Uvezeno {len(copied)} fajlova u {GEN_RESULTS_DIR.name}/")

    facts = load_facts(Path(args.facts))
    runs = load_judged(GEN_RESULTS_DIR, args.tag)
    if not runs:
        raise SystemExit(f"Nema fajlova judge_{args.tag}.jsonl u {GEN_RESULTS_DIR}")
    report = {"tag": args.tag, "facts_file": args.facts, "overview": overview(runs, facts)}

    print(f"=== Ocenjivač '{args.tag}': {len(runs)} pokretanja ===")
    for name, o in report["overview"].items():
        print(f"  {name:<45} {o['grades']}  fact_recall={o['fact_recall']}  "
              f"greške={len(o['judge_errors'])}  alarmi={len(o['flagged'])}")

    if args.compare_tag:
        report["reproducibility"] = reproducibility(runs, load_judged(GEN_RESULTS_DIR, args.compare_tag))
        print(f"\n=== Ponovljivost: '{args.tag}' prema '{args.compare_tag}' ===")
        for name, r in report["reproducibility"].items():
            print(f"  {name:<45} izlazi {r['identical_raw']}/{r['n']}  verdikti {r['identical_verdicts']}/{r['n']}"
                  f"  ocene {r['identical_grades']}/{r['n']}  {r['different_grades'] or ''}")

    ref_path = Path(args.reference)
    if ref_path.exists():
        v = validity(runs, read_jsonl(ref_path), facts)
        report["validity"] = v
        print(f"\n=== Valjanost prema referenci ({ref_path.name}) ===")
        for level in ("verdicts", "grades"):
            x = v[level]
            print(f"  {level:<9} n={x['n']}  slaganje={x['agreement']}  slučajno={x['chance_agreement']}  kappa={x['kappa']}")
        print_matrix("verdikti", v["verdicts"]["confusion"])
        print_matrix("ocene", v["grades"]["confusion"])
        print("  Pristrasnost - broj 'correct' po pokretanju (referenca / ocenjivač):")
        for run, pr in v["per_run"].items():
            print(f"    {run:<45} {pr['ref_correct']:>3} / {pr['judge_correct']:<3}  slaganje ocena {pr['grade_agree']}/{pr['n']}")
        print("  Greške u oceni po dužini odgovora:", v["grade_errors_by_answer_length"])
        print("  Greške u oceni po kategoriji:", {k: f"{x['grade_errors']}/{x['n']}"
                                                  for k, x in v["grade_errors_by_category"].items()})
        print(f"  Neslaganja na nivou verdikta: {sum(len(d['verdict_diffs']) for d in v['disagreements'])} "
              f"(spisak u izveštaju)")
    else:
        print(f"\n(nema reference {ref_path} - valjanost se ne računa)")

    if args.pairs:
        report["comparisons"] = compare_systems(runs, args.pairs, GEN_RESULTS_DIR)
        print(f"\n=== McNemar (ocene ocenjivača '{args.tag}'), Holm za {len(args.pairs)} poređenja ===")
        for c in report["comparisons"]:
            print(f"  {c['a']}  vs  {c['b']}\n    tačnih {c['correct_a']} : {c['correct_b']}  |  samo A {len(c['only_a'])}"
                  f"  samo B {len(c['only_b'])}  |  p={c['p']}  značajno (Holm): {c['significant_holm_0.05']}")

    out = GEN_RESULTS_DIR / f"judge_validation_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nIzveštaj: {out}")


if __name__ == "__main__":
    main()