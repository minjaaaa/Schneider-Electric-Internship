"""Verodostojnost odgovora (faithfulness / groundedness): da li je svaka tvrdnja u odgovoru podržana
delovima dokumenta koje je generator DOBIO (kontekst), a ne očekivanim odgovorom.

Razlika od tačnosti (judge.py):
  tačnost         - odgovor prema očekivanom odgovoru (da li je rekao ono što treba)
  verodostojnost  - odgovor prema kontekstu (da li je izmislio nešto što u izvorima ne piše)
Odgovor može biti verodostojan a netačan (verno prepiše pogrešan red tabele) i tačan a neverodostojan
(pogodi vrednost iz opšteg znanja).

Postupak (jedan poziv ocenjivača po odgovoru):
  1. ocenjivač razloži ODGOVOR na atomske tvrdnje;
  2. za svaku tvrdnju kaže: supported (izvor to kaže, i drugim rečima), contradicted (izvor kaže nešto
     drugo za istu stvar) ili unsupported (nijedan izvor to ne kaže - izmišljeno ili opšte znanje);
  3. metrika se računa u kodu: udeo podržanih tvrdnji po odgovoru i udeo odgovora bez ijedne
     nepodržane tvrdnje. Odgovor bez tvrdnji (samo odbijanje) ne ulazi u prosek.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

CLAIM_VERDICTS = ("supported", "unsupported", "contradicted")

FAITHFULNESS_PROMPT_F1 = """You check whether an ANSWER is grounded in the SOURCES it was given.
The sources are the only allowed evidence. Do not use outside knowledge.

Step 1. Split the ANSWER into atomic claims: each claim states one fact (one value, one rule, one yes/no).
- Keep the answer's own wording as much as possible; resolve "it", "they" from the question.
- A sentence that only says the information is not available (for example "The knowledge base does not
  contain this information") is not a claim - skip it.
- Ignore citations, source labels and formatting.

Step 2. For each claim decide one verdict:
- "supported": a source states this claim, possibly in other words or number formats.
- "contradicted": a source gives a DIFFERENT value for the SAME thing (another number, the opposite
  yes/no, another person, entity, place or category).
- "unsupported": no source states this claim (invented, guessed, derived from a pattern, or general knowledge).
A claim about a different office, company, product or role than the source describes is not supported.
If you are unsure between "supported" and "unsupported", choose "unsupported".
For a supported or contradicted claim, give the label of the source ("Source: <label>").

Respond with one JSON object and nothing else:
{"claims": [{"claim": "<atomic claim>", "verdict": "supported|unsupported|contradicted", "source": "<label or empty>"}]}
If the answer contains no claims, respond with {"claims": []}."""

FAITHFULNESS_PROMPTS = {"f1": FAITHFULNESS_PROMPT_F1}
FAITHFULNESS_VERSION = "f1"


def faithfulness_fingerprint(version: str = FAITHFULNESS_VERSION) -> str:
    """Heš prompta - ide uz rezultate, kao kod generatora i ocenjivača tačnosti."""
    return hashlib.sha256(FAITHFULNESS_PROMPTS[version].encode("utf-8")).hexdigest()[:8]


def build_faithfulness_messages(question: str, context: str, answer: str,
                                version: str = FAITHFULNESS_VERSION) -> list[dict]:
    """Ocenjivač vidi pitanje, iste izvore koje je video generator, i odgovor."""
    user = f"SOURCES:\n\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer.strip() or '(empty answer)'}"
    return [{"role": "system", "content": FAITHFULNESS_PROMPTS[version]}, {"role": "user", "content": user}]


def parse_claims(text: str) -> dict:
    """Izvlači listu tvrdnji iz izlaza ocenjivača. Neispravan izlaz ili nepoznat verdikt -> judge_error."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return {"claims": [], "judge_error": True}
    claims = []
    for c in data["claims"]:
        if not isinstance(c, dict):
            return {"claims": [], "judge_error": True}
        verdict = str(c.get("verdict", "")).strip().lower()
        claims.append({"claim": str(c.get("claim", "")).strip(), "verdict": verdict,
                       "source": re.sub(r"^Source:?\s*", "", str(c.get("source", "") or "")).strip()})
    ok = all(c["verdict"] in CLAIM_VERDICTS and c["claim"] for c in claims)
    return {"claims": claims, "judge_error": not ok}


def score_claims(claims: list[dict]) -> dict:
    """Metrike jednog odgovora. Bez tvrdnji -> faithfulness None (ne ulazi u prosek)."""
    n = len(claims)
    counts = {v: sum(c["verdict"] == v for c in claims) for v in CLAIM_VERDICTS}
    return {"n_claims": n, **{f"n_{v}": k for v, k in counts.items()},
            "faithfulness": round(counts["supported"] / n, 3) if n else None,
            "fully_grounded": (counts["supported"] == n) if n else None}


NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# nazivi i šifre: NS-Guest-PRG, PT-2026-02, SAML2 ... (bar dva dela sa velikim slovom ili cifrom)
NAME_RE = re.compile(r"\b[A-Z0-9][A-Za-z0-9]*(?:-[A-Z0-9][A-Za-z0-9]*)+\b|\b[A-Z]{2,}[0-9]+\b")


def ungrounded_tokens(text: str, context: str, question: str = "") -> dict:
    """Provera bez LLM-a: brojevi i nazivi iz teksta kojih nema ni u kontekstu ni u pitanju.
    Hvata izmišljene šifre i nazive izvedene iz šablona (npr. NS-Guest-LIS iz NS-Guest-PRG),
    što LLM ocenjivač ne hvata pouzdano jer ima istu slabost kao generator."""
    ctx_nums = {n.replace(",", "") for n in NUMBER_RE.findall(context + " " + question)}
    nums = sorted({n.replace(",", "") for n in NUMBER_RE.findall(text)} - ctx_nums)
    names = sorted({t for t in NAME_RE.findall(text) if t not in context and t not in question})
    return {"numbers": nums, "names": names}


def claim_flags(claims: list[dict], context: str, question: str = "") -> list[str]:
    """Alarm (ne menja ocenu): tvrdnja i je 'supported', a sadrži broj ili naziv kog nema u kontekstu ni pitanju."""
    flags = []
    for i, c in enumerate(claims, start=1):
        bad = ungrounded_tokens(c["claim"], context, question)
        if c["verdict"] == "supported" and bad["numbers"]:
            flags.append(f"number_not_in_context:{i}")
        if c["verdict"] == "supported" and bad["names"]:
            flags.append(f"name_not_in_context:{i}")
    return flags


def faithfulness_answer(row: dict, context: str, llm, version: str = FAITHFULNESS_VERSION) -> dict:
    """Ocena jednog odgovora iz answers.jsonl; context = isti tekst izvora koji je video generator."""
    start = time.perf_counter()
    raw = llm(build_faithfulness_messages(row["question"], context, row["answer"], version))
    parsed = parse_claims(raw)
    out = {"id": row["id"], "category": row["category"], "abstained": row.get("abstained", False),
           **parsed, "judge_raw": raw, "judge_latency_s": round(time.perf_counter() - start, 2)}
    if not parsed["judge_error"]:
        out.update(score_claims(parsed["claims"]))
        out["flags"] = claim_flags(parsed["claims"], context, row["question"])
    out["ungrounded_in_answer"] = ungrounded_tokens(row["answer"], context, row["question"])
    return out


def faithfulness_run(run_dir: Path, contexts: dict[str, str], llm, tag: str,
                     version: str = FAITHFULNESS_VERSION) -> Path:
    """Ocenjuje sva pitanja jednog pokretanja; upisuje <run_dir>/faithfulness_<tag>.jsonl."""
    rows = [json.loads(line) for line in open(Path(run_dir) / "answers.jsonl", encoding="utf-8")]
    out = Path(run_dir) / f"faithfulness_{tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(faithfulness_answer(row, contexts[row["id"]], llm, version), ensure_ascii=False) + "\n")
    return out


def summarize(rows: list[dict]) -> dict:
    """Zbirne metrike pokretanja (odgovori bez tvrdnji i greške ocenjivača se izuzimaju iz proseka)."""
    ok = [r for r in rows if not r["judge_error"]]
    scored = [r for r in ok if r["n_claims"]]
    claims = [c for r in scored for c in r["claims"]]
    return {
        "n_answers": len(rows),
        "judge_errors": sorted(r["id"] for r in rows if r["judge_error"]),
        "no_claims": sorted(r["id"] for r in ok if not r["n_claims"]),
        "n_claims": len(claims),
        "claims": {v: sum(c["verdict"] == v for c in claims) for v in CLAIM_VERDICTS},
        "mean_faithfulness": round(sum(r["faithfulness"] for r in scored) / len(scored), 3) if scored else None,
        "fully_grounded_answers": f"{sum(r['fully_grounded'] for r in scored)}/{len(scored)}",
        "not_grounded": sorted(r["id"] for r in scored if not r["fully_grounded"]),
        "flagged": sorted(r["id"] for r in ok if r.get("flags")),
        "answers_with_ungrounded_tokens": {r["id"]: r["ungrounded_in_answer"] for r in rows
                                           if any((r.get("ungrounded_in_answer") or {}).values())},
    }


if __name__ == "__main__":
    import argparse

    from config import GEN_RESULTS_DIR

    ap = argparse.ArgumentParser(description="Zbirne metrike verodostojnosti za sva pokretanja ocenjena tagom.")
    ap.add_argument("--tag", required=True, help="npr. qwen14b_f1")
    ap.add_argument("--details", action="store_true", help="ispiši nepodržane i protivrečne tvrdnje")
    args = ap.parse_args()
    files = sorted(GEN_RESULTS_DIR.glob(f"*/faithfulness_{args.tag}.jsonl"))
    if not files:
        raise SystemExit(f"Nema fajlova faithfulness_{args.tag}.jsonl u {GEN_RESULTS_DIR}")
    from chunking import load_jsonl
    from config import DATA_PROCESSED
    from figures import figure_chunks
    from generate import format_context

    by_id = {c["chunk_id"]: c for c in load_jsonl(DATA_PROCESSED / "chunks_section.jsonl") + figure_chunks()}
    report = {}
    for f in files:
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        answers = {a["id"]: a for a in map(json.loads, open(f.parent / "answers.jsonl", encoding="utf-8"))}
        for r in rows:          # provera bez LLM-a se uvek računa iznova (i za rezultate iz starije verzije koda)
            a = answers[r["id"]]
            ctx = format_context([by_id[c] for c in a["context_ids"]], "labeled")
            r["ungrounded_in_answer"] = ungrounded_tokens(a["answer"], ctx, a["question"])
            if not r["judge_error"]:
                r["flags"] = claim_flags(r["claims"], ctx, a["question"])
        s = report[f.parent.name] = summarize(rows)
        print(f"=== {f.parent.name}")
        print(f"  tvrdnji: {s['n_claims']}  {s['claims']}  |  prosečna verodostojnost: {s['mean_faithfulness']}"
              f"  |  potpuno utemeljeni odgovori: {s['fully_grounded_answers']}")
        print(f"  bez tvrdnji: {s['no_claims']}  |  greške ocenjivača: {s['judge_errors']}  |  alarmi: {s['flagged']}")
        print(f"  odgovori sa brojem/nazivom kog nema u kontekstu (bez LLM-a): {s['answers_with_ungrounded_tokens'] or 'nema'}")
        if args.details:
            for r in rows:
                for c in r.get("claims", []):
                    if c["verdict"] != "supported":
                        print(f"    {r['id']} [{c['verdict']}] {c['claim']}")
    out = GEN_RESULTS_DIR / f"faithfulness_summary_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nIzveštaj: {out}")
