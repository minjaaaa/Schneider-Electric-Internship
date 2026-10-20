"""LLM ocenjivač (LLM-as-a-judge) za tačnost odgovora, po činjenicama.

Za svako pitanje u eval/expected_facts.json stoji lista atomskih činjenica iz očekivanog odgovora:
  core=True  - ono što pitanje direktno traži (bez toga odgovor nije tačan)
  core=False - dodatni detalj (utiče samo na fact_recall, ne na ocenu)
Ocenjivač za svaku činjenicu kaže: present / missing / contradicted. Ocena se zatim izvodi pravilom u kodu
(grade_from_verdicts), a ne prepušta modelu - tako je kriterijum isti za sve modele i proverljiv.

Ocenjivač vidi samo pitanje, činjenice i odgovor (ne kontekst): meri tačnost prema očekivanom odgovoru,
a ne verodostojnost prema izvorima (to je posebna metrika).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

VERDICTS = ("present", "missing", "contradicted")

JUDGE_PROMPT = """You are a strict grader. You compare an ANSWER to numbered reference FACTS for a QUESTION.
For each fact decide one verdict:
- "present": the answer states this fact, possibly in other words, number formats or units
  ("three" = "3", "one hour" = "1 hour", "20th" = "by the 20th of the month").
- "contradicted": the answer states something incompatible with this fact: a different value, the opposite
  yes/no, a different person or entity, or something the fact explicitly excludes.
- "missing": the answer neither states nor contradicts this fact.

Rules:
1. Judge only against the facts. Do not use outside knowledge. Do not judge style, length or politeness.
2. A fact is present only if the answer actually states it. A vague hint is not enough.
3. If the answer gives a wrong value for the thing a fact describes, that fact is contradicted,
   even if the answer also contains correct parts.
4. If a fact says the information is not available and the answer says so without inventing a value,
   the fact is present. If the answer invents a value instead, the fact is contradicted.
5. If the answer says the information is not available but a fact gives a value, that fact is missing.

Write a short note first, then the verdicts. Respond with one JSON object and nothing else:
{"note": "<one short sentence: what is missing or wrong, or 'all facts present'>",
 "verdicts": {"1": "present|missing|contradicted", "2": "..."}}"""

JUDGE_VERSION = "j1"


def judge_fingerprint() -> str:
    """Heš prompta ocenjivača - ide uz rezultate, da se zna tačno kojim promptom je ocenjeno."""
    return hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()[:8]


def load_facts(path: Path) -> dict[str, list[dict]]:
    """eval/expected_facts.json -> {id pitanja: [{"text": ..., "core": bool}, ...]}"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_judge_messages(question: str, facts: list[dict], answer: str) -> list[dict]:
    """Poruke za ocenjivača: pitanje, numerisane činjenice, odgovor koji se ocenjuje."""
    fact_lines = "\n".join(f"{i}. {f['text']}" for i, f in enumerate(facts, start=1))
    user = f"QUESTION: {question}\n\nFACTS:\n{fact_lines}\n\nANSWER: {answer.strip() or '(empty answer)'}"
    return [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}]


def parse_verdicts(text: str, n_facts: int) -> dict:
    """Izvlači note i verdikte iz izlaza ocenjivača. Fali li neki verdikt ili je nevažeći -> judge_error."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("verdicts"), dict):
        raw = {str(k).strip(): str(v).strip().lower() for k, v in data["verdicts"].items()}
        note = str(data.get("note", "")).strip()
    else:                                          # rezerva: "1": "present" parovi bilo gde u tekstu
        raw = {k: v.lower() for k, v in re.findall(r'"(\d+)"\s*:\s*"(present|missing|contradicted)"', text, re.I)}
        note = ""
    verdicts = [raw.get(str(i), "") for i in range(1, n_facts + 1)]
    ok = all(v in VERDICTS for v in verdicts)
    return {"verdicts": verdicts, "note": note, "judge_error": not ok}


def grade_from_verdicts(facts: list[dict], verdicts: list[str]) -> dict:
    """Pravilo ocene (isto za sve):
      wrong   - neka core činjenica je contradicted, ili nijedna core činjenica nije present
      correct - sve core činjenice su present i nijedna činjenica (ni dodatna) nije contradicted
      partial - sve ostalo (fali deo core činjenica, ili je dodatna činjenica contradicted)"""
    core = [v for f, v in zip(facts, verdicts) if f["core"]]
    if "contradicted" in core or "present" not in core:
        grade = "wrong"
    elif all(v == "present" for v in core) and "contradicted" not in verdicts:
        grade = "correct"
    else:
        grade = "partial"
    return {
        "judge_grade": grade,
        "fact_recall": round(verdicts.count("present") / len(verdicts), 3),
        "core_recall": round(core.count("present") / len(core), 3),
        "n_contradicted": verdicts.count("contradicted"),
    }


def judge_answer(row: dict, facts: list[dict], llm) -> dict:
    """Ocena jednog odgovora iz answers.jsonl."""
    start = time.perf_counter()
    raw = llm(build_judge_messages(row["question"], facts, row["answer"]))
    parsed = parse_verdicts(raw, len(facts))
    out = {"id": row["id"], "category": row["category"], **parsed, "judge_raw": raw,
           "judge_latency_s": round(time.perf_counter() - start, 2)}
    if not parsed["judge_error"]:
        out.update(grade_from_verdicts(facts, parsed["verdicts"]))
    return out


def judge_run(run_dir: Path, facts: dict, llm, judge_tag: str) -> Path:
    """Ocenjuje sva pitanja jednog pokretanja; upisuje <run_dir>/judge_<judge_tag>.jsonl i vraća putanju."""
    rows = [json.loads(line) for line in open(Path(run_dir) / "answers.jsonl", encoding="utf-8")]
    out = Path(run_dir) / f"judge_{judge_tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(judge_answer(row, facts[row["id"]], llm), ensure_ascii=False) + "\n")
    return out