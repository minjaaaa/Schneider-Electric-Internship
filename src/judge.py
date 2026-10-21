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

JUDGE_PROMPT_J1 = """You are a strict grader. You compare an ANSWER to numbered reference FACTS for a QUESTION.
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

# j2: uža definicija "contradicted" (j1 ju je koristio i za izostavljenu jedinicu, kraću formulaciju, dodatke),
# pravilo za nesigurne slučajeve i primeri sa IZMIŠLJENIM sadržajem (ne iz evaluacionog skupa).
JUDGE_PROMPT_J2 = """You are a careful grader. You compare an ANSWER to numbered reference FACTS for a QUESTION.
For each fact decide one verdict:
- "present": the answer states the content of this fact. Different wording, number formats, rounding-free
  equivalents ("three" = "3", "one hour" = "1 hour") and a missing unit or currency when the number is
  clearly the asked value ("220" for "220 EUR") all count as present.
- "contradicted": the answer gives a DIFFERENT value for the SAME thing the fact describes: another number,
  the opposite yes/no, another person, entity, place or category.
- "missing": the answer does not state this fact and does not give a different value for it.

These are NOT contradictions (use "present" or "missing" instead):
- a missing unit, currency or detail; shorter or longer wording;
- extra information in the answer that no fact talks about (ignore it completely);
- covering only part of a fact (that is "missing" unless the stated part is the whole fact).

Rules:
1. Judge each fact separately and only against the answer. Do not use outside knowledge.
2. If you are unsure between "contradicted" and "missing", choose "missing".
3. A value that is only derived or computed (for example an amount in EUR) does not make a fact about a
   rule or percentage present; such a fact is "missing" unless the rule itself is stated.
4. A fact that says the answer declares the information unavailable is present if the answer says so,
   and contradicted if the answer gives a specific value instead.
5. An answer that is empty, repeated words or unrelated text makes every fact "missing".

Examples (invented, not from the evaluation):
Facts: 1. The parking fee is 30 EUR per day  2. The fee is charged in USD
Answer: "30, charged in USD."  ->  {"note": "all facts present", "verdicts": {"1": "present", "2": "present"}}
Facts: 1. The response time is 4 hours
Answer: "The response time is 30 minutes."  ->  {"note": "different response time", "verdicts": {"1": "contradicted"}}
Facts: 1. Half of the bonus must be repaid  2. Repayment applies within 6 months
Answer: "You must repay 900 EUR."  ->  {"note": "rule not stated, only an amount", "verdicts": {"1": "missing", "2": "missing"}}
Facts: 1. Printing is not allowed  2. External e-mail is not allowed
Answer: "Printing is prohibited. You may also share it with your team."  ->  {"note": "e-mail rule missing", "verdicts": {"1": "present", "2": "missing"}}

Write the note first, then the verdicts. Respond with one JSON object and nothing else:
{"note": "<one short sentence>", "verdicts": {"1": "present|missing|contradicted", "2": "..."}}"""

JUDGE_PROMPTS = {"j1": JUDGE_PROMPT_J1, "j2": JUDGE_PROMPT_J2}
JUDGE_VERSION = "j2"                             # podrazumevana verzija; j1 ostaje radi ponovljivosti starih rezultata


def judge_fingerprint(version: str = JUDGE_VERSION) -> str:
    """Heš prompta ocenjivača - ide uz rezultate, da se zna tačno kojim promptom je ocenjeno."""
    return hashlib.sha256(JUDGE_PROMPTS[version].encode("utf-8")).hexdigest()[:8]


def load_facts(path: Path) -> dict[str, list[dict]]:
    """eval/expected_facts.json -> {id pitanja: [{"text": ..., "core": bool}, ...]}"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_judge_messages(question: str, facts: list[dict], answer: str, version: str = JUDGE_VERSION) -> list[dict]:
    """Poruke za ocenjivača: pitanje, numerisane činjenice, odgovor koji se ocenjuje."""
    fact_lines = "\n".join(f"{i}. {f['text']}" for i, f in enumerate(facts, start=1))
    user = f"QUESTION: {question}\n\nFACTS:\n{fact_lines}\n\nANSWER: {answer.strip() or '(empty answer)'}"
    return [{"role": "system", "content": JUDGE_PROMPTS[version]}, {"role": "user", "content": user}]


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


NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "half": "50"}


def numbers(text: str, words: bool = False) -> set[str]:
    """Brojevi iz teksta bez separatora hiljada ('1,500' -> '1500').
    words=True: i reči 'one'..'twelve', 'half' (samo za odgovor - činjenice su pisane ciframa)."""
    if words:
        text = re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", lambda m: NUMBER_WORDS[m.group(1).lower()],
                      text or "", flags=re.IGNORECASE)
    return {n.replace(",", "") for n in NUMBER_RE.findall(text or "")}


def consistency_flags(facts: list[dict], verdicts: list[str], note: str, answer: str, question: str = "") -> list[str]:
    """Provera bez LLM-a: znaci da je ocenjivač verovatno pogrešio (alarm za pregled, ne menja ocenu).
    Gledaju se samo brojevi iz činjenice kojih nema u pitanju (npr. 'T2' ili '3,200' iz pitanja se ne broje).
      note_vs_verdicts      - napomena kaže 'all facts present', a neki verdikt nije present
      numbers_in_answer:i   - činjenica i je contradicted, a svi njeni brojevi postoje u odgovoru
      numbers_missing:i     - činjenica i je present, a nijedan njen broj ne postoji u odgovoru"""
    flags = []
    if "all facts present" in note.lower() and any(v != "present" for v in verdicts):
        flags.append("note_vs_verdicts")
    ans, asked = numbers(answer, words=True), numbers(question)
    for i, (f, v) in enumerate(zip(facts, verdicts), start=1):
        fn = numbers(f["text"]) - asked
        if fn and v == "contradicted" and fn <= ans:
            flags.append(f"numbers_in_answer:{i}")
        if fn and v == "present" and not (fn & ans):
            flags.append(f"numbers_missing:{i}")
    return flags


def judge_answer(row: dict, facts: list[dict], llm, version: str = JUDGE_VERSION) -> dict:
    """Ocena jednog odgovora iz answers.jsonl."""
    start = time.perf_counter()
    raw = llm(build_judge_messages(row["question"], facts, row["answer"], version))
    parsed = parse_verdicts(raw, len(facts))
    out = {"id": row["id"], "category": row["category"], **parsed, "judge_raw": raw,
           "judge_latency_s": round(time.perf_counter() - start, 2)}
    if not parsed["judge_error"]:
        out.update(grade_from_verdicts(facts, parsed["verdicts"]))
        out["flags"] = consistency_flags(facts, parsed["verdicts"], parsed["note"], row["answer"], row["question"])
    return out


def judge_run(run_dir: Path, facts: dict, llm, judge_tag: str, version: str = JUDGE_VERSION) -> Path:
    """Ocenjuje sva pitanja jednog pokretanja; upisuje <run_dir>/judge_<judge_tag>.jsonl i vraća putanju."""
    rows = [json.loads(line) for line in open(Path(run_dir) / "answers.jsonl", encoding="utf-8")]
    out = Path(run_dir) / f"judge_{judge_tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(judge_answer(row, facts[row["id"]], llm, version), ensure_ascii=False) + "\n")
    return out