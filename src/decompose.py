"""Razlaganje složenih pitanja na jednostavnija podpitanja (query decomposition) pomoću Gemini API-ja.

Pitanja se šalju u grupama (BATCH_SIZE po pozivu), da bismo ostali u dnevnoj kvoti besplatnog nivoa.
Podpitanja se čuvaju u keš fajlu, pa se API poziva samo jednom, a eksperimenti su ponovljivi.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import httpx

from config import DATA_PROCESSED, EVAL_QUESTIONS, LLM_BATCH_SIZE, LLM_MODEL, LLM_PAUSE_S


PROMPT = """You prepare search queries for a retrieval system over an internal company knowledge base.

Split each user question into simple, self-contained sub-questions, one per fact that must be looked up.
Rules:
- Treat every question independently of the others.
- If a question asks about only one fact, return it unchanged as the only item. Do not split one fact
  into a general and a more specific version of the same question.
- A question about whether a person, role or group may do something usually needs two facts:
  what that role or group is, and who is eligible for or allowed to do that thing.
- Otherwise return 2 to 4 sub-questions.
- Every sub-question must make sense on its own: repeat names of companies, offices, products,
  roles, versions and amounts from the question instead of using "it", "they" or "there".
- Do not answer the questions and do not add facts, names or assumptions that are not in the question.
- Return a JSON object that maps each question number to a JSON array of sub-questions.

Example questions
1. Can contractors in the Berlin office use the company gym, and how is it booked?
2. What is the maximum length of a business trip?
3. Can a G3 analyst join the sabbatical programme?
4. What is the reimbursement rate for taxis at night?
Example output
{"1": ["Can contractors use the company gym?", "Is there a company gym in the Berlin office?", "How is the company gym booked?"],
 "2": ["What is the maximum length of a business trip?"],
 "3": ["What is a G3 analyst?", "Who is eligible for the sabbatical programme?"],
 "4": ["What is the reimbursement rate for taxis at night?"]}

Questions
{questions}
Output:"""


def cache_path(model: str = LLM_MODEL) -> Path:
    """Keš po modelu, verziji prompta i veličini grupe: promena bilo čega od toga pravi novi keš fajl."""
    prompt_id = hashlib.sha256(f"{PROMPT}|{LLM_BATCH_SIZE}".encode("utf-8")).hexdigest()[:8]
    safe_name = re.sub(r"[^\w.-]", "_", model)
    return DATA_PROCESSED / f"decompose_{safe_name}_{prompt_id}.json"


def parse_subqueries(items, question: str) -> list[str]:
    """Lista stringova za jedno pitanje; ako nešto ne valja, vraća samo originalno pitanje."""
    if not isinstance(items, list):
        return [question]
    items = [s.strip() for s in items if isinstance(s, str) and s.strip()]
    return items[:4] or [question]  # maksimalno 4 podpitanja


def decompose_batch(questions: list[str], client, model: str = LLM_MODEL, retries: int = 5) -> list[list[str]]:
    """Šalje grupu pitanja u jednom pozivu i vraća listu podpitanja za svako (istim redom)."""
    from google.genai import errors, types

    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=PROMPT.replace("{questions}", numbered),
                config=types.GenerateContentConfig(temperature=0,  # determinističko: isto pitanje -> ista podpitanja
                                                   response_mime_type="application/json"),
            )
            break
        except (errors.APIError, httpx.TransportError) as e:
            # APIError: server je odgovorio greškom (429 limit, 500/503 preopterećen)
            # TransportError: veza je pukla bez odgovora (mrežni problem) -> isto vredi probati ponovo
            code = getattr(e, "code", None)
            if code == 429 and "PerDay" in str(e):
                raise RuntimeError("Dnevna kvota modela je potrošena - promenite LLM_MODEL ili pokušajte sutra.") from e
            if (code is not None and code not in (429, 500, 503)) or attempt == retries:
                raise
            wait = 30 * 2 ** (attempt - 1)
            print(f"  greška {code or type(e).__name__}, čekam {wait} s (pokušaj {attempt}/{retries})...")
            time.sleep(wait)

    try:
        answer = json.loads(response.text)
    except json.JSONDecodeError:
        answer = {}
    if not isinstance(answer, dict):
        answer = {}
    # odgovor je {"1": [...], "2": [...]}; pitanje bez ispravnog odgovora ostaje nerazloženo
    return [parse_subqueries(answer.get(str(i)), q) for i, q in enumerate(questions, start=1)]


def decompose_all(questions: list[str], model: str = LLM_MODEL) -> dict:
    """Razlaže sva pitanja koja još nisu u kešu, u grupama. Vraća {pitanje: [podpitanja]}."""
    from google import genai

    cache_file = cache_path(model)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    client = genai.Client()   # ključ čita iz GEMINI_API_KEY (.env)

    todo = [q for q in questions if q not in cache]  # ključ je tekst pitanja: preformulisano pitanje = novi poziv
    for start in range(0, len(todo), LLM_BATCH_SIZE):
        batch = todo[start:start + LLM_BATCH_SIZE]
        print(f"Grupa {start // LLM_BATCH_SIZE + 1}: pitanja {start + 1}-{start + len(batch)} od {len(todo)}...")
        for q, subs in zip(batch, decompose_batch(batch, client, model)):
            cache[q] = subs
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        if start + LLM_BATCH_SIZE < len(todo):
            time.sleep(LLM_PAUSE_S)
    return cache


def load_subqueries(model: str = LLM_MODEL) -> dict:
    """Čita samo keš (bez API poziva) - koristi se u evaluaciji."""
    cache_file = cache_path(model)
    if not cache_file.exists():
        raise FileNotFoundError(f"Keš fajl {cache_file.name} ne postoji - prvo pokrenite: python src/decompose.py")
    return json.loads(cache_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    questions = [q["question"] for q in json.loads(EVAL_QUESTIONS.read_text(encoding="utf-8"))]
    result = decompose_all(questions)
    n_split = sum(len(result[q]) > 1 for q in questions)
    print(f"\nRazloženo {n_split} od {len(questions)} pitanja\n")
    for q in questions:
        if len(result[q]) > 1:
            print(q)
            for s in result[q]:
                print("   ->", s)
