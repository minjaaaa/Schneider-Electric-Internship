"""Evaluacija generisanih odgovora (answers.jsonl iz notebooks/04_generation.ipynb), bez LLM-a.

Metrike koriste samo polja koja evaluacioni skup već ima (expected_answer, sources, category),
pa rade za bilo koji dokument obrađen istim pipeline-om.
Upotreba:  python src/evaluate_text.py                      (poslednje generisanje u results/)
           python src/evaluate_text.py <ime_foldera>        (npr. 2026-10-14_1022_gen_hybrid_qwen7b)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[str]:
    """Svi brojevi iz teksta, normalizovani: '3,200' -> '3200', '1.5' ostaje '1.5'."""
    out = set()
    for n in NUMBER_RE.findall(text or ""):
        n = re.sub(r",(?=\d{3}\b)", "", n)      # zarez kao separator hiljada: 3,200 -> 3200
        out.add(n)
    return out


def resolve_citations(citations: list[str], context: list[str]) -> list[str]:
    """Citat-broj bloka ('2', '[2]') -> oznaka izvora iz konteksta (context[1]) - ostalo ostaje kako jeste."""
    """citations = ["2", "14"]
        context   = ["14", "Figure 3", "14", "5.6", "10.3"]"""
    resolved = []
    for c in citations:
        m = re.fullmatch(r"\[?(\d+)\]?", c.strip()) # samo broj sa uglastim zagradama ili bez njih
        if m and 1 <= int(m.group(1)) <= len(context) and c.strip() not in context: # broj mora biti vazeca pozicija bloka
            resolved.append(context[int(m.group(1)) - 1])
        else: # vec prava oznaka iz konteksta, ne diramo ga
            resolved.append(c.strip())
    return list(dict.fromkeys(resolved))      # bez duplikata, redosled ostaje


def score_answer(row: dict, chunk_text: dict) -> dict:
    """Metrike za jedan odgovor. Prima jedan red iz answers.jsonl"""
    is_abstain_q = row["category"] == "abstain" # da li je pitanje bez odgovora
    sources = set(row["sources"])
    context = row["context"]
    cited = set(resolve_citations(row["citations"], context))
    answer_nums = numbers(row["answer"])
    context_nums = set().union(*(numbers(chunk_text.get(cid, "")) for cid in row["context_ids"]))
    # za svaki chunk  uzmemo tekst i iz njega izvucemo brojeve, pa sve spojimo u jedan skup

    out = {
        "id": row["id"], "category": row["category"], "n_sources": len(sources),
        "abstained": row["abstained"], "parse_error": row["parse_error"],
        # odbijanje: tačno je odbiti pitanje bez odgovora i NE odbiti pitanje sa odgovorom
        "abstain_correct": row["abstained"] == is_abstain_q,
        # izmišljeni brojevi: broj u odgovoru koji se ne pojavljuje ni u jednom chunk-u iz konteksta
        "unsupported_numbers": sorted(answer_nums - context_nums), # trazimo razliku brojeva u odgovoru i brojeva u kontekstu
        "latency_s": row["latency_s"],
    }
    if not is_abstain_q: # metrike za pitanja sa odgovorom
        expected_nums = numbers(row["expected_answer"]) # izvlacimo ocekivanje brojeve iz odgovora
        out["number_recall"] = (len(expected_nums & answer_nums) / len(expected_nums)) if expected_nums else None
        # ako je ocekivani odgovor samo tekst, vrednost je None
        out["context_has_all_sources"] = sources <= set(context)    # da li je pretraga donela sve izvore, koliki podskup je context od sources
        out["citation_precision"] = len(cited & sources) / len(cited) if cited else None # ako model nista nije citirao, precision je None
        # presek je isti kad su skupovi cited i sources jednaki
        out["citation_recall"] = len(cited & sources) / len(sources)
    return out


def summarize_answers(df: pd.DataFrame) -> pd.Series:
    """Zbirne metrike za jedno pokretanje."""
    ans = df[df["category"] != "abstain"]
    abst = df[df["category"] == "abstain"]
    s = {
        "n_questions": len(df),
        "parse_error_rate": df["parse_error"].mean(),
        "abstain_recall": abst["abstained"].mean() if len(abst) else None,        # pitanja bez odgovora koja su odbijena
        "false_abstain_rate": ans["abstained"].mean(),                            # pitanja sa odgovorom koja su odbijena
        "number_recall": ans["number_recall"].mean(),                             # NaN (pitanja bez brojeva) se preskaču
        "answers_with_unsupported_numbers": (df["unsupported_numbers"].str.len() > 0).mean(),
        "citation_precision": ans["citation_precision"].mean(),
        "citation_recall": ans["citation_recall"].mean(),
        "context_has_all_sources": ans["context_has_all_sources"].mean(),
        "latency_median_s": df["latency_s"].median(),
    }
    if "manual_grade" in df:
        g = ans["manual_grade"]
        s["manual_correct"] = (g == "correct").mean()
        s["manual_partial"] = (g == "partial").mean()
        s["manual_wrong"] = (g == "wrong").mean()
    return pd.Series(s).round(3)


def error_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Ručna ocena x da li je pretraga donela sve izvore: odvaja greške pretrage od grešaka generisanja."""
    ans = df[df["category"] != "abstain"]
    return pd.crosstab(ans["context_has_all_sources"].map({True: "kontekst ima sve izvore",
                                                          False: "kontekstu fali izvor"}),
                       ans["manual_grade"], margins=True)


if __name__ == "__main__":
    from chunking import load_jsonl
    from config import DATA_PROCESSED, GEN_RESULTS_DIR
    from figures import figure_chunks

    # putanja se računa u odnosu na results/ projekta, a ne u odnosu na folder iz kog je skripta pokrenuta
    if len(sys.argv) > 1:
        run_dir = GEN_RESULTS_DIR / Path(sys.argv[1]).name     # prihvata i "results/<ime>" i samo "<ime>"
    else:
        run_dir = max(GEN_RESULTS_DIR.glob("*_gen_*"))         # bez argumenta: najnovije generisanje (ime počinje datumom)
    print(f"Evaluiram: {run_dir}")
    rows = [json.loads(line) for line in open(run_dir / "answers.jsonl", encoding="utf-8")]
    chunks = load_jsonl(DATA_PROCESSED / "chunks_section.jsonl") + figure_chunks()
    chunk_text = {c["chunk_id"]: c["text"] for c in chunks}

    df = pd.DataFrame([score_answer(r, chunk_text) for r in rows])
    grades_file = run_dir / "manual_grades.csv"                    # opciono: id, manual_grade, note
    if grades_file.exists():
        df = df.merge(pd.read_csv(grades_file), on="id", how="left")

    df.to_csv(run_dir / "answer_metrics.csv", index=False)
    summary = summarize_answers(df)
    summary.to_csv(run_dir / "answer_summary.csv", header=["value"])

    pd.set_option("display.width", 200)
    print("=== Zbirne metrike ===")
    print(summary.to_string())
    print("\n=== Odgovori sa brojevima kojih nema u kontekstu ===")
    print(df[df["unsupported_numbers"].str.len() > 0][["id", "category", "unsupported_numbers"]].to_string(index=False))
    if "manual_grade" in df:
        print("\n=== Ručna ocena x pokrivenost izvora (greške pretrage vs generisanja) ===")
        print(error_attribution(df))
    print(f"\nSačuvano: {run_dir / 'answer_metrics.csv'}, {run_dir / 'answer_summary.csv'}")