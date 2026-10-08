"""Evaluacija retrievera: Hit@k, Recall@k, MRR, nDCG@k i metrike specifične za dokument"""
from __future__ import annotations

import math

import pandas as pd

from datetime import datetime
from pathlib import Path


KS = (1, 3, 5, 10)


def matches(chunk: dict, source: str) -> bool:
    """Da li chunk pokriva izvor iz evaluacionog skupa ('5.2' ili 'Figure 3')."""
    if source.startswith("Figure"):
        return chunk.get("figure") == source       # samo pravi chunk sa opisom slike
    return source in chunk["sections"]


# ---------------- 1. Metrike za jedno pitanje -------------------

def evaluate_question(retrieved: list[dict], sources: list[str], ks=KS) -> dict:
    """Metrike za jedno pitanje 'retrieved' je lista chunkova od najboljeg ka najlosijem"""
    found_at = {} # izvor -> pozicija prvog pogotka (1, 2,3 ..)
    for rank, chunk in enumerate(retrieved, start=1):
        for s in sources:
            if s not in found_at and matches(chunk, s):
                found_at[s] = rank
    row = {}
    for k in ks:
        n_found = sum(1 for r in found_at.values() if r <= k)
        row[f"hit@{k}"] = float(n_found > 0) # koliko pitanja je imalo bar jedan pogodan chunk u top-k
        row[f"recall@{k}"] = n_found / len(sources) # od svih tacnih izvora, koliko ih je bilo u prvih k rezultata

    first = min(found_at.values(), default=None) # vraca None ako je found_at prazan
    row["mrr"] = 1 / first if first is not None else 0

    k = max(ks)
    dcg = sum(1 / math.log2(r + 1) for r in found_at.values() if r <= k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(sources), k) + 1))
    row[f"ndcg@{k}"] = dcg / idcg
    return row

"""pitanje Q02, izvori ["5.2", "16.2"]). 
    Retriever vrati redom chunk iz sekcije 1.1, chunk iz 5.2, chunk iz 16.2. 
    Tada je found_at = {"5.2": 2, "16.2": 3}"""

def run_experiment(name: str, questions: list[dict], search_fn, k: int = max(KS)) -> pd.DataFrame:
    """Pokrece sva pitanja kroz 'search_fn(pitanje, k)' i vraca tabelu sa metrikama za svako pitanje"""
    rows = []
    for q in questions:
        retrieved = search_fn(q["question"], k)
        top5 = retrieved[:5]
        row = {
            "experiment": name,
            "id": q["id"],
            "category": q["category"],
            "top_score": retrieved[0]["score"] if retrieved else 0.0,
            "superseded@5": float(any(s.split(".")[0] == "17" for c in top5 for s in c["sections"])),
            # da li je bar jedan od prvig 5 rezultata iz arhive
            "words@5": sum(len(c["text"].split()) for c in top5),
            # koliko teksta retriever vraca, za poredjenje fix vs section strategije
            "top5": [c["chunk_id"] for c in top5],
            # skor najboljeg rezultata
        }
        if q["sources"]:                           # pitanja bez odgovora (abstain) nemaju izvore
            row.update(evaluate_question(retrieved, q["sources"]))
        rows.append(row)
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Prosek metrika po eksperimentu (i opciono po kategoriji), samo za pitanja sa odgovorom."""
    metrics = [c for c in df.columns if c.startswith(("hit@", "recall@", "mrr", "ndcg@"))]
    metrics += ["superseded@5", "words@5"]
    answerable = df[df["category"] != "abstain"]
    group = ["experiment"] + ([by] if by else [])
    return answerable.groupby(group, sort=False)[metrics].mean().round(3)


def abstain_report(df: pd.DataFrame) -> pd.DataFrame:
    """Prosečan najbolji skor za pitanja bez odgovora vs. pitanja sa odgovorom."""
    kind = df["category"].eq("abstain").map({True: "bez odgovora", False: "sa odgovorom"})
    return df.assign(vrsta=kind).pivot_table(index="experiment", columns="vrsta",
                                             values="top_score", aggfunc=["mean", "max"], sort=False).round(3)


def save_results(df: pd.DataFrame, name: str) -> Path:
    """Čuva rezultate jednog pokretanja u results/<datum_vreme>_<ime>/ (4 CSV fajla)."""
    from config import RESULTS_DIR

    out = RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H%M}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_question.csv", index=False)
    summarize(df).to_csv(out / "summary.csv")
    summarize(df, by="category")["mrr"].unstack(0).to_csv(out / "mrr_by_category.csv")
    abstain_report(df).to_csv(out / "abstain.csv")
    return out

if __name__ == "__main__":
    import json
    import argparse
    from chunking import load_jsonl
    from config import DATA_PROCESSED, EVAL_QUESTIONS
    from figures import descriptions_path, figure_chunks
    from indexing import build_index, get_client, search

    questions = json.loads(EVAL_QUESTIONS.read_text(encoding="utf-8"))
    fixed = load_jsonl(DATA_PROCESSED / "chunks_fixed.jsonl")
    section = load_jsonl(DATA_PROCESSED / "chunks_section.jsonl")

    experiments = [          # (ime, chunk-ovi, putanja naslova, izbaci arhivu)
        ("1_fixed", fixed, False, False),
        ("2_section", section, False, False),
        ("3_section_path", section, True, False),
        ("3b_section_path_filter", section, True, True),
    ]
    if descriptions_path().exists():
        experiments.append(("4_figures", section + figure_chunks(), True, True))
    else:
        print("Nema opisa slika (pokrenite python src/figures.py) - eksperiment 4 se preskače.")

        parser = argparse.ArgumentParser(description="Pokreće eksperimente retrievera i čuva rezultate.")
    parser.add_argument("--only", nargs="+", metavar="IME",
                        help="pokreni samo navedene eksperimente, npr. --only 4_figures")
    args = parser.parse_args()
    if args.only:
        available = [name for name, *_ in experiments]
        unknown = set(args.only) - set(available)
        if unknown:
            parser.error(f"nepoznati eksperimenti: {sorted(unknown)}; dostupni: {available}")
        experiments = [e for e in experiments if e[0] in args.only]
    client = get_client()
    results = []
    for name, chunks, with_path, exclude in experiments:
        print(f"Indeksiram {name} ({len(chunks)} chunk-ova)...")
        build_index(client, name, chunks, with_path=with_path)
        results.append(run_experiment(
            name, questions,
            lambda q, k, c=name, ex=exclude: search(client, c, q, k, exclude_superseded=ex),
        ))
    client.close()

    df = pd.concat(results, ignore_index=True)
    out = save_results(df, "_".join(args.only) if args.only else "all")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n=== Ukupno (pitanja sa odgovorom) ===")
    print(summarize(df))
    print("\n=== MRR po kategoriji ===")
    print(summarize(df, by="category")["mrr"].unstack(0))
    print("\n=== Skor najboljeg rezultata ===")
    print(abstain_report(df))
    print(f"\nRezultati sačuvani u: {out}")