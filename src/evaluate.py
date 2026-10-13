"""Evaluacija retrievera: Hit@k, Recall@k, MRR, nDCG@k, metrike za više izvora i metrike specifične za dokument"""
from __future__ import annotations

import math
import time

import pandas as pd
from tqdm import tqdm

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
        row[f"all_sources@{k}"] = float(n_found == len(sources)) # 1 ako su SVI izvori u prvih k, inače 0

    first = min(found_at.values(), default=None) # vraca None ako je found_at prazan
    row["mrr"] = 1 / first if first is not None else 0

    # cover_rank: pozicija na kojoj je pronađen POSLEDNJI potreban izvor = najmanji k koji bi pokrio sve izvore i za koji bi all_sources@k = 1
    # Ako neki izvor nije pronađen u celoj listi (top 10), vrednost je NaN.
    row["cover_rank"] = max(found_at.values()) if len(found_at) == len(sources) else float("nan")

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
    for q in tqdm(questions, desc=name, unit="pitanje"):
        start = time.perf_counter()
        retrieved = search_fn(q["question"], k)
        latency = time.perf_counter() - start
        top5 = retrieved[:5]
        row = {
            "experiment": name,
            "id": q["id"],
            "category": q["category"],
            "n_sources": len(q["sources"]),          # 0 = bez odgovora, 1 = jedan izvor, 2+ = više izvora
            "top_score": retrieved[0]["score"] if retrieved else 0.0,
            "superseded@5": float(any(s.split(".")[0] == "17" for c in top5 for s in c["sections"])),
            # da li je bar jedan od prvig 5 rezultata iz arhive
            "words@5": sum(len(c["text"].split()) for c in top5),
            # koliko teksta retriever vraca, za poredjenje fix vs section strategije
            "top5": [c["chunk_id"] for c in top5],
            "latency_s": round(latency, 3),   # vreme pretrage za ovo pitanje (sekunde)
            # skor najboljeg rezultata
        }
        if q["sources"]:                           # pitanja bez odgovora (abstain) nemaju izvore
            row.update(evaluate_question(retrieved, q["sources"]))
        rows.append(row)
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Prosek metrika po eksperimentu (i opciono po kategoriji), samo za pitanja sa odgovorom."""
    metrics = [c for c in df.columns if c.startswith(("hit@", "recall@", "all_sources@", "mrr", "ndcg@", "cover_rank"))]
    metrics += ["superseded@5", "words@5"]
    metrics += [c for c in ["latency_s"] if c in df.columns]
    answerable = df[df["category"] != "abstain"]
    group = ["experiment"] + ([by] if by else [])
    return answerable.groupby(group, sort=False)[metrics].mean().round(3)


def multi_source_report(df: pd.DataFrame) -> pd.DataFrame:
    """Metrike samo za pitanja sa 2+ izvora: koliko izvora je pronađeno i koliko duboko treba ići.

    recall@k        prosečan udeo pronađenih izvora (delimična pokrivenost se računa)
    all_sources@k   udeo pitanja kod kojih su SVI izvori u top k (strože: odgovor bez jednog izvora je nepotpun)
    cover_rank      prosečna pozicija poslednjeg potrebnog izvora, samo za pitanja kod kojih su svi pronađeni
    cover_rank_max  najgori slučaj, tj. koliki k bi bio potreban da se pokriju sva pitanja
    missing@10      broj pitanja kod kojih bar jedan izvor nije ni u top 10
    """
    multi = df[df["n_sources"] > 1]
    g = multi.groupby("experiment", sort=False)
    cols = [f"{m}@{k}" for m in ("recall", "all_sources") for k in (3, 5, 10)]
    report = g[cols].mean()
    report["cover_rank"] = g["cover_rank"].mean()
    report["cover_rank_max"] = g["cover_rank"].max()
    report["missing@10"] = g["all_sources@10"].apply(lambda x: int((x == 0).sum()))
    report["n_questions"] = g.size()
    return report.round(3)


def abstain_report(df: pd.DataFrame) -> pd.DataFrame:
    """Prosečan najbolji skor za pitanja bez odgovora vs. pitanja sa odgovorom."""
    kind = df["category"].eq("abstain").map({True: "bez odgovora", False: "sa odgovorom"})
    return df.assign(vrsta=kind).pivot_table(index="experiment", columns="vrsta",
                                             values="top_score", aggfunc=["mean", "max"], sort=False).round(3)


def save_results(df: pd.DataFrame, name: str) -> Path:
    """Čuva rezultate jednog pokretanja u results/<datum_vreme>_<ime>/ (5 CSV fajlova)."""
    from config import RESULTS_DIR

    out = RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H%M}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_question.csv", index=False)
    summarize(df).to_csv(out / "summary.csv")
    summarize(df, by="category")["mrr"].unstack(0).to_csv(out / "mrr_by_category.csv")
    abstain_report(df).to_csv(out / "abstain.csv")
    multi_source_report(df).to_csv(out / "multi_source.csv")
    return out

if __name__ == "__main__":
    import argparse
    import json

    from chunking import load_jsonl
    from config import DATA_PROCESSED, EVAL_QUESTIONS, RERANK_CANDIDATES
    from figures import descriptions_path, figure_chunks
    from indexing import ensure_index, get_client, hybrid_search, multi_query_search, search
    from retrieval import build_links, expand, rerank

    questions = json.loads(EVAL_QUESTIONS.read_text(encoding="utf-8"))
    fixed = load_jsonl(DATA_PROCESSED / "chunks_fixed.jsonl")
    section = load_jsonl(DATA_PROCESSED / "chunks_section.jsonl")

    experiments = [   # (ime, chunk-ovi, putanja naslova, izbaci arhivu, retriever, dorada rezultata)
        ("1_fixed", fixed, False, False, "dense", None),
        ("2_section", section, False, False, "dense", None),
        ("3_section_path", section, True, False, "dense", None),
        ("3b_section_path_filter", section, True, True, "dense", None),
    ]
    if descriptions_path().exists():
        with_figures = section + figure_chunks()
        experiments += [
            ("4_figures", with_figures, True, True, "dense", None),
            ("5_expansion", with_figures, True, True, "dense", "expand"),
            ("6_rerank", with_figures, True, True, "dense", "expand+rerank"),
            ("4b_dense_m3", with_figures, True, True, "m3-dense", None),   # kontrola: treba da = eksp. 4, isti pristup, samo druga biblioteka
            ("7_sparse", with_figures, True, True, "m3-sparse", None),
            ("8_hybrid", with_figures, True, True, "m3-hybrid", None),
            ("9_hybrid_rerank", with_figures, True, True, "m3-hybrid", "expand+rerank"),
            # razlaganje pitanja (decompose.py): pitanje + podpitanja, liste spojene RRF-om
            ("10_decomp", with_figures, True, True, "dense-decomp", None),            # porediti sa 4_figures
            ("11_decomp_hybrid", with_figures, True, True, "m3-hybrid-decomp", None), # porediti sa 8_hybrid
        ]
    else:
        print("Nema opisa slika (pokrenite python src/figures.py) - eksperimenti 4-11 se preskaču.")

    parser = argparse.ArgumentParser(description="Pokreće eksperimente retrievera i čuva rezultate.")
    parser.add_argument("--only", nargs="+", metavar="IME",
                        help="pokreni samo navedene eksperimente, npr. --only 5_expansion 6_rerank")
    args = parser.parse_args()
    if args.only:
        available = [name for name, *_ in experiments]
        unknown = set(args.only) - set(available)
        if unknown:
            parser.error(f"nepoznati eksperimenti: {sorted(unknown)}; dostupni: {available}")
        experiments = [e for e in experiments if e[0] in args.only]

    # podpitanja se čitaju samo iz keša (bez API poziva), i to samo ako se pokreće eksperiment 10 ili 11
    subqueries = {}
    if any(r.endswith("-decomp") for *_, r, _ in experiments):
        from decompose import load_subqueries
        subqueries = load_subqueries()

    client = get_client()

    def make_search_fn(collection: str, retriever: str, exclude: bool, mode: str | None, links: dict):
        """Pravi funkciju pretrage (pitanje, k) -> rezultati za jedan eksperiment."""
        def retrieve(question: str, n: int) -> list[dict]:
            if retriever.endswith("-decomp"):                  # eksp. 10-11
                # originalno pitanje uvek prvo tj. podpitanja se dodaju samo ako je pitanje razloženo
                subs = [q for q in subqueries.get(question, []) if q != question]
                return multi_query_search(client, collection, [question] + subs, n,
                                          hybrid=retriever.startswith("m3"), exclude_superseded=exclude)
            if retriever == "dense":
                return search(client, collection, question, n, exclude_superseded=exclude)
            return hybrid_search(client, collection, question, n, using=retriever.split("-")[1], exclude_superseded=exclude)

        def search_fn(question: str, k: int) -> list[dict]:
            if mode is None:                                   
                return retrieve(question, k)                            # eksp. 1-4
            n = RERANK_CANDIDATES if mode == "expand+rerank" else k
            candidates = retrieve(question, n)  # eksp. 5-6, 8-9
            candidates = expand(candidates, links, k=len(candidates) * 2)   # dodaj memoe i slike
            if mode == "expand":                               # eksperiment 5
                return candidates[:k]
            return rerank(question, candidates, k)             # eksperiment 6
        """
            mode = None            →  search(k)                                  (eksp. 1–4)
            mode = "expand"        →  search(k)  → expand → prvih k               (eksp. 5)
            mode = "expand+rerank" →  search(20) → expand → rerank → prvih k      (eksp. 6)
            retriever: "dense" (1–6), "m3-dense" / "m3-sparse" / "m3-hybrid" (4b, 7–9),
                       "dense-decomp" / "m3-hybrid-decomp" (10–11: pitanje + podpitanja, RRF) """
        return search_fn

    results = []
    for name, chunks, with_path, exclude, retriever, mode in experiments:
        # indeksira samo ako kolekcija sa istim otiskom (chunk-ovi + podešavanja) još ne postoji
        # "m3-..." retriveri koriste kolekciju sa dense + sparse vektorima, ostali običnu dense kolekciju
        collection = ensure_index(client, chunks, with_path=with_path, hybrid=retriever.startswith("m3"))
        print(f"Pokrećem {name}...")
        search_fn = make_search_fn(collection, retriever, exclude, mode, build_links(chunks))
        results.append(run_experiment(name, questions, search_fn))
    client.close()

    df = pd.concat(results, ignore_index=True)
    out = save_results(df, "_".join(args.only) if args.only else "all")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n=== Ukupno (pitanja sa odgovorom) ===")
    print(summarize(df))
    print("\n=== MRR po kategoriji ===")
    print(summarize(df, by="category")["mrr"].unstack(0))
    print("\n=== Pitanja sa više izvora ===")
    print(multi_source_report(df))
    print("\n=== Skor najboljeg rezultata ===")
    print(abstain_report(df))
    print(f"\nRezultati sačuvani u: {out}")
