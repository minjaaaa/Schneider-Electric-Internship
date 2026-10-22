"""Kvalitet konteksta (context precision / recall) 

Svako pitanje u evaluacionom skupu ima označene izvore (sources: oznake sekcija i slika), a uz svaki
sačuvan odgovor stoji lista oznaka delova dokumenta koje je pretraga donela (context, redom po rangu).
Iz ta dva podatka metrike se računaju kao:
  context recall        - koliki deo potrebnih izvora je u kontekstu 
  context precision@k   - koliki deo od k delova konteksta je potreban (relevantni / k)
  context precision     - da li su potrebni delovi PRI VRHU liste

Upotreba:  python src/context_metrics.py                
           python src/context_metrics.py <ime_foldera>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def context_recall(context: list[str], sources: list[str]) -> float:
    """Deo potrebnih izvora koji se nalazi u kontekstu."""
    src = set(sources)
    return len(src & set(context)) / len(src)


def context_precision_at_k(context: list[str], sources: list[str]) -> float:
    """Deo delova konteksta koji su potrebni (relevantni / k); 0 ako kontekst nema delova."""
    src = set(sources)
    return sum(c in src for c in context) / len(context)


def average_precision(context: list[str], sources: list[str]) -> float:
    """Prosek precision@i po pozicijama i na kojima je relevantan deo; 0 ako relevantnih nema."""
    src, hits, precisions = set(sources), 0, []
    for i, label in enumerate(context, start=1):
        if label in src:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(precisions) if precisions else 0.0


def score_question(row: dict) -> dict | None:
    """Metrike za jedno pitanje (red iz answers.jsonl); None za pitanje bez izvora."""
    if not row["sources"]:
        return None
    ctx, src = row["context"], row["sources"]
    return {"id": row["id"], "category": row["category"], "n_sources": len(src),
            "context_recall": round(context_recall(ctx, src), 3),
            "context_precision_at_k": round(context_precision_at_k(ctx, src), 3),
            "average_precision": round(average_precision(ctx, src), 3),
            "missing_sources": sorted(set(src) - set(ctx))}


def summarize(scores: list[dict]) -> dict:
    """Proseci ukupno i odvojeno za pitanja sa jednim i sa više izvora."""
    def mean(items, key):
        return round(sum(s[key] for s in items) / len(items), 3) if items else None

    groups = {"all": scores,
              "one_source": [s for s in scores if s["n_sources"] == 1],
              "multi_source": [s for s in scores if s["n_sources"] > 1]}
    out = {name: {"n": len(g), "context_recall": mean(g, "context_recall"),
                  "context_precision_at_k": mean(g, "context_precision_at_k"),
                  "average_precision": mean(g, "average_precision"),
                  "all_sources_found": sum(s["context_recall"] == 1 for s in g)}
           for name, g in groups.items()}
    out["missing_sources"] = {s["id"]: s["missing_sources"] for s in scores if s["missing_sources"]}
    return out


def evaluate_run(rows: list[dict]) -> dict:
    scores = [s for s in map(score_question, rows) if s is not None]
    return {"k": len(rows[0]["context"]), "n_questions": len(rows), "n_scored": len(scores),
            "summary": summarize(scores), "per_question": scores}


if __name__ == "__main__":
    from config import GEN_RESULTS_DIR

    run = sys.argv[1] if len(sys.argv) > 1 else "2026-10-14_1022_gen_hybrid_qwen7b"
    run_dir = GEN_RESULTS_DIR / Path(run).name
    rows = [json.loads(line) for line in open(run_dir / "answers.jsonl", encoding="utf-8")]
    report = evaluate_run(rows)

    # kontekst je zamrznut: proveri da sva pokretanja generisanja imaju isti kontekst kao ovo
    ctx = {r["id"]: r["context_ids"] for r in rows}
    report["runs_with_different_context"] = [
        f.parent.name for f in sorted(GEN_RESULTS_DIR.glob("*/answers.jsonl"))
        if {r["id"]: r["context_ids"] for r in map(json.loads, open(f, encoding="utf-8"))} != ctx]

    print(f"=== Kvalitet konteksta: {run_dir.name} (k = {report['k']}, "
          f"{report['n_scored']} od {report['n_questions']} pitanja imaju izvore) ===")
    print(f"  {'grupa':<14}{'n':>4}{'recall':>9}{'precision@k':>14}{'AP (rang)':>11}{'svi izvori':>12}")
    for name in ("all", "one_source", "multi_source"):
        s = report["summary"][name]
        print(f"  {name:<14}{s['n']:>4}{s['context_recall']:>9}{s['context_precision_at_k']:>14}"
              f"{s['average_precision']:>11}{s['all_sources_found']:>8}/{s['n']}")
    print("  izvori koji fale:", report["summary"]["missing_sources"] or "nema")
    print("  pokretanja sa drugačijim kontekstom:", report["runs_with_different_context"] or "nema (kontekst zamrznut)")
    out = GEN_RESULTS_DIR / f"context_metrics_{run_dir.name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nIzveštaj: {out}")
