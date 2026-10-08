"""Dorada rezultata pretrage: proširenje povezanim chunkovima (eksperiment 5) i reranker (eksperiment 6)."""
from __future__ import annotations

import re
from functools import lru_cache

from config import RERANKER_MODEL

FIGURE_REF_RE = re.compile(r"\bFigure (\d+)\b")


# ---------- Eksperiment 5: proširenje rezultata ----------

def build_links(chunks: list[dict]) -> dict:
    """Priprema „mapu veza“: memo -> chunkovi koji ga sadrže, slika -> chunk sa njenim opisom."""
    memos, figures = {}, {}
    memo_ids = {m for c in chunks for m in c.get("overridden_by", [])} # PT-2026-02, PT-2026-04
    for c in chunks:
        if c.get("figure"):
            figures[c["figure"]] = c
        if c["status"] == "memo":
            for memo_id in memo_ids:
                if memo_id in c["text"]:
                    memos.setdefault(memo_id, []).append(c)
    return {"memos": memos, "figures": figures}


def linked_chunks(chunk: dict, links: dict) -> list[dict]:
    """Chunkovi povezani sa datim: memoi koji ga menjaju i slike koje pominje."""
    result = []
    for memo_id in chunk.get("overridden_by", []):          # npr. 3.2 -> memo PT-2026-04
        result += links["memos"].get(memo_id, [])
    for n in FIGURE_REF_RE.findall(chunk["text"]):           # npr. "See Figure 3" -> opis slike 3
        fig = links["figures"].get(f"Figure {n}")
        if fig is not None and fig is not chunk:
            result.append(fig)
    return list({c["chunk_id"]: c for c in result}.values())   # bez duplikata, redosled ostaje


def expand(retrieved: list[dict], links: dict, k: int) -> list[dict]:
    """Posle svakog rezultata ubacuje njegove povezane chunkove (bez duplikata), pa skraćuje na k"""
    out, seen = [], set() # seen = chunk_id-ovi koji su već ubačeni
    for chunk in retrieved:
        for c in [chunk] + linked_chunks(chunk, links):
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                # povezani chunk nasleđuje skor chunk-a koji ga je „doveo“
                out.append(c if c is chunk else {**c, "score": chunk["score"], "linked_from": chunk["chunk_id"]})
    return out[:k]


# ---------- Eksperiment 6: reranker ----------

@lru_cache(maxsize=1)
def get_reranker(name: str = RERANKER_MODEL):
    """Učitava cross-encoder samo jednom (prvi put ga i preuzima, ~2 GB)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, max_length=512)


def rerank(query: str, candidates: list[dict], k: int, model=None) -> list[dict]:
    """Ponovo boduje kandidate cross-encoder-om (pitanje + chunk zajedno) i vraća k najboljih."""
    if not candidates:
        return []
    model = model or get_reranker()
    scores = model.predict([(query, c["text"]) for c in candidates], batch_size=8) # vraca skor relevantnosti za svaki par (pitanje, chunk)
    rescored = [{**c, "dense_score": c["score"], "score": float(s)} for c, s in zip(candidates, scores)]
    return sorted(rescored, key=lambda c: c["score"], reverse=True)[:k] # sortira opadajuce, uzima k najboljih
