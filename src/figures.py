"""Opisi slika pomoću vizuelnog modela (Google Gemini API) i pravljenje chunk-ova od tih opisa.

Opisi se čuvaju u keš fajlu (jedan po modelu), pa se API poziva samo jednom po slici.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from chunking import make_chunk
from config import DATA_PROCESSED, VLM_MODEL

FIGURES_JSON = DATA_PROCESSED / "figures" / "figures.json"
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}

PROMPT = """You are converting a figure from an internal company knowledge base into text for a search index.
Figure: {figure_id}: {caption} (in section {section_title})

Write a complete, factual description of the figure:
1. Start with one sentence saying what kind of figure it is.
2. Transcribe ALL visible text exactly: titles, labels, numbers, notes and footnotes.
3. Tables: reproduce them as Markdown tables with all rows and columns.
4. Flowcharts and decision diagrams: write every path as an explicit rule ("If ..., then ...").
5. Timelines: list every milestone as "Day N: event".
6. Floor plans and layouts: list every room or area with all its details.
7. Scanned documents: transcribe the full text, including reference numbers, dates and stamps.
Do not add anything that is not visible in the figure."""


def descriptions_path(model: str = VLM_MODEL) -> Path:
    """Keš fajl za opise jednog modela, npr. figure_descriptions_gemini-3.8-flash.json."""
    safe_name = re.sub(r"[^\w.-]", "_", model)      # npr. ':' nije dozvoljen u imenu fajla na Windowsu
    return DATA_PROCESSED / f"figure_descriptions_{safe_name}.json"


def describe_figure(fig: dict, client, model: str = VLM_MODEL, retries: int = 5) -> str:
    """Šalje jednu sliku Gemini modelu i vraća tekstualni opis."""
    from google.genai import errors, types

    path = Path(fig["image_path"])
    image = types.Part.from_bytes(data=path.read_bytes(), mime_type=MEDIA_TYPES[path.suffix.lower()])

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[image, PROMPT.format(**fig)],
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text
        except errors.APIError as e:
            # 429 = previše zahteva (limit), 500/503 = server preopterećen -> vredi sačekati i probati ponovo
            if e.code not in (429, 500, 503) or attempt == retries:
                raise
            wait = 30 * 2 ** (attempt - 1)         # 30, 60, 120, 240 s
            print(f"  greška {e.code}, čekam {wait} s (pokušaj {attempt}/{retries})...")
            time.sleep(wait)


def describe_all(model: str = VLM_MODEL) -> dict:
    """Opisuje sve slike koje još nisu u kešu. Vraća {figure_id: opis}."""
    from google import genai

    figures = json.loads(FIGURES_JSON.read_text(encoding="utf-8"))
    cache_file = descriptions_path(model)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    client = genai.Client()                        # ključ čita iz GEMINI_API_KEY (.env)

    for fig in figures:
        fid = fig["figure_id"]
        if fid in cache:
            print(f"{fid}: već u kešu")
            continue
        print(f"{fid}: opisujem...")
        cache[fid] = describe_figure(fig, client, model)
        cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return cache


def figure_chunks(model: str = VLM_MODEL) -> list[dict]:
    """Pravi chunkove (type='figure') od keširanih opisa slika."""
    figures = json.loads(FIGURES_JSON.read_text(encoding="utf-8"))
    descriptions = json.loads(descriptions_path(model).read_text(encoding="utf-8"))

    chunks = []
    for fig in figures:
        fid = fig["figure_id"]
        sec = {"section": fig["section"], "section_title": fig["section_title"],
               "heading_path": f"{fig['section_title']} > {fid}: {fig['caption']}"}
        text = f"{fid}: {fig['caption']}\n\n{descriptions[fid]}"
        chunk = make_chunk(f"figure-{fid.split()[-1]}", text, sec, ctype="figure")
        chunk["figure"] = fid
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    descriptions = describe_all()
    for fid, text in descriptions.items():
        print(f"\n===== {fid} =====\n{text[:300]}...")