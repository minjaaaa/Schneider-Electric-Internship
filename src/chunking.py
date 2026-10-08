"""Deljenje Markdown dokumenta na chunk-ove sa metapodacima.

Dve strategije za poređenje:
  A) fiksna dužina (N reči sa preklapanjem) - baseline
  B) po sekcijama dokumenta (x.y), FAp po pitanjima, tabele se nikad ne seku
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from config import OVERRIDES

# "## 5. Travel and Expenses" ili "### 5.2 Hotels"
HEADING_RE = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)*)\.?\s+(.+)$")
# "<!-- image -->" pa prazan red pa "*Figure 3: Expense claim approval*"
FIGURE_PLACEHOLDER_RE = re.compile(r"<!-- image -->\s*\n\s*\*(Figure \d+): (.+?)\*")
# "**Q: Can I carry over unused annual leave?**"
FAQ_Q_RE = re.compile(r"^\*\*Q:\s*(.+?)\*\*$")

# --------------- 1. Ciscenje teksta -------------------

def load_markdown(path: Path) -> str:
    """ Ucitava Markdown i pretvara HTML entitete u normalan tekst. (&amp; -> &) """
    return html.unescape(path.read_text(encoding="utf-8"))

def remove_toc(md: str) -> str:
    """ Uklanja sadrzaj (Contents) sve od 'Contents' do prvog naslova (##). """
    lines = md.splitlines()
    start = next( i for i, line in enumerate(lines) if line.strip()=='## Contents')
    end = next(i for i in range(start + 1, len(lines)) if HEADING_RE.match(lines[i])) # Pronalazi index kraja
    return "\n".join(lines[:start] + lines[end:])

def replace_figure_placeholders(md: str) -> str:
    """'<!-- image -->' + naslov slike -> '[Figure 3: Expense claim approval]'."""
    return FIGURE_PLACEHOLDER_RE.sub(r"[\1: \2]", md)

# --------------- 2. Podela na sekcije -------------------

def split_sections(md: str) -> list[dict]:
    """ Deli dokument na sekcije. Svaka sekcija: section, section_title, geading_path, text """
    sections = []
    current = {"section": "0", "section_title": "Document information",
               "heading_path": "Document information", "lines": []}
    chapter_title = ""

    for line in md.splitlines():
        m = HEADING_RE.match(line)
        if m:
            sections.append(current)
            level, number, title = m.groups()
            section_title = f"{number} {title}"
            if level == '##':                           # poglavlje, npr. "5. Travel and Expenses"
                chapter_title = f"{number}. {title}"
                heading_path = chapter_title
            else:                                       # podsekcija, npr. "5.2 Hotels"
                heading_path = f"{chapter_title} > {section_title}"
            current = {"section": number, "section_title": section_title,
                       "heading_path": heading_path, "lines": []}
        elif not line.startswith("# "):
            current["lines"].append(line)
    sections.append(current)

    result = []
    for s in sections:
        text = "\n".join(s.pop("lines")).strip()
        if text:                                   # poglavlja bez sopstvenog teksta preskačemo
            result.append({**s, "text": text})     # npr. ## 5. Travel and Expenses, posle kojeg odmah ide ### 5.1.
    return result

# --------------- 3. Chunk-ovanje -------------------
# Metoda A: fiksna duzina chunk-ova (baseline) i metapodaci koji zavise samo od sekcije (status, office, overriden_by)

def section_metadata(section: str, section_title: str) -> dict:
    """ Metapodaci koji zavise samo od broja i naslova sekcije, a ne od teksta. """
    chapter = section.split(".")[0]
    if chapter == "17":
        status = "superseded"
    elif chapter == "16.2":
        status = "memo"
    else:
        status = "valid"

    office = None
    if chapter == "8" and section != "8" and section != "8.1":
        office = section_title.split()[1]  # "8.3 Lisbon office" -> "Lisbon"
    return {"status": status, "office": office, "overriden_by": OVERRIDES.get(section, [])}

def content_type(text: str) -> str:
    """ Ako sadrzi Markdown tabelu - 'table', inace 'text' """
    return "table" if any(line.startswith("|") for line in text.splitlines()) else "text"

def make_chunk(chunk_id: str,
               text: str,
               sec: dict,
               ctype: str | None = None,
               sections: list[str] | None = None) -> dict:
    """ Kreira chunk sa metapodacima. """
    return{
        "chunk_id": chunk_id,
        "text": text,
        "section": sec["section"],
        "sections": sections or [sec["section"]],
        "section_title": sec["section_title"],
        "heading_path": sec["heading_path"],
        "type": ctype or content_type(text),
        **section_metadata(sec["section"], sec["section_title"])
    }

def chunk_fixed(sections: list[dict], size: int = 300, overlap: int = 50) -> list[dict]:
    """Prozori od `size` reči sa preklapanjem. Chunk može da pokrije više sekcija."""
    words = [(w, sec) for sec in sections for w in sec["text"].split()]
    chunks, step = [], size - overlap
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start:start + size]
        covered = list(dict.fromkeys(sec["section"] for _, sec in window))  # jedinstvene, redom (uklanja duplikate)
        text = " ".join(w for w, _ in window)
        chunks.append(make_chunk(f"fixed-{i:03d}", text, window[0][1], sections=covered))
        if start + size >= len(words):
            break
    return chunks

# --------------- 3. Chunk-ovanje -------------------
# Metoda B: chunk-ovanje po sekcijama, FAQ pitanjima i tabelama (ne seku se tabele)

def split_faq(text: str) -> list[str]:
    """Deli FAQ sekciju na pitanja. Svaki chunk je pitanje + odgovor. Ako nema pitanja, vraća ceo tekst."""
    pairs, current = [], []
    for block in text.split("\n\n"):
        block = block.strip()
        if FAQ_Q_RE.match(block) and current:
            pairs.append("\n".join(current))
            current = []
        if block:
            current.append(FAQ_Q_RE.sub(r"Q: \1", block))  # uklanja ** i dodaje "Q: " na pocetak pitanja
                                                           # ako nije bilo pitanje, sub dodaje ceo blok u current
    if current: # Posle petlje u current ostaje poslednji par
        pairs.append("\n".join(current))
    return [p for p in pairs if p.startswith("Q: ")] or [text]  # ako nema pitanja, vraca ceo tekst kao jedan chunk

"""
**Q: Can I carry over unused annual leave?**

A: Yes, within limits. NSS employees may carry over up to 5 days ...

**Q: Can I carry over my unused learning budget?**

A: No. The learning budget resets on 1 January. See section 10.2.
"""

def split_long(text: str, max_words: int) -> list[str]:
    """Deli predugačak tekst po blokovima (pasusima). Tabela je jedan blok, pa se ne seče."""
    parts, current = [], []
    for block in text.split("\n\n"):
        if current and len(" ".join(current + [block]).split()) > max_words:
            parts.append("\n\n".join(current))
            current = []
        current.append(block)
    if current:
        parts.append("\n\n".join(current))
    return parts

def chunk_by_section(sections: list[dict], max_words: int = 500) -> list[dict]:
    chunks = []
    for sec in sections:
        if sec["section"] == "14":                 # FAQ: jedno pitanje = jedan chunk
            for i, qa in enumerate(split_faq(sec["text"])):
                chunks.append(make_chunk(f"faq-{i:02d}", qa, sec, ctype="faq"))
            continue
        for i, part in enumerate(split_long(sec["text"], max_words)):
            chunks.append(make_chunk(f"section-{sec['section']}-{i}", part, sec))
    return chunks

# --------------- 4. Pomocne funkcije -------------------

def text_for_embedding(chunk: dict, with_path: bool = True) -> str:
    """Tekst koji ide u embedding model; sa ili bez putanje naslova (eksperiment 2 vs 3)."""
    return f"{chunk['heading_path']}\n\n{chunk['text']}" if with_path else chunk["text"]

def save_jsonl(chunks: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

if __name__ == "__main__":
    from config import DATA_PROCESSED

    md = load_markdown(DATA_PROCESSED / "NorthStar_Knowledge_Base.md")
    md = replace_figure_placeholders(remove_toc(md))
    sections = split_sections(md)

    for name, chunks in [("fixed", chunk_fixed(sections)), ("section", chunk_by_section(sections))]:
        save_jsonl(chunks, DATA_PROCESSED / f"chunks_{name}.jsonl")
        lengths = sorted(len(c["text"].split()) for c in chunks)
        print(f"{name:8} {len(chunks):4} chunk-ova | reči: min {lengths[0]}, "
              f"medijana {lengths[len(lengths) // 2]}, max {lengths[-1]}")