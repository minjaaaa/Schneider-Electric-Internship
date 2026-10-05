"""Parsiranje .docx dokumenta: tekst i tabele u Markdown (Docling) i izdvajanje slika sa naslovima.

Slike u dokumentu nemaju alternativni tekst, a redosled fajlova (image1.png, image2.png...)
ne odgovara brojevima slika (Figure 7 je poslednja slika u fajlu). Zato se svaka slika
povezuje sa naslovom "Figure N: ..." koji stoji odmah posle nje u dokumentu.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
FIGURE_RE = re.compile(r"^(Figure\s+\d+)\s*:\s*(.+)$")
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")


def docx_to_markdown(docx_path: Path, out_path: Path | None = None) -> str:
    """Konvertuje .docx u Markdown pomoću Docling-a (čuva naslove i tabele)."""
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(docx_path))
    md = result.document.export_to_markdown()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
    return md


def _paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(f"{{{NS['w']}}}t")).strip()


def _paragraph_style(p: ET.Element) -> str:
    st = p.find("w:pPr/w:pStyle", NS)
    return st.get(f"{{{NS['w']}}}val", "") if st is not None else ""


def extract_figures(docx_path: Path, out_dir: Path) -> list[dict]:
    """Izdvaja slike iz .docx i vezuje svaku za naslov (Figure N) i sekciju u kojoj se nalazi.

    Vraća listu rečnika: figure_id, caption, section, section_title, image_path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path) as z:
        rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
        rid_to_target = {r.get("Id"): r.get("Target") for r in rels.findall("rel:Relationship", NS)}
        body = ET.fromstring(z.read("word/document.xml")).find("w:body", NS)

        figures, pending_image = [], None
        section, section_title = None, None
        for p in body.iter(f"{{{NS['w']}}}p"):
            text, style = _paragraph_text(p), _paragraph_style(p)

            if style.lower().startswith("heading") and text:
                m = SECTION_RE.match(text)
                if m:
                    section, section_title = m.group(1), text

            blip = p.find(".//a:blip", NS)
            if blip is not None:
                pending_image = rid_to_target[blip.get(f"{{{NS['r']}}}embed")]
                continue

            m = FIGURE_RE.match(text)
            if m and pending_image:
                fig_id = m.group(1)
                img_name = fig_id.lower().replace(" ", "_") + Path(pending_image).suffix
                (out_dir / img_name).write_bytes(z.read(f"word/{pending_image}"))
                figures.append({
                    "figure_id": fig_id,
                    "caption": m.group(2).strip(),
                    "section": section,
                    "section_title": section_title,
                    "image_path": str(out_dir / img_name),
                    "source_file": pending_image,
                })
                pending_image = None

    (out_dir / "figures.json").write_text(json.dumps(figures, indent=2, ensure_ascii=False), encoding="utf-8")
    return figures


if __name__ == "__main__":
    from config import KB_DOCX, DATA_PROCESSED

    figs = extract_figures(KB_DOCX, DATA_PROCESSED / "figures")
    for f in figs:
        print(f"{f['figure_id']:<10} sekcija {f['section']:<6} {f['caption']}  <- {f['source_file']}")
    md = docx_to_markdown(KB_DOCX, DATA_PROCESSED / "NorthStar_Knowledge_Base.md")
    print(f"\nMarkdown: {len(md):,} karaktera")
