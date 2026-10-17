"""Generisanje odgovora: LLM dobija pitanje i pronađene chunkove, vraća odgovor sa citatima (JSON).

Pravila u promptu su opšta,
pa važe za bilo koji dokument obrađen istim pipeline-om.
"""
from __future__ import annotations

import json
import re
import time

SYSTEM_PROMPT = """You answer questions about an internal company knowledge base.
Use ONLY the numbered sources below. Do not use outside knowledge.

Rules:
1. Every fact in your answer must come from the sources. Cite the source labels you used.
2. If the sources do not contain the answer, set "abstained" to true and say that the knowledge base
   does not contain this information. Do not guess.
3. Source status matters:
   - "superseded" sources are historical and must never be used as the current rule;
   - a source "amended by" a memo is overridden by that memo where they differ: give the memo's value;
   - "memo" sources take precedence over the rules they change.
4. If a rule differs by company, office or role, give the value for the one asked about.
   If the question does not say which, give the value for each.
5. Be concise: one to three sentences.

Respond with JSON only, in this format:
{"answer": "...", "citations": ["<source label>", ...], "abstained": false}"""


# ---------- v2: oznake umesto brojeva blokova, uža pravila, dokazi pre odgovora ----------
# Delovi se slažu po verzijama, pa se svaka verzija od prethodne razlikuje samo u onome što je dodato.

V2_INTRO = """You answer questions about an internal company knowledge base.
Use ONLY the sources below. Do not use outside knowledge.
Each source starts with a header line: Source: <label> | Section: <heading path> | Status: <status>."""

V2A_RULES = """Rules:
1. Every fact in your answer must come from the sources. Cite the label written after "Source:"
   exactly as it appears (a section number or a figure name). Do not cite memo names or block positions.
2. If the sources do not contain the answer, set "abstained" to true and say that the knowledge base
   does not contain this information. Do not guess.
3. Source status matters:
   - "superseded" sources are historical and must never be used as the current rule;
   - a source "amended by" a memo is overridden by that memo where they differ: give the memo's value;
   - "memo" sources take precedence over the rules they change.
4. If a rule differs by company, office or role, give the value for the one asked about.
   If the question does not say which, give the value for each.
5. Be concise: one to three sentences."""

V2B_RULES = """Rules:
1. Every fact in your answer must come from the sources. Cite the label written after "Source:"
   exactly as it appears (a section number or a figure name). Do not cite memo names or block positions.
2. Copy values exactly as written (numbers, units, names). Never derive a value from a pattern,
   an example, or a source about a different office, company, product or role.
3. Source status matters:
   - "superseded" sources are historical and must never be used as the current rule;
   - a "memo" changes only the values it explicitly names (for example one city, one company or one limit);
     all other values in the amended source still apply unchanged;
   - "amended by: <memo>" means some values in that source were changed: check the memo, and use the
     memo's value only for what the memo names.
4. If a rule differs by company, office or role, give the value for the one asked about.
   If the question does not say which, give the value for each.
5. In a table, first find the row and the column that match the question (the entity, the range that
   contains the given amount, the direction such as given or received), then read the value.
6. If the question has several parts, answer every part. If one part is not in the sources, answer the
   other parts and say which part the knowledge base does not cover.
7. If the sources do not state the answer, set "abstained" to true and say that the knowledge base
   does not contain this information. Do not guess.
8. Be concise: one to three sentences for each part of the question."""

V2_OUTPUT = """Respond with one JSON object and nothing else. Use double quotes around every key and every
string value, and escape any double quote inside a string as \\".
Format:
{"answer": "<your answer>", "citations": ["<source label>", ...], "abstained": false}"""

V2C_OUTPUT = """Respond with one JSON object and nothing else. Use double quotes around every key and every
string value, and escape any double quote inside a string as \\".
Fill "evidence" first: copy, word for word, each sentence or table row that contains the answer,
starting with its source label. For a table row, pair every cell with its column name.
Then write the answer using only that evidence. If there is no evidence, leave "evidence" empty
and set "abstained" to true.
Format:
{"evidence": ["<source label>: <copied sentence, or column: value pairs of a table row>", ...],
 "answer": "<your answer>", "citations": ["<source label>", ...], "abstained": false}"""

PROMPTS = {
    "v1": SYSTEM_PROMPT,                                              # bazna verzija (nedelja 2), ne menjati
    "v2a": "\n\n".join([V2_INTRO, V2A_RULES, V2_OUTPUT]),            # samo format: oznake, citati, JSON
    "v2b": "\n\n".join([V2_INTRO, V2B_RULES, V2_OUTPUT]),            # + opšta pravila iz analize grešaka
    "v2c": "\n\n".join([V2_INTRO, V2B_RULES, V2C_OUTPUT]),           # + dokazi (evidence) pre odgovora
}
CONTEXT_STYLE = {"v1": "numbered", "v2a": "labeled", "v2b": "labeled", "v2c": "labeled"}


def prompt_fingerprint(version: str) -> str:
    """Kratak heš system prompta i stila konteksta - ide u run_config.json, da se zna tačno šta je pokrenuto."""
    import hashlib
    return hashlib.sha256((PROMPTS[version] + CONTEXT_STYLE[version]).encode("utf-8")).hexdigest()[:8]


def source_label(chunk: dict) -> str:
    """Oznaka izvora koju model citira: 'Figure 3' za slike, broj sekcije za ostalo (npr. '5.2')."""
    return chunk.get("figure") or chunk["section"]


def format_context(chunks: list[dict], style: str = "numbered") -> str:
    """Pronađeni chunkovi sa oznakom, putanjom naslova i statusom.
    numbered (v1): '[i] Source 5.2 | ...'  -  labeled (v2): 'Source: 5.2 | Section: ... | Status: ...', bez rednog broja."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        if style == "numbered":
            meta = f"status: {c['status']}"
            if c.get("overridden_by"):
                meta += f"; amended by: {', '.join(c['overridden_by'])}"
            blocks.append(f"[{i}] Source {source_label(c)} | {c['heading_path']} | {meta}\n{c['text']}")
        else:
            meta = f"Status: {c['status']}"
            if c.get("overridden_by"):
                meta += f"; amended by: {', '.join(c['overridden_by'])}"
            blocks.append(f"Source: {source_label(c)} | Section: {c['heading_path']} | {meta}\n{c['text']}")
    return "\n\n".join(blocks) if style == "numbered" else "\n\n---\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict], prompt_version: str = "v1") -> list[dict]:
    """Poruke u chat formatu (system + user), koji razume svaki instruct model."""
    context = format_context(chunks, CONTEXT_STYLE[prompt_version])
    user = f"Sources:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": PROMPTS[prompt_version]}, {"role": "user", "content": user}]


def _repair_fields(text: str) -> dict | None:
    """Rezerva kad JSON nije ispravan (npr. 14B: {"answer": tekst bez navodnika, "citations": [...]}).
    Polja se traže po imenu ključa, a ne po navodnicima oko vrednosti."""
    ans = re.search(r'"answer"\s*:\s*"?(.*?)"?\s*,\s*"(?:citations|abstained)"', text, re.DOTALL)
    if not ans:
        return None
    cit = re.search(r'"citations"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    abst = re.search(r'"abstained"\s*:\s*(true|false)', text)
    return {"answer": ans.group(1),
            "citations": re.findall(r'"([^"]+)"', cit.group(1)) if cit else [],
            "abstained": bool(abst and abst.group(1) == "true")}


def parse_response(text: str) -> dict:
    """Izvlači JSON iz odgovora modela; ako ne uspe, pokušava popravku po ključevima; na kraju čuva sirovi tekst."""
    match = re.search(r"\{.*\}", text, re.DOTALL)      # prvi {...} blok, i ako model doda tekst oko njega
    repaired = False
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        data = _repair_fields(text)
        repaired = data is not None
    if not isinstance(data, dict):
        return {"answer": text.strip(), "citations": [], "evidence": [], "abstained": False,
                "parse_error": True, "repaired": False}
    citations = data.get("citations") or []
    return {
        "answer": str(data.get("answer", "")).strip(),
        "citations": [re.sub(r"^Source:?\s*", "", str(c)).strip() for c in citations if str(c).strip()],
        "evidence": [str(e).strip() for e in (data.get("evidence") or [])],
        "abstained": bool(data.get("abstained", False)),
        "parse_error": False,
        "repaired": repaired,
    }


class HFChatModel:
    """Otvoreni instruct model sa Hugging Face-a (za Colab GPU). Poziv: model(messages) -> tekst."""

    def __init__(self, model_id: str, load_in_4bit: bool = True, max_new_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                          quantization_config=quant, torch_dtype=torch.float16)

    def __call__(self, messages: list[dict]) -> str:
        import torch
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True).to(self.model.device)
        with torch.no_grad():           # raspakuje recnik u argumente za model.generate()
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        n_prompt = inputs["input_ids"].shape[1]                  # dužina prompta u tokenima
        return self.tokenizer.decode(output[0][n_prompt:], skip_special_tokens=True)

def answer_question(question: str, chunks: list[dict], llm, prompt_version: str = "v1") -> dict:
    """Generiše odgovor za jedno pitanje na osnovu datih chunkova (čuva i ono što je model video)."""
    start = time.perf_counter()
    raw = llm(build_messages(question, chunks, prompt_version))
    result = parse_response(raw)
    result.update({
        "context": [source_label(c) for c in chunks],     # šta je model video (za faithfulness i analizu)
        "context_ids": [c["chunk_id"] for c in chunks],
        "raw": raw,
        "latency_s": round(time.perf_counter() - start, 2),
    })
    return result