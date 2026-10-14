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


def source_label(chunk: dict) -> str:
    """Oznaka izvora koju model citira: 'Figure 3' za slike, broj sekcije za ostalo (npr. '5.2')."""
    return chunk.get("figure") or chunk["section"]


def format_context(chunks: list[dict]) -> str:
    """Pronađeni chunkovi kao numerisani blokovi sa oznakom, putanjom naslova i statusom."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = f"status: {c['status']}"
        if c.get("overridden_by"):
            meta += f"; amended by: {', '.join(c['overridden_by'])}"
        blocks.append(f"[{i}] Source {source_label(c)} | {c['heading_path']} | {meta}\n{c['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Poruke u chat formatu (system + user), koji razume svaki instruct model."""
    user = f"Sources:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_response(text: str) -> dict:
    """Izvlači JSON iz odgovora modela -  ako ne uspe, čuva sirovi tekst i označava grešku."""
    match = re.search(r"\{.*\}", text, re.DOTALL)      # prvi {...} blok, i ako model doda tekst oko njega
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return {"answer": text.strip(), "citations": [], "abstained": False, "parse_error": True}
    citations = data.get("citations") or []
    return {
        "answer": str(data.get("answer", "")).strip(),
        "citations": [str(c).replace("Source ", "").strip() for c in citations if str(c).strip()],
        "abstained": bool(data.get("abstained", False)),
        "parse_error": False,
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

def answer_question(question: str, chunks: list[dict], llm) -> dict: # cuva i ono sto model vidi, a nije odgovorio
    """Generiše odgovor za jedno pitanje na osnovu datih chunkova."""
    start = time.perf_counter()
    raw = llm(build_messages(question, chunks))
    result = parse_response(raw)
    result.update({
        "context": [source_label(c) for c in chunks],     # šta je model video (za faithfulness i analizu)
        "context_ids": [c["chunk_id"] for c in chunks],
        "raw": raw,
        "latency_s": round(time.perf_counter() - start, 2),
    })
    return result