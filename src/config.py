"""Putanje i osnovna podešavanja projekta."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"
QDRANT_PATH = ROOT / "qdrant_data"

KB_DOCX = DATA_RAW / "NorthStar_Knowledge_Base.docx"
EVAL_QUESTIONS = EVAL_DIR / "eval_questions.json"

# Modeli (predlog za prvu nedelju)
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 20            # koliko kandidata iz Qdrant-a reranker preuredjuje (eksperiment 6)

VLM_MODEL = "gemini-3.5-flash"    # opisi slika preko Gemini API-ja (eksperiment 4)

for d in (DATA_PROCESSED, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Memoi koji menjaju pravila iz sekcija (sekcija -> lista memoa).
# Za sada ručno, kasnije se može izvlačiti automatski iz teksta memoa.
OVERRIDES = {
    "3.2": ["PT-2026-04"],
    "5.2": ["PT-2026-02"],
}