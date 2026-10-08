# NorthStar RAG – retriever nad internom bazom znanja

Projekat sa prakse: retriever (a kasnije i agent) koji pronalazi odgovore u dokumentu
*NorthStar Knowledge Base v3.2*. Dokument sadrži tekst, tabele i slike (uključujući skenirani memo).
Svi podaci u njemu su fiktivni.

## Pipeline

```
.docx ─► parsiranje (Docling + izdvajanje slika) ─► Markdown + slike
      ─► opisi slika (Gemini) ─► chunking sa metapodacima
      ─► indeksiranje (BGE-M3 → Qdrant) ─► pretraga (+ filter arhive, proširenje, reranker)
      ─► evaluacija (Hit@k, Recall@k, MRR, nDCG@k ...) ─► results/
```

Stack: **Docling** (parsiranje) · **Qdrant** u lokalnom režimu (vektorska baza) · **BGE-M3** (embedding) ·
**bge-reranker-v2-m3** (reranker) · **Gemini 3.5 Flash** (opisi slika, besplatni nivo API-ja).

## Trenutni rezultati

Najbolja konfiguracija (eksperiment `6_rerank`) na 32 pitanja sa odgovorom:
**Hit@1 = 0,91 · Hit@3 = 1,00 · MRR = 0,95 · nDCG@10 = 0,96**, bez zastarelih pravila u rezultatima.
Glavni trošak: ~11 s po pitanju na procesoru (reranker). Detalji po eksperimentima su u `results/`.

## Struktura

```
northstar-rag/
├── data/
│   ├── raw/                     # originalni dokument (.docx)
│   └── processed/               # sve što pipeline generiše (nije u Git-u, može se ponovo napraviti):
│                                #   Markdown dokumenta, chunk-ovi (.jsonl, po strategiji),
│                                #   izdvojene slike + figures.json, keš opisa slika (po VLM modelu)
├── eval/
│   └── eval_questions.json      # kontrolna pitanja: očekivani odgovor, izvori (sekcija/slika), kategorija
├── notebooks/
│   ├── 01_explore_document.ipynb   # pregled dokumenta, slika i evaluacionog skupa
│   ├── 02_chunks.ipynb             # automatske provere chunk-ova (pokrivenost, TOC, arhiva, tabele)
│   └── 03_experiments.ipynb        # analiza rezultata eksperimenata po pitanjima
├── results/                     # rezultati evaluacije, jedan folder po pokretanju (<datum_vreme>_<ime>/):
│                                #   per_question.csv, summary.csv, mrr_by_category.csv, abstain.csv
├── src/
│   ├── config.py                # putanje, modeli, memoi koji menjaju sekcije (OVERRIDES)
│   ├── parse.py                 # .docx → Markdown (Docling) i izdvajanje slika sa naslovima
│   ├── chunking.py              # chunking: fiksna dužina i po sekcijama, metapodaci (status, office, ...)
│   ├── figures.py               # opisi slika preko Gemini API-ja (sa kešom) i chunk-ovi od opisa
│   ├── indexing.py              # embedding (BGE-M3), indeksiranje u Qdrant i dense pretraga
│   ├── retrieval.py             # proširenje rezultata (memoi, slike) i reranker
│   └── evaluate.py              # metrike, eksperimenti 1–6 i čuvanje rezultata
├── .env.example                 # šablon za API ključeve i podešavanja (kopirati u .env)
├── requirements.txt
└── setup.ps1                    # podešavanje okruženja na Windowsu
```

`qdrant_data/` (lokalna vektorska baza) i `.venv/` se prave automatski i nisu u Git-u.

## Podešavanje (Windows, VS Code)

Potrebno: **Python 3.12**, **Git**, **VS Code** sa ekstenzijama *Python* i *Jupyter*.

```powershell
# u VS Code terminalu, iz foldera projekta
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Skripta pravi virtuelno okruženje `.venv`, instalira pakete i pravi `.env` iz šablona.
Zatim u VS Code-u: `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.

Ručno, ako skripta ne radi:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

## Pokretanje

```powershell
python src/parse.py        # 1. .docx → Markdown + slike                    (~10 s)
python src/figures.py      # 2. opisi slika (Gemini API, samo prvi put)     (~1 min)
python src/chunking.py     # 3. chunk-ovi za obe strategije                  (nekoliko s)
python src/evaluate.py     # 4. svi eksperimenti + rezultati u results/     (~15 min na CPU)
```

Pojedinačni eksperimenti: `python src/evaluate.py --only 5_expansion 6_rerank` (spisak: `--help`).
Prvo pokretanje preuzima modele BGE-M3 i reranker (~2 GB svaki) u Hugging Face keš.

## Eksperimenti

| Ime | Šta se menja u odnosu na prethodni |
|---|---|
| `1_fixed` | baseline: chunk-ovi fiksne dužine (300 reči, preklapanje 50), dense pretraga |
| `2_section` | chunking po sekcijama dokumenta (tabele se ne seku, FAQ po pitanjima) |
| `3_section_path` | putanja naslova dodata u tekst za embedding |
| `3b_section_path_filter` | filter: zastarela pravila (pogl. 17) isključena iz pretrage |
| `4_figures` | dodati chunk-ovi sa opisima slika |
| `5_expansion` | uz rezultat se dodaju povezani memoi i slike na koje upućuje |
| `6_rerank` | 20 kandidata preuređuje cross-encoder (bge-reranker-v2-m3) |

## API ključevi

Ključevi idu samo u `.env` (taj fajl je u `.gitignore` i nikad ne ide na GitHub).
Repozitorijum je javan, pa ključ nikad ne pišite direktno u kod ili notebook.
Za opise slika potreban je `GEMINI_API_KEY` (besplatan ključ iz Google AI Studio).
