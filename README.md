# NorthStar RAG

RAG sistem (pretraga + generisanje odgovora) nad internom bazom znanja fiktivne kompanije NorthStar
(`NorthStar_Knowledge_Base.docx`, 48 strana: tekst, tabele, 9 slika, uključujući skenirani memo).
Projekat nastaje tokom prakse u Schneider Electric-u.

Dokument je namerno težak za pretragu:

- dve firme (NSS i NSFS) sa različitim pravilima za istu temu;
- šest kancelarija sa skoro identičnim tabelama;
- arhiva zastarelih pravila (poglavlje 17);
- memoi koji menjaju postojeća pravila (PT-2026-02, PT-2026-04), jedan samo kao skenirana slika.

## Pipeline

```
.docx ──► parse.py ──► Markdown + slike
                │
                ├──► figures.py ──► opisi slika (Gemini, keširano)
                │
                └──► chunking.py ──► delovi dokumenta (po sekcijama) + metapodaci
                                         │  (status: valid / memo / superseded, overridden_by, office...)
                                         ▼
                        indexing.py ──► Qdrant: dense + sparse vektori (BGE-M3)
                                         │
                   pitanje ──► pretraga (hibrid, RRF) ──► top 5 delova
                                         │
                                         ▼
                        generate.py ──► LLM (Qwen2.5, Colab) ──► odgovor + citati + odbijanje
                                         │
                                         ▼
               evaluate.py (pretraga)  ·  evaluate_text.py (odgovori)
```

## Trenutni rezultati

**Pretraga** – evaluacioni skup od 49 pitanja (45 sa odgovorom, 4 bez odgovora; 17 traži 2–6 izvora).

| Pretraga                     | Hit@1 (32 osnovna pitanja) | Svi izvori u top 5 (17 višeizvornih) | Vreme po pitanju |
| ---------------------------- | -------------------------- | ------------------------------------- | ---------------- |
| Dense                        | 0,69                       | 71%                                   | 0,4 s            |
| Hibrid (dense + sparse, RRF) | 0,72                       | 88%                                   | 0,2 s            |
| Dense + reranker             | 0,91                       | 88%                                   | ~14 s (CPU)      |

- Na jednostavnim pitanjima hibrid ne donosi merljivo poboljšanje; na pitanjima sa više izvora dostiže reranker, a mnogo je brži.
- Razlaganje pitanja preko LLM-a (query decomposition, eksp. 10–11) uz hibrid ne pomaže.

**Generisanje** – hibridna pretraga, top 5, ručna ocena svih 49 odgovora:

| Model                        | Tačno / delimično / netačno | Vreme po odgovoru (T4 GPU) |
| ---------------------------- | ------------------------------ | -------------------------- |
| Qwen2.5-7B-Instruct (4-bit)  | 33 / 8 / 8                     | 3,8 s                      |
| Qwen2.5-14B-Instruct (4-bit) | 39 / 6 / 4                     | 9,3 s                      |

- Od 7 netačnih odgovora 7B modela (na pitanja sa odgovorom), 6 je imalo sve potrebne izvore u kontekstu: greške pravi generator, ne pretraga.
- Razlika 7B / 14B nije statistički pouzdana na 49 pitanja (McNemar, p ≈ 0,18).

Detalji svakog pokretanja su u `results/`.

## Struktura

```
northstar-rag/
├── data/
│   ├── raw/NorthStar_Knowledge_Base.docx
│   └── processed/                     # generisano; u git-u samo ulazi potrebni za ponovljivost:
│                                      #   chunks_section.jsonl, figures/figures.json,
│                                      #   figure_descriptions_*.json, decompose_*.json
├── eval/eval_questions.json           # 49 pitanja: question, expected_answer, sources, category
├── notebooks/
│   ├── 01_explore_document.ipynb
│   ├── 02_chunks.ipynb
│   ├── 03_experiments.ipynb           # analiza rezultata pretrage
│   └── 04_generation.ipynb            # generisanje odgovora u Google Colab-u (GPU)
├── results/                           # jedan folder po pokretanju: <datum_vreme>_<ime>/
├── src/
│   ├── config.py                      # putanje, modeli, parametri, memoi koji menjaju sekcije (OVERRIDES)
│   ├── parse.py                       # .docx → Markdown (Docling) + slike sa naslovima
│   ├── chunking.py                    # chunking (fiksna dužina / po sekcijama) i metapodaci
│   ├── figures.py                     # opisi slika preko Gemini API-ja (keš po modelu)
│   ├── indexing.py                    # BGE-M3 dense + sparse, Qdrant, hibridna i višeupitna pretraga,
│   │                                  #   čuvanje indeksa (otisak chunk-ova u imenu kolekcije)
│   ├── retrieval.py                   # proširenje (memoi, slike) i reranker
│   ├── decompose.py                   # razlaganje pitanja na podpitanja (Gemini, grupno, keširano)
│   ├── evaluate.py                    # metrike pretrage i eksperimenti 1–11
│   ├── generate.py                    # prompt, kontekst sa statusom izvora, HF model, JSON odgovor
│   └── evaluate_text.py               # metrike odgovora bez LLM-a
├── .env.example                       # šablon za ključeve (kopirati u .env)
├── requirements.txt
└── setup.ps1                          # podešavanje okruženja na Windowsu
```

`qdrant_data/` (lokalna baza), `.venv/` i `.env` se ne čuvaju u git-u.

## Podešavanje (Windows, VS Code)

```powershell
.\setup.ps1                                   # ili ručno: python -m venv .venv; pip install -r requirements.txt
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   # jednom, ako aktivacija ne radi
.\.venv\Scripts\Activate.ps1
copy .env.example .env                        # pa upisati ključeve
```

## API ključevi

U `.env` (nikad u kod, notebook ili commit – repo je javan):

- `GEMINI_API_KEY` – opisi slika (`figures.py`) i razlaganje pitanja (`decompose.py`), besplatni nivo;
- `HF_TOKEN` – preuzimanje modela sa Hugging Face-a.

U Colab-u se `HF_TOKEN` unosi preko **Secrets** (ikonica ključa), ne u ćeliju.

Besplatni Gemini ima dnevne kvote po modelu (npr. 20 zahteva dnevno za Flash modele), pa se svi
LLM pozivi keširaju; `decompose.py` šalje 10 pitanja po zahtevu.

## Pokretanje

**Priprema podataka** (lokalno):

```powershell
python src/parse.py          # .docx → Markdown + slike                     (~10 s)
python src/figures.py        # opisi slika (Gemini, samo prvi put)
python src/chunking.py       # delovi dokumenta za obe strategije
python src/decompose.py      # podpitanja za eksp. 10–11 (Gemini, samo prvi put)
```

**Pretraga – eksperimenti** (lokalno):

```powershell
python src/evaluate.py                               # svi eksperimenti
python src/evaluate.py --only 4_figures 8_hybrid     # samo navedeni (spisak: --help)
```

Indeksi se čuvaju u `qdrant_data/` i prave ponovo samo kad se promene delovi dokumenta.
Eksperimenti sa rerankerom (6, 9) su spori na CPU-u (~10–15 s po pitanju).

**Generisanje odgovora** (Google Colab, T4 GPU): otvoriti `notebooks/04_generation.ipynb`,
u ćeliji sa podešavanjima izabrati `GEN_MODEL` i `RUN_NAME`, pa *Run all*. Rezultat
(`answers.jsonl`, `run_config.json`) se preuzima i raspakuje u `results/`.

**Evaluacija odgovora** (lokalno):

```powershell
python src/evaluate_text.py                                    # najnovije generisanje
python src/evaluate_text.py 2026-10-14_1022_gen_hybrid_qwen7b  # određeno pokretanje
```

Ako u folderu pokretanja postoji `manual_grades.csv` (`id, manual_grade, note`), dobija se i
tabela „ručna ocena × pokrivenost izvora“ (greške pretrage naspram grešaka generisanja).

## Eksperimenti

**Pretraga** (`evaluate.py`):

| Ime                        | Šta se menja                                                               |
| -------------------------- | --------------------------------------------------------------------------- |
| `1_fixed`                | baseline: delovi fiksne dužine (300 reči, preklapanje 50), dense pretraga |
| `2_section`              | chunking po sekcijama (tabele se ne seku, FAQ po pitanjima)                 |
| `3_section_path`         | putanja naslova dodata u tekst za embedding                                 |
| `3b_section_path_filter` | zastarela pravila (pogl. 17) isključena iz pretrage                        |
| `4_figures`              | dodati delovi sa opisima slika                                              |
| `5_expansion`            | uz rezultat se dodaju povezani memoi i slike                                |
| `6_rerank`               | 20 kandidata preuređuje cross-encoder (bge-reranker-v2-m3)                 |
| `4b_dense_m3`            | kontrola: isto kao 4, dense iz FlagEmbedding-a                              |
| `7_sparse`               | samo BGE-M3 sparse                                                          |
| `8_hybrid`               | dense + sparse, spojeni RRF-om                                              |
| `9_hybrid_rerank`        | hibrid → proširenje → reranker                                           |
| `10_decomp`              | pitanje + podpitanja (LLM), dense, RRF – porediti sa 4                     |
| `11_decomp_hybrid`       | pitanje + podpitanja (LLM), hibrid, RRF – porediti sa 8                    |

**Generisanje** (`04_generation.ipynb`): `gen_hybrid_qwen7b`, `gen_hybrid_qwen14b` – pretraga kao eksp. 8, top 5.

## Metrike

**Pretraga** (`evaluate.py`):

- Hit@k, Recall@k, MRR, nDCG@10 – na nivou izvora (sekcija ili slika);
- pokrivenost svih izvora (all_sources@k) – da li su svi potrebni izvori u top k;
- cover_rank – pozicija poslednjeg potrebnog izvora (koliki k je potreban);
- superseded@5, words@5, latencija; `multi_source.csv` – samo pitanja sa 2+ izvora.

**Odgovori** (`evaluate_text.py`, bez LLM-a):

- tačno odbijanje (abstain recall) i pogrešno odbijanje (false abstain rate);
- izmišljeni brojevi (unsupported numbers) – broj u odgovoru kog nema u datim izvorima;
- preciznost i odziv citata (citation precision / recall);
- poklapanje brojeva sa očekivanim odgovorom (number recall) – grub signal;
- pokrivenost izvora u kontekstu (context coverage).
