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
                        generate.py ──► LLM (Gemma 4 12B, prompt v2b, Colab) ──► odgovor + citati + odbijanje
                                         │
                                         ▼
     evaluacija:  evaluate.py (pretraga)        evaluate_text.py (odgovori, bez LLM-a)
                  context_metrics.py (kontekst) judge.py → validate_judge.py (tačnost, LLM ocenjivač)
                                                faithfulness.py (verodostojnost prema kontekstu)
```

## Metodologija rada

Svaki korak prati isti obrazac: **probam → analiziram rezultate → postavim pretpostavku → proverim je
eksperimentom → pretpostavka potvrđena ili ne**. Detalji i brojevi su u nedeljnim izveštajima; ovde su
principi koje kod i struktura projekta podržavaju.

- **Jedna promena po eksperimentu.** Pretraga i generisanje mere se odvojeno. Za poređenje generatora
  kontekst je *zamrznut*: svaki model dobija iste delove dokumenta (`BASE_RUN`), pa razlika potiče samo
  od prompta ili modela. Verzije prompta dodaju po jednu grupu izmena (ablacija).
- **Ponovljivost.** Generisanje bez nasumičnosti (greedy); kontrolno pokretanje mora ponoviti bazni
  izlaz; svaki prompt (generator, ocenjivač) ima verziju i heš u konfiguraciji pokretanja; rezultati
  svakog pokretanja u sopstvenom folderu sa `run_config.json`.
- **Automatska ocena umesto ručne.** Očekivani odgovori su razloženi na atomske činjenice; LLM ocenjivač
  odgovara samo na usko pitanje „da li je činjenica prisutna“, a ocenu izvodi pravilo u kodu – isti
  kriterijum za sve modele.
- **Ocenjivač je merni instrument – validira se.** Pouzdanost (dva prolaza daju isto), valjanost
  (slaganje sa ljudskom referencom po *istoj* rubrici, Cohen's kappa) i pristrasnost (sistematske
  greške po modelu, dužini odgovora, kategoriji). Validacija se ponavlja pri svakoj promeni ocenjivača.
- **Kontrolni primeri.** Pre punog ocenjivanja ocenjivač se proverava na odgovoru poznatog ishoda;
  rezultat se gleda pre tumačenja brojeva.
- **Provere bez LLM-a gde je moguće.** Kvalitet konteksta iz označenih izvora; izmišljeni brojevi i
  nazivi proveravaju se doslovnim poklapanjem sa izvorima – LLM ocenjivač može imati istu slabost kao
  generator.
- **Statistika.** Poređenja sistema planirana unapred; egzaktni McNemar na istim pitanjima i Holm
  korekcija za više poređenja. „Nije značajno“ znači „razlika nije pokazana“.
- **Automatski testovi.** Parsiranje, pravila ocene, statistika, metrike i celovitost podataka – bez
  GPU-a i LLM-a, pre svakog commit-a; svaka pronađena greška dobija regresioni test.

## Struktura

```
northstar-rag/
├── data/
│   ├── raw/NorthStar_Knowledge_Base.docx
│   └── processed/                     # generisano; u git-u samo ulazi potrebni za ponovljivost:
│                                      #   chunks_section.jsonl, figures/figures.json,
│                                      #   figure_descriptions_*.json, decompose_*.json
├── eval/
│   ├── eval_questions.json            # 49 pitanja: question, expected_answer, sources, category
│   ├── expected_facts.json            # očekivani odgovori razloženi na atomske činjenice (core / dodatne)
│   └── judge_validation/
│       └── human_verdicts.jsonl       # referenca za validaciju ocenjivača (verdikti po činjenicama)
├── notebooks/
│   ├── 01_explore_document.ipynb
│   ├── 02_chunks.ipynb
│   ├── 03_experiments.ipynb           # analiza rezultata pretrage
│   ├── 04_generation.ipynb            # generisanje odgovora (Colab, GPU)
│   ├── 05_prompt_versions.ipynb       # verzije prompta i modeli na zamrznutom kontekstu (Colab)
│   ├── 06_judge.ipynb                 # LLM ocenjivač tačnosti nad sačuvanim odgovorima (Colab)
│   └── 07_faithfulness.ipynb          # LLM ocena verodostojnosti izabranog sistema (Colab)
├── results/                           # pretraga: jedan folder po pokretanju <datum_vreme>_<ime>/
├── results_generator/                 # generisanje: <datum_vreme>_gen_<ime>/ sa answers.jsonl,
│                                      #   run_config.json, judge_<tag>.jsonl, faithfulness_<tag>.jsonl
├── src/
│   ├── config.py                      # putanje, modeli, parametri, memoi koji menjaju sekcije (OVERRIDES)
│   ├── parse.py                       # .docx → Markdown (Docling) + slike sa naslovima
│   ├── chunking.py                    # chunking (fiksna dužina / po sekcijama) i metapodaci
│   ├── figures.py                     # opisi slika preko Gemini API-ja (keš po modelu)
│   ├── indexing.py                    # BGE-M3 dense + sparse, Qdrant, hibridna i višeupitna pretraga
│   ├── retrieval.py                   # proširenje (memoi, slike) i reranker
│   ├── decompose.py                   # razlaganje pitanja na podpitanja (Gemini, grupno, keširano)
│   ├── evaluate.py                    # metrike pretrage i eksperimenti 1–11
│   ├── generate.py                    # verzije prompta (v1–v2c, heš), kontekst, HF model (Qwen / Gemma 4)
│   ├── evaluate_text.py               # metrike odgovora bez LLM-a
│   ├── context_metrics.py             # context precision / recall iz označenih izvora
│   ├── judge.py                       # LLM ocenjivač tačnosti po činjenicama (prompt j1 / j2)
│   ├── validate_judge.py              # validacija ocenjivača: ponovljivost, kappa, pristrasnost, McNemar + Holm
│   └── faithfulness.py                # verodostojnost: tvrdnje prema kontekstu + provera brojeva i naziva
├── tests/                             # pytest, bez GPU-a i LLM-a
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
pip install pytest                            # za testove (ako nije u requirements.txt)
```

## API ključevi

U `.env` (nikad u kod, notebook ili commit – repo je javan):

- `GEMINI_API_KEY` – opisi slika (`figures.py`) i razlaganje pitanja (`decompose.py`), besplatni nivo;
- `HF_TOKEN` – preuzimanje modela sa Hugging Face-a.

U Colab-u se `HF_TOKEN` unosi preko **Secrets** (ikonica ključa), ne u ćeliju. Za Gemma 4 je potrebno
jednom prihvatiti uslove korišćenja na stranici modela, istim nalogom.

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

**Generisanje i ocenjivanje** (Google Colab, T4 GPU). Colab klonira repo sa GitHub-a, pa izmene
moraju biti push-ovane pre pokretanja. Svaki notebook ima ćeliju *Podešavanja* i kontrolnu probu na
jednom odgovoru – pokrenuti je i proveriti pre punog pokretanja. Rezultati se posle svakog pokretanja
kopiraju na Google Drive, pa preživljavaju prekid sesije.

| Notebook                   | Šta radi                                                               | Podešavanja                                 |
| -------------------------- | ---------------------------------------------------------------------- | ------------------------------------------- |
| `04_generation.ipynb`      | generisanje sa novom pretragom                                         | `GEN_MODEL`, `RUN_NAME`                     |
| `05_prompt_versions.ipynb` | generisanje na zamrznutom kontekstu (`BASE_RUN`), više verzija prompta | `GEN_MODEL`, `VERSIONS`, `MODEL_TAG`        |
| `06_judge.ipynb`           | ocena tačnosti sačuvanih odgovora                                      | `JUDGE_MODEL`, `JUDGE_VERSION`, `JUDGE_TAG` |
| `07_faithfulness.ipynb`    | ocena verodostojnosti izabranog sistema                                | `JUDGE_MODEL`, `FAITH_VERSION`, `TAG`       |

Za Gemma 4 kao generator ili ocenjivač: `multimodal=True` (`MULTIMODAL = True`).

**Evaluacija** (lokalno, bez GPU-a):

```powershell
python src/evaluate_text.py <folder_pokretanja>       # metrike odgovora bez LLM-a
python src/context_metrics.py                         # kvalitet konteksta (izabrani sistem)

# ocene iz Colab zipa raspoređuje u foldere pokretanja, validira ocenjivača i poredi sisteme
python src/validate_judge.py --tag qwen14b_j2 --import "<raspakovani zip>"
python src/validate_judge.py --tag qwen14b_j2 --compare-tag <drugi_prolaz> `
       --pairs v2b_qwen7b:v2b_gemma4-12b v2b_qwen7b:v2c_qwen7b gen_hybrid_qwen14b:v2b_gemma4-12b

python src/faithfulness.py --tag qwen14b_f1 --details # verodostojnost + provera brojeva i naziva
```

Parovi u `--pairs` mogu se zadati krajem imena foldera. Izveštaji se upisuju u `results_generator/`
(`judge_validation_<tag>.json`, `faithfulness_summary_<tag>.json`, `context_metrics_<pokretanje>.json`).

**Testovi** (pre svakog commit-a):

```powershell
python -m pytest -q tests
```

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

**Generisanje** (`generate.py`, notebook 05) – svaka verzija prompta dodaje jednu grupu izmena:

| Verzija | Šta se menja                                                                                           |
| ------- | ------------------------------------------------------------------------------------------------------- |
| `v1`    | bazni prompt (kontrola ponovljivosti)                                                                   |
| `v2a`   | format: oznaka izvora bez rednog broja bloka, citira se oznaka, stroži JSON                             |
| `v2b`   | + pravila: doslovne vrednosti, memo menja samo ono što imenuje, red i kolona tabele, svaki deo pitanja |
| `v2c`   | + dokazi (evidence) pre odgovora                                                                        |

Modeli: Qwen2.5-7B-Instruct (v1–v2c), Qwen2.5-14B-Instruct (v1), Gemma 4 12B (v1, v2b). Heš svakog
prompta se upisuje u `run_config.json`.

## Metrike

**Pretraga** (`evaluate.py`):

- Hit@k, Recall@k, MRR, nDCG@10 – na nivou izvora (sekcija ili slika);
- pokrivenost svih izvora (all_sources@k) – da li su svi potrebni izvori u top k;
- cover_rank – pozicija poslednjeg potrebnog izvora (koliki k je potreban);
- superseded@5, words@5, latencija; `multi_source.csv` – samo pitanja sa 2+ izvora.

**Kontekst** (`context_metrics.py`, bez LLM-a): context recall, context precision@k i precision po
rangu (average precision), na nivou označenih izvora. Kontekst je u eksperimentima generisanja
zamrznut, pa su vrednosti iste za sve generatore (skript to proverava).

**Odgovori, bez LLM-a** (`evaluate_text.py`):

- tačno odbijanje (abstain recall) i pogrešno odbijanje (false abstain rate);
- izmišljeni brojevi (unsupported numbers) – broj u odgovoru kog nema u datim izvorima;
- preciznost i odziv citata (citation precision / recall);
- poklapanje brojeva sa očekivanim odgovorom (number recall) – grub signal.

**Tačnost odgovora** (`judge.py`, LLM ocenjivač): ocenjivač za svaku činjenicu iz
`expected_facts.json` kaže *present* / *missing* / *contradicted*; ocena se izvodi pravilom u kodu –
netačno ako je ključna činjenica u suprotnosti ili nijedna nije prisutna, tačno ako su sve ključne
prisutne i ništa nije u suprotnosti, inače delimično. Uz to: odziv činjenica i provera doslednosti
bez LLM-a (alarm za verovatne greške ocenjivača).

**Validacija ocenjivača** (`validate_judge.py`): ponovljivost između dva prolaza; Cohen's kappa i
matrice neslaganja prema referenci po istoj rubrici; broj tačnih po modelu prema referenci i greške
po dužini odgovora i kategoriji (pristrasnost); egzaktni McNemar sa Holm korekcijom za poređenje
sistema. Ponavlja se pri svakoj promeni ocenjivača ili činjenica.

**Verodostojnost** (`faithfulness.py`): LLM razloži odgovor na tvrdnje i oceni svaku prema kontekstu
(*supported* / *unsupported* / *contradicted*); nezavisno od LLM-a, brojevi i nazivi iz odgovora koji
se ne javljaju ni u kontekstu ni u pitanju označavaju se kao izmišljeni.

**Relevantnost odgovora** (answer relevancy) nije implementirana: tačnost po činjenicama već meri da li
je odgovor dat na postavljeno pitanje; relevantnost je sledeći korak za upotrebu bez očekivanih odgovora.
