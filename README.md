# NorthStar RAG – retriever nad internom bazom znanja

Projekat sa prakse: retriever (a kasnije i agent) koji pronalazi odgovore u dokumentu
*NorthStar Knowledge Base v3.2*. Dokument sadrži tekst, tabele i slike. Svi podaci u njemu su fiktivni.

## Cilj prve nedelje

- Upoznati dokument i istražiti načine čuvanja različitih tipova sadržaja (tekst, tabele, slike).
- Napraviti osnovnu verziju retrievera i uporediti više pristupa.
- Definisati metrike za automatsku evaluaciju retrievera (Hit Rate@k, Recall@k, MRR, nDCG@k i metrike specifične za dokument).

Predlog stack-a: **Docling** (parsiranje) · **Qdrant** (vektorska baza) · **BGE-M3** (embedding, dense + sparse) · **bge-reranker-v2-m3** (reranker) · Claude/GPT-4o (opisi slika).

## Struktura

```
northstar-rag/
├── data/
│   ├── raw/                  # originalni dokumenti (.docx)
│   └── processed/            # Markdown, izdvojene slike, opisi (generiše se, nije u Git-u)
├── eval/
│   └── eval_questions.json   # kontrolna pitanja sa očekivanim odgovorima i izvorima
├── notebooks/
│   └── 01_explore_document.ipynb
├── src/
│   ├── config.py             # putanje i podešavanja
│   └── parse.py              # parsiranje .docx i izdvajanje slika
├── .env.example              # šablon za API ključeve (kopirati u .env)
├── requirements.txt
└── setup.ps1                 # podešavanje okruženja na Windowsu
```

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

## API ključevi

Ključevi idu samo u `.env` (taj fajl je u `.gitignore` i nikad ne ide na GitHub).
Repozitorijum je javan, pa ključ nikad ne pišite direktno u kod ili notebook.
