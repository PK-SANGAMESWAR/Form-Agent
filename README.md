# 📋 Intelligent Form Agent

A **fully local** AI system that reads PDF/DOCX forms, extracts every field automatically,
and answers plain-English questions — no cloud, no API key, everything runs on your machine.

---

## What It Does

| Capability | Description |
|---|---|
| **Extraction** | Pulls structured fields (name, date, status) and free-text notes from any PDF or DOCX form |
| **Single-form Q&A** | Answers questions about one specific form with grounded citations |
| **Multi-form insights** | Aggregates and reasons across all loaded forms simultaneously |
| **Summarization** | Generates a concise 3–5 sentence summary of any form |
| **Exact counts** | Counts/filters forms via SQL — never asks the LLM to count |
| **Anti-hallucination** | Every answer cites the form_id it came from; if the answer isn't in the data, it says so |

---

## Architecture (Quick Overview)

```
Form (PDF/DOCX)
    │
    ▼
[Ingestion]  ─── pdfplumber / pytesseract OCR / python-docx
    │
    ▼
[Extraction] ─── LLM (Ollama) extracts fields → Pydantic validation → retry on failure
    │
    ├──────────────────────────┐
    ▼                          ▼
[SQLite]               [ChromaDB + BM25]
 Exact fields           Vector embeddings of free-text chunks
    │                          │
    └──────────┬───────────────┘
               ▼
           [Router]  ─── keyword fast-path → LLM fallback
               ▼
          [Retrieval] ─── SQL (aggregate) or Hybrid BM25+Embeddings (semantic)
               ▼
          [Synthesis] ─── Grounded LLM answer with [form_id] citations
```

Full design notes: [`APPROACH.MD`](APPROACH.MD) | Architecture diagram: [`ARCHI.MD`](ARCHI.MD)

---

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running
- `uv` package manager (or standard `pip`)

---

## Setup

### 1. Install Ollama and pull models

```bash
# Install Ollama from https://ollama.com then:
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve          # keep this running in a terminal
```

### 2. Install Python dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Generate sample forms (optional — pre-built PDFs already in data/)

```bash
uv run python data/sample_forms/_generate_samples.py
```

---

## Running the App

```bash
uv run streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage Guide

### Step 1 — Upload / Ingest Forms
- Go to the **Upload Forms** tab
- Drop a `.pdf` or `.docx` file and click **Ingest**
- The system extracts all fields, stores them in SQLite, and embeds free-text chunks into ChromaDB

### Step 2 — Single Form Q&A
- Go to **Form Explorer**
- Select a form from the dropdown
- Ask any question or click **Generate Summary**

### Step 3 — Multi-Form Insights
- Go to **Ask the Agent**
- Ask questions that span all forms, e.g.:
  - *"How many membership applications are approved?"*
  - *"Which hospital patients were in Cardiology?"*
  - *"What patterns show up across all forms?"*

---

## Example Queries & Expected Outputs

### Single-Form Q&A
```
Question: What is the status of this application?
Answer:   The application status is "Approved" [membership_001].
```

```
Question: What did the doctor note about the patient?
Answer:   The doctor noted that the patient underwent angioplasty ... [hospital_001].
```

### Form Summary
```
Question: Summarize this form.
Answer:   membership_001 is an approved membership application submitted on 02-01-2026
          by Ananya Rao (Software Engineer). The applicant has a strong credit history
          with no prior defaults, and was recommended for the premium membership tier. [membership_001]
```

### Multi-Form / Holistic Query
```
Question: How many membership applications are approved?
Answer:   2 matching form(s): membership_001, membership_003.
```

```
Question: Which applicants have prior credit defaults?
Answer:   Based on the remarks, membership_002 shows two prior credit defaults on file [membership_002].
```

---

## Running Tests

```bash
uv run pytest tests/ -v
# 73 tests, no Ollama required — all pass in ~4 seconds
```

Tests cover:
- `test_schemas.py` — Pydantic schema validation
- `test_chunking.py` — sentence-group chunking logic
- `test_extraction.py` — form type detection + LLM extraction + retry logic
- `test_router.py` — question routing (keyword + LLM fallback)
- `test_retrieval.py` — aggregate, lookup, semantic, and hybrid retrieval paths
- `test_structured_store.py` — SQLite insert, query, count, filter

---

## CLI Demo (3 Required Runs)

Run all three demo scenarios (single-form Q&A, summary, multi-form insight) and capture output:

```bash
uv run python src/agent.py > docs/demo_run_log.txt
```

See [`docs/demo_run_log.txt`](docs/demo_run_log.txt) for captured output.

---

## Project Structure

```
intelligent-form-agent/
├── src/
│   ├── ingestion.py        # PDF/DOCX/OCR → RawForm
│   ├── llm_client.py       # Ollama REST API wrapper (one place for all LLM calls)
│   ├── extraction.py       # LLM schema-guided extraction → validated JSON
│   ├── schemas.py          # Pydantic schemas per form type (add new types here)
│   ├── chunking.py         # Free-text → sentence-group chunks for embedding
│   ├── vectorstore.py      # Chroma embeddings + similarity search
│   ├── structured_store.py # SQLite for exact fields + aggregate queries
│   ├── router.py           # Classifies question → route (aggregate/lookup/semantic)
│   ├── retrieval.py        # BM25 + embedding hybrid retrieval with RRF merge
│   ├── synthesis.py        # Grounded LLM answer generation with citations
│   └── agent.py            # FormAgent — orchestrates the full pipeline
├── data/
│   └── sample_forms/       # Membership + hospital PDFs (text + scanned)
├── notebooks/              # Exploratory notebooks
├── tests/                  # 73 unit tests (no Ollama needed)
├── docs/
│   ├── architecture.md     # Detailed architecture notes
│   └── demo_run_log.txt    # Captured 3 required demo runs
├── chroma_db/              # Local vector store (auto-created, gitignored)
├── forms.sqlite            # Structured store (auto-created, gitignored)
├── app.py                  # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Creative Extensions

| Extension | Where |
|---|---|
| **Fully local RAG** — no external API, works offline | `src/llm_client.py` + Ollama |
| **Schema-guided extraction** — new form types = new Pydantic schema, nothing else changes | `src/schemas.py` |
| **Hybrid retrieval** — BM25 + embeddings merged with Reciprocal Rank Fusion | `src/retrieval.py` |
| **Grounded synthesis with citations** — every answer tags the form_id(s) used | `src/synthesis.py` |
| **Exact aggregate answers via SQL** — never asks LLM to count | `src/structured_store.py` + `src/router.py` |
| **OCR fallback** — scanned PDFs handled via pytesseract | `src/ingestion.py` |
| **Graceful multi-tier fallback** — keyword router → LLM router; system degrades, never crashes | `src/router.py`, `src/extraction.py` |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not reach Ollama` | Run `ollama serve` in a separate terminal |
| `model not found` | Run `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text` |
| Slow first embedding | Normal — Ollama loads the model into VRAM on first call; retry is built-in |
| `ExtractionError` on a form | The form may be scanned; ensure `pytesseract` + Tesseract binary are installed |
