# APPROACH.md — Local LLM + RAG Implementation Plan

This document details how to implement the Intelligent Form Agent using
**local, self-hosted LLMs** (no external API calls) combined with a
**RAG (Retrieval-Augmented Generation) pipeline**, mapped directly onto the
assignment's required repository structure and deliverables.

---

## 0. Why local models

- No data leaves the machine — important if forms contain PII/PHI (e.g. the
  hospital patient forms).
- No API cost or rate limits during development/grading.
- Fully reproducible "run end-to-end with minimal setup" — grader doesn't
  need an API key.
- Trade-off: local models are smaller/weaker than frontier APIs, so the
  pipeline is designed to lean on **retrieval + structure** to compensate
  for a smaller model's weaker reasoning (see §4).

---

## 1. Model choices

| Role | Model | Why |
|---|---|---|
| Extraction (form → JSON) | `Llama-3.1-8B-Instruct` or `Mistral-7B-Instruct` via **Ollama**, served locally | Strong instruction-following at a size that runs on a laptop CPU/GPU; supports JSON-mode / grammar-constrained output |
| Embeddings | `nomic-embed-text` (via Ollama) or `sentence-transformers/all-MiniLM-L6-v2` (via `sentence-transformers`, fully local) | Small, fast, good semantic quality, no download needed beyond first run |
| Reranker (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (`sentence-transformers`) | Cheap local reranking to sharpen retrieval before synthesis |
| Synthesis / QA / summarization | Same local LLM as extraction (`Llama-3.1-8B-Instruct` / `Mistral-7B-Instruct`) | One model, multiple prompts — simplest to run and grade |

**Serving choice: Ollama.** It runs any of the above models locally via a
simple REST API (`http://localhost:11434`), handles quantization
automatically (so an 8B model runs on consumer hardware), and needs zero
cloud setup. Alternative: `transformers` + `llama-cpp-python` directly if
Ollama isn't allowed in the environment — the prompts and pipeline logic
below are identical either way, only the client call changes.

```bash
# One-time setup
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull nomic-embed-text
```

---

## 2. Repository structure (per assignment rules)

```
intelligent-form-agent/
├── src/
│   ├── ingestion.py        # PDF/DOCX/scan -> raw text (+ OCR fallback)
│   ├── llm_client.py        # thin wrapper around Ollama's REST API
│   ├── extraction.py        # LLM schema-guided extraction -> structured JSON
│   ├── schemas.py            # Pydantic schemas per form template
│   ├── chunking.py           # splits free text into retrieval-sized chunks
│   ├── vectorstore.py        # embedding + Chroma (local, on-disk) wrapper
│   ├── structured_store.py   # SQLite wrapper for exact filters/counts
│   ├── router.py              # classifies a question: aggregate vs semantic vs single/multi-form
│   ├── retrieval.py           # hybrid search (BM25 + embeddings) + reranking
│   ├── synthesis.py           # builds the grounded prompt, calls LLM, parses citations
│   └── agent.py                # FormAgent — orchestrates all of the above
├── data/
│   └── sample_forms/           # mock membership + hospital forms (PDF)
├── notebooks/
│   ├── 01_extraction_eval.ipynb    # test extraction accuracy on sample forms
│   ├── 02_retrieval_eval.ipynb     # compare embedding-only vs hybrid retrieval
│   └── 03_prompt_iteration.ipynb   # iterate on synthesis prompt + citation format
├── tests/
│   ├── test_extraction.py
│   ├── test_router.py
│   ├── test_retrieval.py
│   └── test_agent_e2e.py
├── docs/
│   ├── architecture.md          # diagram + design rationale (this pairs with APPROACH.md)
│   └── demo_run_log.txt          # captured output of the 3 required demo runs
├── chroma_db/                    # local vector store data (gitignored)
├── forms.sqlite                  # local structured store (gitignored)
├── requirements.txt
└── README.md
```

This satisfies the assignment's required folders (`/src`, `/data`,
`/notebooks`, `/tests`, `/docs`) while keeping local-model-specific pieces
(`llm_client.py`, `vectorstore.py`, `structured_store.py`) clearly separated
so each is independently testable.

---

## 3. Pipeline stage by stage

### 3.1 Ingestion (`src/ingestion.py`)
- `.pdf` → `pdfplumber` (text-based) with a `pytesseract` + `pdf2image` OCR
  fallback for scanned/handwritten forms (detect scanned pages by checking
  if `pdfplumber` returns near-empty text).
- `.docx` → `python-docx`.
- Output: one `RawForm(form_id, source_path, text, page_images?)` per file.

### 3.2 Schema-guided LLM extraction (`src/extraction.py`, `src/schemas.py`)
- Define a `Pydantic` schema per known form type, e.g.:

```python
class MembershipForm(BaseModel):
    name: str
    date_of_birth: str
    email: str
    phone: str
    occupation: str
    application_date: str
    status: Literal["Approved", "Pending", "Rejected"]
    remarks: str

class HospitalForm(BaseModel):
    patient_name: str
    patient_id: str
    date_of_birth: str
    doctor: str
    department: str
    admission_date: str
    discharge_date: str | None
    diagnosis: str
    discharge_status: str
    doctors_notes: str
```

- **Form-type detection**: one cheap LLM call — "Given this text, is this a
  membership application or a hospital admission form? Respond with one
  word." (or a simple keyword heuristic as a fast-path before falling back
  to the LLM call).
- **Extraction call**: prompt the local LLM with the schema (as a JSON
  example) + the raw form text, instruct it to output *only* valid JSON
  matching the schema. Use Ollama's `format: "json"` mode (or a grammar via
  `llama-cpp-python`'s GBNF) to force valid JSON — this removes most
  parsing failures.
- Validate the LLM's JSON against the Pydantic schema; on validation
  failure, retry once with the validation error appended to the prompt
  ("your last output failed validation because X — fix it").
- Store validated fields in `structured_store.py` (SQLite) and the free-text
  field (`remarks` / `doctors_notes`) in the chunking/embedding pipeline.

### 3.3 Chunking + embedding (`src/chunking.py`, `src/vectorstore.py`)
- Free text is usually short per form (one remarks paragraph), so chunk at
  the paragraph/sentence-group level (~150–300 tokens) rather than
  arbitrary fixed windows — avoids splitting a single claim across chunks.
- Embed each chunk with `nomic-embed-text` (via Ollama) or
  `all-MiniLM-L6-v2` (via `sentence-transformers`) and store in **Chroma**
  (a local, on-disk, embedded vector DB — no server to run) with metadata:
  `{form_id, form_type, status, department/occupation, date}`.

### 3.4 Structured store (`src/structured_store.py`)
- SQLite table `forms(form_id, form_type, field_name, field_value)` (long
  format) or one table per form type (wide format) — wide format is
  simpler to query and is fine since form types are known in advance.
- This is what makes "how many applicants are approved?" **exact** instead
  of a model guess.

### 3.5 Query router (`src/router.py`)
- A single, cheap local LLM call classifies the incoming question into:
  - `aggregate` — counting/filtering ("how many...", "which forms have...")
  - `single_form_lookup` — a known-field question about one form
  - `single_form_semantic` — an open-ended question about one form's free text
  - `multi_form_semantic` — an open-ended question across many forms
- Prompt: give the model the question + a one-line description of each
  category + ask for just the category label. This is far cheaper and more
  reliable than trying to make one giant prompt do everything.
- Fallback: if the LLM call fails/times out, fall back to the keyword-based
  routing already built in the rule-based version (`FIELD_ALIASES`,
  "how many" detection) — keeps the system degrade-gracefully.

### 3.6 Retrieval (`src/retrieval.py`)
- **Aggregate** path → build a SQL `WHERE` filter from the question
  (either a small text-to-SQL LLM call, or map recognized keywords to
  column filters for the known schema — safer and faster since the schema
  is fixed and small) → run against SQLite → exact count/list.
- **Semantic** path → hybrid search:
  1. BM25 (keyword) search over chunk text (e.g. `rank_bm25` — catches
     exact terms like "surgical" that pure embeddings might miss)
  2. Embedding similarity search via Chroma
  3. Merge + dedupe candidates, optionally rerank with the cross-encoder
  4. Return top-k chunks with their `form_id` metadata
- For `single_form_semantic`, filter the vector search to that one
  `form_id` before searching (Chroma supports metadata filters natively).

### 3.7 Synthesis (`src/synthesis.py`)
- Build a prompt containing: the question, the retrieved chunks (each
  labeled with its `form_id`), and an instruction to **only answer from the
  provided chunks and cite the form_id(s) used**; if the chunks don't
  contain the answer, say so rather than guessing.
- Parse the LLM's response for citation markers (e.g. `[form_004]`) and
  attach them to the returned result so the UI can show "answer, backed by
  these forms" — this is the grounding mechanism that prevents silent
  hallucination.
- For **summarization**, the same synthesis step is used with a
  summarization-style prompt over one form's full content (structured
  fields + free text) instead of a question.
- For **multi-form summaries**, use map-reduce: summarize each relevant
  form individually first (cheap, parallelizable), then a final LLM call
  combines those per-form summaries into one holistic narrative.

### 3.8 Orchestration (`src/agent.py`)
`FormAgent` exposes the same public API as before so the Streamlit UI and
tests don't need to change:

```python
agent = FormAgent()
agent.load_directory("data/sample_forms")

agent.answer_question(form_id, question)   # -> routes internally
agent.summarize_form(form_id)
agent.multi_form_query(question)
```

Internally it now calls `router → (structured_store | retrieval) → synthesis`
instead of the old regex + TF-IDF fallback logic. Keep the old fallback
functions in `models.py` as a **secondary fallback** if Ollama isn't
running — this is a good "system resilience" feature to highlight in the
README's creativity section.

---

## 4. Compensating for a smaller local model

Local 7–8B models are meaningfully weaker than frontier APIs at open-ended
reasoning, especially arithmetic. The pipeline is designed around this:

- **Never ask the LLM to count.** Counting always goes through SQL.
- **Never ask the LLM to extract without a schema.** Constrained JSON
  generation removes most of the failure surface.
- **Keep synthesis prompts short and grounded** (retrieved chunks only, not
  the whole corpus) — smaller models degrade faster with long, noisy
  context than large ones do.
- **Retry-on-validation-failure** for extraction, rather than accepting
  malformed output silently.

---

## 5. requirements.txt additions (on top of the rule-based version)

```
# Local LLM serving
ollama                     # python client for the Ollama REST API

# Structured extraction
pydantic>=2.0

# Vector store + retrieval
chromadb>=0.5
rank-bm25>=0.2
sentence-transformers>=3.0   # embeddings + reranker fallback if not using Ollama

# OCR (scanned forms)
pytesseract>=0.3
pdf2image>=1.17
```

(Ollama itself is installed separately as a system binary —
https://ollama.com — then models are pulled via `ollama pull <model>`.)

---

## 6. Build order (maps to the required demo runs)

1. Ingestion + form-type detection on the existing mock forms (membership +
   hospital).
2. Schema-guided extraction with validation/retry — verify against
   `data/sample_forms` in `notebooks/01_extraction_eval.ipynb`.
3. Structured store (SQLite) + aggregate query path — this alone satisfies
   "how many applicants are approved?"-style holistic questions precisely.
4. Chunking + embedding + Chroma — get semantic single-form QA working.
5. Router — wire question → path selection.
6. Hybrid retrieval + reranking — improve multi-form semantic quality
   (`notebooks/02_retrieval_eval.ipynb`).
7. Synthesis with citations — finalize grounded answer generation
   (`notebooks/03_prompt_iteration.ipynb`).
8. Wire everything into `FormAgent`, keep the old rule-based path as
   fallback.
9. Re-run and capture the three required demos (single-form Q&A, summary,
   multi-form holistic) into `docs/demo_run_log.txt`.
10. Update README: setup steps now include installing Ollama + pulling
    models; document the fallback behavior and the citation format as
    creative extensions.

---

## 7. Testing strategy (`/tests`)

- `test_extraction.py` — run schema-guided extraction against the known
  mock forms and assert exact field matches (this is deterministic enough
  to test precisely, since the mock forms are fixed).
- `test_router.py` — assert a fixed set of questions route to the expected
  category (aggregate vs semantic, single vs multi-form).
- `test_retrieval.py` — assert known queries return the expected `form_id`
  in the top-k results (e.g. "credit defaults" → `form_004`).
- `test_agent_e2e.py` — the same three required demo scenarios as
  integration tests, asserting the final answer contains the expected
  fact and at least one citation.
- Mark LLM-dependent tests with a `@pytest.mark.requires_ollama` decorator
  and skip them gracefully if Ollama isn't running, so `pytest tests/` still
  passes in a minimal-setup grading environment (falls back to the
  rule-based path where relevant).

---

## 8. What to highlight in the README's "Creativity Extensions" section

- Fully local RAG pipeline — no external API calls, works offline after
  initial model download.
- Schema-guided structured extraction instead of regex, generalizes to new
  form templates by adding a Pydantic schema rather than new regex rules.
- Hybrid retrieval (BM25 + embeddings) with reranking.
- Grounded synthesis with explicit per-form citations, reducing
  hallucination risk.
- Graceful multi-tier fallback: LLM router → keyword router; LLM QA → TF-IDF
  extractive QA — the system degrades rather than failing if Ollama isn't
  available.
- Exact aggregate answers via SQL rather than LLM counting.