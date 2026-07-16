# Architecture & Design Notes

## Pipeline Overview

```
                    ┌─────────────────┐
  PDF / DOCX file → │   ingestion.py   │  pdfplumber / pytesseract OCR / python-docx
                    └────────┬────────┘
                             │ RawForm (form_id, raw text)
                             ▼
                    ┌─────────────────┐
                    │  extraction.py   │  form-type detection → schema-guided LLM extraction
                    └────────┬────────┘  Pydantic validation + 1 retry on failure
                             │ ExtractedForm (form_id, form_type, validated data)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌─────────────────────┐      ┌─────────────────────┐
   │  structured_store.py │      │    chunking.py       │
   │  SQLite (forms.sqlite)│      │  sentence grouping   │
   │  ALL structured fields│      │  free-text field only│
   └─────────────────────┘      └──────────┬──────────┘
    exact lookup / SQL counts               │ TextChunk list
                                            ▼
                                  ┌─────────────────────┐
                                  │   vectorstore.py     │
                                  │  Ollama embed() →    │
                                  │  ChromaDB upsert     │
                                  └─────────────────────┘
                                   semantic / meaning search

                    ┌─────────────────┐
  User question  →  │    router.py     │  keyword fast-path → LLM fallback
                    └────────┬────────┘
                             │ RouteDecision (route, form_id, filters)
                             ▼
                    ┌─────────────────┐
                    │  retrieval.py    │  dispatches to SQLite or ChromaDB (or both)
                    └────────┬────────┘  BM25 + embedding RRF for multi-form
                             │ RetrievalResult (rows | form | chunks)
                             ▼
                    ┌─────────────────┐
                    │  synthesis.py    │  builds grounded prompt → Ollama generate()
                    └────────┬────────┘  extracts [form_id] citations from response
                             │ SynthesisResult (answer, cited_form_ids, grounded)
                             ▼
                         User sees answer
```

---

## Key Design Decisions

### 1. Two-store architecture

**Why SQLite AND ChromaDB?**

A pure vector store would hallucinate counts ("about 3 approved forms") and miss exact lookups. A pure SQL store can't answer "what patterns appear across all rejected applicants?" 

The router classifies each question and sends it to the right store:
- Aggregate / exact → SQLite (always precise)
- Open-ended / semantic → ChromaDB (always meaningful)

### 2. Keyword routing first, LLM second

The router tries keyword matching before calling the LLM. Phrase patterns like "how many", "what is the", "list all" cover ~90% of real questions deterministically. The LLM fallback exists for edge cases.

This means routing works even if Ollama is offline, and it's ~100× faster than an LLM call.

### 3. Schema-guided extraction with retry

The LLM is shown a target JSON shape and asked to fill in values from the form text. If the output fails Pydantic validation (wrong type, missing field), the validation error is appended to the prompt and the LLM gets one more attempt. In practice this handles most "almost-right" responses from smaller models.

### 4. Chunking only free-text fields

Only `remarks` (membership) and `doctors_notes` (hospital) get chunked and embedded. Structured fields like `name`, `status`, `department` are already exact — embedding them would waste tokens and add noise to semantic search results.

### 5. Reciprocal Rank Fusion for multi-form search

BM25 and embedding cosine similarity produce incomparable scores. RRF merges the ranked lists by position only (score = Σ 1/(60 + rank)), so no calibration is needed. The top-k results across both methods are always returned.

### 6. Grounding and citation

The synthesis prompt explicitly instructs the LLM to:
- Answer ONLY from the provided context
- Tag every claim with `[form_id]`
- Respond with `NOT_FOUND_IN_CONTEXT` if the answer isn't there

This prevents hallucination and gives users traceability back to the source form.

---

## File-to-stage mapping

| File | Pipeline Stage | Depends on |
|---|---|---|
| `ingestion.py` | Stage 1: raw text | pdfplumber, pytesseract, docx |
| `schemas.py` | Schema registry | pydantic |
| `extraction.py` | Stage 2: structured JSON | llm_client, schemas |
| `chunking.py` | Stage 3a: text chunks | schemas |
| `structured_store.py` | Stage 3b: SQLite | extraction |
| `vectorstore.py` | Stage 3c: ChromaDB | llm_client (embed), chunking |
| `router.py` | Stage 4: route classification | llm_client (fallback only) |
| `retrieval.py` | Stage 5: context fetch | structured_store, vectorstore, router |
| `synthesis.py` | Stage 6: answer generation | llm_client, retrieval |
| `agent.py` | Orchestration | all of the above |
| `app.py` | UI | agent |

---

