# Form-Agent Code Review

---

## Problem Statement vs. What Was Built

### What the Problem Requires (PROBLEM.MD §2)

| Capability | Required | Status |
|---|---|---|
| Extract structured + free-text fields | ✅ Yes | ✅ Built (extraction.py, schemas.py) |
| Answer questions about a **single form** | ✅ Yes | ✅ Built (router → retrieval → synthesis) |
| **Holistic multi-form insights** | ✅ Yes | ✅ Built (multi_form_semantic + aggregate SQL) |
| **Summarize** a form | ✅ Yes | ✅ Built (synthesis.summarize_form) |
| `/src`, `/data`, `/notebooks`, `/tests`, `/docs` folders | ✅ Required (§5) | ❌ `/tests`, `/notebooks`, `/docs` all missing |
| `requirements.txt` with all deps | ✅ Required (§6) | ⚠️ Incomplete (missing `requests`, wrong `docx`) |
| `README.md` with setup + run steps + examples | ✅ Required (§6) | ❌ README is **empty** |
| 3 demo runs (single-form QA, summary, multi-form) | ✅ Required (§7) | ❌ None captured anywhere |
| Creative extensions documented separately | Optional (§3) | ⚠️ In APPROACH.MD, not README |

**The core pipeline logic is well-built. The submission packaging (README, tests, notebooks, docs, demo runs) is almost entirely missing — which directly hits grading criteria §10.**

---

## Summary

The APPROACH.MD describes a solid, well-structured RAG pipeline. The code
**mostly follows the plan**, but has several **critical runtime bugs** and a handful
of design/completeness gaps that would cause failures during grading/demo.

---

## 🔴 Critical Bugs (will break at runtime)

### 1. `agent.py` — Wrong call signature to `extraction.extract_form()`

**File**: [`agent.py` L71](file:///c:/Users/LOQ/Downloads/Form-Agent/src/agent.py#L71-L71)

```python
# agent.py calls it like this:
extracted = extraction.extract_form(raw.form_id, raw.text)
```

But `extract_form()` in [extraction.py](file:///c:/Users/LOQ/Downloads/Form-Agent/src/extraction.py#L147-L157) is defined as:

```python
def extract_form(raw_form: RawForm) -> ExtractedForm:
```

It takes a single `RawForm` object, NOT `(form_id, raw_text)`.  
**Effect**: `TypeError` the moment `load_directory()` or `load_form()` is called — the entire pipeline is broken at startup.

**Fix**: Change to `extraction.extract_form(raw)` (agent.py L71 and L86).

---

### 2. `agent.py` — `ingestion.load_directory()` vs `ingestion.ingest_directory()`

**File**: [`agent.py` L69](file:///c:/Users/LOQ/Downloads/Form-Agent/src/agent.py#L69-L69)

```python
for raw in ingestion.load_directory(dir_path):   # ❌ doesn't exist
```

The function in [ingestion.py](file:///c:/Users/LOQ/Downloads/Form-Agent/src/ingestion.py#L142-L150) is named `ingest_directory()`, not `load_directory()`.  
Same mismatch on L85: `ingestion.load_form()` vs actual function name `ingest_file()`.

**Effect**: `AttributeError` on both `load_directory` and `load_form` calls.

---

### 3. `schemas.py` — `date` type causes Pydantic validation failure with "DD-MM-YYYY" strings

**File**: [`schemas.py` L25, L29, L37, L40-L41](file:///c:/Users/LOQ/Downloads/Form-Agent/src/schemas.py#L23-L44)

```python
date_of_birth: date = Field(description="DD-MM-YYYY as written on the form")
```

Pydantic v2's `date` type expects **ISO 8601** (`YYYY-MM-DD`). But the schema_example
in the same file ([schemas.py L71-L91](file:///c:/Users/LOQ/Downloads/Form-Agent/src/schemas.py#L69-L95)) uses `"DD-MM-YYYY"` strings like `"14-03-1994"`.

The LLM will produce `"14-03-1994"` (matching the example), Pydantic will reject it,
the retry will send the same example back, and extraction will fail every time for date fields.

**Effect**: All forms fail to extract; `ExtractionError` after 2 attempts.

**Fix**: Either use `str` type for date fields (simplest), or use `date` + a custom validator
that accepts `DD-MM-YYYY` format, AND update the schema_example to use ISO format.

---

### 4. `structured_store.py` — `model_dump()` returns `date` objects, but SQLite expects strings

**File**: [`structured_store.py` L77](file:///c:/Users/LOQ/Downloads/Form-Agent/src/structured_store.py#L77)

```python
data = extracted.data.model_dump()
```

If the Pydantic schema fields are `date` type (schemas.py), `model_dump()` returns Python
`datetime.date` objects. Passing these into SQLite `?` parameters causes a
`sqlite3.InterfaceError` — it doesn't know how to serialize them.

**Effect**: Runtime crash in `insert_form()` whenever date fields are present.

**Fix**: Either use `str` fields in the schema (see bug #3), or call
`model_dump(mode="json")` which serializes dates to ISO strings.

---

### 5. `retrieval.py` smoke test — `collection=None` passed to semantic route

**File**: [`retrieval.py` L223](file:///c:/Users/LOQ/Downloads/Form-Agent/src/retrieval.py#L222-L224)

```python
result = retrieve(decision, "...", conn, collection=None)
```

This is only in the `__main__` smoke test block, and the aggregate path happens not to use
`collection`, so it doesn't crash there. But it's a fragile assumption; if route detection
ever returns a semantic route for that question, it would crash with `AttributeError` on
`NoneType`. **Low risk in prod, but the smoke test is misleading.**

---

### 6. `pyproject.toml` — missing `requests` and `sentence-transformers`, wrong `docx` package name

**File**: [`pyproject.toml` L7-L17](file:///c:/Users/LOQ/Downloads/Form-Agent/pyproject.toml)

- `requests` is not listed — but `llm_client.py` imports it directly.
- `docx>=0.2.4` in pyproject.toml is NOT `python-docx`. The correct pip package is `python-docx`.
  `docx` is a different, old package. `ingestion.py` does `import docx` which works with
  `python-docx` but NOT with the `docx` package listed.
- `requirements.txt` also lists `docx` (same problem) and is missing `requests`, `ollama`.

**Effect**: `pip install -r requirements.txt` installs the wrong package; `import docx` in
ingestion.py may still work due to `python-docx` being the correct package, but this is confusing
and fragile. `llm_client.py` will fail with `ModuleNotFoundError: requests` on a clean install.

---

## 🟡 Design / Completeness Issues

### 7. No `/tests` directory exists

APPROACH.MD §7 describes 4 test files:
- `test_extraction.py`
- `test_router.py`
- `test_retrieval.py`
- `test_agent_e2e.py`

**None of them exist.** The `/tests` directory doesn't exist at all.
This is required by the assignment.

---

### 8. No `/notebooks` directory

APPROACH.MD §2 / §6 requires:
- `01_extraction_eval.ipynb`
- `02_retrieval_eval.ipynb`
- `03_prompt_iteration.ipynb`

**None exist.** This is required.

---

### 9. No `/docs` directory (or `architecture.md` / `demo_run_log.txt`)

**None exist** in the Form-Agent folder.

---

### 10. `main.py` is a placeholder

```python
def main():
    print("Hello from form-agent!")
```

The plan says this should either hook into `FormAgent` or at least demonstrate the pipeline.
Not blocking but shows the project is incomplete.

---

### 11. `PROBLEM.MD` is empty (0 bytes)

This file should contain the assignment problem statement. It is completely empty.

---

### 12. `FIELD_ALIASES` in `router.py` is too rigid

**File**: [`router.py` L48-L58](file:///c:/Users/LOQ/Downloads/Form-Agent/src/router.py#L48-L58)

```python
"discharged": ("discharge_status", "Stable, discharged with medication"),
```

The alias hardcodes the **exact value** of `discharge_status`. If a hospital form has
`"Discharged"` or `"Discharged - stable"`, it won't match and the filter will silently
return 0 results. A partial match or `LIKE` query would be more robust.

---

### 13. `ingestion.py` — OCR fallback re-OCRs the ENTIRE doc even for a single blank page

**File**: [`ingestion.py` L94-L100](file:///c:/Users/LOQ/Downloads/Form-Agent/src/ingestion.py#L94-L100)

```python
if any(t is None for t in page_texts):
    ocr_full_text = _ocr_pdf(path)   # OCRs ALL pages
    final_text = ocr_full_text
```

A mixed PDF (e.g. page 1 text, page 2 scanned) throws away the clean text from page 1
and OCRs everything again, potentially degrading quality on the text-extractable pages.
The code has a comment acknowledging this, so the intern is aware — but it's still a gap.

---

### 14. `synthesis.py` — `CITATION_RE` pattern too strict

**File**: [`synthesis.py` L33](file:///c:/Users/LOQ/Downloads/Form-Agent/src/synthesis.py#L33)

```python
CITATION_RE = re.compile(r"\[([a-zA-Z]+_\d{3,})\]")
```

This requires **3+ digits** after the underscore (`\d{3,}`). A form_id like
`membership_01` or `hospital_1` wouldn't be captured, silently dropping citations.
The sample forms use `_001` format so it works for now, but it's fragile.

---

### 15. `vectorstore.py` — no handling when Chroma collection is empty and `where` is set

**File**: [`vectorstore.py` L145](file:///c:/Users/LOQ/Downloads/Form-Agent/src/vectorstore.py#L145)

```python
results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)
```

Chroma raises an exception if `where` is a non-None filter **AND** the collection has
zero chunks (or zero chunks matching the filter). This would crash with a Chroma
`InvalidCollectionException` on a fresh/empty store or a `form_id` that hasn't been
ingested yet. Should be wrapped in a try/except returning `[]`.

---

## 🟢 What's Done Well

- **Architecture clarity**: Every module is well-docstringed, design decisions are
  referenced back to APPROACH.MD sections. Very readable.
- **Keyword-first, LLM-fallback** pattern in both `router.py` and `extraction.py` — correct approach.
- **RRF for hybrid merge** (`retrieval.py`) — non-trivial, done correctly.
- **`_sanitize_metadata`** in `vectorstore.py` — handles `None` values before Chroma insertion.
- **Retry-on-validation-failure** logic in `extraction.py` — exactly as designed.
- **`ingest_directory` skips `_`-prefixed files** — smart, prevents `_generate_samples.py` from being ingested.

---

## Priority Fix Order

| Priority | Issue | Impact |
|---|---|---|
| 🔴 | #1 `extract_form` wrong signature in agent.py | Total failure |
| 🔴 | #2 `load_directory` / `load_form` name mismatch | Total failure |
| 🔴 | #3 `date` type vs DD-MM-YYYY strings in schemas | All extractions fail |
| 🔴 | #4 `model_dump()` returns date objects → SQLite crash | All inserts fail |
| 🔴 | #6 Missing `requests` dep / wrong `docx` package | Won't install/import |
| 🟡 | #7 No `/tests` directory | Missing deliverable |
| 🟡 | #8 No `/notebooks` directory | Missing deliverable |
| 🟡 | #9 No `/docs` directory | Missing deliverable |
| 🟡 | #15 Empty Chroma collection crash | Edge case crash |
| 🟢 | #12 FIELD_ALIASES too rigid | Functional but fragile |
| 🟢 | #13 Full-doc re-OCR on partial scan | Quality degradation |
| 🟢 | #14 Citation regex too strict | Silent citation drops |
