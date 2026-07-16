## This is how i approached the architecture and working

1) Generated pdfs and forms using (data/sample_forms/_generate_samples.py) and stored it in (data/_generate_samples)

2) Ingestion pipeline @ src/ingestion.py

Using PDFPLUMBER to parse the document , pytessaract for OCR, python-docx for DOCX

3) src/llm_client.py - isolate all communication with LLM in this file

4) schemas.py @src/schemas.py - Takes in document -> OCR/Text extraction -> LLM extracts fields -> schemas.py (validate the extracted JSON) -> Structured data

5) src/extraction.py - takes in raw OCR text from a doc and convert to validated, structured obj 

6) src/chunking.py - This comes after we get the extracted form , so suppose we get doctors notes and if we embed all as one big block - the retrivel bcomes less precise - chunking is done

7) src/structured_store.py - This is the one side of extracted form, db layer of the project, it stores all the exact fields from forms in SQLite and lets us answer factual questions using SQL instead of LLM.

We need this because - "How many approved appl are there" - LLM has to go through all docs and count instead we do SQL querying 

8) src/vector_store.py - semantic search db layer, stores embeddings so you can search by meaning

9) src/router.py - It is used to answer questions, it is used to decide the component to answer ie structured_store, vectorstore or both

    Keyword fast-path first: checks for "how many", "count", "list all" → routes to aggregate (SQL)
    If a form_id is in the question + lookup keyword → single_form_lookup
    If a form_id is present but open-ended → single_form_semantic
    If nothing matches → LLM classification fallback → usually resolves to multi_form_semantic

10) src/retrieval.py - Executes the route decided by router.py against the right backend

    aggregate        → structured_store.list_forms() → exact SQL count/filter, no LLM
    single_form_lookup  → structured_store.get_form() → returns the full field dict
    single_form_semantic → vectorstore.similarity_search(where={"form_id": "..."}) → scoped Chroma search
    multi_form_semantic  → HYBRID: BM25 keyword search + Chroma embedding search, merged using
                           Reciprocal Rank Fusion (RRF) — BM25 catches exact terms embeddings miss,
                           RRF merges both ranked lists without needing score calibration

11) src/synthesis.py - Takes the RetrievalResult and builds the final grounded answer

    aggregate results → format_aggregate() — no LLM call, count is already exact from SQL
    semantic/lookup   → answer_question() — builds a prompt with retrieved chunks labeled by form_id,
                        instructs LLM to answer ONLY from context and tag every claim with [form_id]
    summarization     → summarize_form() — full structured fields + free text → 3-5 sentence summary
    NOT_FOUND_IN_CONTEXT marker: if the LLM can't answer from context it says so, no hallucination

12) src/agent.py - FormAgent, the orchestrator — the ONLY public interface for the UI and tests

    load_directory(dir_path)         → ingests all PDFs/DOCX in a folder end-to-end
    load_form(path)                  → single-file version for UI upload-and-ingest
    answer_question(question, form_id) → router → retrieval → synthesis (full pipeline)
    summarize_form(form_id)          → structured_store.get_form + synthesis.summarize_form
    multi_form_query(question)       → forces multi_form_semantic route, bypasses router

13) app.py - Streamlit UI (optional creative extension)

    Tab 1: Upload Forms  — drag-and-drop PDF/DOCX, triggers agent.load_form()
    Tab 2: Form Explorer — dropdown of all loaded form_ids, Q&A + summary per form
    Tab 3: Ask the Agent — multi-form question box, uses agent.answer_question()

14) Tests (tests/) — 73 unit tests, all pass without Ollama running (~4 seconds)

    test_schemas.py         — Pydantic validation (valid + invalid inputs)
    test_chunking.py        — sentence-group chunking, max_words, metadata
    test_extraction.py      — detect_form_type keyword/LLM, extract_fields retry logic
    test_router.py          — aggregate/lookup/semantic routing, keyword + LLM fallback
    test_retrieval.py       — all four route paths (aggregate, lookup, single/multi semantic)
    test_structured_store.py — init_db, insert, count, filter, overwrite

    Run with: uv run pytest tests/ -v

15) Demo runs (docs/demo_run_log.txt) — 3 required scenarios captured with Ollama running

    Demo 1: Single-form Q&A   → agent.answer_question("What is the status?", form_id="membership_001")
    Demo 2: Summary           → agent.summarize_form("membership_001")
    Demo 3: Multi-form insight → agent.multi_form_query("What patterns show up across all forms?")

    Capture with: uv run python src/agent.py > docs/demo_run_log.txt

