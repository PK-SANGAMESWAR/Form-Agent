"""
src/agent.py

Stage 8, the orchestration layer (APPROACH.md §3.8). FormAgent is the
only module that imports every other stage and is the sole public
surface a UI (or the tests) should call:

    agent = FormAgent()
    agent.load_directory("data/sample_forms")
    agent.answer_question(question, form_id=None)   # routes internally
    agent.summarize_form(form_id)
    agent.multi_form_query(question)

Internally: router -> (structured_store | vectorstore via retrieval) ->
synthesis, per APPROACH.md's pipeline diagram.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import extraction
import ingestion
import retrieval
import router
import structured_store
import synthesis
import vectorstore
from chunking import build_chunks
from schemas import FREE_TEXT_FIELD
from synthesis import SynthesisResult


class AgentError(RuntimeError):
    """Raised for agent-level failures not already covered by a stage's
    own exception type (e.g. summarizing a form_id that was never loaded)."""


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (source_path, error message)


class FormAgent:
    def __init__(
        self,
        db_path: str = structured_store.DEFAULT_DB_PATH,
        chroma_dir: str = vectorstore.DEFAULT_PERSIST_DIR,
    ):
        self.conn: sqlite3.Connection = structured_store.connect(db_path)
        structured_store.init_db(self.conn)

        self._chroma_client = vectorstore.get_client(persist_dir=chroma_dir)
        self.collection = vectorstore.get_collection(self._chroma_client)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_directory(self, dir_path: str) -> LoadReport:
        """
        Ingests every form in a directory, extracts + validates each one,
        and writes it to both stores (SQLite for structured fields,
        Chroma for the chunked free-text field). Continues past
        individual failures (bad OCR, extraction validation failure)
        rather than aborting the whole batch.
        """
        report = LoadReport()
        for raw in ingestion.ingest_directory(dir_path):
            try:
                extracted = extraction.extract_form(raw)
                structured_store.insert_form(self.conn, extracted)
                vectorstore.delete_form(self.collection, extracted.form_id)
                vectorstore.add_chunks(self.collection, build_chunks(extracted))
                report.loaded.append(extracted.form_id)
            except (extraction.ExtractionError, vectorstore.VectorStoreError) as e:
                report.failed.append((raw.source_path, str(e)))
        return report

    def load_form(self, path: str, form_id: str | None = None) -> tuple[str, int]:
        """Single-file version of load_directory, for a UI that ingests
        one upload at a time. Returns (form_id, chunk_count) where
        chunk_count is the number of vector chunks embedded — 0 means
        the free-text field was empty (still valid; structured fields
        are always stored). Raises on failure instead of collecting it,
        since there's no batch to keep going for."""
        raw = ingestion.ingest_file(path)
        if form_id is not None:
            raw.form_id = form_id
        extracted = extraction.extract_form(raw)
        structured_store.insert_form(self.conn, extracted)
        chunks = build_chunks(extracted)
        vectorstore.delete_form(self.collection, extracted.form_id)
        chunk_count = vectorstore.add_chunks(self.collection, chunks)
        return extracted.form_id, chunk_count

    # ------------------------------------------------------------------
    # Public QA / summarization API
    # ------------------------------------------------------------------
    def answer_question(self, question: str, form_id: str | None = None) -> SynthesisResult:
        """
        Routes internally via router.route_question: aggregate -> exact
        SQL answer (no LLM call), single_form_lookup/semantic -> that
        one form's context, multi_form_semantic -> hybrid retrieval
        across all forms. `form_id` should be passed when the caller
        already has one form open (e.g. a UI showing a form's detail
        page) — mirrors router.route_question's own parameter.
        """
        decision = router.route_question(question, form_id=form_id)
        result = retrieval.retrieve(decision, question, self.conn, self.collection)

        if decision.route == router.RouteType.aggregate:
            return synthesis.format_aggregate(result)
        return synthesis.answer_question(result, question)

    def summarize_form(self, form_id: str) -> SynthesisResult:
        form = structured_store.get_form(self.conn, form_id)
        if form is None:
            raise AgentError(f"No form found with form_id={form_id!r}")
        free_text_field = FREE_TEXT_FIELD[form["form_type"]]
        return synthesis.summarize_form(form, free_text_field)

    def multi_form_query(self, question: str) -> SynthesisResult:
        """Convenience wrapper for a question that should always be
        treated as multi-form semantic regardless of keyword routing
        (e.g. a UI's dedicated 'ask across all forms' box) — bypasses
        router.py's classification entirely."""
        decision = router.RouteDecision(
            route=router.RouteType.multi_form_semantic, detection_method="forced"
        )
        result = retrieval.retrieve(decision, question, self.conn, self.collection)
        return synthesis.answer_question(result, question)

    def list_form_ids(self, form_type: str | None = None) -> list[str]:
        return structured_store.all_form_ids(self.conn, form_type)

    def close(self) -> None:
        self.conn.close()
        # Release ChromaDB file handles (important on Windows — the persistent
        # client holds a lock on the chroma dir until the object is deleted).
        try:
            del self.collection
            del self._chroma_client
        except Exception:
            pass


if __name__ == "__main__":
    # Demo runner — produces the three required demo scenarios when Ollama is
    # running. Redirect to docs/demo_run_log.txt to capture required output:
    #   uv run python src/agent.py > docs/demo_run_log.txt
    import sys

    import llm_client

    if not llm_client.is_available():
        print(f"Ollama/{llm_client.DEFAULT_MODEL} not reachable — run `ollama serve` first.")
        sys.exit(1)

    # Use in-memory SQLite (fresh each run) + persistent chroma_db.
    # load_directory() upserts — delete-then-add per form — so re-running
    # never duplicates chunks in chroma_db.
    agent = FormAgent(db_path=":memory:")
    report = agent.load_directory("data/sample_forms")
    print(f"Loaded {len(report.loaded)} form(s): {report.loaded}")
    if report.failed:
        print(f"Failed {len(report.failed)} form(s):")
        for path, err in report.failed:
            print(f"  {path}: {err}")

    if not report.loaded:
        print("No forms loaded — nothing to demo.")
        sys.exit(0)

    # Pick a membership form for demos 1 & 2 (has status + remarks fields)
    membership_ids = [fid for fid in report.loaded if "membership" in fid]
    hospital_ids   = [fid for fid in report.loaded if "hospital" in fid]
    demo_id = membership_ids[0] if membership_ids else report.loaded[0]

    print("\n" + "="*60)
    print("DEMO 1: Single-form Q&A")
    print("="*60)
    q1 = "What is the applicant's name and what is the status of their application?"
    print(f"Question: {q1}")
    r1 = agent.answer_question(q1, form_id=demo_id)
    print(f"Answer:   {r1.answer}")
    print(f"Cited:    {r1.cited_form_ids}")
    print(f"Grounded: {r1.grounded}")

    print("\n" + "="*60)
    print("DEMO 2: Single-form Summary")
    print("="*60)
    print(f"Form: {demo_id}")
    r2 = agent.summarize_form(demo_id)
    print(f"Summary:\n{r2.answer}")

    print("\n" + "="*60)
    print("DEMO 3: Multi-form Holistic Insight")
    print("="*60)
    q3 = "How many membership applications are approved?"
    print(f"Question: {q3}")
    r3 = agent.answer_question(q3)
    print(f"Answer:   {r3.answer}")
    print(f"Cited:    {r3.cited_form_ids}")

    if hospital_ids:
        print("\n" + "="*60)
        print("DEMO 3b: Multi-form Aggregate (Hospital)")
        print("="*60)
        q3b = "How many hospital forms are there?"
        print(f"Question: {q3b}")
        r3b = agent.answer_question(q3b)
        print(f"Answer:   {r3b.answer}")
        print(f"Cited:    {r3b.cited_form_ids}")

        print()
        q3c = "List all hospital patients"
        print(f"Question: {q3c}")
        r3c = agent.answer_question(q3c)
        print(f"Answer:   {r3c.answer}")
        print(f"Cited:    {r3c.cited_form_ids}")


    agent.close()
