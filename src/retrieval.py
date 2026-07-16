"""
src/retrieval.py

Stage 6 of the pipeline: turn a router.RouteDecision into an actual
answer-grounding result (see APPROACH.md §3.6). This is the only module
that touches BOTH structured_store.py and vectorstore.py — everything
upstream of it just classifies, everything downstream (synthesis.py)
just consumes a RetrievalResult.

Route -> backend mapping:
  - aggregate            -> structured_store.list_forms (count = len(rows))
  - single_form_lookup   -> structured_store.get_form
  - single_form_semantic -> vectorstore.similarity_search(where={form_id})
  - multi_form_semantic  -> hybrid: BM25 (rank_bm25) + Chroma embeddings,
                            merged via reciprocal rank fusion (RRF)

Why RRF for the hybrid merge: BM25 scores and cosine-distance scores live
on different, incomparable scales, so averaging them directly is
meaningless without calibration. RRF only uses each list's RANK, not its
raw score, so it needs no tuning and is the standard cheap fix for
this (see APPROACH.md §3.6 "merge + dedupe candidates").
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from chromadb.api.models.Collection import Collection

import structured_store
import vectorstore
from router import RouteDecision, RouteType
from vectorstore import RetrievedChunk

DEFAULT_TOP_K = 5
RRF_K = 60  # standard RRF damping constant; de-emphasizes rank-1-only hits

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RetrievalError(RuntimeError):
    """Raised when a route can't be fulfilled (e.g. unknown form_id, missing BM25 corpus)."""


@dataclass
class RetrievalResult:
    route: RouteType
    count: int | None = None
    rows: list[dict] = field(default_factory=list)   # aggregate
    form: dict | None = None                          # single_form_lookup
    chunks: list[RetrievedChunk] = field(default_factory=list)  # semantic routes


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_corpus(collection: Collection) -> tuple[list[str], list[str], list[dict]]:
    """
    Pulls every chunk out of Chroma to build a BM25 index. Chroma is the
    single source of truth for chunk text (chunking.py never persists
    chunks anywhere else), so BM25 has to rebuild its index from it each
    call rather than maintaining a separate store — fine at this corpus
    size (form remarks fields, not full documents); revisit if the corpus
    grows large enough that this becomes slow.
    """
    got = collection.get(include=["documents", "metadatas"])
    ids = got.get("ids", [])
    documents = got.get("documents", [])
    metadatas = got.get("metadatas", [])
    return ids, documents, metadatas


def _bm25_search(collection: Collection, query_text: str, top_k: int) -> list[RetrievedChunk]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise RetrievalError(
            "rank_bm25 not installed — run `pip install rank-bm25` (see requirements.txt)"
        ) from e

    ids, documents, metadatas = _load_corpus(collection)
    if not ids:
        return []

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query_text))

    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk_id=ids[i],
            form_id=metadatas[i].get("form_id", ""),
            form_type=metadatas[i].get("form_type", ""),
            field_name=metadatas[i].get("field_name", ""),
            text=documents[i],
            metadata=metadatas[i],
            distance=float(-scores[i]),  # not a real distance metric; BM25 hits are merged by RANK (see RRF below), not this value
        )
        for i in ranked
        if scores[i] > 0
    ]


def _reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    top_k: int,
) -> list[RetrievedChunk]:
    """Merges multiple ranked chunk lists by RRF score = sum(1 / (RRF_K + rank))
    per chunk_id, keeping the first-seen RetrievedChunk instance for each id."""
    rrf_scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            first_seen.setdefault(chunk.chunk_id, chunk)

    merged_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    return [first_seen[cid] for cid in merged_ids]


def _aggregate(conn: sqlite3.Connection, decision: RouteDecision) -> RetrievalResult:
    if decision.form_type is None:
        raise RetrievalError(
            "Aggregate route needs a form_type to query (none detected in the question — "
            "mention 'membership' or 'hospital' explicitly)."
        )
    rows = structured_store.list_forms(conn, decision.form_type, decision.filters or None)
    return RetrievalResult(route=decision.route, count=len(rows), rows=rows)


def _single_form_lookup(conn: sqlite3.Connection, decision: RouteDecision) -> RetrievalResult:
    if decision.form_id is None:
        raise RetrievalError("single_form_lookup route requires a form_id.")
    form = structured_store.get_form(conn, decision.form_id)
    if form is None:
        raise RetrievalError(f"No form found with form_id={decision.form_id!r}.")
    return RetrievalResult(route=decision.route, form=form)


def _single_form_semantic(
    collection: Collection,
    decision: RouteDecision,
    question: str,
    top_k: int,
    conn: sqlite3.Connection | None = None,
) -> RetrievalResult:
    if decision.form_id is None:
        raise RetrievalError("single_form_semantic route requires a form_id.")
    chunks = vectorstore.similarity_search(
        collection, question, top_k=top_k, where={"form_id": decision.form_id}
    )
    if chunks:
        return RetrievalResult(route=decision.route, chunks=chunks)
    # No chunks in vector store for this form — fall back to the structured
    # field lookup so the user still gets an answer instead of an error.
    if conn is not None:
        form = structured_store.get_form(conn, decision.form_id)
        if form is not None:
            return RetrievalResult(
                route=RouteType.single_form_lookup,
                form=form,
            )
    raise RetrievalError(
        f"No chunks found for form_id={decision.form_id!r} and no structured "
        "fallback available. Try re-ingesting the form."
    )


def _multi_form_semantic(collection: Collection, question: str, top_k: int) -> RetrievalResult:
    bm25_hits = _bm25_search(collection, question, top_k=top_k * 2)
    embedding_hits = vectorstore.similarity_search(collection, question, top_k=top_k * 2, where=None)
    merged = _reciprocal_rank_fusion([embedding_hits, bm25_hits], top_k=top_k)
    return RetrievalResult(route=RouteType.multi_form_semantic, chunks=merged)


def retrieve(
    decision: RouteDecision,
    question: str,
    conn: sqlite3.Connection,
    collection: Collection,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalResult:
    """Single entry point synthesis.py/agent.py should call — dispatches
    on decision.route so callers never need to know which backend serves
    which route."""
    if decision.route == RouteType.aggregate:
        return _aggregate(conn, decision)
    if decision.route == RouteType.single_form_lookup:
        return _single_form_lookup(conn, decision)
    if decision.route == RouteType.single_form_semantic:
        return _single_form_semantic(collection, decision, question, top_k, conn=conn)
    if decision.route == RouteType.multi_form_semantic:
        return _multi_form_semantic(collection, question, top_k)
    raise RetrievalError(f"Unhandled route: {decision.route!r}")


if __name__ == "__main__":
    # Manual smoke test. The aggregate/lookup paths need no LLM; the
    # semantic paths need Ollama for embeddings, same caveat as
    # vectorstore.py's own smoke test.
    import shutil
    import tempfile

    import llm_client
    from extraction import ExtractedForm
    from chunking import build_chunks
    from router import route_question
    from schemas import MembershipForm

    conn = structured_store.connect(":memory:")
    structured_store.init_db(conn)

    forms = [
        ExtractedForm(
            form_id="membership_001", form_type="membership",
            data=MembershipForm(
                name="Ananya Rao", date_of_birth="14-03-1994", email="a@example.com",
                phone="+91-9000000001", occupation="Engineer", application_date="02-01-2026",
                status="Approved", remarks="Strong credit history, no defaults.",
            ),
            detection_method="keyword", extraction_attempts=1,
        ),
        ExtractedForm(
            form_id="membership_002", form_type="membership",
            data=MembershipForm(
                name="Vikram Nair", date_of_birth="22-07-1988", email="v@example.com",
                phone="+91-9000000002", occupation="Designer", application_date="15-01-2026",
                status="Rejected", remarks="Two prior credit defaults on file.",
            ),
            detection_method="keyword", extraction_attempts=1,
        ),
    ]
    for f in forms:
        structured_store.insert_form(conn, f)

    decision = route_question("How many membership applications are approved?")
    result = retrieve(decision, "How many membership applications are approved?", conn, collection=None)
    print("Aggregate ->", result.count, "row(s)")

    decision = route_question("What is the status of membership_002?", form_id="membership_002")
    result = retrieve(decision, "status?", conn, collection=None)
    print("Lookup ->", result.form)

    if not llm_client.is_available(llm_client.DEFAULT_EMBED_MODEL):
        print(f"Ollama/{llm_client.DEFAULT_EMBED_MODEL} not reachable — skipping semantic routes.")
    else:
        tmp_dir = tempfile.mkdtemp()
        try:
            client = vectorstore.get_client(persist_dir=tmp_dir)
            collection = vectorstore.get_collection(client)
            for f in forms:
                vectorstore.add_chunks(collection, build_chunks(f))

            decision = route_question("Tell me about the credit history for membership_001", form_id="membership_001")
            result = retrieve(decision, "credit history", conn, collection)
            print("Single-form semantic ->", [c.chunk_id for c in result.chunks])

            decision = route_question("What patterns show up across all applicants' credit history?")
            result = retrieve(decision, "credit history patterns", conn, collection)
            print("Multi-form semantic ->", [c.chunk_id for c in result.chunks])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)