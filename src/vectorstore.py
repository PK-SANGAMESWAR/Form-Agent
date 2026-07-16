"""
src/vectorstore.py

Embedding + storage half of Stage 3 (see APPROACH.md §3.3). chunking.py
turns an ExtractedForm's free-text field into TextChunks; this module
embeds each chunk (via llm_client.embed -> Ollama's nomic-embed-text) and
stores it in a local, on-disk Chroma collection with metadata for
filtering.

retrieval.py (next) will call similarity_search() twice conceptually:
  - single_form_semantic -> where={"form_id": "<one form>"}
  - multi_form_semantic  -> where=None (search everything), then merge
    with BM25 results and optionally rerank.

Chroma runs embedded/file-based (chroma_db/, gitignored) — no server
process, matching the "minimal setup" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

import llm_client
from chunking import TextChunk

DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "form_chunks"


class VectorStoreError(RuntimeError):
    """Raised when embedding or Chroma operations fail."""


@dataclass
class RetrievedChunk:
    chunk_id: str
    form_id: str
    form_type: str
    field_name: str
    text: str
    metadata: dict
    distance: float  # lower = more similar (cosine distance, see get_collection)


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(
    client: chromadb.ClientAPI,
    name: str = DEFAULT_COLLECTION_NAME,
) -> Collection:
    """
    Forces cosine distance (Chroma defaults to L2) since that's the
    convention nomic-embed-text / all-MiniLM-L6-v2 are tuned for.
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _sanitize_metadata(raw: dict) -> dict:
    """
    Chroma metadata values must be str/int/float/bool — no None, no
    nested dicts/lists. ExtractedForm fields can be None (e.g.
    discharge_date for an ongoing hospital stay). Coerce rather than
    drop, so the field stays visible/filterable.
    """
    clean = {}
    for key, value in raw.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def add_chunks(collection: Collection, chunks: list[TextChunk]) -> int:
    """
    Embeds and upserts a batch of TextChunks. Returns count written.

    Uses upsert (not add) so re-running ingestion on the same forms
    doesn't fail on duplicate IDs — chunk_id is stable per
    (form_id, field_name, index), so re-ingesting a form overwrites its
    old chunks rather than duplicating them.
    """
    if not chunks:
        return 0

    ids, embeddings, documents, metadatas = [], [], [], []
    for chunk in chunks:
        if not chunk.text.strip():
            continue
        try:
            vector = llm_client.embed(chunk.text)
        except llm_client.LLMError as e:
            raise VectorStoreError(f"Failed to embed chunk {chunk.chunk_id!r}: {e}") from e

        metadata = _sanitize_metadata(
            {
                **chunk.metadata,
                "form_id": chunk.form_id,
                "form_type": chunk.form_type,
                "field_name": chunk.field_name,
            }
        )
        ids.append(chunk.chunk_id)
        embeddings.append(vector)
        documents.append(chunk.text)
        metadatas.append(metadata)

    if not ids:
        return 0

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(ids)


def delete_form(collection: Collection, form_id: str) -> None:
    """Removes all chunks for one form_id — call before re-ingesting that form."""
    collection.delete(where={"form_id": form_id})


def similarity_search(
    collection: Collection,
    query_text: str,
    top_k: int = 5,
    where: dict | None = None,
) -> list[RetrievedChunk]:
    """
    Embeds query_text and returns the top_k nearest chunks. `where` scopes
    the search, e.g. where={"form_id": "hospital_003"} for
    single_form_semantic questions (retrieval.py's job to decide when).
    """
    try:
        query_vector = llm_client.embed(query_text)
    except llm_client.LLMError as e:
        raise VectorStoreError(f"Failed to embed query: {e}") from e

    results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            form_id=metadata.get("form_id", ""),
            form_type=metadata.get("form_type", ""),
            field_name=metadata.get("field_name", ""),
            text=text,
            metadata=metadata,
            distance=distance,
        )
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
    ]


def count_chunks(collection: Collection) -> int:
    return collection.count()


if __name__ == "__main__":
    # Manual smoke test. Unlike structured_store.py/chunking.py, this
    # stage genuinely needs Ollama running — embeddings ARE the LLM call.
    import shutil
    import tempfile

    from schemas import MembershipForm
    from extraction import ExtractedForm
    from chunking import build_chunks

    if not llm_client.is_available(llm_client.DEFAULT_EMBED_MODEL):
        print(
            f"Ollama / {llm_client.DEFAULT_EMBED_MODEL} not reachable — "
            f"run `ollama serve` and `ollama pull {llm_client.DEFAULT_EMBED_MODEL}` first."
        )
    else:
        tmp_dir = tempfile.mkdtemp()
        try:
            client = get_client(persist_dir=tmp_dir)
            collection = get_collection(client)

            fake = ExtractedForm(
                form_id="membership_001",
                form_type="membership",
                data=MembershipForm(
                    name="Ananya Rao", date_of_birth="14-03-1994",
                    email="ananya.rao@example.com", phone="+91-9876543210",
                    occupation="Software Engineer", application_date="02-01-2026",
                    status="Approved",
                    remarks="Applicant has a strong credit history and no prior defaults.",
                ),
                detection_method="keyword", extraction_attempts=1,
            )
            chunks = build_chunks(fake)
            written = add_chunks(collection, chunks)
            print(f"Wrote {written} chunk(s). Collection count: {count_chunks(collection)}")

            hits = similarity_search(collection, "credit history", top_k=3)
            for hit in hits:
                print(f"- {hit.chunk_id} (distance={hit.distance:.4f}): {hit.text[:80]}...")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)