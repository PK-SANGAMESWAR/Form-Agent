"""
src/llm_client.py

Thin wrapper around Ollama's local REST API (see APPROACH.md §1, §5).

Every other module (extraction.py, router.py, synthesis.py) talks to the
local LLM only through this file. That means:
  - if you ever swap Ollama for llama-cpp-python or vLLM, only this file
    changes.
  - retry/timeout/JSON-mode behavior is defined once, not duplicated per call
    site.

Requires Ollama running locally (default http://localhost:11434) with the
target model already pulled, e.g.:
    ollama pull llama3.1:8b-instruct-q4_K_M
    ollama pull nomic-embed-text
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests

# Defaults — override per-call if needed.
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_TIMEOUT_S = 60
DEFAULT_EMBED_TIMEOUT_S = 120  # embedding model may need time to load into VRAM
DEFAULT_MAX_RETRIES = 2
DEFAULT_EMBED_MAX_RETRIES = 3  # extra retries for transient 404/503 during model load


class LLMError(RuntimeError):
    """Raised when the local LLM can't be reached or returns something unusable."""


@dataclass
class LLMResponse:
    text: str
    raw: dict
    model: str


def _post(endpoint: str, payload: dict, timeout: int) -> dict:
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. "
            f"Is `ollama serve` running and is the model pulled? ({e})"
        ) from e
    except requests.exceptions.Timeout as e:
        raise LLMError(f"Ollama request to {endpoint} timed out after {timeout}s") from e
    except requests.exceptions.HTTPError as e:
        raise LLMError(f"Ollama returned an error for {endpoint}: {e}") from e
    return resp.json()


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> LLMResponse:
    """
    Single-turn completion via /api/generate.

    json_mode=True sets Ollama's `format: "json"`, which constrains the
    model's output to syntactically valid JSON (see APPROACH.md §3.2 —
    this is what removes most extraction parsing failures). It does NOT
    guarantee the JSON matches your Pydantic schema — that's still
    validated by the caller (extraction.py), which retries with the
    validation error appended to the prompt on failure.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            data = _post("/api/generate", payload, timeout)
            text = data.get("response", "")
            if not text.strip():
                raise LLMError("Model returned an empty response.")
            return LLMResponse(text=text, raw=data, model=model)
        except LLMError as e:
            last_err = e
            if attempt == max_retries:
                raise
    # unreachable, but keeps type checkers happy
    raise last_err  # type: ignore[misc]


def generate_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """
    Convenience wrapper: calls generate() with json_mode=True and parses
    the result. Raises LLMError if the model's output isn't valid JSON
    even in JSON mode (rare, but Ollama's json-mode guarantees syntactic
    validity, not non-empty/sane content).
    """
    resp = generate(
        prompt=prompt,
        model=model,
        system=system,
        json_mode=True,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model output wasn't valid JSON: {resp.text[:200]!r}") from e


def embed(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    timeout: int = DEFAULT_EMBED_TIMEOUT_S,
    max_retries: int = DEFAULT_EMBED_MAX_RETRIES,
) -> list[float]:
    """
    Get an embedding vector for one piece of text via /api/embed (Ollama v0.4+).

    Retries with exponential backoff on 404/503 errors — these are transient
    when Ollama is loading the embedding model into VRAM for the first time
    in a session (load_duration can exceed 7 seconds for MoE models).
    """
    payload = {"model": model, "input": text}
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            data = _post("/api/embed", payload, timeout)
            # New API returns {"embeddings": [[...]]}
            embeddings = data.get("embeddings")
            if not embeddings or not embeddings[0]:
                raise LLMError("Ollama returned no embedding vector.")
            return embeddings[0]
        except LLMError as e:
            last_err = e
            err_str = str(e)
            # Retry on transient errors (404 = model loading, 503 = busy)
            is_transient = "404" in err_str or "503" in err_str or "502" in err_str
            if attempt < max_retries and is_transient:
                wait = 2 ** attempt  # 1s, 2s, 4s backoff
                time.sleep(wait)
                continue
            raise
    raise last_err  # type: ignore[misc]


def is_available(model: str = DEFAULT_MODEL) -> bool:
    """
    Quick health check — used by callers (agent.py) to decide whether to
    take the LLM path or fall back to the rule-based path (APPROACH.md §8).
    Does not raise; returns False on any failure.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        tags = [m["name"] for m in resp.json().get("models", [])]
        return any(model in t or t in model for t in tags)
    except Exception:
        return False


if __name__ == "__main__":
    # Quick manual smoke test.
    print("Ollama reachable + model available:", is_available())
    try:
        out = generate("Reply with exactly the word: pong", temperature=0.0)
        print("generate() ->", out.text.strip())
    except LLMError as e:
        print("generate() failed:", e)