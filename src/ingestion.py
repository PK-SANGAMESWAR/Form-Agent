"""
src/ingestion.py

Stage 1 of the pipeline: turn a form file on disk (PDF or DOCX) into raw text.

Design notes (see APPROACH.md §3.1):
- Text-based PDFs are extracted directly with pdfplumber (fast, no OCR needed).
- If pdfplumber returns near-empty text for a page, we assume it's a scanned
  image and fall back to OCR (pdf2image -> pytesseract).
- DOCX files are read with python-docx.
- Everything downstream (extraction.py, chunking.py) only ever sees a
  RawForm, so it doesn't care whether OCR was used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# OCR deps are optional at import time — only needed if a scanned page is hit.
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import docx  # python-docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# Below this many characters, a pdfplumber page is treated as "no extractable
# text" and routed to OCR instead. Tuned empirically — real text pages with
# any content clear this easily; blank/scanned pages don't.
MIN_CHARS_PER_PAGE = 20


@dataclass
class RawForm:
    """Container for one ingested form, before any LLM extraction happens."""
    form_id: str          # derived from filename, e.g. "membership_001"
    source_path: str
    text: str
    used_ocr: bool = False
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (f"RawForm(form_id={self.form_id!r}, chars={len(self.text)}, "
                f"used_ocr={self.used_ocr}, preview={preview!r}...)")


def _form_id_from_path(path: str) -> str:
    return Path(path).stem


def _ocr_pdf(path: str) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR fallback needed but pytesseract/pdf2image are not installed. "
            "Install them (see requirements.txt) or provide a text-based PDF."
        )
    images = convert_from_path(path)
    pages_text = [pytesseract.image_to_string(img) for img in images]
    return "\n".join(pages_text)


def ingest_pdf(path: str) -> RawForm:
    """Extract text from a PDF, falling back to OCR per-page if needed."""
    form_id = _form_id_from_path(path)
    warnings: list[str] = []
    page_texts: list[str] = []
    used_ocr = False

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < MIN_CHARS_PER_PAGE:
                warnings.append(
                    f"page {i+1}: near-empty text ({len(text.strip())} chars) "
                    f"— falling back to OCR"
                )
                page_texts.append(None)  # placeholder, filled in below
            else:
                page_texts.append(text)

    if any(t is None for t in page_texts):
        used_ocr = True
        ocr_full_text = _ocr_pdf(path)
        # Simplification: if ANY page needed OCR, OCR the whole doc and use
        # that. Mixed text/scanned multi-page docs could be handled page-by-
        # page with pdf2image(first_page=, last_page=) if needed later.
        final_text = ocr_full_text
    else:
        final_text = "\n".join(page_texts)

    return RawForm(
        form_id=form_id,
        source_path=path,
        text=final_text.strip(),
        used_ocr=used_ocr,
        page_count=page_count,
        warnings=warnings,
    )


def ingest_docx(path: str) -> RawForm:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed.")
    form_id = _form_id_from_path(path)
    document = docx.Document(path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    return RawForm(
        form_id=form_id,
        source_path=path,
        text=text.strip(),
        used_ocr=False,
        page_count=1,
        warnings=[],
    )


def ingest_file(path: str) -> RawForm:
    """Dispatch to the right ingestor based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return ingest_pdf(path)
    elif ext == ".docx":
        return ingest_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} ({path})")


def ingest_directory(dir_path: str) -> list[RawForm]:
    """Ingest every .pdf/.docx in a directory (non-recursive)."""
    results = []
    for name in sorted(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, name)
        ext = Path(name).suffix.lower()
        if ext in (".pdf", ".docx") and not name.startswith("_"):
            results.append(ingest_file(full_path))
    return results


if __name__ == "__main__":
    # Quick manual smoke test: ingest the sample_forms directory and print
    # what was extracted, including whether OCR kicked in.
    sample_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample_forms",
    )
    forms = ingest_directory(sample_dir)
    for f in forms:
        print(f)
        if f.warnings:
            for w in f.warnings:
                print("   !", w)
        print("   ---")
        print("  ", f.text.replace("\n", "\n   ")[:300])
        print()