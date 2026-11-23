"""Utilities for the Kim Han-Nom OCR API module.

Adapted from ``ocr_tool/utils.py``: PDF discovery, page selection,
PDF page rendering, image encoding, and JSONL result persistence.
"""
from __future__ import annotations

import base64
import glob
import io
import json
import re
from pathlib import Path
from typing import Any

# Cache open PDF handles across page renders.
_DOCS: dict[str, tuple[str, Any]] = {}


def parse_pages(spec: str, page_count: int) -> list[int]:
    """Parse 1-based page selection, e.g. '1-10,15'; return zero-based indices."""
    if not spec.strip():
        return list(range(page_count))

    selected: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start, separator, end = part.partition("-")
        first = int(start)
        last = int(end) if separator else first
        if first < 1 or last < first:
            raise ValueError(f"Invalid page range: {part!r}")
        selected.extend(range(first - 1, min(last, page_count)))
    return sorted(set(selected))


def _open_pdf(source: str | Path) -> tuple[str, Any]:
    """Open a PDF using pypdfium2 (preferred) or PyMuPDF (fitz) as fallback."""
    key = str(Path(source).resolve())
    if key in _DOCS:
        return _DOCS[key]

    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(key)
        _DOCS[key] = ("pdfium", doc)
        return _DOCS[key]
    except ImportError:
        pass

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(key)
        _DOCS[key] = ("fitz", doc)
        return _DOCS[key]
    except ImportError as exc:
        raise ImportError(
            "No PDF rendering backend found. Install pypdfium2 (`pip install pypdfium2`) "
            "or PyMuPDF (`pip install pymupdf`)."
        ) from exc


def get_pdf_page_count(source: str | Path) -> int:
    """Return the number of pages in a PDF file."""
    backend, doc = _open_pdf(source)
    return len(doc)


def render_pdf_page(
    source: str | Path,
    page: int,
    dpi: int = 200,
    max_side: int = 1600,
):
    """Render a zero-based PDF page as an RGB PIL image."""
    from PIL import Image

    backend, doc = _open_pdf(source)
    if page < 0 or page >= len(doc):
        raise IndexError(
            f"Page {page} is outside the PDF ({len(doc)} pages)."
        )

    if backend == "pdfium":
        image = doc[page].render(scale=dpi / 72).to_pil().convert("RGB")
    else:
        pix = doc[page].get_pixmap(dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    if max_side > 0 and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return image


def close_pdf_documents() -> None:
    """Close all cached PDF handles."""
    for _, document in _DOCS.values():
        try:
            document.close()
        except Exception:
            pass
    _DOCS.clear()


def image_to_bytes(image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    """Encode a PIL Image to JPEG/PNG bytes."""
    fmt_upper = fmt.upper()
    if fmt_upper == "JPG":
        fmt_upper = "JPEG"
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if fmt_upper == "JPEG":
        image = image.convert("RGB")
        save_kwargs["quality"] = quality
    image.save(buffer, format=fmt_upper, **save_kwargs)
    return buffer.getvalue()


def image_to_data_uri(image, fmt: str = "JPEG", quality: int = 92) -> str:
    """Encode a PIL Image to a base64 data URI for the /viet-ocr endpoint."""
    fmt_upper = fmt.upper()
    if fmt_upper == "JPG":
        fmt_upper = "JPEG"
    mime = "image/jpeg" if fmt_upper == "JPEG" else f"image/{fmt_upper.lower()}"
    raw = image_to_bytes(image, fmt=fmt_upper, quality=quality)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def discover_pdfs(
    roots: str | Path | list[str | Path],
    page_spec: str = "",
    recursive: bool = True,
    skip_macosx: bool = True,
) -> dict[str, tuple[str, list[int]]]:
    """Find PDFs and return {book_name: (source_path, selected_zero_based_pages)}."""
    if isinstance(roots, (str, Path)):
        roots = [roots]

    paths: list[str] = []
    for root in roots:
        candidate = Path(root)
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            paths.append(str(candidate))
        elif candidate.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            paths.extend(glob.glob(str(candidate / pattern), recursive=recursive))

    books: dict[str, tuple[str, list[int]]] = {}
    for path in sorted(set(paths)):
        if skip_macosx and "__MACOSX" in path:
            continue
        name = re.sub(r"[^\w\-]+", "_", Path(path).stem).strip("_") or "document"
        while name in books:
            name += "_"
        page_count = get_pdf_page_count(path)
        pages = parse_pages(page_spec, page_count)
        books[name] = (path, pages)
    return books


def result_file(output_dir: str | Path, book: str, name: str = "api_hannom") -> Path:
    """Return the JSONL result path for a given book."""
    return Path(output_dir) / book / f"{name}.jsonl"


def load_results(
    output_dir: str | Path,
    book: str,
    name: str = "api_hannom",
) -> dict[int, dict[str, Any]]:
    """Load completed page records from JSONL for one book."""
    path = result_file(output_dir, book, name=name)
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records[int(record["page"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return records


def append_result(
    output_dir: str | Path,
    book: str,
    record: dict[str, Any],
    name: str = "api_hannom",
) -> Path:
    """Append a single page OCR record to the book's JSONL file."""
    path = result_file(output_dir, book, name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def page_lines(record: dict[str, Any]) -> list[str]:
    """Extract clean text lines from an API OCR page record."""
    lines = record.get("lines")
    if isinstance(lines, list) and lines:
        return [str(line).strip() for line in lines if str(line).strip()]
    text = record.get("text", "")
    if isinstance(text, str) and text.strip():
        return [line.strip() for line in text.splitlines() if line.strip()]
    return []
