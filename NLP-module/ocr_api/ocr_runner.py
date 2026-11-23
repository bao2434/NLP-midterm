"""High-level PDF/folder OCR runner using the Kim Han-Nom API.

Mirrors the workflow of ``ocr_tool`` (discover -> run -> export_txt) with
incremental JSONL persistence for resumability. Because the API enforces
strict rate limits (25/min, 50/hr, 200/day, 600/week) and explicitly asks
callers not to send concurrent bursts, pages are processed sequentially with
a configurable inter-request delay.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from .api_client import APIConfig, APIError, HanNomAPIClient, RateLimitError
from .utils import (
    append_result,
    close_pdf_documents,
    discover_pdfs,
    get_pdf_page_count,
    load_results,
    page_lines,
    parse_pages,
    render_pdf_page,
)


class HanNomOCR:
    """Discover PDFs in a folder, OCR each page via API, and export plain text."""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.client = HanNomAPIClient(self.config)

    @property
    def run_name(self) -> str:
        return f"api_{self.config.mode}"

    def discover(
        self,
        input_dir: str | Path | list[str | Path],
        pages: str = "",
        recursive: bool = True,
    ) -> dict[str, tuple[str, list[int]]]:
        """Find PDF files under ``input_dir`` and select pages for each document."""
        return discover_pdfs(
            roots=input_dir,
            page_spec=pages,
            recursive=recursive,
        )

    def process_image(self, image_path: str | Path) -> dict[str, Any]:
        """OCR a single image file via the API."""
        from PIL import Image

        path = Path(image_path)
        started = time.time()
        with Image.open(path) as img:
            image = img.convert("RGB")
            if self.config.max_side > 0 and max(image.size) > self.config.max_side:
                scale = self.config.max_side / max(image.size)
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            result = self.client.ocr_pil_image(image, filename=path.name)
            width, height = image.size

        return {
            "source": str(path),
            "width": width,
            "height": height,
            "seconds": round(time.time() - started, 2),
            **result,
        }

    def process_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path = "ocr_api_raw",
        pages: Optional[list[int] | str] = None,
        resume: bool = True,
        budget_hours: Optional[float] = None,
        progress: bool = True,
    ) -> dict[int, dict[str, Any]]:
        """OCR a single PDF file sequentially with JSONL checkpointing."""
        pdf_path = Path(pdf_path)
        page_count = get_pdf_page_count(pdf_path)
        if pages is None:
            selected = list(range(page_count))
        elif isinstance(pages, str):
            selected = parse_pages(pages, page_count)
        else:
            selected = sorted(set(pages))

        books = {pdf_path.stem: (str(pdf_path), selected)}
        all_results = self.run(
            books,
            output_dir=output_dir,
            resume=resume,
            budget_hours=budget_hours,
            progress=progress,
        )
        return all_results.get(pdf_path.stem, {})

    def run(
        self,
        books: dict[str, tuple[str, list[int]]],
        output_dir: str | Path = "ocr_api_raw",
        resume: bool = True,
        budget_hours: Optional[float] = None,
        progress: bool = True,
    ) -> dict[str, dict[int, dict[str, Any]]]:
        """Sequentially OCR all selected pages across discovered PDFs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        started_run = time.time()
        all_records: dict[str, dict[int, dict[str, Any]]] = {}
        todo_items: list[tuple[str, str, int, int]] = []

        for book, (src, selected) in books.items():
            existing = load_results(output_dir, book, name=self.run_name) if resume else {}
            all_records[book] = existing
            total_pages = len(selected)
            for page_index in selected:
                if page_index not in existing:
                    todo_items.append((book, src, page_index, total_pages))

        if progress:
            total_selected = sum(len(sel) for _, sel in books.values())
            already_done = total_selected - len(todo_items)
            print(
                f"OCR API ({self.config.mode}): {len(todo_items)} page(s) to process "
                f"({already_done} already completed, delay={self.config.request_delay}s, "
                f"calls={self.config.rate_limit_summary()})."
            )

        consecutive_errors = 0
        max_consecutive_errors = self.config.max_consecutive_errors

        for idx, (book, src, page_index, _) in enumerate(todo_items, 1):
            if budget_hours is not None and (time.time() - started_run) > budget_hours * 3600:
                if progress:
                    print(
                        f"[STOP] Time budget ({budget_hours}h) reached after {idx - 1} page(s). "
                        "Progress saved; re-run to resume."
                    )
                break

            page_started = time.time()
            try:
                image = render_pdf_page(
                    src,
                    page_index,
                    dpi=self.config.dpi,
                    max_side=self.config.max_side,
                )
                filename = f"{book}_p{page_index + 1:04d}.jpg"
                ocr_res = self.client.ocr_pil_image(image, filename=filename)
                record = {
                    "book": book,
                    "source": src,
                    "page": page_index,
                    "width": image.width,
                    "height": image.height,
                    "seconds": round(time.time() - page_started, 2),
                    **ocr_res,
                }
                append_result(output_dir, book, record, name=self.run_name)
                all_records[book][page_index] = record
                consecutive_errors = 0
                if progress:
                    print(
                        f"[{idx}/{len(todo_items)}] {book} p.{page_index + 1}: "
                        f"{record['seconds']}s, {len(record['text'])} chars "
                        f"(api_reqs={self.client.request_count})"
                    )
            except RateLimitError as exc:
                if progress:
                    print(
                        f"[RATE LIMIT] Stopped at {book} p.{page_index + 1}: {exc}. "
                        "Completed pages are saved; re-run later to resume."
                    )
                break
            except APIError as exc:
                consecutive_errors += 1
                if progress:
                    print(f"[API ERROR] {book} p.{page_index + 1}: {exc}")
                if max_consecutive_errors > 0 and consecutive_errors >= max_consecutive_errors:
                    if progress:
                        print(
                            f"[STOP] Reached {consecutive_errors} consecutive API errors. "
                            "Stopping early to preserve your API rate-limit quota."
                        )
                    break
            except Exception as exc:
                consecutive_errors += 1
                if progress:
                    print(
                        f"[ERROR] {book} p.{page_index + 1}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                if max_consecutive_errors > 0 and consecutive_errors >= max_consecutive_errors:
                    if progress:
                        print(
                            f"[STOP] Reached {consecutive_errors} consecutive errors. "
                            "Stopping early to preserve your API rate-limit quota."
                        )
                    break

        return {
            book: {p: all_records[book][p] for p in selected if p in all_records[book]}
            for book, (_, selected) in books.items()
        }

    def export_txt(
        self,
        books: dict[str, tuple[str, list[int]]],
        output_dir: str | Path = "ocr_api_raw",
        txt_dir: str | Path = "ocr_api_txt",
        progress: bool = True,
    ) -> dict[str, Path]:
        """Export completed OCR results for each discovered book to .txt files."""
        txt_path = Path(txt_dir)
        txt_path.mkdir(parents=True, exist_ok=True)
        exported: dict[str, Path] = {}

        for book, (_, selected) in books.items():
            records = load_results(output_dir, book, name=self.run_name)
            chunks = [
                "\n".join(page_lines(records[p]))
                for p in selected
                if p in records
            ]
            out_file = txt_path / f"{book}.txt"
            content = "\n\n".join(chunk for chunk in chunks if chunk)
            out_file.write_text((content + "\n") if content else "", encoding="utf-8")
            exported[book] = out_file

            if progress:
                done = sum(1 for p in selected if p in records)
                missing = len(selected) - done
                suffix = f" (missing {missing})" if missing else ""
                print(f"Exported {book}: {done}/{len(selected)} pages -> {out_file}{suffix}")

        return exported

    def close(self) -> None:
        self.client.close()
        close_pdf_documents()

    def __enter__(self) -> "HanNomOCR":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
