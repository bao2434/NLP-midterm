"""High-level folder sentence segmentation runner using the Kim Han-Nom API.

Discovers all text files in a folder, divides each non-empty text file into
line-based chunks (each non-empty line before a newline) to stay within the
API's 10,000-token input limit, sends each chunk sequentially to the API, and
concatenates the segmented results (ensuring a newline between chunk results
so two sentences never share a line at concatenation boundaries) into the
output directory.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

from .api_client import (
    APIError,
    HanNomSegmentationClient,
    RateLimitError,
    SegmentationAPIConfig,
)
from .utils import (
    concat_chunk_results,
    discover_text_files,
    extract_result_text,
    split_text_into_line_chunks,
)


class HanNomSegmenter:
    """Discover text files in a folder, segment each file line-chunk by line-chunk via the API, and write concatenated results."""

    def __init__(self, config: Optional[SegmentationAPIConfig] = None):
        self.config = config or SegmentationAPIConfig()
        self.client = HanNomSegmentationClient(self.config)

    @property
    def run_name(self) -> str:
        return f"seg_{self.config.endpoint}_{self.config.genre}"

    def discover(
        self,
        input_dir: str | Path | list[str | Path],
        extensions: Sequence[str] = (".txt",),
        recursive: bool = True,
    ) -> dict[str, str]:
        """Discover all text files under ``input_dir``.

        Returns:
            ``{doc_name: source_path}`` sorted deterministically by path.
        """
        return discover_text_files(
            roots=input_dir,
            extensions=extensions,
            recursive=recursive,
        )

    def split_chunks(self, text: str) -> list[str]:
        """Divide text into chunks where each chunk is a non-empty line before a newline."""
        return split_text_into_line_chunks(text)

    def segment_chunk(self, chunk: str) -> dict:
        """Send a single text chunk to the Kim Han-Nom API."""
        return self.client.segment_text(chunk)

    def segment_text(self, text: str) -> dict:
        """Segment a text (splitting into non-empty line chunks if multiple lines exist) and concatenate results."""
        chunks = self.split_chunks(text)
        if not chunks:
            return {
                "text": "",
                "sentences": [],
                "chunks": 0,
                "endpoint": self.config.endpoint,
                "genre": self.config.genre,
                "punctuation_group": self.config.punctuation_group,
            }

        chunk_outputs: list[str] = []
        all_sentences: list[str] = []

        for chunk in chunks:
            seg_res = self.segment_chunk(chunk)
            chunk_text = extract_result_text(seg_res)
            if chunk_text:
                if not chunk_text.endswith("\n"):
                    chunk_text += "\n"
                chunk_outputs.append(chunk_text)
                for line in chunk_text.splitlines():
                    stripped = line.strip()
                    if stripped:
                        all_sentences.append(stripped)

        combined_text = concat_chunk_results(chunk_outputs)
        return {
            "text": combined_text,
            "sentences": all_sentences,
            "chunks": len(chunks),
            "endpoint": self.config.endpoint,
            "genre": self.config.genre,
            "punctuation_group": self.config.punctuation_group,
        }

    def process_file(
        self,
        file_path: str | Path,
        output_dir: str | Path = "seg_api_output",
        resume: bool = True,
        budget_hours: Optional[float] = None,
        progress: bool = True,
    ) -> Optional[Path]:
        """Segment a single text file by line chunks and write the concatenated result to ``output_dir``."""
        path = Path(file_path)
        docs = self.discover(path, extensions=(path.suffix or ".txt",), recursive=False)
        written = self.run(
            docs,
            output_dir=output_dir,
            resume=resume,
            budget_hours=budget_hours,
            progress=progress,
        )
        return next(iter(written.values()), None)

    def run(
        self,
        docs: dict[str, str],
        output_dir: str | Path = "seg_api_output",
        resume: bool = True,
        budget_hours: Optional[float] = None,
        progress: bool = True,
    ) -> dict[str, Path]:
        """Divide each text file in ``docs`` into non-empty line chunks, call the API for each chunk, and write concatenated results to ``output_dir``."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        started_run = time.time()
        written: dict[str, Path] = {}
        todo_items: list[tuple[str, str, Path]] = []

        for doc_name, src in docs.items():
            dest_file = out_path / f"{doc_name}.txt"
            if resume and dest_file.exists() and dest_file.stat().st_size > 0:
                written[doc_name] = dest_file
            else:
                todo_items.append((doc_name, src, dest_file))

        if progress:
            already_done = len(docs) - len(todo_items)
            print(
                f"Segmentation API (endpoint={self.config.endpoint}, genre={self.config.genre}, "
                f"punctuation={self.config.punctuation_group}): {len(todo_items)} file(s) to process "
                f"({already_done} already completed, delay={self.config.request_delay}s, "
                f"calls={self.config.rate_limit_summary()})."
            )

        consecutive_errors = 0
        max_consecutive_errors = self.config.max_consecutive_errors
        unlimited = self.config.is_unlimited

        for idx, (doc_name, src, dest_file) in enumerate(todo_items, 1):
            if budget_hours is not None and (time.time() - started_run) > budget_hours * 3600:
                if progress:
                    print(
                        f"[STOP] Time budget ({budget_hours}h) reached after {idx - 1} file(s). "
                        "Re-run to resume remaining files."
                    )
                break

            file_started = time.time()
            raw_text = Path(src).read_text(encoding="utf-8-sig", errors="replace")
            chunks = self.split_chunks(raw_text)

            if not chunks:
                dest_file.write_text("", encoding="utf-8")
                written[doc_name] = dest_file
                if progress:
                    print(f"[{idx}/{len(todo_items)}] {doc_name}: empty file, wrote empty output -> {dest_file}")
                continue

            chunk_results: list[str] = []
            total_sentences = 0
            file_aborted = False
            stop_all = False

            for chunk_idx, chunk in enumerate(chunks, 1):
                if budget_hours is not None and (time.time() - started_run) > budget_hours * 3600:
                    if progress:
                        print(
                            f"[STOP] Time budget ({budget_hours}h) reached during {doc_name} "
                            f"(chunk {chunk_idx}/{len(chunks)}). Re-run to resume remaining files."
                        )
                    file_aborted = True
                    stop_all = True
                    break

                try:
                    seg_res = self.segment_chunk(chunk)
                    chunk_text = extract_result_text(seg_res)
                    if chunk_text:
                        if not chunk_text.endswith("\n"):
                            chunk_text += "\n"
                        chunk_results.append(chunk_text)
                        total_sentences += sum(
                            1 for line in chunk_text.splitlines() if line.strip()
                        )
                    consecutive_errors = 0
                except RateLimitError as exc:
                    if unlimited:
                        if progress:
                            print(
                                f"[RATE LIMIT - CONTINUING] {doc_name} (chunk {chunk_idx}/{len(chunks)}): {exc}"
                            )
                        file_aborted = True
                        break
                    if progress:
                        print(
                            f"[RATE LIMIT] Stopped at {doc_name} (chunk {chunk_idx}/{len(chunks)}): {exc}. "
                            "Completed files are saved; re-run later to resume."
                        )
                    file_aborted = True
                    stop_all = True
                    break
                except APIError as exc:
                    consecutive_errors += 1
                    file_aborted = True
                    if progress:
                        print(f"[API ERROR] {doc_name} (chunk {chunk_idx}/{len(chunks)}): {exc}")
                    if not unlimited and max_consecutive_errors > 0 and consecutive_errors >= max_consecutive_errors:
                        if progress:
                            print(
                                f"[STOP] Reached {consecutive_errors} consecutive API errors. "
                                "Stopping early to preserve your API rate-limit quota."
                            )
                        stop_all = True
                    break
                except Exception as exc:
                    consecutive_errors += 1
                    file_aborted = True
                    if progress:
                        print(
                            f"[ERROR] {doc_name} (chunk {chunk_idx}/{len(chunks)}): "
                            f"{type(exc).__name__}: {exc}"
                        )
                    if not unlimited and max_consecutive_errors > 0 and consecutive_errors >= max_consecutive_errors:
                        if progress:
                            print(
                                f"[STOP] Reached {consecutive_errors} consecutive errors. "
                                "Stopping early to preserve your API rate-limit quota."
                            )
                        stop_all = True
                    break

            if not file_aborted:
                final_text = concat_chunk_results(chunk_results)
                dest_file.write_text(final_text, encoding="utf-8")
                written[doc_name] = dest_file
                elapsed = round(time.time() - file_started, 2)
                if progress:
                    print(
                        f"[{idx}/{len(todo_items)}] {doc_name}: {elapsed}s, "
                        f"{len(raw_text)} chars ({len(chunks)} chunk(s)) -> "
                        f"{total_sentences} sentence(s) -> {dest_file} "
                        f"(api_reqs={self.client.request_count})"
                    )

            if stop_all:
                break

        return written

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "HanNomSegmenter":
        return self
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
