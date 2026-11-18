from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .config import OCRConfig
from .visualizer import run_layout_preview
from .worker import close_cached_docs, load_results, page_lines


def detect_gpus() -> int:
    try:
        res = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
        return len([ln for ln in res.stdout.strip().splitlines() if ln.strip()])
    except Exception:
        return 0


def parse_pages(spec: str, total_pages: int) -> list[int]:
    """Convert '1-10,15' (1-indexed) into sorted 0-indexed page indices."""
    if not spec.strip():
        return list(range(total_pages))
    out = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out.extend(range(int(a) - 1, min(int(b or a), total_pages)))
    return sorted(set(out))


class OCRPipeline:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.txt_dir.mkdir(parents=True, exist_ok=True)

        self.num_gpus = cfg.num_gpus if cfg.num_gpus is not None else detect_gpus()
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs detected. Ensure CUDA/GPU is available or set num_gpus.")

    def prefetch_model(self) -> None:
        """Download model weights once prior to spawning multi-GPU workers."""
        if self.cfg.engine == "hunyuan":
            from huggingface_hub import snapshot_download
            print(f"Prefetching model weights for {self.cfg.model}...")
            snapshot_download(self.cfg.model, ignore_patterns=["v1.0/*"])

    def discover_books(self) -> dict[str, tuple[str, list[int]]]:
        """Scan input_dir for PDFs and build tasks.jsonl."""
        import pypdfium2 as pdfium

        books: dict[str, tuple[str, list[int]]] = {}
        pdfs = sorted(self.cfg.input_dir.rglob("*.pdf"))
        for p in pdfs:
            if "__MACOSX" in str(p):
                continue
            b = re.sub(r"[^\w\-]+", "_", p.stem)
            while b in books:
                b += "_"
            doc = pdfium.PdfDocument(str(p))
            selected = parse_pages(self.cfg.pages, len(doc))
            doc.close()
            books[b] = (str(p), selected)
            print(f"{b:<50} {len(selected):5d} pages")

        if not books:
            raise FileNotFoundError(f"No PDF files found in {self.cfg.input_dir}")

        with open(self.cfg.tasks_file, "w", encoding="utf-8") as f:
            for b, (src, sel) in books.items():
                for pg in sel:
                    f.write(json.dumps({"book": b, "src": src, "page": pg}, ensure_ascii=False) + "\n")

        total = sum(len(s) for _, s in books.values())
        print(f"Total: {total} pages across {self.num_gpus} GPU(s)")
        return books

    def restore_previous_results(self, books: dict[str, tuple[str, list[int]]]) -> int:
        """Copy prior shard JSONL/log files into raw_dir so completed pages are skipped."""
        if not self.cfg.prev_results_dir or not self.cfg.prev_results_dir.exists():
            return 0

        n_prev = 0
        for f in self.cfg.prev_results_dir.rglob("*"):
            if (
                f.is_file()
                and f.parent.name in books
                and f.name.startswith(self.cfg.engine)
                and f.suffix in (".jsonl", ".log")
            ):
                dst = self.cfg.raw_dir / f.parent.name / f"{f.stem}_prev{f.suffix}"
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, dst)
                    n_prev += 1
        print(f"Restored {n_prev} previous result file(s)")
        return n_prev

    def run_workers(
        self,
        tasks_file: Optional[Path] = None,
        name: Optional[str] = None,
        extra_args: list[str] | tuple[str, ...] = (),
        limit: Optional[int] = None,
        budget_hours: Optional[float] = None,
    ) -> None:
        """Spawn 1 worker subprocess per GPU, monitor progress, and auto-restart stalled workers."""
        tasks_path = str(tasks_file or self.cfg.tasks_file)
        run_name = name or self.cfg.engine
        budget = budget_hours if budget_hours is not None else self.cfg.budget_hours

        engine_args = [
            "--model", self.cfg.model,
            "--prompt", self.cfg.prompt,
            "--dtype", self.cfg.dtype,
            "--rep-penalty", str(self.cfg.repetition_penalty),
            "--paddle-version", self.cfg.paddle_version,
            "--paddle-args", json.dumps(self.cfg.paddle_args),
            "--page-timeout", str(self.cfg.page_timeout),
        ]

        jobs = []
        for k in range(self.num_gpus):
            cmd = [
                sys.executable, "-m", "ocr_pipeline.worker",
                "--engine", self.cfg.engine,
                "--name", run_name,
                "--tasks", tasks_path,
                "--out", str(self.cfg.raw_dir),
                "--shard", str(k),
                "--nshard", str(self.num_gpus),
                "--dpi", str(self.cfg.dpi),
                "--max-side", str(self.cfg.max_side),
                "--max-new", str(self.cfg.max_new_tokens),
                *engine_args,
                *extra_args,
            ]
            if limit is not None:
                cmd += ["--limit", str(limit)]

            log = self.cfg.raw_dir / f"_{run_name}_{k}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}
            log_fh = open(log, "a", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            jobs.append({"cmd": cmd, "env": env, "log": log, "fh": log_fh, "restart": 0, "p": proc})

        t0 = t_print = time.time()
        try:
            while any(j["p"].poll() is None for j in jobs):
                time.sleep(30)
                if time.time() - t0 > budget * 3600:
                    print(f"!!! Time budget ({budget}h) reached -> terminating workers, saving completed pages")
                    for j in jobs:
                        j["p"].terminate()
                    break

                for j in jobs:
                    if (
                        j["p"].poll() is None
                        and time.time() - os.path.getmtime(j["log"]) > self.cfg.stall_minutes * 60
                        and j["restart"] < 5
                    ):
                        j["p"].kill()
                        j["p"].wait()
                        j["restart"] += 1
                        j["p"] = subprocess.Popen(j["cmd"], stdout=j["fh"], stderr=subprocess.STDOUT, env=j["env"])
                        print(f"!!! {j['log'].name}: stalled -> restarted worker (attempt {j['restart']}/5)")

                if time.time() - t_print > 120:
                    t_print = time.time()
                    last = []
                    for j in jobs:
                        lines = j["log"].read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                        last.append((lines[-1] if lines else "...")[-120:])
                    print(f"[{(time.time() - t0) / 60:5.1f} min] " + " | ".join(last), flush=True)

            for j in jobs:
                rc = j["p"].wait()
                if rc not in (0, -15):
                    print(f"\n!!! {j['log'].name} exited with code {rc}. Last 20 lines:")
                    lines = j["log"].read_text(encoding="utf-8", errors="ignore").splitlines()
                    print("\n".join(lines[-20:]))
        finally:
            for j in jobs:
                j["fh"].close()

    def preview_layout(self, books: dict[str, tuple[str, list[int]]]) -> list[Path]:
        """Generate layout bounding-box preview images on sample pages."""
        return run_layout_preview(self, books)

    def test_run(self, books: dict[str, tuple[str, list[int]]], pages_per_gpu: int = 1) -> None:
        """Run a quick test pass and print snippets of extracted text."""
        self.run_workers(limit=pages_per_gpu, budget_hours=1.0)
        for b in books:
            for r in list(load_results(self.cfg.raw_dir, b, self.cfg.engine).values())[:1]:
                print(f"--- {b} — p.{r['page'] + 1} ({r['sec']}s) ---\n{r['text'][:500]}\n")

    def export_txt(self, books: dict[str, tuple[str, list[int]]], create_zip: bool = True) -> Optional[Path]:
        """Assemble recognized pages into per-book .txt files and optionally archive into txt.zip."""
        for b, (_, sel) in books.items():
            recs = load_results(self.cfg.raw_dir, b, self.cfg.engine)
            pages = ["\n".join(page_lines(recs[p])) for p in sel if p in recs]
            out_txt = self.cfg.txt_dir / f"{b}.txt"
            out_txt.write_text("\n\n".join(p for p in pages if p) + "\n", encoding="utf-8")
            miss = len(sel) - len(pages)
            print(f"{b:<50} {len(pages)}/{len(sel)} pages" + (f"  -- MISSING {miss} pages" if miss else ""))

        close_cached_docs()
        if create_zip:
            zip_path = shutil.make_archive(str(self.cfg.work_dir / "txt"), "zip", self.cfg.txt_dir)
            print(f"\nSaved output text files to {self.cfg.txt_dir} and archive to {zip_path}")
            return Path(zip_path)
        return None

    def run_all(self, preview_layout: bool = False, test_first: bool = False) -> Optional[Path]:
        """Execute the full end-to-end OCR pipeline."""
        self.prefetch_model()
        books = self.discover_books()
        self.restore_previous_results(books)

        if preview_layout:
            self.preview_layout(books)
        if test_first:
            self.test_run(books, pages_per_gpu=1)

        self.run_workers()
        return self.export_txt(books, create_zip=True)