"""PaddleOCR-VL/Hunyuan OCR module extracted from the original notebook.

The module consolidates the notebook's OCR worker, PDF workflow, GPU sharding,
resume/skip logic, layout visualization, and TXT export into PaddleOCR_Tool.

Typical usage:
    from paddle_ocr_tool import PaddleOCR_Tool

    tool = PaddleOCR_Tool(
        engine="paddle",
        paddle_version="v1.6",
        paddle_args={},
        dpi=200,
        max_side=1600,
        max_new_tokens=6000,
    )
    books = tool.discover("/kaggle/input")
    tool.run(books, output_dir="/kaggle/working/ocr_raw")
    tool.export_txt(books, output_dir="/kaggle/working/ocr_raw",
                    txt_dir="/kaggle/working/txt")

For a single image:
    text, metadata = tool.ocr_image("page.png")

"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from . import utils


class PaddleOCR_Tool:
    """OCR engine and complete PDF-to-TXT workflow.

    Supports PaddleOCR-VL and HunyuanOCR, matching the notebook's two engines.

    Args:
        engine: ``"paddle"`` or ``"hunyuan"``.
        paddle_version: PaddleOCR-VL pipeline version (v1.5 or v1.6).
        paddle_args: Extra keyword arguments passed to PaddleOCRVL.
        max_new_tokens: Generation token budget per page.
        dpi: PDF render DPI.
        max_side: Maximum image edge after rendering.
        model: Hunyuan model name.
        dtype: Hunyuan dtype (fp32/fp16/bf16).
        prompt: Optional Hunyuan OCR prompt override.
        repetition_penalty: Hunyuan repetition penalty.
        ignore_labels: Paddle layout labels to omit from extracted text.
    """

    def __init__(
        self,
        engine: str = "paddle",
        paddle_version: str = "v1.6",
        paddle_args: dict[str, Any] | str | None = None,
        max_new_tokens: int = 6000,
        dpi: int = 200,
        max_side: int = 1600,
        model: str = "tencent/HunyuanOCR",
        dtype: str = "fp32",
        prompt: str = "",
        repetition_penalty: float = 1.08,
        ignore_labels: set[str] | None = None,
    ) -> None:
        if engine not in {"paddle", "hunyuan"}:
            raise ValueError("engine must be 'paddle' or 'hunyuan'")
        if dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("dtype must be fp32, fp16, or bf16")

        if isinstance(paddle_args, str):
            try:
                paddle_args = json.loads(paddle_args)
            except json.JSONDecodeError as exc:
                raise ValueError("paddle_args must be valid JSON") from exc
        if paddle_args is None:
            paddle_args = {}
        if not isinstance(paddle_args, dict):
            raise TypeError("paddle_args must be a dictionary or JSON string")

        self.engine_name = engine
        self.paddle_version = paddle_version
        self.paddle_args = paddle_args
        self.max_new_tokens = max_new_tokens
        self.dpi = dpi
        self.max_side = max_side
        self.model_name = model
        self.dtype = dtype
        self.prompt = prompt or utils.HUNYUAN_PROMPT
        self.repetition_penalty = repetition_penalty
        self.ignore_labels = (
            set(ignore_labels) if ignore_labels is not None
            else set(utils.PADDLE_IGNORE)
        )
        self.engine = None

    # ------------------------------------------------------------------
    # Model initialization and single-image inference
    # ------------------------------------------------------------------
    def initialize(self, cache_dir: str | Path = './cache'):
        """Load the selected OCR model once and return the engine instance."""
        if self.engine is not None:
            return self.engine

        if self.engine_name == "paddle":
            try:
                cache_path = Path(cache_dir).resolve()
                cache_path.mkdir(parents=True, exist_ok=True)
                os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_path)
                from paddleocr import PaddleOCRVL
            except ImportError as exc:
                raise ImportError(
                    'Install PaddleOCR-VL with: pip install "paddleocr[doc-parser]"'
                ) from exc
            self.engine = _PaddleBackend(
                PaddleOCRVL,
                pipeline_version=self.paddle_version,
                paddle_args=self.paddle_args,
                max_new_tokens=self.max_new_tokens,
                ignore_labels=self.ignore_labels,
            )
        else:
            self.engine = _HunyuanBackend(
                model_name=self.model_name,
                dtype=self.dtype,
                max_new_tokens=self.max_new_tokens,
                prompt=self.prompt,
                repetition_penalty=self.repetition_penalty,
                cache_dir=cache_dir
            )
        return self.engine

    def ocr_image(self, image_path: str | Path, **predict_kwargs):
        """Run OCR on one image and return ``(text, metadata)``."""
        return self.initialize().run(str(image_path), **predict_kwargs)

    __call__ = ocr_image

    def parse(self, result: dict[str, Any]):
        """Parse a PaddleOCR-VL result JSON object."""
        return _PaddleBackend.parse(result, self.ignore_labels)

    # ------------------------------------------------------------------
    # PDF discovery, task creation, and workflow
    # ------------------------------------------------------------------
    def discover(
        self,
        roots: str | Path | list[str | Path],
        pages: str = "",
        recursive: bool = True,
    ) -> dict[str, tuple[str, list[int]]]:
        """Find PDFs; return book -> (PDF path, selected zero-based page indices)."""
        return utils.discover_pdfs(roots, page_spec=pages, recursive=recursive)

    def create_tasks(
        self,
        books: dict[str, tuple[str, list[int]]],
        task_file: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        tasks = utils.make_tasks(books)
        if task_file is not None:
            utils.write_tasks(tasks, task_file)
        return tasks

    def import_previous_results(
        self,
        previous_roots: str | Path | list[str | Path],
        books: dict[str, tuple[str, list[int]]],
        output_dir: str | Path = "ocr_raw",
        name: str | None = None,
    ) -> int:
        """Copy earlier OCR shard/log files into the current output directory.

        This mirrors the notebook workflow where a previous Kaggle Output is
        added as an Input before rerunning. Existing destination copies are
        left untouched.
        """
        import shutil

        if isinstance(previous_roots, (str, Path)):
            previous_roots = [previous_roots]
        output_dir = Path(output_dir)
        name = name or self.engine_name
        copied = 0

        for root in previous_roots:
            root = Path(root)
            if not root.exists():
                continue
            for source in root.glob("**/ocr_raw/*/*"):
                if source.parent.name not in books:
                    continue
                if not source.name.startswith(name):
                    continue
                if source.suffix not in {".jsonl", ".log"}:
                    continue
                destination_dir = output_dir / source.parent.name
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / (
                    source.stem + "_prev" + source.suffix
                )
                if not destination.exists():
                    shutil.copy2(source, destination)
                    copied += 1
        print(f"Imported {copied} previous OCR result/log files.")
        return copied

    def run(
        self,
        books: dict[str, tuple[str, list[int]]] | None = None,
        tasks: list[dict[str, Any]] | str | Path | None = None,
        output_dir: str | Path = "ocr_raw",
        name: str | None = None,
        num_gpus: int | None = None,
        budget_hours: float | None = None,
        limit: int | None = None,
        page_timeout: int = 300,
        stall_minutes: int = 15,
        max_restarts: int = 5,
        workers_use_subprocess: bool = True,
    ) -> None:
        """OCR the selected PDF pages.

        By default, one subprocess is launched per visible GPU. Each worker is
        pinned to one GPU, splits tasks by shard, resumes completed pages, and
        records timed-out pages in skip logs. ``workers_use_subprocess=False``
        runs sequentially in the current process (one GPU/model).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = name or self.engine_name

        if tasks is None:
            if books is None:
                raise ValueError("Pass either books or tasks")
            tasks = utils.make_tasks(books)
        elif isinstance(tasks, (str, Path)):
            tasks = utils.read_tasks(tasks)

        if not tasks:
            print("No pages to OCR.")
            return

        gpu_total = num_gpus if num_gpus is not None else utils.gpu_count()
        if workers_use_subprocess and gpu_total > 1:
            self._run_subprocess_workers(
                tasks, output_dir, name, gpu_total, budget_hours,
                limit, page_timeout, stall_minutes, max_restarts,
            )
        else:
            self._run_local(
                tasks, output_dir, name, limit=limit,
                page_timeout=page_timeout, budget_hours=budget_hours,
            )

    def _worker_command(
        self,
        tasks_file: Path,
        output_dir: Path,
        name: str,
        shard: int,
        shard_count: int,
        page_timeout: int,
    ) -> list[str]:
        # Child process reconstructs this exact object from JSON config.
        config = {
            "engine": self.engine_name,
            "paddle_version": self.paddle_version,
            "paddle_args": self.paddle_args,
            "max_new_tokens": self.max_new_tokens,
            "dpi": self.dpi,
            "max_side": self.max_side,
            "model": self.model_name,
            "dtype": self.dtype,
            "prompt": self.prompt,
            "repetition_penalty": self.repetition_penalty,
            "ignore_labels": sorted(self.ignore_labels),
        }
        config_path = output_dir / f"_{name}_worker_config_{shard}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--config", str(config_path),
            "--tasks", str(tasks_file),
            "--out", str(output_dir),
            "--name", name,
            "--shard", str(shard),
            "--nshard", str(shard_count),
            "--page-timeout", str(page_timeout),
        ]

    def _run_subprocess_workers(
        self, tasks, output_dir, name, gpu_total, budget_hours,
        limit, page_timeout, stall_minutes, max_restarts,
    ):
        tasks_file = output_dir / f"_{name}_tasks.jsonl"
        utils.write_tasks(tasks, tasks_file)
        jobs = []
        for shard in range(gpu_total):
            command = self._worker_command(
                tasks_file, output_dir, name, shard, gpu_total, page_timeout
            )
            log = output_dir / f"_{name}_{shard}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)}
            if limit is not None:
                command += ["--limit", str(limit)]
            process = subprocess.Popen(
                command, stdout=log.open("a", encoding="utf-8"),
                stderr=subprocess.STDOUT, env=env,
            )
            jobs.append({
                "command": command, "env": env, "log": log,
                "process": process, "restarts": 0,
            })

        started = last_report = time.time()
        while any(job["process"].poll() is None for job in jobs):
            time.sleep(30)
            if budget_hours is not None and time.time() - started > budget_hours * 3600:
                print(f"Time budget ({budget_hours} h) reached; stopping workers.")
                for job in jobs:
                    if job["process"].poll() is None:
                        job["process"].terminate()
                break

            for job in jobs:
                process = job["process"]
                if process.poll() is not None:
                    continue
                age = time.time() - job["log"].stat().st_mtime if job["log"].exists() else 0
                if (
                    age > stall_minutes * 60
                    and job["restarts"] < max_restarts
                ):
                    process.kill()
                    process.wait()
                    job["restarts"] += 1
                    log_handle = job["log"].open("a", encoding="utf-8")
                    job["process"] = subprocess.Popen(
                        job["command"], stdout=log_handle,
                        stderr=subprocess.STDOUT, env=job["env"],
                    )
                    print(
                        f"{job['log'].name} stalled; restarted worker "
                        f"({job['restarts']}/{max_restarts})."
                    )

            if time.time() - last_report >= 120:
                last_report = time.time()
                tails = []
                for job in jobs:
                    try:
                        lines = job["log"].read_text(
                            encoding="utf-8", errors="ignore"
                        ).splitlines()
                        tails.append(lines[-1][-120:] if lines else "...")
                    except OSError:
                        tails.append("...")
                print(f"[{(time.time() - started) / 60:.1f} min] " + " | ".join(tails))

        for job in jobs:
            code = job["process"].wait()
            if code not in (0, -15, 1):
                print(f"Worker failed: {job['log']}")
                try:
                    print("\n".join(job["log"].read_text(errors="ignore").splitlines()[-20:]))
                except OSError:
                    pass

    def _run_local(
        self, tasks, output_dir, name, limit=None,
        page_timeout=300, budget_hours=None,
    ):
        shard_tasks = tasks[:limit] if limit is not None else tasks
        engine = self.initialize()
        started = time.time()
        for index, task in enumerate(shard_tasks, 1):
            book, page = task["book"], int(task["page"])
            folder = output_dir / book
            folder.mkdir(parents=True, exist_ok=True)
            if page in utils.load_results(output_dir, book, name):
                continue
            if page in utils.load_skipped(output_dir, book, name):
                continue

            begin = time.time()
            try:
                image = utils.render_pdf_page(
                    task["src"], page, self.dpi, self.max_side
                )
                with tempfile.TemporaryDirectory() as temp:
                    image_path = Path(temp) / "page.png"
                    image.save(image_path)
                    text, metadata = _run_with_timeout(
                        lambda: engine.run(str(image_path)),
                        page_timeout,
                    )
                record = {
                    "page": page,
                    "text": text,
                    "w": image.width,
                    "h": image.height,
                    "sec": round(time.time() - begin, 1),
                    **metadata,
                }
                with utils.result_file(output_dir, book, name, 0).open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                eta = (time.time() - started) / index * (len(shard_tasks) - index) / 60
                print(
                    f"[{index}/{len(shard_tasks)}] {book} page {page + 1}: "
                    f"{record['sec']}s, {len(text)} chars; ETA ~{eta:.0f} min"
                )
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    with (folder / f"{name}_skip.log").open("a") as stream:
                        stream.write(f"{page}\n")
                print(f"{book} page {page + 1}: {type(exc).__name__}: {str(exc)[:200]}")
            if budget_hours is not None and time.time() - started > budget_hours * 3600:
                print(f"Time budget ({budget_hours} h) reached.")
                break
        utils.close_pdf_documents()

    def export_txt(
        self,
        books: dict[str, tuple[str, list[int]]],
        output_dir: str | Path = "ocr_raw",
        txt_dir: str | Path = "txt",
        name: str | None = None,
        create_zip: bool = True,
    ) -> dict[str, dict[str, int]]:
        """Merge shard results into one UTF-8 TXT file per book."""
        import shutil

        output_dir, txt_dir = Path(output_dir), Path(txt_dir)
        txt_dir.mkdir(parents=True, exist_ok=True)
        name = name or self.engine_name
        summary = {}

        for book, (_, selected_pages) in books.items():
            records = utils.load_results(output_dir, book, name)
            page_texts = [
                "\n".join(utils.page_lines(records[page]))
                for page in selected_pages if page in records
            ]
            (txt_dir / f"{book}.txt").write_text(
                "\n\n".join(text for text in page_texts if text) + "\n",
                encoding="utf-8",
            )
            missing = len(selected_pages) - len(page_texts)
            summary[book] = {
                "completed": len(page_texts),
                "total": len(selected_pages),
                "missing": missing,
            }
            suffix = f" -- MISSING {missing} pages" if missing else ""
            print(f"{book:<50} {len(page_texts)}/{len(selected_pages)} pages{suffix}")

        if create_zip:
            shutil.make_archive(str(txt_dir), "zip", txt_dir)
        return summary

    def demo_pages(
        self,
        books: dict[str, tuple[str, list[int]]],
        count: int = 3,
    ) -> list[dict[str, Any]]:
        """Select interior pages across books for layout visualization."""
        selected = []
        for fraction in (1 / 3, 1 / 2, 2 / 3, 1 / 4, 3 / 4):
            for book, (source, pages) in books.items():
                task = (
                    {"book": book, "src": source,
                     "page": pages[int(len(pages) * fraction)]}
                    if pages else None
                )
                if task and task not in selected and len(selected) < count:
                    selected.append(task)
        return selected

    def visualize_layout(
        self,
        task: dict[str, Any],
        record: dict[str, Any],
        output_dir: str | Path = "layout",
        show: bool = False,
    ):
        """Draw the Paddle layout detector's boxes for one OCR'd page."""
        image = utils.render_pdf_page(
            task["src"], task["page"], self.dpi, self.max_side
        )
        boxes = sorted(
            record.get("boxes", []),
            key=lambda box: (
                box.get("order") is None,
                box.get("order") or 0,
            ),
        )
        destination = (
            Path(output_dir)
            / f"{self.engine_name}_{task['book']}_tr{task['page'] + 1}.png"
        )
        image = utils.draw_layout_boxes(image, boxes, destination)
        print(
            f"{task['book']} — page {task['page'] + 1}: "
            f"{len(boxes)} layout boxes"
        )
        if show:
            try:
                from IPython.display import display
                display(image)
            except ImportError:
                pass
        return image

    def close(self):
        """Release cached PDF handles and the loaded OCR model."""
        utils.close_pdf_documents()
        self.engine = None


class _PaddleBackend:
    def __init__(
        self, paddle_class, pipeline_version, paddle_args,
        max_new_tokens, ignore_labels,
    ):
        self.pipeline = paddle_class(
            pipeline_version=pipeline_version, **paddle_args
        )
        self.max_new_tokens = max_new_tokens
        self.ignore_labels = ignore_labels

    def run(self, image_path: str, **kwargs):
        try:
            results = list(self.pipeline.predict(
                image_path,
                max_new_tokens=kwargs.pop("max_new_tokens", self.max_new_tokens),
                **kwargs,
            ))
        except TypeError:
            # Older PaddleOCR versions may not accept max_new_tokens.
            kwargs.pop("max_new_tokens", None)
            results = list(self.pipeline.predict(image_path, **kwargs))
        if not results:
            raise RuntimeError("PaddleOCR-VL returned no result")
        raw = results[0].json
        if callable(raw):
            raw = raw()
        return self.parse(raw, self.ignore_labels)

    @staticmethod
    def parse(result: dict[str, Any], ignore_labels=None):
        """Convert PaddleOCR-VL JSON into text, parsed blocks, and layout boxes."""
        ignore_labels = set(utils.PADDLE_IGNORE if ignore_labels is None else ignore_labels)
        data = result.get("res", result) if isinstance(result, dict) else {}

        blocks = [
            {
                "label": block.get("block_label"),
                "box": [
                    round(value)
                    for value in block.get("block_bbox", [0, 0, 0, 0])
                ],
                "order": block.get("block_order"),
                "text": utils.clean_degenerate(
                    block.get("block_content") or ""
                ),
            }
            for block in data.get("parsing_res_list", [])
            if block.get("block_label") not in ignore_labels
        ]
        blocks = [block for block in blocks if block["text"]]
        text = "\n".join(utils.page_lines({"blocks": blocks}))

        layout_boxes = (data.get("layout_det_res") or {}).get("boxes", [])
        boxes = [
            {
                "label": box.get("label"),
                "box": [
                    round(value)
                    for value in box.get("coordinate", [])
                ],
                "order": box.get("order"),
                "score": round(box.get("score", 0), 2),
            }
            for box in layout_boxes
        ]
        return text, {"blocks": blocks, "boxes": boxes}


class _HunyuanBackend:
    def __init__(
        self, model_name, dtype, max_new_tokens, prompt, repetition_penalty, cache_dir
    ):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import HunYuanVLForConditionalGeneration as Model
        except ImportError:
            from transformers import AutoModelForImageTextToText as Model

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[dtype]
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False, cache_dir=cache_dir
        )
        self.model = Model.from_pretrained(
            model_name, dtype=torch_dtype, trust_remote_code=True, cache_dir=cache_dir
        ).to(self.device).eval()

    def run(self, image_path: str, **kwargs):
        torch = self.torch
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": self.prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]
        return utils.clean_degenerate(text), {
            "hit_max": int(generated.shape[1]) >= self.max_new_tokens
        }


def _run_with_timeout(function, seconds: int):
    """Run a function with SIGALRM timeout on platforms that support it."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return function()

    def timeout_handler(signum, frame):
        raise TimeoutError(f"page exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        return function()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _worker_main(args):
    """Internal subprocess worker; invoked by this module's --worker CLI."""
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tool = PaddleOCR_Tool(**config)
    engine = tool.initialize()
    tasks = utils.read_tasks(args.tasks)
    mine = [
        task for index, task in enumerate(tasks)
        if index % args.nshard == args.shard
    ]
    name, output = args.name, Path(args.out)
    completed: set[tuple[str, int]] = set()
    skipped: set[tuple[str, int]] = set()
    for book in {task["book"] for task in mine}:
        completed.update(
            (book, page) for page in utils.load_results(output, book, name)
        )
        skipped.update(
            (book, page) for page in utils.load_skipped(output, book, name)
        )

    # A page left in-flight after a killed worker is skipped, matching notebook behavior.
    inflight = output / f"_inflight_{name}_{args.shard}"
    if inflight.exists():
        previous = inflight.read_text(encoding="utf-8")
        book, separator, page_text = previous.rpartition("|")
        if separator:
            page = int(page_text)
            if (book, page) not in completed:
                folder = output / book
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / f"{name}_skip.log").open("a") as stream:
                    stream.write(f"{page}\n")
                skipped.add((book, page))
                print(f"{book} page {page + 1}: previous in-flight page skipped.")
        inflight.unlink(missing_ok=True)

    todo = [
        task for task in mine
        if (task["book"], int(task["page"])) not in completed | skipped
    ]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(
        f"[{name}#{args.shard}] {len(mine)} assigned, "
        f"{len(todo)} remaining",
        flush=True,
    )

    started = time.time()
    try:
        for index, task in enumerate(todo, 1):
            book, page = task["book"], int(task["page"])
            folder = output / book
            folder.mkdir(parents=True, exist_ok=True)
            begin = time.time()
            inflight.write_text(f"{book}|{page}", encoding="utf-8")
            try:
                image = utils.render_pdf_page(
                    task["src"], page, tool.dpi, tool.max_side
                )
                with tempfile.TemporaryDirectory() as temp:
                    image_path = Path(temp) / "page.png"
                    image.save(image_path)
                    text, extra = _run_with_timeout(
                        lambda: engine.run(str(image_path)),
                        args.page_timeout,
                    )
                record = {
                    "page": page,
                    "text": text,
                    "w": image.width,
                    "h": image.height,
                    "sec": round(time.time() - begin, 1),
                    **extra,
                }
                with utils.result_file(output, book, name, args.shard).open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                eta = (
                    (time.time() - started) / index
                    * (len(todo) - index) / 60
                )
                print(
                    f"[{name}#{args.shard}] {index}/{len(todo)} "
                    f"{book} page {page + 1}: {record['sec']}s, "
                    f"{len(text)} chars | ETA ~{eta:.0f} min",
                    flush=True,
                )
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    with (folder / f"{name}_skip.log").open("a") as stream:
                        stream.write(f"{page}\n")
                print(
                    f"[{name}#{args.shard}] {book} page {page + 1}: "
                    f"{type(exc).__name__}: {str(exc)[:200]}",
                    flush=True,
                )
    finally:
        inflight.unlink(missing_ok=True)
        utils.close_pdf_documents()



