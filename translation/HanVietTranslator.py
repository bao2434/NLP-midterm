"""Class dịch Hán -> Việt bằng Transformers; giao diện tương tự OCR_Paddle.py.

Chỉ cần import từ package::

    from han_viet_translator import HanVietTranslator

    tool = HanVietTranslator(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_per_model=2,  # Kaggle T4 x2; model 7B trải trên 2 GPU
        batch=8,
        chunk_chars=300,
        max_new_tokens=1200,
    )

    # Dịch thử 2 đoạn; lưu lại checkpoint để dùng tiếp.
    result = tool.translate("/kaggle/input", "/kaggle/working", limit=2)

    # Dịch hết: tự bỏ qua những đoạn đã dịch.
    result = tool.translate("/kaggle/input", "/kaggle/working")
    print(result.translated_dir)    # /kaggle/working/dich
    print(result.zip_path)          # /kaggle/working/dich.zip
    print(result.book_stats)        # Thống kê các sách

Trước khi dùng, cài thư viện::

    pip install -U torch transformers accelerate huggingface_hub
    # Nếu load_4bit=True: pip install bitsandbytes

Lưu ý:
- Input là thư mục có file *.txt chứa chữ Hán (tìm cả thư mục con).
- Prompt mặc định theo notebook gốc, chuyên sách Y học cổ truyền.
- Để đổi prompt, truyền ``system_prompt=...`` và ``user_prompt=...`` vào class;
  user_prompt bắt buộc chứa chuỗi ``{text}``.
- Checkpoint ở work_dir/dich_raw; đổi model/prompt thì dùng work_dir mới
  (checkpoint cũ không ghi model/prompt nên không tự nhận biết khác biệt).
- Import package không tự tải model hoặc chạy GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .utils import (
    TranslationConfig,
    TranslationResult,
    build_prompt,
    export_translations,
    import_previous_checkpoints,
    load_translations,
    prepare_books,
    read_jsonl,
    write_jsonl,
)


class HanVietTranslator:
    """Công cụ dịch các file sách chữ Hán sang tiếng Việt bằng LLM.

    Cách tạo (giống ``PaddleOCR_Tool``)::

        translator = HanVietTranslator(model="Qwen/Qwen2.5-7B-Instruct", batch=4)
        result = translator.translate(input_dir="./ocr_txt", work_dir="./output")

    Có thể truyền trực tiếp các tham số ``TranslationConfig`` như ``batch``,
    ``load_4bit`` hoặc truyền sẵn một ``config=TranslationConfig(...)``.
    """

    def __init__(self, config: TranslationConfig | None = None, **model_options: Any):
        if config is not None and model_options:
            raise ValueError("Chọn config=TranslationConfig(...) HOẶC các tham số trực tiếp")
        self.config = config if config is not None else TranslationConfig(**model_options)
        self.config.validate()

    def translate(
        self,
        input_dir: str | Path = "/kaggle/input",
        work_dir: str | Path = "/kaggle/working",
        *,
        limit: int | None = None,
        budget_h: float | None = None,
    ) -> TranslationResult:
        """Dịch các file .txt chữ Hán, lưu checkpoint và xuất kết quả .txt + ZIP.

        Args:
            input_dir: Thư mục chứa *.txt (tìm đệ quy).
            work_dir: Thư mục lưu kết quả, file tạm và checkpoint.
            limit: Chạy thử tối đa N đoạn CHƯA DỊCH trên tất cả các sách.
                   Nếu None: dịch toàn bộ các đoạn còn thiếu.
            budget_h: Giới hạn thời gian chạy worker, đơn vị giờ.
                      Mặc định config.budget_h = 10.5.

        Returns:
            TranslationResult gồm zip_path, translated_dir, book_stats...

        Ví dụ::

            tool.translate("/kaggle/input", "/kaggle/working", limit=2)
            result = tool.translate("/kaggle/input", "/kaggle/working")
            print(result.zip_path)
        """
        if limit is not None and limit < 1:
            raise ValueError("limit phải >= 1 hoặc None")
        budget_h = self.config.budget_h if budget_h is None else budget_h
        if budget_h <= 0:
            raise ValueError("budget_h phải > 0")

        input_dir = Path(input_dir).expanduser().resolve()
        work_dir = Path(work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = work_dir / "dich_raw"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        books = prepare_books(input_dir, self.config.chunk_chars)
        chunks = [row for book in books.values() for row in book]
        prompt_path = work_dir / "prompt.json"
        prompt_path.write_text(
            json.dumps({"system": self.config.system_prompt, "user": self.config.user_prompt},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        chunks_path = work_dir / "chunks.jsonl"
        write_jsonl(chunks_path, chunks)

        # Đọc checkpoint phiên trước, cả trong work_dir và Kaggle Add Input.
        import_previous_checkpoints(input_dir, checkpoint_dir)
        done_ids = set(load_translations(checkpoint_dir))
        pending = [row for row in chunks if row["id"] not in done_ids]
        if limit is not None:
            pending = pending[:limit]
        already_translated = sum(row["id"] in done_ids for row in chunks)
        print(f"Tổng {len(chunks)} đoạn | đã dịch {already_translated}"
              f" | cần dịch lượt này {len(pending)}")

        errors: list[str] = []
        if pending:
            gpu_count = _count_gpus()
            if gpu_count < self.config.gpu_per_model:
                raise RuntimeError(
                    f"Cần {self.config.gpu_per_model} GPU, nhưng chỉ phát hiện {gpu_count}. "
                    "Kaggle: Settings > Accelerator > GPU T4 x2. "
                    "Hoặc chọn model nhỏ hơn với gpu_per_model=1."
                )
            _setup_hf_token()
            if self.config.predownload_model:
                _download_model(self.config.model)

            pending_path = work_dir / "dich_pending.jsonl"
            write_jsonl(pending_path, pending)
            errors = _launch_workers(
                pending_path, prompt_path, checkpoint_dir, self.config,
                budget_h, gpu_count,
            )
        else:
            print("Không còn đoạn cần dịch; dùng checkpoint hiện có.")

        result = export_translations(books, checkpoint_dir, work_dir)
        result.worker_errors.extend(errors)
        return result

    def translate_books(
        self,
        input_dir: str | Path = "/kaggle/input",
        work_dir: str | Path = "/kaggle/working",
        *,
        limit: int | None = None,
        budget_h: float | None = None,
    ) -> TranslationResult:
        """Tên thay thế cho ``translate()`` nếu muốn rõ đang dịch nhiều file."""
        return self.translate(input_dir, work_dir, limit=limit, budget_h=budget_h)

    def export_only(
        self,
        input_dir: str | Path = "/kaggle/input",
        work_dir: str | Path = "/kaggle/working",
    ) -> TranslationResult:
        """Xuất lại sách từ checkpoint đã có, không cần CUDA/LLM.

        Phải giữ ``chunk_chars`` giống lúc dịch để ID đoạn khớp.
        """
        input_dir, work_dir = Path(input_dir), Path(work_dir)
        books = prepare_books(input_dir, self.config.chunk_chars)
        checkpoint_dir = work_dir / "dich_raw"
        import_previous_checkpoints(input_dir, checkpoint_dir)
        return export_translations(books, checkpoint_dir, work_dir)


def _count_gpus() -> int:
    """Đếm GPU khả dụng qua nvidia-smi mà không import torch khi chỉ inspect."""
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 0
    if proc.returncode:
        return 0
    return sum(line.lstrip().startswith("GPU ") for line in proc.stdout.splitlines())


def _setup_hf_token() -> None:
    """Lấy HF_TOKEN từ Kaggle Secrets khi người dùng chưa đặt trong env."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass


def _download_model(name: str) -> None:
    """Cache model trước để worker không tải trùng nhau."""
    from huggingface_hub import snapshot_download
    print(f"Kiểm tra / tải model: {name}")
    snapshot_download(name, ignore_patterns=["*.gguf", "*.pth", "*.onnx", "original/*"])


def _launch_workers(
    chunks_path: Path,
    prompt_path: Path,
    checkpoint_dir: Path,
    config: TranslationConfig,
    budget_h: float,
    gpu_count: int,
) -> list[str]:
    """Mỗi worker chạy một model trên nhóm GPU riêng, ghi checkpoint JSONL.

    Dùng ``python -c '... worker_main()'`` để import bằng đường dẫn package,
    không cần tạo file dich_worker.py riêng như notebook gốc.
    """
    count = gpu_count // config.gpu_per_model
    package_parent = str(Path(__file__).resolve().parent.parent)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(p for p in (package_parent, inherited_pythonpath) if p)
    jobs: list[tuple[Path, subprocess.Popen[Any]]] = []
    errors: list[str] = []
    stopped_for_budget = False

    try:
        for shard in range(count):
            visible_devices = ",".join(
                str(i) for i in range(shard * config.gpu_per_model, (shard + 1) * config.gpu_per_model)
            )
            command = [
                sys.executable, "-c",
                "from han_viet_translator.HanVietTranslator import worker_main; worker_main()",
                "--model", config.model,
                "--chunks", str(chunks_path),
                "--prompt", str(prompt_path),
                "--out", str(checkpoint_dir),
                "--shard", str(shard), "--nshard", str(count),
                "--batch", str(config.batch),
                "--max-new", str(config.max_new_tokens),
                "--temperature", str(config.temperature),
                "--rep-penalty", str(config.repetition_penalty),
            ]
            if config.load_4bit:
                command.append("--load-4bit")
            log_path = checkpoint_dir / f"_dich_{shard}.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": visible_devices, "PYTHONPATH": pythonpath},
                )
            jobs.append((log_path, process))

        start = last_report = time.monotonic()
        while any(process.poll() is None for _, process in jobs):
            time.sleep(5)
            elapsed = time.monotonic() - start
            if elapsed > budget_h * 3600:
                stopped_for_budget = True
                print(f"Hết thời gian {budget_h} giờ; dừng worker, giữ checkpoint đã ghi.")
                for _, process in jobs:
                    if process.poll() is None:
                        process.terminate()
                break
            if time.monotonic() - last_report > 120:
                last_report = time.monotonic()
                for log_path, _ in jobs:
                    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    print("\n".join(lines[-3:]) or "Đang tải model...", flush=True)
    finally:
        # Hủy tiến trình GPU nếu notebook bị interrupt, không để worker chạy ngầm.
        for _, process in jobs:
            if process.poll() is None:
                process.terminate()
        for log_path, process in jobs:
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            if code != 0 and not (stopped_for_budget and code == -15):
                message = f"Worker {log_path.name} kết thúc với mã {code}"
                errors.append(message)
                print(f"CẢNH BÁO: {message}")
                print("\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-20:]))
    return errors


def worker_main() -> None:
    """Entry point riêng cho worker subprocess; không gọi trực tiếp từ notebook."""
    parser = argparse.ArgumentParser(description="Worker dịch Hán-Việt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--rep-penalty", type=float, default=1.05)
    parser.add_argument("--load-4bit", action="store_true")
    _worker(parser.parse_args())


def _worker(args: argparse.Namespace) -> None:
    """Một worker: load model, dịch batch và append checkpoint từng batch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(args.out)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(read_jsonl(args.chunks))
    done_ids = set(load_translations(checkpoint_dir))
    todo = [
        row for index, row in enumerate(chunks)
        if index % args.nshard == args.shard and row.get("id") not in done_ids
    ]
    print(f"[dich#{args.shard}] còn {len(todo)} đoạn", flush=True)
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    options: dict[str, Any] = {"dtype": torch.float32}
    if torch.cuda.is_available():
        options = {"dtype": torch.float16, "device_map": "auto"}
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
    model = AutoModelForCausalLM.from_pretrained(args.model, **options).eval()
    prompt = json.loads(Path(args.prompt).read_text(encoding="utf-8"))

    # Nhóm các đoạn dài với nhau để hạn chế padding, tiết kiệm VRAM.
    todo.sort(key=lambda row: len(row["zh"]), reverse=True)
    start = time.monotonic()
    n_translated = 0
    with (checkpoint_dir / f"dich_shard{args.shard}.jsonl").open("a", encoding="utf-8") as out:
        for begin in range(0, len(todo), args.batch):
            batch = todo[begin:begin + args.batch]
            prompts = [
                build_prompt(tokenizer, prompt["system"], prompt["user"].format(text=row["zh"]))
                for row in batch
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False,
            ).to(model.device)
            generation_options = (
                {"do_sample": True, "temperature": args.temperature}
                if args.temperature > 0 else {"do_sample": False}
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=args.max_new,
                    repetition_penalty=args.rep_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_options,
                )
            generated = generated[:, encoded["input_ids"].shape[1]:]

            last_vi = ""
            for row, tokens in zip(batch, generated):
                # EOS có thể bằng PAD: kiểm tra EOS thay vì chỉ đếm token != PAD.
                raw = tokens.tolist()
                eos = model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
                ended = any(t in eos_ids for t in raw)
                hit_max = not ended and len(raw) >= args.max_new
                last_vi = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                out.write(json.dumps(
                    {"id": row["id"], "vi": last_vi, "hit_max": hit_max},
                    ensure_ascii=False,
                ) + "\n")
            out.flush()  # Chạy lại sẽ tự bỏ qua batch đã ghi.

            n_translated += len(batch)
            remaining = (time.monotonic() - start) / n_translated * (len(todo) - n_translated) / 60
            print(
                f"[dich#{args.shard}] {n_translated}/{len(todo)} đoạn | còn ~{remaining:.0f} phút\n"
                f"   ZH: {batch[-1]['zh'][:60]}\n   VI: {last_vi[:120]}",
                flush=True,
            )
