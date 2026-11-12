"""Các hàm hỗ trợ cho HanVietTranslator (không tự tải/chạy LLM).

Không cần gọi trực tiếp nếu chỉ muốn dịch sách; hãy dùng ``HanVietTranslator``.
Có thể dùng riêng để chia đoạn hoặc xuất kết quả từ checkpoint::

    from han_viet_translator.utils import make_chunks, load_translations
    chunks = make_chunks("黃帝曰：...", max_chars=300)
    done = load_translations("/kaggle/working/dich_raw")
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# Prompt gốc của notebook 2_dich_han_viet.ipynb (sách Y học cổ truyền).
DEFAULT_SYSTEM_PROMPT = (
    "Bạn là dịch giả chuyên dịch sách Y học cổ truyền từ chữ Hán "
    "(văn ngôn và bạch thoại) sang tiếng Việt."
)
DEFAULT_USER_PROMPT = (
    "Văn bản chữ Hán:\n{text}\n\n"
    "Dịch toàn bộ văn bản trên sang tiếng Việt, đầy đủ ý, văn phong tự nhiên. "
    "Tên huyệt, vị thuốc, bài thuốc, bệnh danh giữ theo âm Hán-Việt "
    "(vd: 足三里 -> Túc tam lý). Chỉ viết tiếng Việt: không chép lại chữ Hán, "
    "không dịch sang tiếng Anh, không lời dẫn, không giải thích."
)
HAN_PATTERN = re.compile(r"[㐀-鿿\U00020000-\U0002ffff]")


@dataclass
class TranslationConfig:
    """Cấu hình dịch; các giá trị mặc định giống notebook gốc.

    Lưu ý: model 7B FP16 thường cần hai GPU T4; nếu thiếu VRAM,
    hãy giảm ``batch`` hoặc chọn model nhỏ hơn.
    """

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    gpu_per_model: int = 2
    load_4bit: bool = False
    batch: int = 8
    chunk_chars: int = 300
    max_new_tokens: int = 1200
    temperature: float = 0.0
    repetition_penalty: float = 1.05
    budget_h: float = 10.5
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT
    predownload_model: bool = True

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model không được rỗng")
        if min(self.gpu_per_model, self.batch, self.chunk_chars, self.max_new_tokens) < 1:
            raise ValueError("gpu_per_model, batch, chunk_chars, max_new_tokens phải >= 1")
        if self.temperature < 0 or self.repetition_penalty <= 0 or self.budget_h <= 0:
            raise ValueError("temperature >= 0, repetition_penalty > 0, budget_h > 0")
        if "{text}" not in self.user_prompt:
            raise ValueError("user_prompt cần có placeholder {text}")


@dataclass
class TranslationResult:
    """Kết quả trả về từ ``translator.translate()`` hoặc ``export_only()``."""

    translated_dir: Path
    zip_path: Path
    checkpoint_dir: Path
    # {'ten_sach': {'total': N, 'translated': M, 'hit_max': K}}
    book_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    worker_errors: list[str] = field(default_factory=list)


def split_long(line: str, max_chars: int) -> list[str]:
    """Cắt một dòng dài theo dấu câu; đoạn không có dấu câu thì cắt cứng."""
    if max_chars < 1:
        raise ValueError("max_chars phải >= 1")
    parts: list[str] = []
    current = ""
    for segment in re.split(r"(?<=[。！？；!?;])", line):
        while len(segment) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(segment[:max_chars])
            segment = segment[max_chars:]
        if current and len(current) + len(segment) > max_chars:
            parts.append(current)
            current = ""
        current += segment
    return parts + ([current] if current else [])


def make_chunks(text: str, max_chars: int = 300) -> list[str]:
    """Gộp dòng liên tiếp thành các đoạn tối đa ``max_chars`` ký tự.

    Giữ thuật toán gốc để ID đoạn trùng với checkpoint từ notebook.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        for piece in split_long(line, max_chars):
            if current and current_length + len(piece) > max_chars:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
            current.append(piece)
            current_length += len(piece)
    if current:
        chunks.append("\n".join(current))
    return chunks


def prepare_books(input_dir: str | Path, chunk_chars: int = 300) -> dict[str, list[dict[str, str]]]:
    """Tìm đệ quy *.txt chứa >=50 chữ Hán, chia đoạn và gán ID ổn định.

    Hai file trùng tên (khác thư mục) chỉ lấy file đầu tiên như notebook.
    """
    folder = Path(input_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {folder}")

    books: dict[str, list[dict[str, str]]] = {}
    for path in sorted(folder.rglob("*.txt")):
        name = path.stem
        content = path.read_text(encoding="utf-8", errors="ignore")
        if name in books or len(HAN_PATTERN.findall(content)) < 50:
            continue
        books[name] = [
            {"id": f"{name}|{i}|{hashlib.md5(chunk.encode()).hexdigest()[:8]}", "zh": chunk}
            for i, chunk in enumerate(make_chunks(content, chunk_chars))
        ]
        print(f"{name:<50} {len(content):8,} ký tự -> {len(books[name]):5d} đoạn")

    if not books:
        raise ValueError(f"Không có file .txt chữ Hán hợp lệ trong {folder}")
    return books


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Lưu từng dictionary trên một dòng JSON, giữ nguyên chữ Hán/Việt."""
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Đọc JSONL; bỏ qua dòng lỗi (checkpoint bị ngắt khi đang ghi)."""
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_translations(checkpoint_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Đọc checkpoint ``dich_shard*.jsonl`` để resume hoặc ghép bản dịch."""
    translated: dict[str, dict[str, Any]] = {}
    files = list(Path(checkpoint_dir).glob("dich_shard*.jsonl"))
    # File import từ phiên trước đọc trước; file hiện tại ghi đè nếu trùng ID.
    files.sort(key=lambda p: ("_prev" not in p.stem, p.name))
    for path in files:
        for row in read_jsonl(path):
            if isinstance(row.get("id"), str) and isinstance(row.get("vi"), str):
                translated[row["id"]] = row
    return translated


def import_previous_checkpoints(input_dir: str | Path, checkpoint_dir: str | Path) -> int:
    """Tự copy checkpoint từ Kaggle Add Input sang thư mục làm việc."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(Path(input_dir).rglob("dich_raw/dich_shard*.jsonl")):
        if src.parent.resolve() == checkpoint_dir.resolve():
            continue
        suffix = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:8]
        dst = checkpoint_dir / f"{src.stem}_prev_{suffix}.jsonl"
        if not dst.exists():
            shutil.copyfile(src, dst)
            count += 1
            print(f"Nạp checkpoint cũ: {src}")
    return count


def build_prompt(tokenizer: Any, system: str, user: str) -> str:
    """Tạo chat prompt cho model Instruct/Chat; fallback nếu thiếu system role."""
    messages_options = (
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        [{"role": "user", "content": system + "\n\n" + user}],
    )
    for messages in messages_options:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            continue
    raise RuntimeError("Model không có chat template; hãy dùng model *-Instruct/*-Chat")


def export_translations(
    books: dict[str, list[dict[str, str]]],
    checkpoint_dir: str | Path,
    work_dir: str | Path,
) -> TranslationResult:
    """Ghép checkpoint theo thứ tự đoạn thành ``dich/*.txt`` và ``dich.zip``."""
    checkpoint_dir = Path(checkpoint_dir)
    work_dir = Path(work_dir)
    translated_dir = work_dir / "dich"
    translated_dir.mkdir(parents=True, exist_ok=True)
    done = load_translations(checkpoint_dir)
    stats: dict[str, dict[str, int]] = {}

    for name, chunks in books.items():
        texts = [done[c["id"]]["vi"] if c["id"] in done else "[chưa dịch]" for c in chunks]
        (translated_dir / f"{name}.txt").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
        translated = sum(c["id"] in done for c in chunks)
        hit_max = sum(bool(done[c["id"]].get("hit_max")) for c in chunks if c["id"] in done)
        stats[name] = {"total": len(chunks), "translated": translated, "hit_max": hit_max}
        print(f"{name:<50} {translated}/{len(chunks)} đoạn" +
              (f", {hit_max} đoạn chạm MAX_NEW_TOKENS" if hit_max else ""))

    zip_path = Path(shutil.make_archive(str(work_dir / "dich"), "zip", translated_dir))
    return TranslationResult(translated_dir, zip_path, checkpoint_dir, book_stats=stats)
