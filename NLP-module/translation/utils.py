"""Các hàm hỗ trợ cho HanVietTranslator (không tự tải/chạy LLM).

Không cần gọi trực tiếp nếu chỉ muốn dịch sách; hãy dùng ``HanVietTranslator``.
Có thể dùng riêng để chia đoạn hoặc xuất kết quả từ checkpoint::

    from han_viet_translator.utils import make_chunks, load_translations
    chunks = make_chunks("黃帝曰：...", max_chars=300)
    done = load_translations("/kaggle/working/dich_raw")
"""

from __future__ import annotations

import bisect
import csv
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
    zip_path: Path | None
    checkpoint_dir: Path
    # {'ten_sach': {'total': N, 'translated': M, 'hit_max': K}}
    book_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    worker_errors: list[str] = field(default_factory=list)
    annotated_dir: Path | None = None
    locator_dir: Path | None = None


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


def _line_number(newline_positions: list[int], offset: int) -> int:
    """Dòng 1-based của offset ký tự (không phải byte) trong TXT gốc."""
    return bisect.bisect_left(newline_positions, offset) + 1


def make_chunks_located(text: str, max_chars: int = 300) -> list[dict[str, Any]]:
    """Giữ nguyên thuật toán make_chunks và định vị các mảnh trên TXT Hán gốc.

    source_start/end là half-open [start, end) tính theo codepoint Unicode của
    file đã giải mã UTF-8; source_spans loại bỏ các khoảng trắng/dòng trống nằm
    giữa các mảnh. Không được dùng offset này làm BYTE offset.
    """
    newline_positions = [i for i, ch in enumerate(text) if ch == "\n"]
    results: list[dict[str, Any]] = []
    current: list[str] = []
    spans: list[list[int]] = []
    length = 0

    def flush() -> None:
        nonlocal current, spans, length
        if not current:
            return
        results.append({
            "zh": "\n".join(current),
            "source_start": spans[0][0],
            "source_end": spans[-1][1],
            "source_line_start": _line_number(newline_positions, spans[0][0]),
            "source_line_end": _line_number(newline_positions, spans[-1][1] - 1),
            "source_spans": [span[:] for span in spans],
        })
        current, spans, length = [], [], 0

    abs_pos = 0
    for raw_line in text.splitlines(keepends=True):
        content = raw_line.strip()
        if not content:
            abs_pos += len(raw_line)
            continue
        # strip() có thể bỏ các khoảng trắng ở cả đầu và cuối dòng.
        content_start = abs_pos + len(raw_line) - len(raw_line.lstrip())
        rel_pos = 0
        for piece in split_long(content, max_chars):
            if current and length + len(piece) > max_chars:
                flush()
            start = content_start + rel_pos
            end = start + len(piece)
            if text[start:end] != piece:
                raise ValueError("Không khớp vị trí chunk với TXT gốc")
            current.append(piece)
            if spans and spans[-1][1] == start:
                spans[-1][1] = end
            else:
                spans.append([start, end])
            length += len(piece)
            rel_pos += len(piece)
        abs_pos += len(raw_line)
    flush()
    if [row["zh"] for row in results] != make_chunks(text, max_chars):
        raise AssertionError("Dóng vị trí làm thay đổi chuỗi chunk; dừng để bảo vệ checkpoint")
    return results


def prepare_books(input_dir: str | Path, chunk_chars: int = 300) -> dict[str, list[dict[str, Any]]]:
    """Tìm *.txt chữ Hán, chia chunk ổn định và thêm vị trí nguồn chính xác.

    Chỉ số source_start/end tương ứng với file TXT Hán thực tế, đọc dưới dạng
    UTF-8 không tự đổi CRLF. File trùng stem tiếp tục giữ quy tắc cũ: lấy file đầu.
    """
    folder = Path(input_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {folder}")

    books: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(folder.rglob("*.txt")):
        name = path.stem
        # newline='' để vị trí luôn đo được trên ký tự thật trong TXT đầu vào.
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            content = f.read()
        if name in books or len(HAN_PATTERN.findall(content)) < 50:
            continue
        source_file = path.relative_to(folder).as_posix()
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        book_rows = make_chunks_located(content, chunk_chars)
        books[name] = [
            {
                "id": f"{name}|{i}|{hashlib.md5(row['zh'].encode()).hexdigest()[:8]}",
                "chunk_index": i,
                "chunk_number": i + 1,
                "source_file": source_file,
                "source_sha256": raw_hash,
                **row,
            }
            for i, row in enumerate(book_rows)
        ]
        print(f"{name:<50} {len(content):8,} ký tự -> {len(books[name]):5d} đoạn")

    if not books:
        raise ValueError(f"Không có file .txt chữ Hán hợp lệ trong {folder}")
    return books


def _safe_book_filename(book: str) -> str:
    return re.sub(r'[/\\\x00-\x1f]', '_', book).strip('. ') or 'unknown_book'


def render_book_with_index(book: str, rows: list[dict[str, Any]],
                           output_file: str | Path, annotated_file: str | Path,
                           index_file: str | Path) -> list[dict[str, Any]]:
    """Xuất TXT sạch, TXT có nhãn CHUNK và TSV tra cứu vị trí hai phía.

    rows theo thứ tự sách, có 'text' (văn bản hiển thị), 'status', cùng metadata
    nguồn. Không đính nhãn vào TXT sạch dùng để dóng hàng.
    """
    output_file = Path(output_file)
    annotated_file = Path(annotated_file)
    index_file = Path(index_file)
    for p in (output_file, annotated_file, index_file):
        p.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    annotated: list[str] = []
    positions: list[dict[str, Any]] = []
    offset = 0
    line = 1
    for i, row in enumerate(rows):
        if i:
            parts.append("\n\n")
            offset += 2
            line += 2
        content = str(row.get("text") or "").replace("\r\n", "\n").replace("\r", "\n")
        start, start_line = offset, line
        parts.append(content)
        offset += len(content)
        line += content.count("\n")
        idx = row.get("chunk_index")
        if not isinstance(idx, int):
            try:
                idx = int(row["id"].rsplit('|', 2)[-2])
            except (ValueError, IndexError, KeyError):
                idx = i
        number = idx + 1
        tag = f"CHUNK {number:06d}"
        status = row.get("status") or "UNKNOWN"
        positions.append({
            "book": book, "chunk_number": number, "chunk_index": idx,
            "search_tag": tag, "chunk_id": row["id"], "status": status,
            "source_file": row.get("source_file", ""),
            "source_start": row.get("source_start"),
            "source_end": row.get("source_end"),
            "source_line_start": row.get("source_line_start"),
            "source_line_end": row.get("source_line_end"),
            "translation_start": start, "translation_end": offset,
            "translation_line_start": start_line,
            "translation_line_end": line,
            "source_preview": row.get("zh", "")[:100].replace("\n", " "),
            "translation_preview": content[:100].replace("\n", " "),
        })
        annotated.extend([
            f"===== {tag} | {status} | {row['id']} =====\n",
            f"Hán gốc: {row.get('source_file', '?')} | "
            f"dòng {row.get('source_line_start', '?')}–{row.get('source_line_end', '?')} | "
            f"ký tự [{row.get('source_start', '?')}, {row.get('source_end', '?')})\n",
            f"Vị trí TXT dịch: dòng {start_line}–{line} | ký tự [{start}, {offset})\n",
        ])
        if row.get('reasons'):
            annotated.append("Lý do cần xem: " + "; ".join(str(x) for x in row['reasons']) + "\n")
        if row.get('candidate_vi') and row.get('candidate_vi') != content:
            annotated.append("Bản dịch chưa đạt (để kiểm tra, không thuộc corpus):\n"
                             + str(row['candidate_vi']) + "\n")
        annotated.append("Bản dịch đã xuất trong TXT sạch:\n" + content + "\n\n")
    output_file.write_text("".join(parts) + ("\n" if parts else ""), encoding="utf-8")
    annotated_file.write_text("".join(annotated), encoding="utf-8")
    fields = [
        "book", "chunk_number", "chunk_index", "search_tag", "chunk_id", "status",
        "source_file", "source_start", "source_end", "source_line_start", "source_line_end",
        "translation_start", "translation_end", "translation_line_start", "translation_line_end",
        "source_preview", "translation_preview",
    ]
    with index_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(positions)
    return positions


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
    books: dict[str, list[dict[str, Any]]],
    checkpoint_dir: str | Path,
    work_dir: str | Path,
    *, create_zip: bool = False,
) -> TranslationResult:
    """Ghép checkpoint, xuất TXT sạch + TXT có CHUNK tag + vị trí TSV.

    File dịch sạch vẫn đúng định dạng cũ; file .annotated.txt dùng Ctrl+F
    "CHUNK 000006" để tìm nhanh. Index.tsv ánh xạ gốc <-> bản dịch.
    """
    checkpoint_dir = Path(checkpoint_dir)
    work_dir = Path(work_dir)
    translated_dir = work_dir / "dich"
    translated_dir.mkdir(parents=True, exist_ok=True)
    done = load_translations(checkpoint_dir)
    stats: dict[str, dict[str, int]] = {}

    for name, chunks in books.items():
        records = []
        for chunk in chunks:
            result = done.get(chunk["id"])
            status = ("NOT_TRANSLATED" if result is None else
                      "HIT_MAX_NEW_TOKENS" if result.get("hit_max") else "TRANSLATED")
            records.append({**chunk, "status": status,
                            "text": result["vi"] if result else "[chưa dịch]"})
        safe = _safe_book_filename(name)
        render_book_with_index(
            name, records,
            translated_dir / f"{safe}.txt",
            work_dir / "dich_annotated" / f"{safe}.annotated.txt",
            work_dir / "dich_index" / f"{safe}.tsv",
        )
        translated = sum(r["status"] != "NOT_TRANSLATED" for r in records)
        hit_max = sum(r["status"] == "HIT_MAX_NEW_TOKENS" for r in records)
        stats[name] = {"total": len(chunks), "translated": translated, "hit_max": hit_max}
        print(f"{name:<50} {translated}/{len(chunks)} đoạn" +
              (f", {hit_max} đoạn chạm MAX_NEW_TOKENS" if hit_max else ""))

    zip_path = (Path(shutil.make_archive(str(work_dir / "dich"), "zip", translated_dir))
                if create_zip else None)
    return TranslationResult(
        translated_dir, zip_path, checkpoint_dir, book_stats=stats,
        annotated_dir=work_dir / "dich_annotated",
        locator_dir=work_dir / "dich_index",
    )


def attach_source_locations(chunks: list[dict[str, Any]],
                            source_text_dir: str | Path) -> list[dict[str, Any]]:
    """Khôi phục source offsets từ Hán TXT cho chunks.jsonl bản cũ.

    Dò tìm lần lượt trong toàn bộ văn bản, không đoán qua tỷ lệ ký tự.
    Gặp thiếu/trùng file stem hoặc không khớp nội dung thì báo lỗi rõ ràng.
    """
    source_root = Path(source_text_dir)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục Hán gốc: {source_root}")
    names = {row['id'].rsplit('|', 2)[0] for row in chunks}
    files: dict[str, Path] = {}
    for name in names:
        matches = [p for p in source_root.rglob('*.txt') if p.stem == name]
        if len(matches) != 1:
            raise ValueError(f"Sách {name}: cần duy nhất 1 TXT Hán gốc, thấy {matches}")
        files[name] = matches[0]
    all_books: dict[str, list[dict[str, Any]]] = {}
    for row in chunks:
        name = row['id'].rsplit('|', 2)[0]
        all_books.setdefault(name, []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for name, book_chunks in all_books.items():
        src = files[name]
        with src.open('r', encoding='utf-8', newline='') as f:
            text = f.read()
        newlines = [i for i, ch in enumerate(text) if ch == '\n']
        cursor = 0
        sorted_rows = sorted(book_chunks, key=lambda x: int(x['id'].rsplit('|', 2)[-2]))
        for row in sorted_rows:
            spans = []
            for piece in row['zh'].split('\n'):
                if not piece:
                    raise ValueError(f"Chunk {row['id']} chứa dòng rỗng không thể khôi phục an toàn")
                start = text.find(piece, cursor)
                if start < 0:
                    raise ValueError(f"Không xác định được vị trí chính xác của {row['id']} trong {src}")
                end = start + len(piece)
                if spans and spans[-1][1] == start:
                    spans[-1][1] = end
                else:
                    spans.append([start, end])
                cursor = end
            start, end = spans[0][0], spans[-1][1]
            output[row['id']] = {
                **row,
                'chunk_index': int(row['id'].rsplit('|', 2)[-2]),
                'chunk_number': int(row['id'].rsplit('|', 2)[-2]) + 1,
                'source_file': src.relative_to(source_root).as_posix(),
                'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                'source_start': start, 'source_end': end,
                'source_line_start': _line_number(newlines, start),
                'source_line_end': _line_number(newlines, end - 1),
                'source_spans': spans,
            }
    return [output[row['id']] for row in chunks]
