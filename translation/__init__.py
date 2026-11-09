"""Package dịch sách Hán -> Việt; import class từ thư mục như OCR_Paddle.

CÀI THƯ VIỆN (trên Kaggle, bật GPU và Internet)::

    !pip install -q -U transformers accelerate huggingface_hub
    # Chỉ cài thêm nếu load_4bit=True:
    !pip install -q bitsandbytes

CÁCH SỬ DỤNG (đặt thư mục han_viet_translator cạnh notebook)::

    from han_viet_translator import HanVietTranslator

    translator = HanVietTranslator(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_per_model=2,  # Kaggle GPU T4 x2
        batch=8,           # Nếu hết VRAM: giảm xuống 4 / 2
        chunk_chars=300,
    )

    # Chạy thử 2 đoạn; tự lưu checkpoint.
    test = translator.translate(
        input_dir="/kaggle/input", work_dir="/kaggle/working", limit=2,
    )

    # Chạy tất cả; bỏ qua các đoạn đã dịch, xuất .txt và .zip.
    result = translator.translate(
        input_dir="/kaggle/input", work_dir="/kaggle/working",
    )
    print(result.zip_path)

    # Nếu chỉ muốn ghép lại checkpoint, không gọi GPU/model:
    result = translator.export_only("/kaggle/input", "/kaggle/working")

LƯU Ý: Package import nhanh và không tải model cho đến khi thực sự dịch.
"""

from .HanVietTranslator import HanVietTranslator
from .utils import TranslationConfig, TranslationResult, make_chunks, prepare_books, load_translations

__all__ = [
    "HanVietTranslator", "TranslationConfig", "TranslationResult",
    "make_chunks", "prepare_books", "load_translations",
]
