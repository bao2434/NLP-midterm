from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_HUNYUAN_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)
DEFAULT_LAYOUT_PROMPT = "按照阅读顺序解析图中的版式信息。"


@dataclass
class OCRConfig:
    # Paths
    input_dir: Path = Path("./input")
    work_dir: Path = Path("./working")
    prev_results_dir: Optional[Path] = None  # Optional path to resume from previous ocr_raw

    # Engine & Model
    engine: str = "hunyuan"  # "hunyuan" or "paddle"
    model: str = "tencent/HunyuanOCR"
    dtype: str = "fp32"      # "fp32", "fp16", "bf16"
    prompt: str = DEFAULT_HUNYUAN_PROMPT
    repetition_penalty: float = 1.08

    # PaddleOCR-VL specific (if engine == "paddle")
    paddle_version: str = "v1.6"
    paddle_args: dict = field(default_factory=dict)

    # Rendering & Generation
    pages: str = ""          # "" = all pages; "1-10,15" = specific pages (1-indexed)
    dpi: int = 200
    max_side: int = 1600
    max_new_tokens: int = 6000

    # Execution & Fault Tolerance
    num_gpus: Optional[int] = None  # Auto-detected if None
    budget_hours: float = 10.5
    stall_minutes: int = 15
    page_timeout: int = 300
    n_layout_samples: int = 3

    @property
    def raw_dir(self) -> Path:
        return self.work_dir / "ocr_raw"

    @property
    def txt_dir(self) -> Path:
        return self.work_dir / "txt"

    @property
    def layout_dir(self) -> Path:
        return self.work_dir / "layout"

    @property
    def tasks_file(self) -> Path:
        return self.work_dir / "tasks.jsonl"