from pathlib import Path
from hunyuan_ocr import OCRConfig, OCRPipeline


def main():
    config = OCRConfig(
        input_dir=Path("./pdfs"),
        work_dir=Path("./output"),
        engine="hunyuan",
        pages="",            # "" = all pages; or e.g. "1-20"
        dtype="fp16",        # "fp16" is ~2x faster than "fp32"
        dpi=200,
        max_side=1600,
        max_new_tokens=6000,
        budget_hours=10.5,
    )

    pipeline = OCRPipeline(config)
    zip_archive = pipeline.run_all(
        preview_layout=True,  # Saves layout bbox visualizations to ./output/layout/
        test_first=True,      # Runs 1 page per GPU and prints sample text first
    )
    print(f"Done! Archive created at: {zip_archive}")


if __name__ == "__main__":
    main()