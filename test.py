import os

# Set this FIRST, before importing PaddleX or PaddleOCR
os.environ["PADDLE_PDX_CACHE_HOME"] = r"D:\PaddleModels"

from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL(pipeline_version="v1.6", device="gpu:0")
output = pipeline.predict(["./test_1.jpg", "./test_2.jpg"])
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
