import os
from pathlib import Path

class PaddleOCR_Tool:
    def __init__(
        self,
        paddle_version="v1.6",
        cache_dir="./cache",
        **paddle_additional_arg
    ):
        cache_path = Path(cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)

        os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_path)

        from paddleocr import PaddleOCRVL

        self.model = PaddleOCRVL(
            pipeline_version=paddle_version,
            **paddle_additional_arg
        )

    def process_image(
            self, 
            image_path_list, 
            save_to_json=True,
            save_to_md=True,
            save_path="./res"
        ):
        for image_path in image_path_list:
            path = Path(image_path)
            if not path.exists():
                raise ValueError("image path list contain unexisted path")
            
        res_list = self.model.predict(image_path_list)
        for res in res_list:
            if save_to_json:
                res.save_to_json(save_path=save_path)
            if save_to_md:
                res.save_to_markdown(save_path=save_path)