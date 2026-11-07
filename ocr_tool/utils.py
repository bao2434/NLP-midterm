from pathlib import Path
from PIL import Image, ImageDraw, ImageFont 

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0b0", "#f032e6", "#9a6324", "#808000", "#000075"]

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_boxes(
    image_path: str | Path,
    res_dir: str | Path,
    label_list: list[str],
    box_list: list[tuple[int, int, int, int]]
):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if len(label_list) != len(box_list):
        raise ValueError("Unequal label_list and box_list lengths")

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default(
        size=max(14, img.width // 60)
    )

    for i, (label, box) in enumerate(zip(label_list, box_list)):
        c = COLORS[i % len(COLORS)]

        draw.rectangle(box, outline=c, width=3)
        draw.text(
            (box[0] + 4, box[1] + 2),
            f"{i}.{label}",
            fill=c,
            font=font
        )

    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    output_path = res_dir / image_path.name
    img.save(output_path)

    return output_path