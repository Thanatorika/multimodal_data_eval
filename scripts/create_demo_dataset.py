"""Create a tiny local image-text dataset for smoke testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="demo_dataset", help="Directory where the demo dataset will be created.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("red_square.png", (220, 60, 60), "A clean red square centered on a white background."),
        ("green_circle.png", (70, 180, 120), "A green circle with sharp edges and simple composition."),
        ("blue_triangle.png", (70, 120, 210), "A blue triangle on white, useful for a toy geometry caption."),
        ("blurred_noise.png", (150, 150, 150), "A noisy blurry patch with little semantic value."),
    ]

    records = []
    for name, color, text in specs:
        image = Image.new("RGB", (256, 256), color="white")
        draw = ImageDraw.Draw(image)
        if "square" in name:
            draw.rectangle((48, 48, 208, 208), fill=color)
        elif "circle" in name:
            draw.ellipse((48, 48, 208, 208), fill=color)
        elif "triangle" in name:
            draw.polygon([(128, 40), (36, 216), (220, 216)], fill=color)
        else:
            for offset in range(0, 256, 8):
                draw.line((0, offset, 255, 255 - offset), fill=color, width=6)
        image_path = images_dir / name
        image.save(image_path)
        records.append({"id": name.rsplit(".", 1)[0], "image": f"images/{name}", "text": text})

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Created demo dataset at: {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
