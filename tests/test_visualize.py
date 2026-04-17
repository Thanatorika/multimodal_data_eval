from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import visualize as visualize_module


class VisualizeTest(unittest.TestCase):
    def test_visualize_with_precomputed_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            records = []
            for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), start=1):
                image_path = images_dir / f"{index}.png"
                Image.new("RGB", (64, 64), color=color).save(image_path)
                records.append(
                    {
                        "sample_id": str(index),
                        "normalized_text": f"sample {index}",
                        "resolved_image_path": str(image_path),
                        "composite_score": 0.2 * index,
                        "alignment_score": 0.2 * index,
                        "image_quality_score": 20.0 * index,
                        "text_quality_score": 30.0 * index,
                        "clip_pair_embedding": [0.1 * index, 0.2 * index, 0.3 * index],
                    }
                )

            scored_path = root / "scored_metadata.jsonl"
            with scored_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            argv = [
                "visualize.py",
                "--scored_metadata_path",
                str(scored_path),
                "--output_dir",
                str(root / "outputs"),
                "--cache_dir",
                str(root / ".cache"),
            ]

            with patch.object(sys, "argv", argv):
                visualize_module.main()

            self.assertTrue((root / "outputs" / "composite_score_histogram.png").exists())
            self.assertTrue((root / "outputs" / "quality_scatter.png").exists())
            self.assertTrue((root / "outputs" / "best_worst_examples.png").exists())


if __name__ == "__main__":
    unittest.main()
