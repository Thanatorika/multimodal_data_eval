from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from multimodal_quality_eval.data_loader import MultiModalDataLoader


class DataLoaderTest(unittest.TestCase):
    def test_local_dataset_normalization_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=(255, 0, 0)).save(images_dir / "red.png")

            records = [
                {"id": "a", "image": "images/red.png", "caption": "a red square"},
                {
                    "id": "b",
                    "image": "images/missing.png",
                    "conversations": [
                        {"from": "human", "value": "<image> what is this?"},
                        {"from": "gpt", "value": "a missing image sample"},
                    ],
                },
            ]
            metadata_path = root / "metadata.jsonl"
            with metadata_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            loader = MultiModalDataLoader(str(root), cache_dir=root / ".cache")
            samples = list(loader.iter_samples())

            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].text, "a red square")
            self.assertTrue(samples[0].image_path.endswith("red.png"))
            self.assertIsNone(samples[0].error)

            self.assertEqual(samples[1].text, "a missing image sample")
            self.assertIn("missing_image_file", samples[1].error or "")


if __name__ == "__main__":
    unittest.main()
