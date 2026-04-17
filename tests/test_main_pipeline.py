from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import main as main_module
import multimodal_quality_eval


class FakeScorer:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def score_batch(self, samples, return_clip_embeddings=False):
        outputs = []
        for index, sample in enumerate(samples):
            score = 0.8 if "high" in sample["text"] else 0.4
            record = {
                "alignment_score": score,
                "image_quality_score": score * 100.0,
                "text_quality_score": score * 100.0,
                "composite_score": score,
                "image_quality_reason": "ok",
                "text_quality_reason": "ok",
            }
            if return_clip_embeddings:
                record["clip_pair_embedding"] = [float(index), float(index + 1)]
            outputs.append(record)
        return outputs

    def score_sample(self, image_path, text):
        return self.score_batch([{"image_path": image_path, "text": text}], return_clip_embeddings=False)[0]


class MainPipelineTest(unittest.TestCase):
    def test_main_writes_scored_filtered_and_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset"
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            Image.new("RGB", (64, 64), color=(255, 0, 0)).save(images_dir / "one.png")
            Image.new("RGB", (64, 64), color=(0, 255, 0)).save(images_dir / "two.png")

            metadata_path = dataset_dir / "metadata.jsonl"
            records = [
                {"id": "1", "image": "images/one.png", "text": "high quality sample"},
                {"id": "2", "image": "images/two.png", "text": "low quality sample"},
            ]
            with metadata_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            argv = [
                "main.py",
                "--dataset_path",
                str(dataset_dir),
                "--output_dir",
                str(root / "outputs"),
                "--cache_dir",
                str(root / ".cache"),
                "--filter",
                "--composite_threshold",
                "0.65",
            ]

            with patch.object(sys, "argv", argv):
                original_scorer = multimodal_quality_eval.__dict__.get("MultiModalQualityScorer")
                multimodal_quality_eval.MultiModalQualityScorer = FakeScorer
                try:
                    main_module.main()
                finally:
                    if original_scorer is None:
                        delattr(multimodal_quality_eval, "MultiModalQualityScorer")
                    else:
                        multimodal_quality_eval.MultiModalQualityScorer = original_scorer

            scored_path = root / "outputs" / "scored_metadata.jsonl"
            filtered_path = root / "outputs" / "filtered_metadata.jsonl"
            summary_path = root / "outputs" / "scoring_summary.json"

            self.assertTrue(scored_path.exists())
            self.assertTrue(filtered_path.exists())
            self.assertTrue(summary_path.exists())

            scored_records = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines()]
            filtered_records = [json.loads(line) for line in filtered_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(len(scored_records), 2)
            self.assertEqual(len(filtered_records), 1)
            self.assertEqual(filtered_records[0]["sample_id"], "1")
            self.assertEqual(summary["successfully_scored"], 2)
            self.assertEqual(summary["filtered_records"], 1)


if __name__ == "__main__":
    unittest.main()
