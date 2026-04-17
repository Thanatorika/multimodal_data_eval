"""Main entry point for multimodal dataset quality scoring."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multimodal_quality_eval.runtime import ensure_dir, prepare_runtime_environment, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True, help="Local dataset folder/metadata.jsonl or Hugging Face dataset ID.")
    parser.add_argument("--dataset_split", default="train", help="Split to use when --dataset_path is a Hugging Face dataset.")
    parser.add_argument("--image_root", default=None, help="Optional directory containing image files referenced by the metadata.")
    parser.add_argument("--metadata_filename", default=None, help="Optional metadata filename for local datasets.")
    parser.add_argument("--output_dir", default="outputs", help="Directory where scored JSONL files will be written.")
    parser.add_argument("--cache_dir", default=".cache", help="Writable cache directory for Hugging Face and matplotlib.")
    parser.add_argument("--summary_filename", default="scoring_summary.json", help="Filename for the run summary JSON.")
    parser.add_argument("--subset_size", type=int, default=None, help="Only score the first N samples.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for CLIP-based alignment computation.")
    parser.add_argument("--local_files_only", action="store_true", help="Load model files only from the local Hugging Face cache.")
    parser.add_argument("--filter", action="store_true", help="Also write filtered_metadata.jsonl using the composite threshold.")
    parser.add_argument("--composite_threshold", type=float, default=0.65, help="Threshold used when --filter is enabled.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32", help="CLIP model used for alignment.")
    parser.add_argument(
        "--judge_model_name",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Vision-language judge model. Use the 3B checkpoint by default for a 24GB GPU.",
    )
    parser.add_argument("--disable_judge", action="store_true", help="Skip the judge model and only compute alignment scores.")
    parser.add_argument("--disable_4bit", action="store_true", help="Load the judge model without bitsandbytes 4-bit quantization.")
    parser.add_argument(
        "--offload_between_models",
        action="store_true",
        help="Move CLIP to CPU while the judge model runs to reduce peak VRAM.",
    )
    parser.add_argument(
        "--save_clip_embeddings",
        action="store_true",
        help="Store CLIP pair embeddings in the scored JSONL for faster visualization later.",
    )
    return parser.parse_args()


@dataclass
class RunningMetric:
    """Streaming summary statistics for one scalar metric."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, value: float | None) -> None:
        if value is None:
            return
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def to_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        variance = max(0.0, (self.total_sq / self.count) - (mean * mean))
        return {
            "count": self.count,
            "mean": round(mean, 6),
            "std": round(variance**0.5, 6),
            "min": round(self.minimum, 6) if self.minimum is not None else None,
            "max": round(self.maximum, 6) if self.maximum is not None else None,
        }


@dataclass
class RunSummary:
    """Aggregate counters and metric summaries for one scoring run."""

    total_seen: int = 0
    invalid_records: int = 0
    successfully_scored: int = 0
    filtered_records: int = 0
    alignment_score: RunningMetric = field(default_factory=RunningMetric)
    image_quality_score: RunningMetric = field(default_factory=RunningMetric)
    text_quality_score: RunningMetric = field(default_factory=RunningMetric)
    composite_score: RunningMetric = field(default_factory=RunningMetric)

    def update_invalid(self) -> None:
        self.total_seen += 1
        self.invalid_records += 1

    def update_scored(self, record: dict[str, Any], passed_filter: bool) -> None:
        self.total_seen += 1
        self.successfully_scored += 1
        self.alignment_score.update(record.get("alignment_score"))
        self.image_quality_score.update(record.get("image_quality_score"))
        self.text_quality_score.update(record.get("text_quality_score"))
        self.composite_score.update(record.get("composite_score"))
        if passed_filter:
            self.filtered_records += 1

    def update_failed_valid(self) -> None:
        self.total_seen += 1
        self.invalid_records += 1

    def to_dict(self, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
        return {
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "subset_size": args.subset_size,
            "batch_size": args.batch_size,
            "filter_enabled": args.filter,
            "composite_threshold": args.composite_threshold,
            "output_dir": str(output_dir),
            "total_seen": self.total_seen,
            "successfully_scored": self.successfully_scored,
            "invalid_or_failed_records": self.invalid_records,
            "filtered_records": self.filtered_records,
            "metrics": {
                "alignment_score": self.alignment_score.to_dict(),
                "image_quality_score": self.image_quality_score.to_dict(),
                "text_quality_score": self.text_quality_score.to_dict(),
                "composite_score": self.composite_score.to_dict(),
            },
        }


def combine_record(sample: Any, scores: dict[str, Any] | None = None, extra_error: str | None = None) -> dict[str, Any]:
    """Merge normalized sample metadata with scoring outputs."""

    record = sample.to_output_record()
    record["sample_id"] = sample.sample_id
    record["normalized_text"] = sample.text
    record["resolved_image_path"] = sample.image_path
    record["source_dataset"] = sample.source

    if scores is None:
        record["alignment_score"] = None
        record["image_quality_score"] = None
        record["text_quality_score"] = None
        record["composite_score"] = None
        record["image_quality_reason"] = ""
        record["text_quality_reason"] = ""
    else:
        record.update(scores)

    errors = [error for error in (sample.error, extra_error) if error]
    if errors:
        record["quality_error"] = ";".join(errors)
    return record


def score_with_fallback(
    scorer: Any,
    samples: list[Any],
    save_clip_embeddings: bool,
) -> tuple[list[tuple[Any, dict[str, Any]]], list[tuple[Any, str]]]:
    """Score a batch, then fall back to per-sample scoring if needed."""

    failures: list[tuple[Any, str]] = []
    if not samples:
        return [], failures

    payload = [{"image_path": sample.image_path, "text": sample.text} for sample in samples]
    try:
        batch_scores = scorer.score_batch(payload, return_clip_embeddings=save_clip_embeddings)
        return list(zip(samples, batch_scores)), failures
    except Exception as batch_error:
        outputs: list[tuple[Any, dict[str, Any]]] = []
        for sample in samples:
            try:
                if save_clip_embeddings:
                    result = scorer.score_batch(
                        [{"image_path": sample.image_path, "text": sample.text}],
                        return_clip_embeddings=True,
                    )[0]
                else:
                    result = scorer.score_sample(sample.image_path, sample.text)
                outputs.append((sample, result))
            except Exception as sample_error:
                failures.append((sample, f"{type(batch_error).__name__}:{batch_error};{type(sample_error).__name__}:{sample_error}"))
        return outputs, failures


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    prepare_runtime_environment(project_root / args.cache_dir)
    set_random_seed(args.seed)

    from multimodal_quality_eval import MultiModalDataLoader, MultiModalQualityScorer

    output_dir = ensure_dir(project_root / args.output_dir)
    scored_path = output_dir / "scored_metadata.jsonl"
    filtered_path = output_dir / "filtered_metadata.jsonl"
    summary_path = output_dir / args.summary_filename
    summary = RunSummary()

    loader = MultiModalDataLoader(
        dataset_path=args.dataset_path,
        split=args.dataset_split,
        image_root=args.image_root,
        metadata_filename=args.metadata_filename,
        cache_dir=project_root / args.cache_dir,
        validate_images=True,
    )
    scorer = MultiModalQualityScorer(
        clip_model_name=args.clip_model_name,
        judge_model_name=args.judge_model_name,
        cache_dir=project_root / args.cache_dir,
        load_judge_model=not args.disable_judge,
        use_4bit=not args.disable_4bit,
        offload_between_models=args.offload_between_models,
        local_files_only=args.local_files_only,
    )

    with scored_path.open("w", encoding="utf-8") as scored_handle:
        filtered_handle = filtered_path.open("w", encoding="utf-8") if args.filter else None
        try:
            for batch in loader.iter_batches(
                batch_size=args.batch_size,
                subset_size=args.subset_size,
                show_progress=True,
                description="Scoring dataset",
            ):
                valid_samples = [sample for sample in batch if not sample.error]
                invalid_samples = [sample for sample in batch if sample.error]

                for sample in invalid_samples:
                    record = combine_record(sample)
                    scored_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary.update_invalid()

                scored_pairs, failures = score_with_fallback(
                    scorer=scorer,
                    samples=valid_samples,
                    save_clip_embeddings=args.save_clip_embeddings,
                )

                scored_ids = set()
                for sample, score in scored_pairs:
                    record = combine_record(sample, score)
                    scored_ids.add(sample.sample_id)
                    scored_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    passed_filter = (
                        filtered_handle is not None
                        and record["composite_score"] is not None
                        and record["composite_score"] > args.composite_threshold
                    )
                    summary.update_scored(record, passed_filter=passed_filter)
                    if (
                        passed_filter
                    ):
                        filtered_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                failed_by_id = {sample.sample_id: error for sample, error in failures}
                for sample in valid_samples:
                    if sample.sample_id in scored_ids:
                        continue
                    record = combine_record(sample, extra_error=failed_by_id.get(sample.sample_id, "scoring_failed"))
                    scored_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary.update_failed_valid()
        finally:
            if filtered_handle is not None:
                filtered_handle.close()

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(args=args, output_dir=output_dir), handle, ensure_ascii=False, indent=2)

    print(f"Saved scored metadata to: {scored_path}")
    if args.filter:
        print(f"Saved filtered metadata to: {filtered_path}")
    print(f"Saved scoring summary to: {summary_path}")


if __name__ == "__main__":
    main()
