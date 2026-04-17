"""Visualize quality-scoring outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

from multimodal_quality_eval.runtime import chunked, ensure_dir, prepare_runtime_environment, read_jsonl, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_metadata_path", required=True, help="Path to scored_metadata.jsonl.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for generated PNG files.")
    parser.add_argument("--cache_dir", default=".cache", help="Writable cache directory.")
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32", help="CLIP model used for embeddings.")
    parser.add_argument("--score_threshold", type=float, default=0.65, help="Threshold separating high vs. low quality points.")
    parser.add_argument("--embedding_batch_size", type=int, default=16, help="Batch size for CLIP embedding recomputation.")
    parser.add_argument("--example_count", type=int, default=6, help="Number of best and worst examples to show.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def save_histogram(records: list[dict], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    scores = [record["composite_score"] for record in records if record.get("composite_score") is not None]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(scores, bins=30, color="#20639B", edgecolor="white", alpha=0.9)
    axis.set_title("Composite Score Distribution")
    axis.set_xlabel("Composite score")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.25, linestyle="--")
    output_path = output_dir / "composite_score_histogram.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def collect_embeddings(
    records: list[dict],
    scorer: Any | None,
    batch_size: int,
) -> np.ndarray:
    precomputed = [
        np.asarray(record.get("clip_pair_embedding"), dtype=np.float32)
        for record in records
        if record.get("clip_pair_embedding") is not None
    ]
    if len(precomputed) == len(records):
        return np.stack(precomputed, axis=0)
    if scorer is None:
        raise ValueError(
            "CLIP embeddings were not saved in scored_metadata.jsonl. "
            "Re-run scoring with --save_clip_embeddings or allow visualize.py to load CLIP."
        )

    image_paths = [record["resolved_image_path"] for record in records]
    texts = [record["normalized_text"] for record in records]

    embeddings = []
    for batch_indices in chunked(list(range(len(records))), batch_size):
        batch_image_paths = [image_paths[index] for index in batch_indices]
        batch_texts = [texts[index] for index in batch_indices]
        embeddings.append(scorer.compute_pair_embeddings(batch_image_paths, batch_texts))
    return np.concatenate(embeddings, axis=0)


def save_scatter_plot(
    records: list[dict],
    embeddings: np.ndarray,
    output_dir: Path,
    score_threshold: float,
) -> Path:
    import matplotlib.pyplot as plt

    if len(records) < 2:
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.text(0.5, 0.5, "Need at least two valid samples for PCA.", ha="center", va="center")
        axis.set_axis_off()
        output_path = output_dir / "quality_scatter.png"
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return output_path

    reduced = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    labels = np.array([record["composite_score"] > score_threshold for record in records])

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        reduced[~labels, 0],
        reduced[~labels, 1],
        c="#D1495B",
        alpha=0.7,
        label=f"<= {score_threshold:.2f}",
    )
    axis.scatter(
        reduced[labels, 0],
        reduced[labels, 1],
        c="#2A9D8F",
        alpha=0.7,
        label=f"> {score_threshold:.2f}",
    )
    axis.set_title("CLIP Pair Embeddings (PCA)")
    axis.set_xlabel("PC 1")
    axis.set_ylabel("PC 2")
    axis.legend()
    axis.grid(alpha=0.2, linestyle="--")
    output_path = output_dir / "quality_scatter.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def save_example_grid(records: list[dict], output_dir: Path, example_count: int) -> Path:
    import matplotlib.pyplot as plt

    ranked = [record for record in records if record.get("composite_score") is not None and record.get("resolved_image_path")]
    best_records = sorted(ranked, key=lambda record: record["composite_score"], reverse=True)[:example_count]
    worst_records = sorted(ranked, key=lambda record: record["composite_score"])[:example_count]
    columns = max(len(best_records), len(worst_records), 1)

    figure, axes = plt.subplots(2, columns, figsize=(3.2 * columns, 6.5))
    axes = np.asarray(axes, dtype=object).reshape(2, columns)

    def render_row(row_index: int, row_records: list[dict], title_prefix: str) -> None:
        for column_index in range(columns):
            axis = axes[row_index, column_index]
            axis.axis("off")
            if column_index >= len(row_records):
                continue
            record = row_records[column_index]
            image_path = Path(record["resolved_image_path"])
            try:
                with Image.open(image_path) as image:
                    rendered = ImageOps.fit(image.convert("RGB"), size=(320, 320))
            except OSError:
                rendered = Image.new("RGB", (320, 320), color=(245, 245, 245))
            axis.imshow(rendered)
            axis.set_title(f"{title_prefix} {column_index + 1}\nscore={record['composite_score']:.3f}", fontsize=10)

    render_row(0, best_records, "Best")
    render_row(1, worst_records, "Worst")
    output_path = output_dir / "best_worst_examples.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    prepare_runtime_environment(project_root / args.cache_dir)
    set_random_seed(args.seed)

    scored_metadata_path = Path(args.scored_metadata_path).expanduser().resolve()
    output_dir = ensure_dir(project_root / args.output_dir)
    records = read_jsonl(scored_metadata_path)
    valid_records = [
        record
        for record in records
        if record.get("composite_score") is not None
        and record.get("normalized_text")
        and record.get("resolved_image_path")
        and Path(record["resolved_image_path"]).exists()
    ]

    if not valid_records:
        raise ValueError("No valid scored records with image paths were found.")

    histogram_path = save_histogram(valid_records, output_dir)
    scorer = None
    if not all(record.get("clip_pair_embedding") is not None for record in valid_records):
        from multimodal_quality_eval import MultiModalQualityScorer

        scorer = MultiModalQualityScorer(
            clip_model_name=args.clip_model_name,
            cache_dir=project_root / args.cache_dir,
            load_judge_model=False,
        )
    embeddings = collect_embeddings(valid_records, scorer, args.embedding_batch_size)
    scatter_path = save_scatter_plot(valid_records, embeddings, output_dir, args.score_threshold)
    grid_path = save_example_grid(valid_records, output_dir, args.example_count)

    print(f"Saved histogram to: {histogram_path}")
    print(f"Saved scatter plot to: {scatter_path}")
    print(f"Saved example grid to: {grid_path}")


if __name__ == "__main__":
    main()
