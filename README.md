# Multimodal Data Quality Evaluation

This project scores image-text pairs for multimodal instruction-tuning data quality on a single 24GB GPU. It combines:

- CLIP alignment scoring for semantic image-text consistency
- A quantized Qwen2.5-VL judge for image quality and text quality
- A CLI pipeline for scoring/filtering datasets
- Visualization scripts for quick research analysis

## Project Structure

```text
multimodal_data_eval/
├── main.py
├── scripts/
│   └── create_demo_dataset.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_main_pipeline.py
│   └── test_visualize.py
├── visualize.py
├── requirements.txt
├── README.md
└── multimodal_quality_eval/
    ├── __init__.py
    ├── data_loader.py
    ├── quality_scorer.py
    └── runtime.py
```

## Supported Dataset Formats

### 1. Local folder with `metadata.jsonl`

Expected layout:

```text
your_dataset/
├── metadata.jsonl
└── images/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

Example `metadata.jsonl` line:

```json
{"id": "0001", "image": "images/0001.jpg", "text": "A brown dog running through a grassy field."}
```

The loader also tries to handle common fields such as `caption`, `response`, `file_name`, `image_path`, and LLaVA-style `conversations`.

### 2. Hugging Face datasets

If `--dataset_path` does not exist locally, it is treated as a Hugging Face dataset ID and loaded with `datasets.load_dataset(...)`.

Example:

```bash
python main.py \
  --dataset_path liuhaotian/LLaVA-Pretrain \
  --subset_size 100
```

For public datasets with image filenames instead of resolved image objects, pass `--image_root` to point to the extracted image directory.

## Setup

Use Python 3.10+ and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- This project has been validated with `transformers==4.56.2`.
- Newer Transformers releases may require `torch>=2.6` for older `.bin` checkpoints such as some CLIP variants.
- `bitsandbytes` and `accelerate` are required for 4-bit loading.
- For the most reliable Qwen2.5-VL support, the model card recommends a recent Transformers build, and older versions may fail with `KeyError: 'qwen2_5_vl'`.
- The scripts automatically redirect Hugging Face and matplotlib caches into a local writable `.cache/` directory.
- `qwen-vl-utils` is optional. If installed, the scorer uses it for more faithful Qwen image-input preprocessing.

Optional extra:

```bash
pip install qwen-vl-utils
```

## Quick Demo Dataset

Create a tiny local dataset for smoke testing:

```bash
python scripts/create_demo_dataset.py --output_dir demo_dataset
```

Then run:

```bash
python main.py \
  --dataset_path demo_dataset \
  --subset_size 4 \
  --batch_size 2 \
  --filter \
  --save_clip_embeddings
```

## Scoring Pipeline

Run the full scorer:

```bash
python main.py \
  --dataset_path /path/to/dataset \
  --subset_size 100 \
  --batch_size 4 \
  --filter \
  --composite_threshold 0.65
```

Recommended single-GPU settings:

- Keep the default judge model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Use `--offload_between_models` if GPU memory is tight
- Start with `--subset_size 100` before scaling up

Useful options:

- `--disable_judge`: alignment-only dry run
- `--disable_4bit`: load the judge model without bitsandbytes quantization
- `--local_files_only`: load models only from the local Hugging Face cache
- `--save_clip_embeddings`: save CLIP pair embeddings into `scored_metadata.jsonl`

Practical recommendation:

- On a first run, allow network access so CLIP and Qwen checkpoints can be cached locally.
- On later runs, `--local_files_only` avoids extra Hub requests and is useful in restricted environments.
- If `bitsandbytes` 4-bit loading fails on your Torch build, rerun with `--disable_4bit`.

Outputs:

- `outputs/scored_metadata.jsonl`
- `outputs/filtered_metadata.jsonl` when `--filter` is enabled
- `outputs/scoring_summary.json`

Each scored record contains:

- `alignment_score` in `[0, 1]`
- `image_quality_score` in `[0, 100]`
- `text_quality_score` in `[0, 100]`
- `composite_score` in `[0, 1]`
- `resolved_image_path`, `normalized_text`, and optional error metadata

Composite score formula:

```text
0.4 * alignment_score
+ 0.3 * (image_quality_score / 100)
+ 0.3 * (text_quality_score / 100)
```

## Visualization

After scoring:

```bash
python visualize.py \
  --scored_metadata_path outputs/scored_metadata.jsonl \
  --score_threshold 0.65
```

Saved files:

- `outputs/composite_score_histogram.png`
- `outputs/quality_scatter.png`
- `outputs/best_worst_examples.png`

The scatter plot uses PCA on CLIP pair embeddings and colors samples by the composite-score threshold.

## Tests

Run the lightweight test suite:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility

- Random seeds are fixed via `--seed`
- Model inference runs under `torch.inference_mode()`
- The pipeline skips broken records instead of crashing and stores `quality_error` when a sample cannot be processed

## Practical Notes

- First run will download CLIP and Qwen checkpoints, which can take time and disk space.
- The 7B Qwen2.5-VL model may fit on 24GB VRAM in 4-bit mode, but the 3B model is the safer default.
- If your dataset is large, iterate with `--subset_size` first and scale up once the scoring loop is stable.
