#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash coco.sh
#   CONDA_ENV_NAME=rep_lm_cuda bash coco.sh
#   SUBSET_SIZE=1000 OUTDIR=outputs/coco_val_1k bash coco.sh
#   USE_LOCAL_FILES_ONLY=1 bash coco.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
DATASET_ID="${DATASET_ID:-astro21/coco-caption-split}"
DATASET_SPLIT="${DATASET_SPLIT:-val}"
SUBSET_SIZE="${SUBSET_SIZE:-5000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
THRESHOLD="${THRESHOLD:-0.65}"
OUTDIR="${OUTDIR:-outputs/coco_val_5k}"
CACHE_DIR="${CACHE_DIR:-.cache}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
USE_LOCAL_FILES_ONLY="${USE_LOCAL_FILES_ONLY:-0}"
ENABLE_FILTER="${ENABLE_FILTER:-1}"
SAVE_CLIP_EMBEDDINGS="${SAVE_CLIP_EMBEDDINGS:-1}"
OFFLOAD_BETWEEN_MODELS="${OFFLOAD_BETWEEN_MODELS:-1}"
DISABLE_4BIT="${DISABLE_4BIT:-1}"

export HF_HOME="$PROJECT_ROOT/$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MPLCONFIGDIR="$PROJECT_ROOT/$CACHE_DIR/matplotlib"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

run_python() {
  if [[ -n "$CONDA_ENV_NAME" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "Error: CONDA_ENV_NAME is set but 'conda' is not available in PATH." >&2
      exit 1
    fi
    conda run -n "$CONDA_ENV_NAME" python "$@"
    return
  fi

  if ! command -v python >/dev/null 2>&1; then
    echo "Error: python is not available in PATH. Activate your conda env first or set CONDA_ENV_NAME." >&2
    exit 1
  fi
  python "$@"
}

main_args=(
  main.py
  --dataset_path "$DATASET_ID"
  --dataset_split "$DATASET_SPLIT"
  --subset_size "$SUBSET_SIZE"
  --batch_size "$BATCH_SIZE"
  --composite_threshold "$THRESHOLD"
  --output_dir "$OUTDIR"
  --cache_dir "$CACHE_DIR"
  --clip_model_name "$CLIP_MODEL_NAME"
  --judge_model_name "$JUDGE_MODEL_NAME"
)

if [[ "$ENABLE_FILTER" == "1" ]]; then
  main_args+=(--filter)
fi

if [[ "$SAVE_CLIP_EMBEDDINGS" == "1" ]]; then
  main_args+=(--save_clip_embeddings)
fi

if [[ "$OFFLOAD_BETWEEN_MODELS" == "1" ]]; then
  main_args+=(--offload_between_models)
fi

if [[ "$DISABLE_4BIT" == "1" ]]; then
  main_args+=(--disable_4bit)
fi

if [[ "$USE_LOCAL_FILES_ONLY" == "1" ]]; then
  main_args+=(--local_files_only)
fi

echo "Running multimodal quality evaluation with:"
echo "  dataset: $DATASET_ID [$DATASET_SPLIT]"
echo "  subset_size: $SUBSET_SIZE"
echo "  batch_size: $BATCH_SIZE"
echo "  threshold: $THRESHOLD"
echo "  output_dir: $OUTDIR"
if [[ -n "$CONDA_ENV_NAME" ]]; then
  echo "  conda_env: $CONDA_ENV_NAME"
else
  echo "  conda_env: current shell"
fi

run_python "${main_args[@]}"

run_python visualize.py \
  --scored_metadata_path "$OUTDIR/scored_metadata.jsonl" \
  --output_dir "$OUTDIR" \
  --cache_dir "$CACHE_DIR"

echo "Done. Outputs are in $OUTDIR"
