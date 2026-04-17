"""Dataset loading utilities for local folders and Hugging Face datasets."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from .runtime import ensure_dir, read_jsonl

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency at runtime.
    load_dataset = None


IMAGE_FIELDS = (
    "image_path",
    "image",
    "file_name",
    "filename",
    "image_file",
    "file",
    "path",
)
TEXT_FIELDS = (
    "text",
    "caption",
    "response",
    "answer",
    "output",
    "description",
    "blip_caption",
)
METADATA_FILENAMES = (
    "metadata.jsonl",
    "scored_metadata.jsonl",
    "data.jsonl",
    "annotations.jsonl",
)
ASSISTANT_ROLES = {"assistant", "gpt", "model"}
USER_ROLES = {"user", "human"}


@dataclass
class DatasetSample:
    """Normalized representation of one multimodal example."""

    sample_id: str
    text: str
    image_path: str | None
    raw_record: dict[str, Any]
    source: str
    error: str | None = None

    def to_output_record(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the original record."""

        return dict(self.raw_record)


class MultiModalDataLoader:
    """Load multimodal samples from a local folder or Hugging Face dataset."""

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        image_root: str | None = None,
        metadata_filename: str | None = None,
        cache_dir: str | Path = ".cache",
        validate_images: bool = True,
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.validate_images = validate_images
        self.cache_dir = ensure_dir(Path(cache_dir))
        self.materialized_image_dir = ensure_dir(self.cache_dir / "materialized_images")
        self.dataset_kind = "local" if Path(dataset_path).exists() else "huggingface"
        self.image_root = Path(image_root).expanduser().resolve() if image_root else None
        self.metadata_filename = metadata_filename

        self._local_records: list[dict[str, Any]] | None = None
        self._hf_dataset: Any = None
        self._local_dataset_root: Path | None = None
        self._metadata_path: Path | None = None

        if self.dataset_kind == "local":
            self._bootstrap_local_dataset()
        else:
            self._bootstrap_huggingface_dataset()

    def _bootstrap_local_dataset(self) -> None:
        path = Path(self.dataset_path).expanduser().resolve()
        if path.is_file():
            self._metadata_path = path
            self._local_dataset_root = path.parent
        else:
            self._local_dataset_root = path
            self._metadata_path = self._find_metadata_path(path)
        self._local_records = read_jsonl(self._metadata_path)

    def _bootstrap_huggingface_dataset(self) -> None:
        if load_dataset is None:
            raise ImportError(
                "Loading Hugging Face datasets requires the `datasets` package. "
                "Install the project dependencies from requirements.txt first."
            )
        self._hf_dataset = load_dataset(
            self.dataset_path,
            split=self.split,
            cache_dir=str(self.cache_dir / "hf_datasets"),
        )

    def _find_metadata_path(self, dataset_root: Path) -> Path:
        if self.metadata_filename:
            candidate = dataset_root / self.metadata_filename
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Metadata file not found: {candidate}")

        for filename in METADATA_FILENAMES:
            candidate = dataset_root / filename
            if candidate.exists():
                return candidate

        matches = sorted(dataset_root.glob("*.jsonl"))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(
            "Could not locate metadata.jsonl automatically. "
            "Pass --metadata_filename to point at the right JSONL file."
        )

    def __len__(self) -> int | None:
        if self.dataset_kind == "local":
            return len(self._local_records or [])
        try:
            return len(self._hf_dataset)
        except TypeError:
            return None

    def get_total(self, subset_size: int | None = None) -> int | None:
        """Return the number of records that will be processed."""

        total = len(self)
        if total is None:
            return subset_size
        if subset_size is None:
            return total
        return min(total, subset_size)

    def iter_samples(self, subset_size: int | None = None) -> Iterator[DatasetSample]:
        """Yield normalized dataset samples."""

        if self.dataset_kind == "local":
            records: Iterable[dict[str, Any]] = self._local_records or []
        else:
            records = self._hf_dataset

        for index, record in enumerate(records):
            if subset_size is not None and index >= subset_size:
                break
            yield self._normalize_record(record, index)

    def iter_batches(
        self,
        batch_size: int,
        subset_size: int | None = None,
        show_progress: bool = True,
        description: str = "Loading samples",
    ) -> Iterator[list[DatasetSample]]:
        """Yield batches of normalized samples with an optional progress bar."""

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        iterator: Iterable[DatasetSample] = self.iter_samples(subset_size=subset_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=self.get_total(subset_size=subset_size),
                desc=description,
                unit="sample",
            )

        batch: list[DatasetSample] = []
        for sample in iterator:
            batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _normalize_record(self, record: dict[str, Any], index: int) -> DatasetSample:
        sample_id = self._extract_sample_id(record, index)
        text = self._extract_text(record)
        raw_record = self._sanitize_for_json(record)
        image_path, error = self._resolve_image_path(record, sample_id)

        if not text:
            error = self._append_error(error, "missing_text")
        if image_path and self.validate_images:
            validation_error = self._validate_image(image_path)
            if validation_error:
                error = self._append_error(error, validation_error)
        elif not image_path:
            error = self._append_error(error, "missing_image")

        return DatasetSample(
            sample_id=sample_id,
            text=text,
            image_path=image_path,
            raw_record=raw_record,
            source=self.dataset_path,
            error=error,
        )

    def _extract_sample_id(self, record: dict[str, Any], index: int) -> str:
        for key in ("id", "sample_id", "uid", "image_id"):
            value = record.get(key)
            if isinstance(value, (str, int)):
                return str(value)
        image_ref = self._extract_image_reference(record)
        if isinstance(image_ref, str):
            return Path(image_ref).stem or f"sample_{index:06d}"
        return f"sample_{index:06d}"

    def _extract_text(self, record: dict[str, Any]) -> str:
        for field in TEXT_FIELDS:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()

        conversations = record.get("conversations")
        if isinstance(conversations, list):
            assistant_text = self._extract_from_conversations(conversations, ASSISTANT_ROLES)
            if assistant_text:
                return assistant_text
            user_text = self._extract_from_conversations(conversations, USER_ROLES)
            if user_text:
                return user_text

        for field in ("messages", "dialog", "conversation"):
            value = record.get(field)
            if isinstance(value, list):
                extracted = self._extract_from_conversations(value, ASSISTANT_ROLES | USER_ROLES)
                if extracted:
                    return extracted

        return ""

    def _extract_from_conversations(self, turns: list[Any], target_roles: set[str]) -> str:
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or turn.get("from") or "").strip().lower()
            if role not in target_roles:
                continue
            content = turn.get("content", turn.get("value"))
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text.strip())
                joined = " ".join(part for part in parts if part)
                if joined:
                    return joined.replace("<image>", "").strip()
            if isinstance(content, str) and content.strip():
                return content.replace("<image>", "").strip()
        return ""

    def _extract_image_reference(self, record: dict[str, Any]) -> Any:
        for field in IMAGE_FIELDS:
            if field in record:
                return record[field]
        return None

    def _resolve_image_path(self, record: dict[str, Any], sample_id: str) -> tuple[str | None, str | None]:
        image_reference = self._extract_image_reference(record)
        if image_reference is None:
            return None, "missing_image"

        if isinstance(image_reference, str):
            return self._resolve_string_image_reference(image_reference), None

        if isinstance(image_reference, Image.Image):
            return str(self._materialize_pil_image(image_reference, sample_id)), None

        if isinstance(image_reference, dict):
            if "path" in image_reference and isinstance(image_reference["path"], str):
                return self._resolve_string_image_reference(image_reference["path"]), None
            if "bytes" in image_reference and image_reference["bytes"] is not None:
                try:
                    image = Image.open(io.BytesIO(image_reference["bytes"])).convert("RGB")
                except (UnidentifiedImageError, OSError) as error:
                    return None, f"corrupted_image:{error}"
                return str(self._materialize_pil_image(image, sample_id)), None

        return None, "unsupported_image_format"

    def _resolve_string_image_reference(self, image_reference: str) -> str | None:
        image_reference = image_reference.strip()
        if not image_reference:
            return None

        candidate = Path(image_reference).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return str(candidate.resolve())

        if self.image_root is not None:
            rooted = self.image_root / image_reference
            if rooted.exists():
                return str(rooted.resolve())

        if self._local_dataset_root is not None:
            rooted = self._local_dataset_root / image_reference
            if rooted.exists():
                return str(rooted.resolve())

        if candidate.exists():
            return str(candidate.resolve())

        return str(candidate)

    def _materialize_pil_image(self, image: Image.Image, sample_id: str) -> Path:
        target_path = self.materialized_image_dir / f"{sample_id}.png"
        if not target_path.exists():
            image.convert("RGB").save(target_path)
        return target_path

    def _validate_image(self, image_path: str) -> str | None:
        path = Path(image_path)
        if not path.exists():
            return "missing_image_file"
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            return f"corrupted_image:{error}"
        return None

    def _append_error(self, current_error: str | None, new_error: str | None) -> str | None:
        if not new_error:
            return current_error
        if not current_error:
            return new_error
        if new_error in current_error:
            return current_error
        return f"{current_error};{new_error}"

    def _sanitize_for_json(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Image.Image):
            return {"_type": "PIL.Image", "mode": value.mode, "size": list(value.size)}
        if isinstance(value, bytes):
            return {"_type": "bytes", "length": len(value)}
        if isinstance(value, dict):
            return {str(key): self._sanitize_for_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_for_json(item) for item in value]
        return str(value)
