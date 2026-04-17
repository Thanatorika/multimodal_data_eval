"""Model-backed scoring for multimodal training examples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, CLIPModel, CLIPProcessor

from .runtime import clamp, ensure_dir

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover - optional dependency at runtime.
    process_vision_info = None


DEFAULT_IMAGE_PROMPT = """You are rating the visual quality of an image for multimodal instruction-tuning data.
Score the image from 0 to 100.

Criteria:
- Clear, recognizable main subject
- Sufficient sharpness, lighting, and resolution
- No severe corruption, heavy artifacts, or unreadable content
- Useful visual evidence for supervision

Return strict JSON with keys:
{"score": <number>, "reason": "<short reason>"}"""

DEFAULT_TEXT_PROMPT = """You are rating the text quality of a caption or answer for multimodal instruction-tuning data.
Score the text from 0 to 100.

Criteria:
- Fluent and grammatically coherent
- Specific and informative
- Not repetitive, templated, or nonsensical
- Suitable as supervision text for vision-language training

Return strict JSON with keys:
{"score": <number>, "reason": "<short reason>"}"""


class MultiModalQualityScorer:
    """Compute alignment, image quality, text quality, and composite scores."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        judge_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        cache_dir: str | Path = ".cache",
        device: str | None = None,
        clip_device: str | None = None,
        judge_device: str | None = None,
        load_judge_model: bool = True,
        use_4bit: bool = True,
        offload_between_models: bool = False,
        alignment_weight: float = 0.4,
        image_quality_weight: float = 0.3,
        text_quality_weight: float = 0.3,
        max_new_tokens: int = 96,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        local_files_only: bool = False,
    ) -> None:
        self.cache_dir = ensure_dir(Path(cache_dir))
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clip_device = clip_device or self.device
        self.judge_device = judge_device or self.device
        self.clip_model_name = clip_model_name
        self.judge_model_name = judge_model_name
        self.load_judge_model_flag = load_judge_model
        self.use_4bit = use_4bit and torch.cuda.is_available()
        self.offload_between_models = offload_between_models and torch.cuda.is_available()
        self.alignment_weight = alignment_weight
        self.image_quality_weight = image_quality_weight
        self.text_quality_weight = text_quality_weight
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.local_files_only = local_files_only

        self.clip_model: CLIPModel | None = None
        self.clip_processor: CLIPProcessor | None = None
        self.judge_model: Any = None
        self.judge_processor: Any = None
        self._clip_target_device = self.clip_device

        self._load_clip_components()
        if self.load_judge_model_flag:
            self._load_judge_components()

    def _resolve_model_source(self, model_name: str, required_filename: str | None = None) -> str:
        if not self.local_files_only:
            return model_name

        model_cache_dir = self.cache_dir / "hf_models" / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = model_cache_dir / "snapshots"
        refs_main = model_cache_dir / "refs" / "main"

        snapshot_candidates: list[Path] = []
        if snapshots_dir.exists():
            snapshot_candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True)

        if required_filename:
            for snapshot_path in snapshot_candidates:
                if (snapshot_path / required_filename).exists():
                    return str(snapshot_path)

        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_path = snapshots_dir / revision
            if snapshot_path.exists():
                return str(snapshot_path)

        if snapshot_candidates:
            return str(snapshot_candidates[0])

        return model_name

    def _load_clip_components(self) -> None:
        clip_processor_source = self._resolve_model_source(
            self.clip_model_name,
            required_filename="preprocessor_config.json",
        )
        clip_model_source = self._resolve_model_source(
            self.clip_model_name,
            required_filename="model.safetensors",
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            clip_processor_source,
            cache_dir=str(self.cache_dir / "hf_models"),
            local_files_only=self.local_files_only,
        )
        self.clip_model = CLIPModel.from_pretrained(
            clip_model_source,
            cache_dir=str(self.cache_dir / "hf_models"),
            local_files_only=self.local_files_only,
            use_safetensors=True,
        )
        self.clip_model.eval()
        self.clip_model.to(self._clip_target_device)

    def _load_judge_components(self) -> None:
        if self.judge_model is not None and self.judge_processor is not None:
            return

        model_class = self._resolve_judge_model_class()
        judge_processor_source = self._resolve_model_source(
            self.judge_model_name,
            required_filename="processor_config.json",
        )
        judge_model_source = self._resolve_model_source(
            self.judge_model_name,
            required_filename="model.safetensors",
        )
        self.judge_processor = AutoProcessor.from_pretrained(
            judge_processor_source,
            cache_dir=str(self.cache_dir / "hf_models"),
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        quantization_config = None
        torch_dtype: torch.dtype | str | None = torch.float16 if torch.cuda.is_available() else torch.float32
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir / "hf_models"),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "use_safetensors": True,
            "local_files_only": self.local_files_only,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["low_cpu_mem_usage"] = True

        model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
        try:
            self.judge_model = model_class.from_pretrained(judge_model_source, **model_kwargs)
        except Exception as error:
            if self.use_4bit and "bitsandbytes" in str(error).lower():
                raise RuntimeError(
                    "Failed to load the judge model in 4-bit mode. "
                    "Install `bitsandbytes` and `accelerate`, or rerun with --disable_4bit."
                ) from error
            raise
        self.judge_model.eval()

    def _resolve_judge_model_class(self) -> Any:
        import transformers

        for class_name in (
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            model_class = getattr(transformers, class_name, None)
            if model_class is not None:
                return model_class
        raise ImportError(
            "Could not find a Qwen2.5-VL-compatible model class. "
            "Install `transformers>=4.57.0` to use the judge model."
        )

    def score_sample(self, image_path: str | Path, text: str) -> dict[str, Any]:
        """Score one image-text pair."""

        scores = self.score_batch(
            [{"image_path": str(image_path), "text": text}],
            return_clip_embeddings=False,
        )
        return scores[0]

    def score_batch(
        self,
        samples: Sequence[dict[str, Any]],
        return_clip_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """Score a batch of normalized samples."""

        if not samples:
            return []

        image_paths = [str(sample["image_path"]) for sample in samples]
        texts = [str(sample["text"]) for sample in samples]
        alignment_scores, pair_embeddings = self.compute_alignment_batch(
            image_paths=image_paths,
            texts=texts,
            return_pair_embeddings=return_clip_embeddings,
        )

        clip_offloaded = False
        if self.offload_between_models:
            self._move_clip("cpu")
            clip_offloaded = True

        outputs: list[dict[str, Any]] = []
        try:
            for index, sample in enumerate(samples):
                image_quality_score: float | None = None
                text_quality_score: float | None = None
                image_reason = ""
                text_reason = ""

                if self.load_judge_model_flag:
                    image_quality_score, image_reason = self._score_image_quality(sample["image_path"])
                    text_quality_score, text_reason = self._score_text_quality(sample["text"])

                record = {
                    "alignment_score": round(float(alignment_scores[index]), 6),
                    "image_quality_score": image_quality_score,
                    "text_quality_score": text_quality_score,
                    "composite_score": round(
                        self._compute_composite_score(
                            alignment_score=float(alignment_scores[index]),
                            image_quality_score=image_quality_score,
                            text_quality_score=text_quality_score,
                        ),
                        6,
                    ),
                    "image_quality_reason": image_reason,
                    "text_quality_reason": text_reason,
                }
                if return_clip_embeddings and pair_embeddings is not None:
                    record["clip_pair_embedding"] = [round(float(value), 6) for value in pair_embeddings[index]]
                outputs.append(record)
        finally:
            if clip_offloaded:
                self._move_clip(self._clip_target_device)
        return outputs

    @torch.inference_mode()
    def compute_alignment_batch(
        self,
        image_paths: Sequence[str | Path],
        texts: Sequence[str],
        return_pair_embeddings: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute CLIP alignment scores and optional pair embeddings."""

        if self.clip_model is None or self.clip_processor is None:
            raise RuntimeError("CLIP components are not initialized.")

        images = [self._load_image(path) for path in image_paths]
        inputs = self.clip_processor(
            text=list(texts),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {name: tensor.to(self._clip_target_device) for name, tensor in inputs.items()}

        outputs = self.clip_model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds.float(), dim=-1)
        text_embeds = F.normalize(outputs.text_embeds.float(), dim=-1)
        cosine_scores = (image_embeds * text_embeds).sum(dim=-1)
        alignment_scores = ((cosine_scores + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()

        pair_embeddings = None
        if return_pair_embeddings:
            pair_embeddings = F.normalize(image_embeds + text_embeds, dim=-1).cpu().numpy()
        return alignment_scores, pair_embeddings

    def compute_pair_embeddings(
        self,
        image_paths: Sequence[str | Path],
        texts: Sequence[str],
    ) -> np.ndarray:
        """Convenience wrapper used by the visualization script."""

        _, pair_embeddings = self.compute_alignment_batch(
            image_paths=image_paths,
            texts=texts,
            return_pair_embeddings=True,
        )
        if pair_embeddings is None:
            raise RuntimeError("Pair embeddings were not generated.")
        return pair_embeddings

    def _score_image_quality(self, image_path: str | Path) -> tuple[float, str]:
        prompt = DEFAULT_IMAGE_PROMPT
        raw_response = self._generate_judge_response(prompt=prompt, image_path=image_path)
        return self._parse_judge_response(raw_response)

    def _score_text_quality(self, text: str) -> tuple[float, str]:
        prompt = f"{DEFAULT_TEXT_PROMPT}\n\nText to rate:\n{text.strip()}"
        raw_response = self._generate_judge_response(prompt=prompt, image_path=None)
        return self._parse_judge_response(raw_response)

    def _build_messages(self, prompt: str, image_path: str | Path | None) -> list[dict[str, Any]]:
        content: list[dict[str, str]] = []
        if image_path is not None:
            content.append({"type": "image", "image": Path(image_path).expanduser().resolve().as_uri()})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @torch.inference_mode()
    def _generate_judge_response(self, prompt: str, image_path: str | Path | None) -> str:
        if self.judge_model is None or self.judge_processor is None:
            raise RuntimeError("Judge model was not loaded. Initialize with load_judge_model=True.")

        messages = self._build_messages(prompt=prompt, image_path=image_path)
        chat_text = self.judge_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_inputs: dict[str, Any] = {
            "text": [chat_text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_path is not None:
            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
                if image_inputs is not None:
                    processor_inputs["images"] = image_inputs
                if video_inputs is not None:
                    processor_inputs["videos"] = video_inputs
            else:
                processor_inputs["images"] = [self._load_image(image_path)]

        inputs = self.judge_processor(**processor_inputs)
        inputs = {name: tensor.to(self.judge_device) for name, tensor in inputs.items()}

        generated_ids = self.judge_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        trimmed_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return self.judge_processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

    def _parse_judge_response(self, response_text: str) -> tuple[float, str]:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        reason = response_text.strip()
        if match:
            try:
                payload = json.loads(match.group(0))
                score = float(payload["score"])
                reason = str(payload.get("reason", "")).strip()
                return clamp(score, 0.0, 100.0), reason
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        number_match = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)", response_text)
        if not number_match:
            raise ValueError(f"Could not parse judge response: {response_text}")
        score = float(number_match.group(1))
        return clamp(score, 0.0, 100.0), reason

    def _compute_composite_score(
        self,
        alignment_score: float,
        image_quality_score: float | None,
        text_quality_score: float | None,
    ) -> float:
        weighted_sum = alignment_score * self.alignment_weight
        total_weight = self.alignment_weight

        if image_quality_score is not None:
            weighted_sum += (image_quality_score / 100.0) * self.image_quality_weight
            total_weight += self.image_quality_weight
        if text_quality_score is not None:
            weighted_sum += (text_quality_score / 100.0) * self.text_quality_weight
            total_weight += self.text_quality_weight

        return weighted_sum / total_weight if total_weight else 0.0

    def _move_clip(self, target_device: str) -> None:
        if self.clip_model is None:
            return
        self.clip_model.to(target_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_image(self, image_path: str | Path) -> Image.Image:
        path = Path(image_path)
        with Image.open(path) as image:
            return image.convert("RGB")
