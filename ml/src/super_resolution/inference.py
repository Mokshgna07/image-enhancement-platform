from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from .checkpoints import CheckpointManager
from .config import load_config
from .image_utils import pil_to_tensor
from .model import EDSR


class InferenceError(RuntimeError):
    """Base exception for inference failures."""


class ImageValidationError(InferenceError):
    """Raised when an input image is invalid."""


class ModelConfigurationError(InferenceError):
    """Raised when model configuration/checkpoint is invalid."""


@dataclass(frozen=True)
class InferenceResult:
    output_path: str
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    input_format: str
    output_format: str
    processing_time_ms: float
    model_version: str
    scale: int
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "scale": self.scale,
            "device": self.device,
        }


class SuperResolutionService:
    """
    Production-style interface around the trained super-resolution model.

    The frontend/backend only interacts with this service and never needs
    access to the underlying PyTorch model.
    """

    SUPPORTED_SCALES = {2, 4}
    SUPPORTED_INPUT_FORMATS = {
        "JPEG",
        "PNG",
        "WEBP",
        "BMP",
        "TIFF",
    }
    MAX_IMAGE_PIXELS = 25_000_000

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_paths: dict[int, str | Path],
        device: str = "auto",
        model_version: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise ModelConfigurationError(
                f"Model configuration not found: {self.config_path}"
            )

        try:
            self.config = load_config(self.config_path)
        except Exception as exc:
            raise ModelConfigurationError(
                f"Failed to load model configuration: {exc}"
            ) from exc

        self.device = self._resolve_device(device)
        self.checkpoint_paths = {
            int(scale): Path(path)
            for scale, path in checkpoint_paths.items()
        }

        self.model_version = (
            model_version
            or self.config.get("experiment", {}).get(
                "name",
                "unknown",
            )
        )

        self._models: dict[int, EDSR] = {}

    def _resolve_device(self, device: str) -> torch.device:
        device = device.lower()

        if device not in {"auto", "cpu", "cuda"}:
            raise ModelConfigurationError(
                "device must be one of: auto, cpu, cuda"
            )

        if device == "cuda":
            if not torch.cuda.is_available():
                raise ModelConfigurationError(
                    "CUDA was requested but no CUDA device is available."
                )
            return torch.device("cuda")

        if device == "cpu":
            return torch.device("cpu")

        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _load_model(self, scale: int) -> EDSR:
        if scale not in self.SUPPORTED_SCALES:
            raise InferenceError(
                f"Unsupported scale '{scale}'. "
                f"Supported scales: {sorted(self.SUPPORTED_SCALES)}"
            )

        if scale in self._models:
            return self._models[scale]

        checkpoint_path = self.checkpoint_paths.get(scale)

        if checkpoint_path is None:
            raise ModelConfigurationError(
                f"No checkpoint configured for {scale}x inference."
            )

        if not checkpoint_path.exists():
            raise ModelConfigurationError(
                f"Checkpoint not found for {scale}x: {checkpoint_path}"
            )

        model_config = self.config.get("model", {})

        model = EDSR(
            scale=scale,
            channels=int(model_config.get("channels", 64)),
            num_blocks=int(model_config.get("num_blocks", 16)),
            residual_scale=float(
                model_config.get("residual_scale", 0.1)
            ),
        )

        try:
            CheckpointManager.load(
                checkpoint_path,
                model=model,
                device="cpu",
            )
        except KeyError as exc:
            raise ModelConfigurationError(
                f"Checkpoint does not contain the expected model weights: "
                f"{checkpoint_path}"
            ) from exc
        except RuntimeError as exc:
            raise ModelConfigurationError(
                f"Checkpoint is incompatible with the {scale}x model: "
                f"{checkpoint_path}"
            ) from exc
        except Exception as exc:
            raise ModelConfigurationError(
                f"Failed to load checkpoint: {checkpoint_path}"
            ) from exc

        model.to(self.device)
        model.eval()

        self._models[scale] = model

        return model

    def validate_image(self, image_path: str | Path) -> Image.Image:
        path = Path(image_path)

        if not path.exists():
            raise ImageValidationError(
                f"Input image does not exist: {path}"
            )

        if not path.is_file():
            raise ImageValidationError(
                f"Input path is not a file: {path}"
            )

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                image.load()
                image_format = image.format
                width, height = image.size

                if image_format not in self.SUPPORTED_INPUT_FORMATS:
                    raise ImageValidationError(
                        f"Unsupported image format: {image_format}"
                    )

                if width <= 0 or height <= 0:
                    raise ImageValidationError(
                        "Image dimensions must be positive."
                    )

                if width * height > self.MAX_IMAGE_PIXELS:
                    raise ImageValidationError(
                        "Image exceeds the maximum supported pixel count."
                    )

                converted = image.convert("RGB").copy()
                converted.info["source_format"] = image_format

                return converted

        except UnidentifiedImageError as exc:
            raise ImageValidationError(
                f"File is not a valid image: {path}"
            ) from exc

        except ImageValidationError:
            raise

        except Exception as exc:
            raise ImageValidationError(
                f"Failed to validate image: {path}"
            ) from exc

    @staticmethod
    def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.detach().cpu().clamp(0.0, 1.0)

        array = (
            tensor.permute(1, 2, 0).numpy() * 255.0
        ).round().astype("uint8")

        return Image.fromarray(array, mode="RGB")

    def enhance(
        self,
        image_path: str | Path,
        output_path: str | Path,
        scale: int = 4,
    ) -> InferenceResult:
        """
        Enhance an image and save the result.

        This is the main public interface intended for FastAPI.
        """

        if scale not in self.SUPPORTED_SCALES:
            raise InferenceError(
                f"Unsupported scale '{scale}'. "
                f"Supported scales: 2x and 4x."
            )

        image = self.validate_image(image_path)

        input_width, input_height = image.size
        input_format = image.info.get("source_format", "UNKNOWN")

        model = self._load_model(scale)

        tensor = pil_to_tensor(image).unsqueeze(0)
        tensor = tensor.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            output = model(tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        enhanced = self._tensor_to_image(output[0])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            enhanced.save(
                output_path,
                format="PNG",
            )
        except Exception as exc:
            raise InferenceError(
                f"Failed to save enhanced image: {output_path}"
            ) from exc

        output_width, output_height = enhanced.size

        return InferenceResult(
            output_path=str(output_path),
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
            input_format=input_format,
            output_format="PNG",
            processing_time_ms=round(elapsed_ms, 3),
            model_version=self.model_version,
            scale=scale,
            device=str(self.device),
        )


def build_service_from_environment() -> SuperResolutionService:
    """
    Build the inference service from environment variables.

    Required:
        SR_CONFIG_PATH

    4x checkpoint:
        SR_CHECKPOINT_4X

    Optional 2x checkpoint:
        SR_CHECKPOINT_2X

    Device:
        SR_DEVICE=auto|cpu|cuda
    """

    config_path = os.getenv(
        "SR_CONFIG_PATH",
        "ml/configs/training.yaml",
    )

    checkpoint_paths: dict[int, str] = {}

    checkpoint_4x = os.getenv("SR_CHECKPOINT_4X")
    checkpoint_2x = os.getenv("SR_CHECKPOINT_2X")

    if checkpoint_4x:
        checkpoint_paths[4] = checkpoint_4x

    if checkpoint_2x:
        checkpoint_paths[2] = checkpoint_2x

    return SuperResolutionService(
        config_path=config_path,
        checkpoint_paths=checkpoint_paths,
        device=os.getenv("SR_DEVICE", "auto"),
        model_version=os.getenv(
            "SR_MODEL_VERSION",
            "edsr_x4_baseline",
        ),
    )
