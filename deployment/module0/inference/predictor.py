from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from .model import create_model
from .transforms import build_inference_transforms

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for Module 0 inference.") from exc


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "inference.yaml"


def _resolve_package_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PACKAGE_ROOT / candidate


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _resolve_package_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Module 0 configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid Module 0 configuration file: {path}")
    return config


class MRIGatePredictor:
    """EfficientNet-B0 gatekeeper that accepts only likely brain MRI images."""

    VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = _load_config(config_path)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module_config = self.config["module0"]
        self.checkpoint_path = _resolve_package_path(checkpoint_path or module_config["checkpoint"])
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Module 0 checkpoint not found: {self.checkpoint_path}")
        self.threshold = float(module_config["brain_mri_threshold"])
        self.model_name = str(module_config["model_name"])
        self.model_version = str(module_config["version"])
        self.class_to_idx = {str(k): int(v) for k, v in self.config["class_to_idx"].items()}
        self.idx_to_key = {idx: key for key, idx in self.class_to_idx.items()}
        self.display_names = {str(k): str(v) for k, v in self.config["display_names"].items()}

        self.model = create_model(num_classes=2, pretrained=False).to(self.device)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform = build_inference_transforms(self.config)

    def _validate_and_open_image(self, image_path: str | Path) -> Image.Image:
        if not isinstance(image_path, (str, Path)):
            raise TypeError("image_path must be a string or pathlib.Path.")
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Invalid image path: {path}")
        suffix = path.suffix.lower()
        if suffix not in self.VALID_SUFFIXES:
            raise ValueError(f"Unsupported image format '{suffix}'. Expected one of: {sorted(self.VALID_SUFFIXES)}.")
        try:
            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise ValueError(f"Image has invalid dimensions: {path}")
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Image could not be opened or decoded: {path}") from exc

    def predict_pil(self, image: Image.Image) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL.Image.Image object.")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
        brain_prob = float(probabilities[self.class_to_idx["brain_mri"]])
        non_brain_prob = float(probabilities[self.class_to_idx["non_brain_mri"]])
        accepted = brain_prob >= self.threshold
        class_key = "brain_mri" if accepted else "non_brain_mri"
        class_id = self.class_to_idx[class_key]
        confidence = brain_prob if accepted else non_brain_prob
        result: dict[str, Any] = {
            "status": "accepted" if accepted else "rejected",
            "module": "module0",
            "model": {"name": self.model_name, "version": self.model_version},
            "validation": {
                "class": self.display_names[class_key],
                "class_id": int(class_id),
                "confidence": float(confidence),
                "brain_mri_probability": float(brain_prob),
                "threshold": float(self.threshold),
            },
        }
        if not accepted:
            result["reason"] = "Input image was classified as non-brain-MRI"
        return result

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        return self.predict_pil(self._validate_and_open_image(image_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 0 brain MRI gatekeeper inference.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    predictor = MRIGatePredictor(args.checkpoint, args.config, args.device)
    print(json.dumps(predictor.predict(args.image), indent=2))


if __name__ == "__main__":
    main()
