from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .model import create_model
from .transforms import build_inference_transforms

try:
    import yaml
except ImportError:
    yaml = None


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "inference.yaml"


def _resolve_package_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _resolve_package_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        if yaml is not None:
            config = yaml.safe_load(file)
        else:
            config = _load_simple_yaml(file.read())
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration file: {path}")
    return config


def _load_simple_yaml(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            key, value = _split_yaml_key_value(line)
            if value == "":
                current_section = {}
                config[key] = current_section
            else:
                config[key] = _parse_yaml_scalar(value)
                current_section = None
            continue
        if current_section is None:
            raise ValueError("Invalid inference YAML structure.")
        key, value = _split_yaml_key_value(line.strip())
        current_section[_parse_yaml_scalar(key)] = _parse_yaml_scalar(value)

    return config


def _split_yaml_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError("Invalid inference YAML line.")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_yaml_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value


class TumorClassifier:
    """Reusable inference wrapper for the validated EfficientNetV2-S checkpoint."""

    VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = _load_config(config_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint_config = self.config["deployment"]["checkpoint"]
        self.checkpoint_path = _resolve_package_path(checkpoint_path or checkpoint_config)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.checkpoint_path.is_file():
            raise ValueError(f"Invalid checkpoint path: {self.checkpoint_path}")

        self.num_classes = int(self.config["model"]["num_classes"])
        self.idx_to_display = {int(idx): str(name) for idx, name in self.config["classes"].items()}
        self.class_names = [self.idx_to_display[idx] for idx in range(self.num_classes)]
        self.model_name = str(self.config["model"]["display_name"])
        self.model_version = str(self.config["model"]["version"])

        try:
            self.model = create_model(num_classes=self.num_classes, pretrained=False)
            self.model.to(self.device)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and any(
                key.startswith("features") or key.startswith("classifier") for key in checkpoint
            ):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from checkpoint '{self.checkpoint_path}': {exc}") from exc

        self.model.eval()
        self.transform = build_inference_transforms(self.config)

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: str | torch.device) -> None:
        self._device = torch.device(value)

    def _validate_and_open_image(self, image_path: str | Path) -> Image.Image:
        if not isinstance(image_path, (str, Path)):
            raise TypeError("image_path must be a string or pathlib.Path pointing to an image file.")

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
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Image could not be opened or decoded: {path}") from exc

    def _predict_from_rgb(self, image: Image.Image) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("Invalid PIL image. Expected a PIL.Image.Image object.")

        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        probabilities_np = probabilities.detach().numpy()
        top1_idx = int(np.argmax(probabilities_np))
        top2_idx = int(np.argsort(probabilities_np)[-2])
        top1_probability = float(probabilities_np[top1_idx])
        top2_probability = float(probabilities_np[top2_idx])

        probability_map: dict[str, float] = {}
        for idx in range(self.num_classes):
            class_name = self.idx_to_display[idx]
            probability_map[class_name] = float(probabilities_np[idx])

        return {
            "status": "success",
            "model": {
                "name": self.model_name,
                "version": self.model_version,
            },
            "prediction": {
                "class": self.idx_to_display[top1_idx],
                "class_id": int(top1_idx),
                "confidence": top1_probability,
            },
            "probabilities": probability_map,
            "uncertainty": {
                "top1_probability": top1_probability,
                "top2_probability": top2_probability,
                "prediction_margin": top1_probability - top2_probability,
            },
        }

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """Run inference on an image stored at a filesystem path."""
        rgb_image = self._validate_and_open_image(image_path)
        return self._predict_from_rgb(rgb_image)

    def predict_pil(self, image: Image.Image) -> dict[str, Any]:
        """Run inference on a PIL.Image.Image object."""
        if not isinstance(image, Image.Image):
            raise TypeError("Unsupported image type. Expected a PIL.Image.Image object.")
        return self._predict_from_rgb(image)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference for the NeuroScan AI Module 1 tumor classifier.")
    parser.add_argument("--image", required=True, help="Path to a JPG or PNG MRI image.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override.")
    parser.add_argument("--config", default=None, help="Optional inference config path override.")
    parser.add_argument("--device", default=None, help="Override device: cuda or cpu.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    classifier = TumorClassifier(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    result = classifier.predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
