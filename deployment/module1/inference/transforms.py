from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RemoveBlackMargin:
    def __call__(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("L"))
        mask = arr > 5
        if not mask.any():
            return image
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        return image.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


class NotImplementedPreprocess:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError(
            f"{self.name} is intentionally disabled in the baseline. "
            "Enable only after validating it is appropriate for these 2D JPG/PNG images."
        )


def build_inference_transforms(config: dict):
    image_size = int(config["input"]["image_size"])
    preprocessing = config.get("preprocessing", {})
    ops: list = []

    if preprocessing.get("enable_margin_removal", False):
        ops.append(RemoveBlackMargin())
    if preprocessing.get("enable_skull_stripping", False):
        ops.append(NotImplementedPreprocess("Skull stripping"))
    if preprocessing.get("enable_n4", False):
        ops.append(NotImplementedPreprocess("N4 bias-field correction"))

    ops.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)
