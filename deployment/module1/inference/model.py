from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


def create_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    try:
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Could not load EfficientNetV2-S pretrained weights. "
                "For deployment inference, use pretrained=False before loading the checkpoint."
            ) from exc
        raise
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
