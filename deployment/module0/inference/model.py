from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """Create EfficientNet-B0 with a two-class classifier head for inference."""
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
