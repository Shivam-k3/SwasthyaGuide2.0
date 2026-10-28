"""DenseNet121 pneumonia classifier with consistent grayscale adaptation."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_model(pretrained: bool = True) -> 'PneumoniaNet':
    """torchvision DenseNet121, ImageNet weights optional, 1-logit head.

    `pretrained=True` loads DenseNet121_Weights.IMAGENET1K_V1 (downloads on
    first use - never during unit tests, which always pass False).
    """
    from torchvision.models import densenet121, DenseNet121_Weights
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = densenet121(weights=weights)
    backbone.classifier = nn.Linear(backbone.classifier.in_features, 1)
    return PneumoniaNet(backbone)


class PneumoniaNet(nn.Module):
    """Wraps a 3-channel classifier for single-channel input.

    The grayscale image is replicated to 3 channels so the ImageNet-pretrained
    convolution weights are used unchanged - one fixed adaptation applied
    identically at train, validation and test time. Input (B, 1, H, W) ->
    logits (B, 1).
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f'expected (B, 1, H, W) grayscale batch, got {tuple(x.shape)}')
        return self.net(x.repeat(1, 3, 1, 1))
