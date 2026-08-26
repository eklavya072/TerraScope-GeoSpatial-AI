"""Model construction. One function, so no architecture can get special treatment."""

import timm
import torch

from bench.config import CLASSES, MODEL_ZOO


def build(name: str, pretrained: bool = True) -> torch.nn.Module:
    """Instantiate a zoo model with a fresh 10-class head.

    Every model is created the same way: ImageNet-pretrained weights, classifier
    replaced, all layers trainable. No per-model freezing schedules.
    """
    if name not in MODEL_ZOO:
        raise SystemExit(f"unknown model {name!r}; choose from {list(MODEL_ZOO)}")
    model = timm.create_model(MODEL_ZOO[name], pretrained=pretrained,
                              num_classes=len(CLASSES))
    for p in model.parameters():
        p.requires_grad = True
    return model


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
