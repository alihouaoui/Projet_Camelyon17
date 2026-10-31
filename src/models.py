"""Modèles de classification de patchs."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """CNN entraîné de zéro : 4 blocs conv-BN-ReLU-pool, puis pooling global."""

    def __init__(self, n_classes: int = 2, width: int = 32):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (width, width * 2, width * 4, width * 8):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, pretrained: bool = True, n_classes: int = 2) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(n_classes)
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    raise ValueError(f"Modèle inconnu : {name}")
