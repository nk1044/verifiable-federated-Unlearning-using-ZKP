"""Lightweight MNIST CNN. Small enough for SVD-based SubID and for an eventual ZK circuit."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)   # 28 -> 24 -> pool 12
        self.conv2 = nn.Conv2d(16, 32, 5)  # 12 -> 8  -> pool 4
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)

    def features(self, x, upto: int):
        """Forward through the first `upto` weight layers (0-indexed), returning the feature map.
        Used by the attack to compute z_in = f_{1:hmin-1}(x). Layer order: conv1,conv2,fc1,fc2."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        if upto <= 1:
            return x
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if upto <= 2:
            return x
        x = F.relu(self.fc1(x.flatten(1)))
        return x  # upto >= 3


def weight_layers(model: nn.Module):
    """Ordered (name, module) list of trainable weight layers (the 'layers' h=1..H in the paper)."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def get_vec(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def set_vec(model: nn.Module, vec: torch.Tensor) -> None:
    vector_to_parameters(vec.clone(), model.parameters())


def load(template: nn.Module, vec: torch.Tensor) -> nn.Module:
    m = copy.deepcopy(template)
    set_vec(m, vec)
    return m
