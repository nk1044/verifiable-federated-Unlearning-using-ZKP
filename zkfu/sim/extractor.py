"""Frozen feature extractor phi(x), pre-trained ONCE on public MNIST data, then frozen. Not proved.

Two variants (D = 64 in both):
  * "cnn+grid" (default): 48 CNN features (centred, L2-normalised to 2.5) + 16 fixed region
    intensities (mean of each 7x7 cell, centred, x2). The region features keep coarse spatial
    information, so a head-only backdoor with the paper's 3x3 corner trigger is POSSIBLE.
  * "cnn": 64 CNN features (centred, L2-normalised to 3). The clean-pretrained CNN discards the
    corner trigger: even a head trained on 10% poisoned data reaches only ~1% BA. Kept as an
    ablation: with this extractor the frozen-feature design already blunts FUsion.
Scales are chosen so the proved head trains stably with eta/B = 2^-8 (DESIGN_NOTES.md §2).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import D, WORK, MNIST_DIR, quantize


class CNN(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc = nn.Linear(32 * 4 * 4, out_dim)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.flatten(1))


def load_mnist():
    from torchvision import datasets
    tr = datasets.MNIST(MNIST_DIR, train=True, download=True)
    te = datasets.MNIST(MNIST_DIR, train=False, download=True)
    f = lambda d: (d.data.float().div(255).unsqueeze(1), d.targets.long())  # noqa: E731
    return (*f(tr), *f(te))


def add_trigger(x: torch.Tensor) -> torch.Tensor:
    """The FUsion paper's MNIST trigger: 3x3 white square in the bottom-right corner."""
    x = x.clone()
    x[..., -3:, -3:] = 1.0
    return x


class Extractor:
    def __init__(self, mode: str, Xpub: torch.Tensor, ypub: torch.Tensor, seed: int = 0):
        assert mode in ("cnn+grid", "cnn")
        self.mode = mode
        self.cnn_dim = 48 if mode == "cnn+grid" else D
        path = WORK / f"extractor_{mode.replace('+', '_')}.pt"
        self.net = CNN(self.cnn_dim)
        if path.exists():
            st = torch.load(path)
            self.net.load_state_dict(st["net"])
            self.mu, self.gmu = st["mu"], st["gmu"]
        else:
            self._pretrain(Xpub, ypub, seed)
            WORK.mkdir(parents=True, exist_ok=True)
            torch.save({"net": self.net.state_dict(), "mu": self.mu, "gmu": self.gmu}, path)
        self.net.eval()

    def _pretrain(self, Xpub, ypub, seed, epochs=2):
        torch.manual_seed(seed)
        head = nn.Linear(self.cnn_dim, 10)
        opt = torch.optim.Adam(list(self.net.parameters()) + list(head.parameters()), lr=2e-3)
        for _ in range(epochs):
            perm = torch.randperm(len(ypub))
            for i in range(0, len(perm), 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                F.cross_entropy(head(self.net(Xpub[idx])), ypub[idx]).backward()
                opt.step()
        self.net.eval()
        with torch.no_grad():
            self.mu = self.net(Xpub).mean(0)
            self.gmu = F.avg_pool2d(Xpub, 7).flatten(1).mean(0)

    @torch.no_grad()
    def features(self, X: torch.Tensor) -> np.ndarray:
        """Quantised features (int64, value * 2^16) of shape (n, D)."""
        out = []
        for i in range(0, len(X), 4096):
            x = X[i:i + 4096]
            h = self.net(x) - self.mu
            if self.mode == "cnn+grid":
                h = 2.5 * h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
                g = 2.0 * (F.avg_pool2d(x, 7).flatten(1) - self.gmu)
                h = torch.cat([h, g], 1)
            else:
                h = 3.0 * h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
            out.append(h.numpy())
        return quantize(np.concatenate(out))
