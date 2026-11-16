"""Datasets, client partitioning and the backdoor trigger."""
from __future__ import annotations

import numpy as np
import torch


def load_dataset(cfg):
    """Return (Xtr, ytr, Xte, yte); images in [0,1], shape (N,1,28,28)."""
    if cfg.dataset == "mnist":
        from torchvision import datasets

        tr = datasets.MNIST(cfg.data_dir, train=True, download=True)
        te = datasets.MNIST(cfg.data_dir, train=False, download=True)
        f = lambda d: (d.data.float().div(255).unsqueeze(1), d.targets.long())  # noqa: E731
        return (*f(tr), *f(te))
    if cfg.dataset == "fake":  # learnable toy data: a bright row whose position encodes the label
        g = torch.Generator().manual_seed(cfg.seed)

        def mk(n):
            y = torch.randint(0, 10, (n,), generator=g)
            x = torch.rand(n, 1, 28, 28, generator=g) * 0.3
            x[torch.arange(n), 0, 2 * y + 2, :] += 0.7
            return x, y

        return (*mk(3000), *mk(1000))
    raise ValueError(cfg.dataset)


def add_trigger(x: torch.Tensor) -> torch.Tensor:
    """MNIST trigger from the paper: 3x3 white square at the bottom-right corner."""
    x = x.clone()
    x[..., -3:, -3:] = 1.0
    return x


def partition(y: torch.Tensor, n_clients: int, iid: bool, alpha: float, rng: np.random.Generator):
    """Index split per client. IID = random equal shards; non-IID = Dirichlet(alpha) label skew."""
    if iid:
        return [torch.as_tensor(p) for p in np.array_split(rng.permutation(len(y)), n_clients)]
    labels = y.numpy()
    parts = [[] for _ in range(n_clients)]
    for _ in range(200):  # resample until every client has >= 10 samples
        parts = [[] for _ in range(n_clients)]
        for c in range(int(labels.max()) + 1):
            ids = rng.permutation(np.where(labels == c)[0])
            cuts = (np.cumsum(rng.dirichlet([alpha] * n_clients)) * len(ids)).astype(int)[:-1]
            for i, part in enumerate(np.split(ids, cuts)):
                parts[i].extend(part.tolist())
        if min(len(p) for p in parts) >= 10:
            break
    return [torch.as_tensor(rng.permutation(p)) for p in parts]


def make_backdoor_set(X, y, frac, target, rng):
    """Poisoned copies (x', y_t) of `frac` of a client's non-target-class samples."""
    cand = np.where(y.numpy() != target)[0]
    if len(cand) == 0:
        cand = np.arange(len(y))
    n = min(len(cand), max(1, int(round(frac * len(y)))))
    pick = torch.as_tensor(rng.choice(cand, n, replace=False))
    return add_trigger(X[pick]), torch.full((n,), target, dtype=torch.long)
