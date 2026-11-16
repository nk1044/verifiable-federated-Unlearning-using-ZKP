"""Server aggregation rules: FedAvg (mean), coordinate-wise trimmed mean, coordinate-wise median."""
from __future__ import annotations

import torch


def aggregate(updates: list[torch.Tensor], method: str, trim_frac: float = 0.1) -> torch.Tensor:
    """Combine client updates into the global delta. `updates` are weight deltas (new - base)."""
    G = torch.stack(updates, 0)  # (n_clients, d)
    if method == "mean":
        return G.mean(0)
    if method == "median":
        return G.median(0).values
    if method == "trim":
        n = G.shape[0]
        k = int(trim_frac * n)
        s, _ = torch.sort(G, dim=0)
        if 2 * k >= n:
            return s.median(0).values
        return s[k:n - k].mean(0)
    raise ValueError(method)
