"""Entry point: python -m fubd [--preset quick|paper|tiny] [--attack fusion] ..."""
from __future__ import annotations

import torch

from .config import parse_args, to_dict
from .runner import run


def main():
    cfg = parse_args()
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    print("[CONFIG]", {k: v for k, v in to_dict(cfg).items()
                       if k in ("name", "preset", "dataset", "n_clients", "n_malicious",
                                "train_rounds", "unlearn_rounds", "fu_method", "attack",
                                "verifier", "iid", "aggregator")})
    run(cfg)


if __name__ == "__main__":
    main()
