"""Float training of the SAME head under the SAME schedule, to quantify the cost of fixed point.
Same features, same per-round batch size B with sampling over the padded N positions (padding and
forgotten positions masked), same update rule W -= 2^-S * sum_j m_j e_j h_j^T, FedAvg, no attack.

    ../../.venv/bin/python float_baseline.py --fl_rounds 20 --fu_rounds 20 --name float_baseline
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from common import B, C, D, N, ONE, ROOT, S, WORK


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="float_baseline")
    p.add_argument("--features", default="cnn+grid")
    p.add_argument("--clients", type=int, default=50)
    p.add_argument("--per_client", type=int, default=1000)
    p.add_argument("--malicious", type=int, default=10)
    p.add_argument("--benign_unlearn", type=int, default=5)
    p.add_argument("--forget_frac", type=float, default=0.02)
    p.add_argument("--fl_rounds", type=int, default=20)
    p.add_argument("--fu_rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    tag = f"{a.features.replace('+', '_')}_c{a.clients}_n{a.per_client}_s{a.seed}"
    z = np.load(WORK / f"data_{tag}.npz", allow_pickle=True)
    feats, labels, idx = z["feats"] / ONE, z["labels"], z["client_idx"]
    tf, tl = z["test_feats"] / ONE, z["test_labels"]
    rng = np.random.default_rng(a.seed)
    W, b = np.zeros((C, D)), np.zeros(C)
    lr = 2.0 ** -S
    forgot = {c: set() for c in range(a.clients)}
    acc = lambda: float(((tf @ W.T + b).argmax(1) == tl).mean())  # noqa: E731
    out = {}
    for r in range(a.fl_rounds + a.fu_rounds):
        if r == a.fl_rounds:
            out["FL_end_MA"] = acc()
            for c in range(a.malicious + a.benign_unlearn):
                forgot[c] = set(random.Random(1000 + c).sample(range(a.per_client), max(1, int(a.forget_frac * a.per_client))))
        heads = []
        for c in range(a.clients):
            pos = rng.integers(0, N, B)                       # same distribution as the server's indices
            m = np.array([(i < a.per_client and i not in forgot[c]) for i in pos], float)
            pos = np.minimum(pos, a.per_client - 1)
            h, y = feats[idx[c][pos]], labels[idx[c][pos]]
            e = (h @ W.T + b - np.eye(C)[y]) * m[:, None]
            heads.append((W - lr * e.T @ h, b - lr * e.sum(0)))
        W = np.mean([h[0] for h in heads], 0)
        b = np.mean([h[1] for h in heads], 0)
    out["final_MA"] = acc()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{a.name}.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
