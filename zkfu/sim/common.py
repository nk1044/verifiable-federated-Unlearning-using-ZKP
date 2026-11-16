"""Constants mirrored from zkfu/params (read-only here) and float helpers for evaluation.

Python never computes an honest weight update: honest steps come from the Go native trainer.
Python only (a) extracts and quantises features, (b) aggregates accepted heads (FedAvg, not
proved), (c) evaluates accuracy, and (d) plays the *attacker*, whose patched update is by
definition not the output of the fixed procedure.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

D, C, B, N, DEPTH = 64, 10, 32, 2048, 11
FRAC, S = 16, 8
ONE = 1 << FRAC
VMIN, VMAX = -(1 << 24), (1 << 24) - 1  # private values must lie in [-2^24, 2^24)

ROOT = Path(__file__).resolve().parent.parent          # zkfu/
BIN = ROOT / "bin"
KEYS = ROOT / "keys"
WORK = ROOT / "sim" / "work"
MNIST_DIR = Path(os.environ.get("MNIST_DIR", ROOT.parent / "data"))


def quantize(x: np.ndarray) -> np.ndarray:
    """float -> fixed point int64 (round to nearest), clipped to the allowed range."""
    return np.clip(np.rint(x * ONE), VMIN, VMAX).astype(np.int64)


def head_to_float(W, b):
    return np.asarray(W, dtype=np.float64) / ONE, np.asarray(b, dtype=np.float64) / ONE


def head_in_range(W, b) -> bool:
    Wa, ba = np.asarray(W), np.asarray(b)
    return (Wa.shape == (C, D) and ba.shape == (C,) and Wa.min() >= VMIN and Wa.max() <= VMAX
            and ba.min() >= VMIN and ba.max() <= VMAX)


def accuracy(W, b, feats_int: np.ndarray, labels: np.ndarray) -> float:
    Wf, bf = head_to_float(W, b)
    z = (feats_int / ONE) @ Wf.T + bf
    return float((z.argmax(1) == labels).mean())


def fedavg(heads: list[tuple[list, list]]):
    """FedAvg over accepted heads with deterministic re-quantisation:
    q = floor((sum + n//2) / n) using exact Python integers (round half up), then clip."""
    n = len(heads)
    Ws = np.array([h[0] for h in heads], dtype=object)
    bs = np.array([h[1] for h in heads], dtype=object)
    W = [[min(max((int(s) + n // 2) // n, VMIN), VMAX) for s in row] for row in Ws.sum(0)]
    b = [min(max((int(s) + n // 2) // n, VMIN), VMAX) for s in bs.sum(0)]
    return W, b
