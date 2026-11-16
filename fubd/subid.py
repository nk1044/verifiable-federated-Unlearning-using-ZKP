"""SubID: Subspace-based deviation IDentification (Sec. IV).

Training phase : build the benign subspace S_benign from SVD over the last Tr rounds of updates.
Unlearning     : per window of tau rounds, residual to S_benign -> deviation subspace S_dev (SVD)
                 -> projection score phi_i -> MAD flag (Eq. 15). A client flagged in > phi fraction
                 of windows is classified malicious.
"""
from __future__ import annotations

import numpy as np
import torch


def _topk_left(M: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k left singular vectors of d x N matrix M (columns = updates)."""
    k = max(1, min(k, M.shape[1], M.shape[0]))
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, :k]


class SubID:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_hist = []        # list of (round, {cid: update_vec})
        self.S_benign = None        # d x k basis
        self.windows = []           # per-window dict {cid: flagged bool}
        self._buf = []              # current window's per-round {cid: update}

    # ---- training phase ----
    def observe_training(self, rnd, updates: dict):
        self.train_hist.append((rnd, updates))

    def build_benign_subspace(self):
        last = self.train_hist[-self.cfg.subid_Tr:]
        cols = [u for _, ups in last for u in ups.values()]
        G = torch.stack(cols, 1)                               # d x (n*Tr)
        self.S_benign = _topk_left(G, self.cfg.subid_k)
        return self.S_benign

    # ---- unlearning phase ----
    def observe_unlearning(self, rnd, updates: dict):
        self._buf.append(updates)
        if len(self._buf) >= self.cfg.subid_tau:
            self._close_window()

    def finalize(self):
        if self._buf:
            self._close_window()

    def _close_window(self):
        P = self.S_benign                                      # projection onto benign subspace
        clients = sorted({c for ups in self._buf for c in ups})
        residuals, per_client = [], {c: [] for c in clients}
        for ups in self._buf:
            for c, g in ups.items():
                r = g - P @ (P.t() @ g)                        # Eq.(13) residual
                residuals.append(r)
                per_client[c].append(g)
        Da = torch.stack(residuals, 1)                         # d x (n*tau)
        Sdev = _topk_left(Da, self.cfg.subid_k)                # deviation subspace (SVD of residuals)
        scores = {c: torch.linalg.norm(Sdev.t() @ torch.stack(gs, 1).mean(1)).item()  # Eq.(14)
                  for c, gs in per_client.items()}
        v = np.array(list(scores.values()))
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-12
        thr = med + self.cfg.subid_kappa * mad                 # Eq.(15)
        self.windows.append({c: (scores[c] > thr) for c in clients})
        self._buf = []

    def classify(self):
        """Return (predicted_malicious_set, per_client_flag_fraction)."""
        clients = sorted({c for w in self.windows for c in w})
        frac = {c: np.mean([w.get(c, False) for w in self.windows]) for c in clients}
        pred = {c for c, f in frac.items() if f > self.cfg.subid_phi}
        return pred, frac


def metrics(pred: set, truth: set, all_clients: set):
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    tn = len(all_clients - pred - truth)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    return dict(TPR=tpr, FPR=fpr, F1=f1, TP=tp, FP=fp, FN=fn, TN=tn)
