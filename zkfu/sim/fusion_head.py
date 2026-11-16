"""FUsion backdoor (Malicious Forgetting, INFOCOM 2026) adapted to the head-only model of the ZK
design. This is ATTACKER code: its output is deliberately NOT the fixed procedure.

Adaptation (the only trainable/proved part is the linear head z = W h + b):
  * BC layers  -> the head is the single trainable layer, so H_BC = {head}.
  * TS units   -> the head's input units (feature dims). Grad-CAM score of dim d on backdoor inputs:
                  mean_j relu( dz_t/dh_d * h_j[d] ) = mean_j relu( W_bd[t,d] * h_j[d] );  top-rho.
  * Subnetwork -> only W[:, TS] is trained on backdoor data, with the Eq.(10) clean-activation
                  regulariser (change of clean logits through the TS units).
  * Injection  -> every FU round: honest step (from the Go trainer)  ->  W~ = W' + beta*S_bd
                  (Eq.11)  ->  fine-tune on clean + lam*backdoor (Eq.12)  ->  quantise and submit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common import C, D, ONE, VMAX, VMIN


def _t(x):
    return torch.tensor(np.asarray(x, dtype=np.float64) / ONE, dtype=torch.float32)


def _mse(W, b, h, y):
    return 0.5 * ((h @ W.T + b - F.one_hot(y, C).float()) ** 2).sum(1).mean()


def _q(x: torch.Tensor) -> np.ndarray:
    return np.clip(np.rint(x.detach().numpy().astype(np.float64) * ONE), VMIN, VMAX).astype(np.int64)


class FusionAttacker:
    def __init__(self, target=0, rho=0.25, beta=1.0, lam=1.0, gamma=1.0, lambda1=1.0, lambda2=0.5,
                 prep_steps=1500, sub_steps=1500, ft_steps=60, lr=0.05, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.g = torch.Generator().manual_seed(seed)
        self.S_bd = None
        self.ts = None

    def _batch(self, n, k):
        return torch.randint(0, n, (k,), generator=self.g)

    def prepare(self, W, b, clean_h, clean_y, bd_h):
        """Collaborative preparation by the malicious clients at the end of FL (pooled data)."""
        Wg, bg = _t(W), _t(b)
        hc, yc, hb = _t(clean_h), torch.tensor(clean_y), _t(bd_h)
        yb = torch.full((len(hb),), self.target, dtype=torch.long)
        # 1) local backdoor head g_bd (Eq.2)
        Wbd, bbd = Wg.clone().requires_grad_(True), bg.clone().requires_grad_(True)
        opt = torch.optim.SGD([Wbd, bbd], lr=self.lr)
        for _ in range(self.prep_steps):
            i, j = self._batch(len(hc), 256), self._batch(len(hb), 64)
            opt.zero_grad()
            (_mse(Wbd, bbd, hc[i], yc[i]) + self.gamma * _mse(Wbd, bbd, hb[j], yb[j])).backward()
            opt.step()
        # 2) trigger-sensitive input units by Grad-CAM (grad of target logit x activation)
        score = torch.relu(Wbd.detach()[self.target] * hb).mean(0)
        k = max(1, int(round(self.rho * D)))
        self.ts = torch.topk(score, k).indices.sort().values
        # 3) subnetwork: train only W[:, TS] (Eq.10)
        mask = torch.zeros(C, D)
        mask[:, self.ts] = 1.0
        Ws = Wg.clone().requires_grad_(True)
        opt = torch.optim.SGD([Ws], lr=self.lr)
        for _ in range(self.sub_steps):
            i, j = self._batch(len(hc), 256), self._batch(len(hb), 64)
            opt.zero_grad()
            l_bd = _mse(Ws, bg, hb[j], yb[j])
            l_clean = (((Ws - Wg) * mask) @ hc[i].T).pow(2).sum(0).mean()  # clean-logit change via TS units
            (self.lambda1 * l_bd + self.lambda2 * l_clean).backward()
            Ws.grad *= mask
            opt.step()
        self.S_bd = _q((Ws.detach() - Wg) * mask)  # nonzero only on the TS units of the head
        with torch.no_grad():
            gbd_ba = float(((hb @ Wbd.T + bbd).argmax(1) == self.target).float().mean())
            sub_ba = float(((hb @ Ws.T + bg).argmax(1) == self.target).float().mean())
        return {"ts_units": self.ts.tolist(), "gbd_BA": gbd_ba, "subnet_BA": sub_ba,
                "S_bd_l2": float(np.linalg.norm(self.S_bd / ONE))}

    def patch(self, W2, b2, clean_h, clean_y, bd_h):
        """One malicious FU update: honest W' -> fuse beta*S_bd -> fine-tune (Eq.12) -> quantise."""
        W = _t(np.asarray(W2) + np.rint(self.beta * self.S_bd).astype(np.int64)).requires_grad_(True)
        b = _t(b2).requires_grad_(True)
        hc, yc, hb = _t(clean_h), torch.tensor(clean_y), _t(bd_h)
        yb = torch.full((len(hb),), self.target, dtype=torch.long)
        opt = torch.optim.SGD([W, b], lr=self.lr)
        for _ in range(self.ft_steps):
            i, j = self._batch(len(hc), 64), self._batch(len(hb), 32)
            opt.zero_grad()
            (_mse(W, b, hc[i], yc[i]) + self.lam * _mse(W, b, hb[j], yb[j])).backward()
            opt.step()
        return _q(W).tolist(), _q(b).tolist()
