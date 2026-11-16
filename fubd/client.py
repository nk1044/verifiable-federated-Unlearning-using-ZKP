"""A federated client: local data, dataset commitment, forget bitmask, deterministic honest rounds.

Honest rounds are PURE functions of (base weights, client data, mask, round) with no hidden
randomness, so the ReplayVerifier / a future ZK circuit can recompute them exactly.
"""
from __future__ import annotations

import copy
import os

import torch
import torch.nn.functional as F

from . import models
from .zk_hooks import DatasetCommitment


class Client:
    def __init__(self, cid, X, y, template, cfg, malicious=False, unlearner=False, rng=None):
        self.id = cid
        self.X, self.y = X, y
        self.cfg = cfg
        self.template = template
        self.malicious = malicious
        self.unlearner = unlearner          # requests unlearning during the FU phase
        self.net = copy.deepcopy(template)

        # forget bitmask b over A: 1 = use (retain), 0 = forget. Default all-ones.
        self.mask = torch.ones(len(y), dtype=torch.uint8)
        if (malicious or unlearner) and len(y) > 1 and rng is not None:
            k = max(1, int(round(cfg.forget_frac * len(y))))
            idx = rng.choice(len(y), min(k, len(y) - 1), replace=False)
            self.mask[idx] = 0
        self.commit = DatasetCommitment(X, y, salt=os.urandom(16)) if cfg.commit else None

        # attacker state (filled by attack.prepare); None for benign clients
        self.attack = None

    # ---------------------------------------------------------------- helpers
    def _retain(self):
        m = self.mask.bool()
        return self.X[m], self.y[m]

    def _forget(self):
        m = ~self.mask.bool()
        return self.X[m], self.y[m]

    def _sgd(self, net, X, y, steps, lr, seed, sign=1.0):
        """`steps` fixed-order minibatch SGD steps. sign=-1 => gradient ascent (unlearn)."""
        net.train()
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.cfg.momentum)
        g = torch.Generator().manual_seed(seed)
        n = len(y)
        for s in range(steps):
            if n <= self.cfg.batch_size:
                bx, by = X, y
            else:
                idx = torch.randint(0, n, (self.cfg.batch_size,), generator=g)
                bx, by = X[idx], y[idx]
            opt.zero_grad()
            loss = F.cross_entropy(net(bx), by)
            (sign * loss).backward()
            opt.step()
        return net

    def _seed(self, rnd, tag):
        return (self.id * 1_000_003 + rnd * 97 + tag) % (2**31)

    # ---------------------------------------------------------------- honest procedures
    def honest_fl_round(self, base_vec, rnd):
        """k local SGD steps on retained data -> weight update (new - base)."""
        net = models.load(self.template, base_vec)
        Xr, yr = self._retain()
        self._sgd(net, Xr, yr, self.cfg.local_steps, self.cfg.lr, self._seed(rnd, 1))
        return models.get_vec(net) - base_vec

    def honest_fu_round(self, base_vec, rnd):
        """Honest active unlearning update, per cfg.fu_method. Deterministic in base_vec."""
        Xr, yr = self._retain()
        if self.cfg.fu_method == "not":                      # NoT: negate chosen layers ONCE (rnd 0),
            net = models.load(self.template, base_vec)        # then fine-tune on retained data each round
            if rnd == 0:
                layers = [int(i) for i in str(self.cfg.not_layers).split(",") if i != ""]
                wl = models.weight_layers(net)
                with torch.no_grad():
                    for li in layers:
                        if 0 <= li < len(wl):
                            wl[li][1].weight.mul_(-1)
            self._sgd(net, Xr, yr, self.cfg.local_steps, self.cfg.lr, self._seed(rnd, 2))
            return models.get_vec(net) - base_vec
        if self.cfg.fu_method == "ga":                       # gradient ascent on forget + descent on retain
            net = models.load(self.template, base_vec)
            Xf, yf = self._forget()
            if len(yf):
                self._sgd(net, Xf, yf, max(1, self.cfg.local_steps // 2),
                          self.cfg.ga_alpha, self._seed(rnd, 3), sign=-1.0)
            self._sgd(net, Xr, yr, self.cfg.local_steps, self.cfg.lr, self._seed(rnd, 4))
            return models.get_vec(net) - base_vec
        if self.cfg.fu_method == "retrain":                  # fine-tune on retained data only
            net = models.load(self.template, base_vec)
            self._sgd(net, Xr, yr, self.cfg.local_steps, self.cfg.lr, self._seed(rnd, 5))
            return models.get_vec(net) - base_vec
        raise ValueError(self.cfg.fu_method)
