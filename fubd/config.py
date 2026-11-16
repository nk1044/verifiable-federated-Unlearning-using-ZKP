"""Experiment configuration. Defaults = the paper's default scenario (Sec. III-D / IV-B)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # ---- general ----
    name: str = "run"
    seed: int = 0
    dataset: str = "mnist"          # mnist | fake (random toy data, for tests)
    data_dir: str = "./data"
    out_dir: str = "./runs"
    threads: int = 0                # 0 = torch default

    # ---- federated learning (FL phase) ----
    n_clients: int = 50
    n_malicious: int = 10           # m < n/2
    iid: bool = True
    dirichlet_alpha: float = 0.5    # non-IID concentration (paper: 0.5)
    train_rounds: int = 500
    local_steps: int = 20           # k fixed local SGD iterations per round (fixed => ZK-circuit friendly)
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.9           # paper uses 0.9; set 0 for a simpler ZK circuit
    aggregator: str = "mean"        # mean (FedAvg) | trim | median
    trim_frac: float = 0.1
    eval_every: int = 10

    # ---- active federated unlearning (FU phase) ----
    unlearn_rounds: int = 200
    fu_method: str = "not"          # not (Negation-of-Training) | ga (gradient ascent + retain) | retrain (fine-tune on retained)
    not_layers: str = "0"           # comma-separated weight-layer indices negated by NoT
    ga_alpha: float = 0.1
    forget_frac: float = 0.02       # fraction of local data each target client asks to forget
    n_benign_unlearn: int = 5       # benign clients that ALSO request unlearning (makes detection non-trivial)
    fu_benign_participate: bool = True   # True (paper-faithful): all clients participate in FU aggregation
    #                                      every round, so malicious stay a minority and SubID's median/MAD
    #                                      baseline is benign-anchored. False => only unlearning (target)
    #                                      clients send updates; then malicious may be a majority of FU
    #                                      participants and MAD-based SubID detection degrades.

    # ---- FUsion backdoor attack ----
    attack: str = "fusion"          # none | baseline | fusion
    target_label: int = 0
    poison_frac: float = 0.10       # backdoor samples = 10% of a malicious client's local data
    gamma: float = 1.0              # Eq.(2) weight for g_bd training
    prep_steps: int = 200           # SGD steps to train each local backdoor model g_bd
    theta: float = 0.6              # Eq.(7) BC-layer expansion threshold
    rho: float = 0.10               # top-rho fraction of trigger-sensitive units per BC layer
    sub_rounds: int = 5             # collaborative (averaged) subnetwork training rounds among attackers
    sub_steps: int = 20
    lambda1: float = 1.0            # Eq.(10)
    lambda2: float = 0.5            # Eq.(10)
    beta: float = 0.2               # Eq.(11) fusion strength
    lam: float = 0.5                # Eq.(12)
    ft_steps: int = 20              # fine-tuning steps after fusion (Eq.12)
    fuse_mode: str = "delta"        # delta: S_bd = trained units - global ; weights: S_bd = trained unit weights

    # ---- SubID detection ----
    subid: bool = True
    subid_k: int = 5
    subid_Tr: int = 5               # last Tr training rounds form the benign subspace
    subid_tau: int = 10             # window size (rounds) in the unlearning phase
    subid_kappa: float = 7.0        # MAD threshold
    subid_phi: float = 0.5          # fraction of windows flagged => malicious

    # ---- ZK-compatibility hooks ----
    verifier: str = "none"          # none | replay (cleartext oracle that emulates what a ZK proof certifies)
    commit: bool = True             # Merkle commitment of each client's dataset
    save_updates: bool = False      # dump every update tensor (large!) for offline proving experiments


PRESETS = {
    # paper default scenario (slow: ~700 rounds x 50 clients)
    "paper": {},
    # ~1-2 min on a laptop CPU; reliably shows attack success + perfect SubID detection.
    # 2/10 = 20% malicious matches the paper's ratio and keeps malicious a minority for SubID.
    "quick": dict(n_clients=10, n_malicious=2, train_rounds=25, unlearn_rounds=60, local_steps=15,
                  eval_every=10, n_benign_unlearn=2, prep_steps=150, subid_tau=10),
    # seconds; random data; for smoke tests only
    "tiny": dict(dataset="fake", n_clients=6, n_malicious=2, train_rounds=4, unlearn_rounds=6, local_steps=3,
                 eval_every=2, prep_steps=10, sub_rounds=2, sub_steps=3, ft_steps=3, subid_Tr=2, subid_tau=3,
                 subid_k=2, n_benign_unlearn=1),
}


def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="FUsion backdoor + SubID on active federated unlearning")
    p.add_argument("--preset", choices=list(PRESETS), default="quick")
    for f in fields(Config):
        typ = _str2bool if isinstance(f.default, bool) else type(f.default)
        p.add_argument(f"--{f.name}", type=typ, default=None)
    a = p.parse_args(argv)
    cfg = Config(**PRESETS[a.preset])
    for f in fields(Config):
        v = getattr(a, f.name)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
