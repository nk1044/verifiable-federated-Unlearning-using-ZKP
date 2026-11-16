"""Fast correctness tests (CPU, random data). Run: .venv/bin/python -m pytest -q"""
import copy

import torch

from fubd import attack, models, subid
from fubd.aggregators import aggregate
from fubd.config import Config, PRESETS
from fubd.data import add_trigger
from fubd.runner import build, run
from fubd.zk_hooks import DatasetCommitment, ReplayVerifier, ClientMessage, tensor_digest


def tiny_cfg(**kw):
    c = Config(**PRESETS["tiny"])
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---- building blocks ----
def test_aggregators_shapes():
    ups = [torch.randn(10) for _ in range(6)]
    for m in ("mean", "median", "trim"):
        assert aggregate(ups, m).shape == (10,)
    # trimmed mean drops extremes: a huge outlier barely moves it
    base = [torch.zeros(4) for _ in range(5)]
    outlier = base + [torch.full((4,), 1e6)]
    assert aggregate(outlier, "trim", 0.2).abs().max() < 1.0


def test_trigger_is_localized():
    x = torch.zeros(2, 1, 28, 28)
    xt = add_trigger(x)
    assert xt[..., -3:, -3:].min() == 1.0 and xt[..., :-3, :-3].max() == 0.0


def test_commitment_binds_and_hides():
    X, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    salt = b"s" * 16
    r1 = DatasetCommitment(X, y, salt).root_hex
    assert r1 == DatasetCommitment(X, y, salt).root_hex          # deterministic
    X2 = X.clone(); X2[0, 0, 0, 0] += 1.0
    assert DatasetCommitment(X2, y, salt).root_hex != r1          # any change flips the root


# ---- honest procedures are deterministic (needed for ZK replay) ----
def test_honest_round_deterministic():
    cfg = tiny_cfg()
    _, clients, _, _ = build(cfg)
    c = clients[-1]
    base = models.get_vec(c.template)
    assert torch.allclose(c.honest_fl_round(base, 3), c.honest_fl_round(base, 3))
    assert torch.allclose(c.honest_fu_round(base, 1), c.honest_fu_round(base, 1))


# ---- attack preparation produces a localized subnetwork ----
def test_fusion_prepare_localized():
    cfg = tiny_cfg()
    template, clients, _, _ = build(cfg)
    g = models.get_vec(template)
    mal = [c for c in clients if c.malicious]
    Xc = torch.cat([c._retain()[0] for c in mal]); yc = torch.cat([c._retain()[1] for c in mal])
    Xb = torch.cat([c.attack_bd[0] for c in mal]); yb = torch.cat([c.attack_bd[1] for c in mal])
    prep = attack.prepare(template, g, (Xc, yc), (Xb, yb), cfg)
    assert len(prep.bc_layers) >= 1
    assert prep.bc_layers == list(range(prep.bc_layers[0], prep.bc_layers[-1] + 1))  # contiguous
    assert torch.linalg.norm(prep.S_bd) > 0
    # S_bd is sparse: only TS units in BC layers are nonzero
    assert (prep.S_bd != 0).float().mean() < 0.9


# ---- replay verifier accepts honest, rejects patched ----
def test_replay_rejects_patched_update():
    cfg = tiny_cfg(attack="fusion")
    template, clients, _, _ = build(cfg)
    g = models.get_vec(template)
    c = clients[0]  # malicious

    def honest_fn(cid, msg):
        cc = next(x for x in clients if x.id == cid)
        return cc.honest_fu_round(msg.meta["base_vec"], msg.meta["rnd"])

    v = ReplayVerifier(honest_fn)
    honest = c.honest_fu_round(g, 0)
    patched = honest + 0.05 * torch.randn_like(honest)
    mk = lambda u: ClientMessage(c.id, "fu", tensor_digest(g), u, meta={"base_vec": g, "rnd": 0})
    assert v.verify(mk(honest)) is True
    assert v.verify(mk(patched)) is False


# ---- SubID flags the coherent (aligned) deviation, not unstructured noise ----
def test_subid_flags_aligned_deviation():
    cfg = tiny_cfg(subid_k=2, subid_tau=4, subid_kappa=3.0, subid_phi=0.3)
    d = subid.SubID(cfg)
    torch.manual_seed(0)
    dim = 50
    benign_dirs = torch.randn(dim, 2)
    for r in range(cfg.subid_Tr):                                 # benign training subspace
        d.observe_training(r, {i: benign_dirs @ torch.randn(2) for i in range(6)})
    d.build_benign_subspace()
    trigger = torch.randn(dim); trigger /= trigger.norm()         # shared malicious direction
    for r in range(cfg.subid_tau):
        ups = {}
        for i in range(6):
            if i < 2:                                             # malicious: aligned off-subspace push
                ups[i] = benign_dirs @ torch.randn(2) + 4.0 * trigger
            else:                                                 # benign unlearners: unstructured noise
                ups[i] = benign_dirs @ torch.randn(2) + 0.5 * torch.randn(dim)
        d.observe_unlearning(r, ups)
    d.finalize()
    pred, _ = d.classify()
    m = subid.metrics(pred, {0, 1}, set(range(6)))
    assert m["TPR"] == 1.0 and m["FPR"] == 0.0


# ---- full tiny run, both defenses ----
def test_end_to_end_replay_blocks_attack():
    cfg = tiny_cfg(attack="fusion", verifier="replay", subid=False, name="t_replay",
                   out_dir="./runs")
    res = run(cfg, log=lambda *a: None)
    assert res["final_BA"] < 0.5 and res["rejected_updates"] > 0


def test_end_to_end_attack_runs():
    cfg = tiny_cfg(attack="fusion", verifier="none", name="t_attack", out_dir="./runs")
    res = run(cfg, log=lambda *a: None)
    assert "subid" in res and set(res["subid"]["truth"]) == {0, 1}
