"""End-to-end orchestration: FL training -> backdoor prep -> active FU (attack + SubID + verifier)."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from . import attack, models, subid
from .aggregators import aggregate
from .client import Client
from .data import add_trigger, load_dataset, make_backdoor_set, partition
from .zk_hooks import ClientMessage, make_verifier, tensor_digest


def _set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def _evaluate(template, vec, Xte, yte, target):
    net = models.load(template, vec).eval()
    with torch.no_grad():
        ma = (net(Xte).argmax(1) == yte).float().mean().item()
        nz = yte != target                              # BA on non-target samples with the trigger
        ba = (net(add_trigger(Xte[nz])).argmax(1) == target).float().mean().item()
    return ma, ba


def build(cfg):
    _set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    Xtr, ytr, Xte, yte = load_dataset(cfg)
    template = models.CNN()
    idxs = partition(ytr, cfg.n_clients, cfg.iid, cfg.dirichlet_alpha, rng)

    mal_ids = set(range(cfg.n_malicious))
    ben_unlearn_ids = set(range(cfg.n_malicious, cfg.n_malicious + cfg.n_benign_unlearn))
    clients = []
    for cid, idx in enumerate(idxs):
        is_mal = cid in mal_ids
        is_unl = is_mal or cid in ben_unlearn_ids
        c = Client(cid, Xtr[idx], ytr[idx], template, cfg,
                   malicious=is_mal, unlearner=is_unl, rng=rng)
        if is_mal:                                       # per-client backdoor data for fine-tuning
            Xb, yb = make_backdoor_set(c.X, c.y, cfg.poison_frac, cfg.target_label, rng)
            c.attack_bd = (Xb, yb)
        clients.append(c)
    return template, clients, (Xte, yte), mal_ids


def run(cfg, log=print):
    t0 = time.time()
    template, clients, (Xte, yte), mal_ids = build(cfg)
    global_vec = models.get_vec(template)
    det = subid.SubID(cfg) if cfg.subid else None
    all_ids = {c.id for c in clients}

    # ---------------------------------------------------------------- FL training phase
    log(f"[FL] {cfg.n_clients} clients ({len(mal_ids)} malicious), {cfg.train_rounds} rounds, "
        f"agg={cfg.aggregator}, {cfg.dataset}, {'IID' if cfg.iid else 'nonIID'}")
    for rnd in range(cfg.train_rounds):
        base = global_vec
        updates = {c.id: c.honest_fl_round(base, rnd) for c in clients}   # all behave honestly
        if det and rnd >= cfg.train_rounds - cfg.subid_Tr:
            det.observe_training(rnd, updates)
        global_vec = base + aggregate(list(updates.values()), cfg.aggregator, cfg.trim_frac)
        if (rnd + 1) % cfg.eval_every == 0 or rnd == cfg.train_rounds - 1:
            ma, ba = _evaluate(template, global_vec, Xte, yte, cfg.target_label)
            log(f"  round {rnd+1:4d}  MA={ma:.3f}  BA={ba:.3f}")
    if det:
        det.build_benign_subspace()
        log(f"[SubID] benign subspace built (k={cfg.subid_k}) from last {cfg.subid_Tr} rounds")

    # ---------------------------------------------------------------- backdoor preparation
    prep = None
    if cfg.attack == "fusion":
        mc = [(c._retain()) for c in clients if c.malicious]
        Xc = torch.cat([x for x, _ in mc]); yc = torch.cat([y for _, y in mc])
        Xb = torch.cat([c.attack_bd[0] for c in clients if c.malicious])
        yb = torch.cat([c.attack_bd[1] for c in clients if c.malicious])
        prep = attack.prepare(template, global_vec, (Xc, yc), (Xb, yb), cfg)
        log(f"[Attack] FUsion prepared: g_bd BA={prep.gbd_ba:.3f}  BC layers={prep.bc_layers}  "
            f"TS units={{{', '.join(f'{k}:{len(v)}' for k,v in prep.ts_units.items())}}}  "
            f"||S_bd||={torch.linalg.norm(prep.S_bd):.3f}")

    # ---------------------------------------------------------------- verifier (ZK hook)
    client_by_id = {c.id: c for c in clients}

    def honest_fn(cid, msg):
        c = client_by_id[cid]
        base = msg.meta["base_vec"]
        return (c.honest_fu_round(base, msg.meta["rnd"]) if c.unlearner
                else c.honest_fl_round(base, msg.meta["rnd"]))

    verifier = make_verifier(cfg, honest_fn)

    # ---------------------------------------------------------------- active FU phase
    log(f"[FU] method={cfg.fu_method}  attack={cfg.attack}  verifier={verifier.name}  "
        f"{cfg.unlearn_rounds} rounds")
    rejected_total = 0
    for rnd in range(cfg.unlearn_rounds):
        base = global_vec
        base_dig = tensor_digest(base)
        submitted, accepted = {}, []
        for c in clients:
            if c.malicious and cfg.attack != "none":
                if cfg.attack == "fusion":
                    upd = attack.inject_fusion(c, base, rnd, prep, cfg)
                elif cfg.attack == "baseline":
                    upd = attack.inject_baseline(c, base, rnd, cfg)
                elif cfg.attack == "signflip":
                    upd = attack.inject_signflip(c, base, rnd, cfg)
                else:
                    raise ValueError(cfg.attack)
            elif c.unlearner:
                upd = c.honest_fu_round(base, rnd)
            elif cfg.fu_benign_participate:
                upd = c.honest_fl_round(base, rnd)       # benign non-unlearners keep training
            else:
                continue                                 # converged benign non-targets sit out the FU phase
            submitted[c.id] = upd
            msg = ClientMessage(c.id, "fu", base_dig, upd,
                                commit_root=c.commit.root_hex if c.commit else None,
                                meta={"base_vec": base, "rnd": rnd})
            if verifier.verify(msg):
                accepted.append(upd)
            else:
                rejected_total += 1
        if det:
            det.observe_unlearning(rnd, submitted)       # server sees every submitted update
        if accepted:
            global_vec = base + aggregate(accepted, cfg.aggregator, cfg.trim_frac)
        if (rnd + 1) % cfg.eval_every == 0 or rnd == cfg.unlearn_rounds - 1:
            ma, ba = _evaluate(template, global_vec, Xte, yte, cfg.target_label)
            log(f"  round {rnd+1:4d}  MA={ma:.3f}  BA={ba:.3f}"
                + (f"  rejected={rejected_total}" if verifier.name != "null" else ""))

    # ---------------------------------------------------------------- results
    ma, ba = _evaluate(template, global_vec, Xte, yte, cfg.target_label)
    result = dict(config=cfg.__dict__.copy(), final_MA=ma, final_BA=ba,
                  attack_success=bool(ba > 0.60), rejected_updates=rejected_total,
                  seconds=round(time.time() - t0, 1))
    if cfg.attack == "fusion" and prep is not None:
        result["bc_layers"] = prep.bc_layers
        result["ts_units"] = {k: v.tolist() for k, v in prep.ts_units.items()}
    if det:
        det.finalize()
        pred, frac = det.classify()
        m = subid.metrics(pred, mal_ids, all_ids)
        result["subid"] = dict(predicted=sorted(pred), truth=sorted(mal_ids),
                               flag_fraction={int(k): round(v, 3) for k, v in frac.items()}, **m)
        log(f"[SubID] predicted malicious={sorted(pred)}  truth={sorted(mal_ids)}")
        log(f"[SubID] TPR={m['TPR']:.2f} FPR={m['FPR']:.2f} F1={m['F1']:.2f}")
    log(f"[RESULT] MA={ma:.3f} BA={ba:.3f} attack_success={result['attack_success']} "
        f"rejected={rejected_total} ({result['seconds']}s)")

    os.makedirs(cfg.out_dir, exist_ok=True)
    out = os.path.join(cfg.out_dir, f"{cfg.name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"[RESULT] saved -> {out}")
    return result
