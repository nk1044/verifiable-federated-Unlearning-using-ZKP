"""Verifiable FL + FU simulation: 1 server, N clients, Groth16 proofs via the Go CLIs, random auditing,
and the FUsion backdoor attacker. Run from zkfu/sim:

    ../../.venv/bin/python fl_sim.py --audit_p 0.0  --name noverify      # attack, no verification
    ../../.venv/bin/python fl_sim.py --audit_p 0.1  --name audit10       # random auditing, p = 10%
    ../../.venv/bin/python fl_sim.py --audit_p 1.0  --name full ...      # every update proved

audit_p = 0 means the server accepts every update unverified (the setting FUsion attacks);
audit_p = 1 means every update of every round must carry a valid proof (the ZK paper's main mode);
0 < p < 1 is the paper's random auditing: the server picks rounds to audit AFTER updates arrive.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

import gocli
from common import C, D, N, WORK, accuracy, fedavg, head_in_range
from extractor import Extractor, add_trigger, load_mnist
from fusion_head import FusionAttacker


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--features", default="cnn+grid", choices=["cnn+grid", "cnn"])
    p.add_argument("--clients", type=int, default=50)
    p.add_argument("--per_client", type=int, default=1000)
    p.add_argument("--malicious", type=int, default=10)
    p.add_argument("--benign_unlearn", type=int, default=5)
    p.add_argument("--forget_frac", type=float, default=0.02)
    p.add_argument("--fl_rounds", type=int, default=20)
    p.add_argument("--fu_rounds", type=int, default=20)
    p.add_argument("--audit_p", type=float, default=0.0, help="0 = no verification, 1 = prove every update")
    p.add_argument("--on_catch", default="rollback", choices=["rollback", "exclude"])
    p.add_argument("--attack", default="fusion", choices=["fusion", "none"])
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--poison_frac", type=float, default=0.10)
    p.add_argument("--rho", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--ft_steps", type=int, default=60)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--resume_fl", default="", help="checkpoint JSON of an FL-final head: skip FL")
    return p.parse_args()


def prepare_data(args):
    """Features (cached per extractor mode), client split, backdoor sets, commitments (cached)."""
    tag = f"{args.features.replace('+', '_')}_c{args.clients}_n{args.per_client}_s{args.seed}"
    cache = WORK / f"data_{tag}.npz"
    Xtr, ytr, Xte, yte = load_mnist()
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        d = {k: z[k] for k in z.files}
    else:
        ext = Extractor(args.features, Xtr[50000:], ytr[50000:])   # public data: last 10k of train
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(50000)[: args.clients * args.per_client].reshape(args.clients, args.per_client)
        feats = ext.features(Xtr[: 50000])
        d = {"client_idx": idx, "feats": feats, "labels": ytr[:50000].numpy(),
             "test_feats": ext.features(Xte), "test_labels": yte.numpy()}
        nz = (yte != args.target)
        d["test_bd_feats"] = ext.features(add_trigger(Xte[nz]))
        # backdoor samples of each potential attacker: poison_frac of its data (non-target), NOT committed
        bd = []
        for c in range(args.clients):
            own = idx[c][ytr[idx[c]].numpy() != args.target]
            pick = rng.choice(own, int(round(args.poison_frac * args.per_client)), replace=False)
            bd.append(ext.features(add_trigger(Xtr[pick])))
        d["bd_feats"] = np.stack(bd)
        WORK.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **d)
    # commitments Com(A) (client side), cached
    com_path = WORK / f"commit_{tag}.json"
    if com_path.exists():
        com = json.loads(com_path.read_text())
    else:
        t = time.time()
        clients = [{"client_id": c, "features": d["feats"][d["client_idx"][c]].tolist(),
                    "labels": d["labels"][d["client_idx"][c]].tolist()} for c in range(args.clients)]
        com = gocli.commit(str(WORK / f"datasets_{tag}"), clients)
        com["seconds"] = time.time() - t
        com_path.write_text(json.dumps(com))
    return d, {c["client_id"]: c for c in com["clients"]}, com.get("seconds")


def main():
    args = parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t_all = time.time()
    d, com, commit_s = prepare_data(args)
    mal = set(range(args.malicious))
    targets = set(range(args.malicious + args.benign_unlearn))       # clients that request unlearning
    audit_rng = random.Random(args.seed + 777)                       # server-side, unpredictable to clients

    def ev(W, b):
        return (accuracy(W, b, d["test_feats"], d["test_labels"]),
                accuracy(W, b, d["test_bd_feats"], np.full(len(d["test_bd_feats"]), args.target)))

    W = [[0] * D for _ in range(C)]
    b = [0] * C
    forgotten = {c: [] for c in com}          # server-stored forget masks (as lists of cleared positions)
    flagged, caught_round = set(), {}
    hist, T = [], {"native_ms": 0.0, "proofs": 0, "prove_ms": [], "witness_ms": [], "verify_ms": [],
                   "pk_load_ms": 0.0, "compile_ms": 0.0, "attack_ms": 0.0, "server_round_ms": [], "rounds": 0}
    attacker, prep_info, rollbacks = None, None, 0
    first_round = 0
    if args.resume_fl:
        ck = json.loads(open(args.resume_fl).read())
        W, b, first_round = ck["W"], ck["b"], ck["fl_rounds"]
        args.fl_rounds = first_round
    fu_ckpt = None

    def client_view(c):
        return d["feats"][d["client_idx"][c]], d["labels"][d["client_idx"][c]]

    total_rounds = args.fl_rounds + args.fu_rounds
    for r in range(first_round, total_rounds):
        phase = "FL" if r < args.fl_rounds else "FU"
        if r == args.fl_rounds:   # ---------------- start of FU: unlearning requests + attacker prep
            ma, ba = ev(W, b)
            hist.append({"round": r, "phase": "FL-end", "MA": ma, "BA": ba})
            json.dump({"W": W, "b": b, "fl_rounds": r}, open(WORK / f"ckpt_fl_{args.name}.json", "w"))
            for c in targets:
                nreal = com[c]["n_real"]
                cleared = sorted(random.Random(1000 + c).sample(range(nreal), max(1, int(args.forget_frac * nreal))))
                # server-side validation of the request (length N implied by positions < N)
                assert all(0 <= i < N for i in cleared) and any(i < nreal for i in cleared)
                assert set(forgotten[c]) <= set(cleared)          # may only clear bits
                forgotten[c] = cleared
            fu_ckpt = (W, b)
            if args.attack == "fusion" and mal:
                attacker = FusionAttacker(target=args.target, rho=args.rho, beta=args.beta, lam=args.lam,
                                          ft_steps=args.ft_steps, seed=args.seed)
                ch = np.concatenate([client_view(c)[0] for c in mal])
                cy = np.concatenate([client_view(c)[1] for c in mal])
                prep_info = attacker.prepare(W, b, ch, cy, np.concatenate([d["bd_feats"][c] for c in mal]))
                print(f"[attack] FUsion prepared: {prep_info}")

        t_round = time.time()
        active = [c for c in sorted(com) if c not in flagged]
        jobs = [{"client_id": c, "dataset": com[c]["dataset"], "round": r, "W": W, "b": b,
                 "forgotten": forgotten[c], "prove": False} for c in active]
        nat = gocli.prove(jobs)                                     # honest native steps (Go, fixed point)
        T["native_ms"] += nat["total_ms"]
        received = {}
        for res in nat["results"]:
            c = res["client_id"]
            if res.get("error"):
                raise RuntimeError(f"client {c}: {res['error']}")
            W2, b2 = res["W2"], res["b2"]
            if phase == "FU" and attacker is not None and c in mal:
                ta = time.time()
                ch, cy = client_view(c)
                keep = np.ones(len(cy), bool)
                keep[[i for i in forgotten[c] if i < len(cy)]] = False
                W2, b2 = attacker.patch(W2, b2, ch[keep], cy[keep], d["bd_feats"][c])
                T["attack_ms"] += (time.time() - ta) * 1000
            if not head_in_range(W2, b2):
                flagged.add(c)
                continue
            received[c] = (W2, b2)

        # ---------------- server: choose audits AFTER updates arrived ----------------
        t_srv = time.time()
        audited = [c for c in received if args.audit_p > 0 and audit_rng.random() < args.audit_p]
        caught_now = []
        if audited:
            pr = gocli.prove([{**j, "prove": True} for j in jobs if j["client_id"] in audited])
            T["pk_load_ms"] += pr["pk_load_ms"]
            T["compile_ms"] += pr["compile_ms"]
            proofs = {}
            for res in pr["results"]:
                T["proofs"] += 1
                T["prove_ms"].append(res["prove_ms"])
                T["witness_ms"].append(res["witness_ms"])
                proofs[res["client_id"]] = res.get("proof", "")
            t_srv = time.time()
            ver = gocli.verify([{"client_id": c, "round": r, "root": com[c]["root"], "n_real": com[c]["n_real"],
                                 "W": W, "b": b, "W2": received[c][0], "b2": received[c][1],
                                 "forgotten": forgotten[c], "proof": proofs[c]} for c in audited])
            for v in ver["results"]:
                T["verify_ms"].append(v["verify_ms"])
                if not v["ok"]:
                    flagged.add(v["client_id"])
                    caught_round[v["client_id"]] = r
                    caught_now.append(v["client_id"])
        accepted = [received[c] for c in received if c not in flagged]
        W, b = fedavg(accepted)
        if caught_now and phase == "FU" and args.on_catch == "rollback":
            W, b = fu_ckpt                                           # discard the possibly-poisoned FU progress
            rollbacks += 1
        T["server_round_ms"].append((time.time() - t_srv) * 1000)
        T["rounds"] += 1

        row = {"round": r, "phase": phase, "audited": len(audited), "caught": caught_now,
               "accepted": len(accepted), "round_s": round(time.time() - t_round, 2)}
        if (r + 1) % args.eval_every == 0 or r == total_rounds - 1 or r == args.fl_rounds - 1:
            row["MA"], row["BA"] = ev(W, b)
        hist.append(row)
        print(f"[{phase} {r:3d}] accepted={len(accepted):2d} audited={len(audited):2d} caught={caught_now} "
              + (f"MA={row['MA']:.3f} BA={row['BA']:.3f} " if "MA" in row else "") + f"({row['round_s']}s)")

    ma, ba = ev(W, b)
    fl_end = next((h for h in hist if h["phase"] == "FL-end"), None)
    fu_rows = [h for h in hist if h["phase"] == "FU" and "BA" in h]
    m_rounds = next((h["round"] - args.fl_rounds + 1 for h in fu_rows if h["BA"] > 0.6), None)
    pm = np.array(T["prove_ms"]) if T["prove_ms"] else np.array([np.nan])
    out = {
        "args": vars(args), "final_MA": ma, "final_BA": ba, "attack_success": ba > 0.6,
        "FL_end": fl_end, "rounds_to_BA60": m_rounds, "attack_prep": prep_info,
        "malicious": sorted(mal), "flagged": sorted(flagged), "caught_round": caught_round,
        "TP": len(flagged & mal), "FP": len(flagged - mal), "rollbacks": rollbacks,
        "timing": {
            "wall_s": time.time() - t_all, "commit_s_once": commit_s, "rounds": T["rounds"],
            "native_total_s": T["native_ms"] / 1000, "attacker_total_s": T["attack_ms"] / 1000,
            "proofs": T["proofs"], "prove_s_mean": float(np.nanmean(pm)) / 1000,
            "prove_s_std": float(np.nanstd(pm)) / 1000,
            "witness_ms_mean": float(np.mean(T["witness_ms"])) if T["witness_ms"] else None,
            "client_proving_total_s": (sum(T["prove_ms"]) + sum(T["witness_ms"])) / 1000,
            "pk_load_total_s": T["pk_load_ms"] / 1000, "compile_total_s": T["compile_ms"] / 1000,
            "verify_ms_mean": float(np.mean(T["verify_ms"])) if T["verify_ms"] else None,
            "server_verify_total_s": sum(T["verify_ms"]) / 1000,
            "server_round_ms_mean": float(np.mean(T["server_round_ms"])),
        },
        "history": hist,
    }
    (WORK.parent / "results").mkdir(exist_ok=True)
    path = WORK.parent / "results" / f"{args.name}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[result] MA={ma:.3f} BA={ba:.3f} success={ba > 0.6} flagged={sorted(flagged)} "
          f"proofs={T['proofs']} wall={out['timing']['wall_s']:.0f}s -> {path}")


if __name__ == "__main__":
    main()
