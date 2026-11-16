"""Build the result tables of ZK_REPORT.md from results/*.json (no hand-copied numbers).

    ../../.venv/bin/python summarize.py   ->  results/summary.md
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    out = []
    st, br, bl = load("setup"), load("bench_random"), load("bench_real")
    if st:
        out += ["## Circuit and keys", "", "| item | value |", "|---|---|",
                f"| constraints (R1CS) | {st['constraints']:,} |", f"| public inputs | {st['public_inputs']} |",
                f"| compile time | {st['compile_s']:.2f} s |", f"| Groth16 setup time | {st['setup_s']:.1f} s |",
                f"| proving key | {st['pk_bytes']/1e6:.1f} MB |", f"| verifying key | {st['vk_bytes']:,} B |", ""]
        if "breakdown" in st:
            b = st["breakdown"]
            tot = st["constraints"]
            out += ["### Constraint breakdown (marginal cost per component)", "", "| component | constraints | share |",
                    "|---|---:|---:|"]
            for k, lab in [("all_samples_B", "32 samples: range checks, one-hot, leaf hash, Merkle path"),
                           ("head_hashes_x2", "2 head hashes: range checks + H(W,b), H(W',b')"),
                           ("multiplications_fwd_grad", "forward + gradient multiplications"),
                           ("truncations_total", "970 truncations (hint + 2 range checks each)")]:
                out.append(f"| {lab} | {b[k]:,} | {100*b[k]/tot:.1f}% |")
            out += [f"| *(range-check lookup amortisation difference)* | {b['remainder_overhead']:,} | |", "",
                    f"Poseidon2 compression = {b['poseidon2_compression']} constraints; one sample "
                    f"(leaf + 11-level path + 64 range checks + one-hot) = {b['per_sample_leaf_rangecheck_merkle']:,}.", ""]
    for name, bm in [("random features", br), ("real client dataset", bl)]:
        if bm:
            out += [f"## Proving benchmark ({name}, {bm['runs']} proofs, {bm['cpus']} cores)", "",
                    "| metric | value |", "|---|---|",
                    f"| proving time per step | **{bm['prove_s_mean']:.3f} s ± {bm['prove_s_std']:.3f}** "
                    f"(min {bm['prove_s_min']:.3f}, max {bm['prove_s_max']:.3f}) |",
                    f"| verification time | **{bm['verify_ms_mean']:.2f} ms ± {bm['verify_ms_std']:.2f}** |",
                    f"| proof size | {bm['proof_bytes']} B |",
                    f"| native fixed-point step | {bm['native_step_ms_mean']:.2f} ms |",
                    f"| Merkle tree rebuild (2048 leaves) | {bm['merkle_tree_build_ms']:.0f} ms |",
                    f"| witness assembly | {bm['witness_build_ms_mean']:.2f} ms |",
                    f"| proving-key load | {bm['pk_load_s']:.2f} s |",
                    f"| Go runtime peak memory | {bm['go_runtime_sys_peak_mb']:.0f} MB |", ""]

    runs = [("E1_noverify_noattack", "no verification, no attack"), ("E0_noverify_fusion", "no verification, FUsion"),
            ("E2_cnn_features_fusion", "ablation: pure-CNN features, FUsion"),
            ("A05_audit", "auditing p = 5 %"), ("A10_audit", "auditing p = 10 %"), ("A20_audit", "auditing p = 20 %"),
            ("A10_exclude", "auditing p = 10 %, exclude only"), ("F_full_proofs", "every update proved (p = 1)")]
    out += ["## Attack vs. defence (50 clients, 10 malicious, 5 benign unlearners)", "",
            "| run | setting | FL-end MA | final MA | final BA | attack OK | caught | FP | rollbacks | proofs | client proving (CPU s) | server verify (s) | wall (s) |",
            "|---|---|---:|---:|---:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|"]
    for k, lab in runs:
        r = load(k)
        if not r:
            continue
        t = r["timing"]
        fl = r.get("FL_end") or {}
        out.append(f"| {k} | {lab} | {f(fl.get('MA'))} | {f(r['final_MA'])} | **{f(r['final_BA'])}** | "
                   f"{'yes' if r['attack_success'] else 'no'} | {r['TP']}/{len(r['malicious'])} | {r['FP']} | "
                   f"{r['rollbacks']} | {t['proofs']} | {t['client_proving_total_s']:.0f} | "
                   f"{t['server_verify_total_s']:.2f} | {t['wall_s']:.0f} |")
    fb = load("float_baseline")
    e1 = load("E1_noverify_noattack")
    if fb and e1:
        out += ["", "## Fixed point vs float (same head, same schedule, no attack)", "",
                "| | after FL | after FU |", "|---|---:|---:|",
                f"| float | {fb['FL_end_MA']:.4f} | {fb['final_MA']:.4f} |",
                f"| fixed point (proved procedure) | {e1['FL_end']['MA']:.4f} | {e1['final_MA']:.4f} |",
                f"| difference | {e1['FL_end']['MA']-fb['FL_end_MA']:+.4f} | {e1['final_MA']-fb['final_MA']:+.4f} |"]
    e0 = load("E0_noverify_fusion")
    if e0:
        traj = [(h["round"] - e0["args"]["fl_rounds"] + 1, h["MA"], h["BA"]) for h in e0["history"]
                if h["phase"] == "FU" and "BA" in h]
        out += ["", "## FUsion trajectory without verification (E0)", "", "| FU round | " + " | ".join(str(a) for a, _, _ in traj) + " |",
                "|---|" + "---:|" * len(traj), "| MA | " + " | ".join(f"{m:.3f}" for _, m, _ in traj) + " |",
                "| BA | " + " | ".join(f"{b:.3f}" for _, _, b in traj) + " |",
                f"", f"Rounds of patching to reach BA > 0.6: **m = {e0['rounds_to_BA60']}**. Attack prep: {e0['attack_prep']}"]
        m = e0["rounds_to_BA60"] or 1
        R_fu = e0["args"]["fu_rounds"]
        out += ["", "## Auditing: analytic catch probability with the measured m", "",
                f"| p | P(one attacker caught within m={m}) | paper's figure (m=40) | P(one attacker caught within {R_fu} FU rounds) | P(all 10 caught within {R_fu}) | proving saved |",
                "|---|---:|---:|---:|---:|---:|"]
        for p in (0.05, 0.10, 0.20, 0.60, 1.0):
            out.append(f"| {p:.2f} | {1-(1-p)**m:.3f} | {1-(1-p)**40:.3f} | {1-(1-p)**R_fu:.3f} | {(1-(1-p)**R_fu)**10:.3f} | {100*(1-p):.0f}% |")
    full = load("F_full_proofs")
    if full and bl:
        n, rounds = full["args"]["clients"], e0["args"]["fl_rounds"] + e0["args"]["fu_rounds"] if e0 else 40
        per = bl["prove_s_mean"] + bl["merkle_tree_build_ms"] / 1000
        out += ["", "## Cost of full verification vs auditing (whole 40-round schedule, 50 clients)", "",
                f"Measured with p = 1 (F_full_proofs): server time per round = verify {n} proofs = "
                f"{full['timing']['server_verify_total_s']/full['timing']['rounds']:.3f} s; mean proving time "
                f"{full['timing']['prove_s_mean']:.3f} s per client per round.", "",
                "| mode | proofs | client proving, total CPU | per client (parallel clients) |", "|---|---:|---:|---:|",
                f"| every update (p = 1) | {n*rounds} | {n*rounds*per/60:.1f} min | {rounds*per:.0f} s |"]
        for k in ("A05_audit", "A10_audit", "A20_audit"):
            a = load(k)
            if a:
                out.append(f"| {k} (measured) | {a['timing']['proofs']} | {a['timing']['client_proving_total_s']/60:.1f} min | "
                           f"{a['timing']['client_proving_total_s']/n:.0f} s |")
    (R / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
