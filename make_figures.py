"""Generate every figure of final_report.tex from the raw result files (no hand-copied numbers).

    .venv/bin/python make_figures.py      ->  figures/*.pdf

Sources: runs/part1_canonical.log (Part I), zkfu/results/*.json (Part II).
Palette: validated categorical slots (blue, orange, aqua, yellow, magenta, green) in fixed order;
each series also has its own marker shape (secondary encoding; three slots are < 3:1 on white).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "zkfu" / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W, FULL_W = 3.45, 7.1  # IEEE column / page width (inches)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ------------------------------------------------------------------ Part I trajectories
def part1():
    text = (ROOT / "runs" / "part1_canonical.log").read_text()
    sections = re.split(r"^### ", text, flags=re.M)[1:]
    series = {}
    for sec in sections:
        key = sec[0]
        fu = sec.split("[FU]", 1)
        fl_pts = re.findall(r"round\s+(\d+)\s+MA=([\d.]+)\s+BA=([\d.]+)", fu[0])
        fu_pts = re.findall(r"round\s+(\d+)\s+MA=([\d.]+)\s+BA=([\d.]+)", fu[1]) if len(fu) > 1 else []
        pts = [(0, float(fl_pts[-1][1]), float(fl_pts[-1][2]))] + [(int(r), float(m), float(b)) for r, m, b in fu_pts]
        series[key] = pts
    spec = [("A", "FUsion, no defence", BLUE, "o"), ("D", "Baseline attack", ORANGE, "s"),
            ("C", "FUsion + replay oracle", AQUA, "^"), ("E", "No attack", YELLOW, "D")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True)
    for k, lab, c, mk in spec:
        if k not in series:
            continue
        r, m, b = zip(*series[k])
        a1.plot(r, b, color=c, marker=mk, label=lab)
        a2.plot(r, m, color=c, marker=mk, label=lab)
    a1.axhline(0.6, color=INK2, lw=0.8, ls="--")
    a1.text(41, 0.5, "success threshold (BA = 0.6)", color=INK2, fontsize=6.5)
    a1.set_ylabel("Backdoor accuracy (BA)")
    a1.set_ylim(-0.03, 1.05)
    a2.set_ylabel("Main-task accuracy (MA)")
    a2.set_ylim(0.93, 0.985)
    a2.set_xlabel("Unlearning round (0 = end of FL)")
    a1.legend(loc="upper left")
    a1.set_title("Part I: full CNN, 10 clients (2 malicious), NoT unlearning")
    save(fig, "fig_part1")


# ------------------------------------------------------------------ Part II
P2 = [("E0_noverify_fusion", "No verification", BLUE, "o"), ("A05_audit", "Audit p = 5 %", ORANGE, "s"),
      ("A10_audit", "Audit p = 10 %", AQUA, "^"), ("A20_audit", "Audit p = 20 %", YELLOW, "D"),
      ("A10_exclude", "Audit p = 10 %, no rollback", MAGENTA, "v"), ("F_full_proofs", "Every update proved", GREEN, "P")]


def fu_curve(r):
    fl = r["args"]["fl_rounds"]
    pts = [(0, r["FL_end"]["MA"], r["FL_end"]["BA"])] + [
        (h["round"] - fl + 1, h["MA"], h["BA"]) for h in r["history"] if h["phase"] == "FU" and "BA" in h]
    return zip(*pts)


def part2_ba():
    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
    for name, lab, c, mk in P2:
        x, _, b = fu_curve(load(name))
        ax.plot(x, b, color=c, marker=mk, label=lab, ms=5 if name == "F_full_proofs" else 4)
    ax.axhline(0.6, color=INK2, lw=0.8, ls="--")
    ax.text(0.3, 0.62, "attack success threshold (BA = 0.6)", color=INK2, fontsize=6.5)
    ax.set_xlabel("Unlearning round (0 = end of FL)")
    ax.set_ylabel("Backdoor accuracy (BA)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(range(0, 21, 2))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig_part2_ba")


def part2_ma():
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for name, lab, c, mk in [("E1_noverify_noattack", "No attack (fixed point)", BLUE, "o"),
                             ("E0_noverify_fusion", "FUsion, no verification", ORANGE, "s"),
                             ("A10_audit", "FUsion, audit p = 10 %", AQUA, "^")]:
        r = load(name)
        pts = [(h["round"] + 1, h["MA"]) for h in r["history"] if "MA" in h and h["phase"] in ("FL", "FU")]
        x, m = zip(*pts)
        ax.plot(x, m, color=c, marker=mk, label=lab)
    fb = load("float_baseline")
    ax.plot([20, 40], [fb["FL_end_MA"], fb["final_MA"]], ls="none", marker="*", ms=8, color=YELLOW,
            markeredgecolor=INK, markeredgewidth=0.4, label="Float training (same schedule)")
    ax.axvline(20.5, color=INK2, lw=0.8, ls=":")
    ax.text(21, 0.949, "FU starts", color=INK2, fontsize=6.5)
    ax.set_xlabel("Global round")
    ax.set_ylabel("Main-task accuracy (MA)")
    ax.set_ylim(0.85, 0.955)
    ax.legend(loc="lower right")
    save(fig, "fig_part2_ma")


def constraints():
    st = json.loads((RES / "setup.json").read_text())
    b = st["breakdown"]
    items = [("Truncations (970)", b["truncations_total"]),
             ("Head hashes H(W,b), H(W',b')", b["head_hashes_x2"]),
             ("Forward + gradient products", b["multiplications_fwd_grad"]),
             ("32 samples: range checks,\nleaf hash, Merkle path", b["all_samples_B"])]
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    labels, vals = zip(*items)
    y = np.arange(len(items))
    ax.barh(y, vals, color=BLUE, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 3000, yi, f"{v/1000:.1f}k ({100*v/st['constraints']:.0f}%)", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.set_xlabel(f"R1CS constraints (total {st['constraints']:,})")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_constraints")


def timing():
    bm, st, pt, F = load("bench_real"), load("setup"), load("pipeline_timing"), load("F_full_proofs")
    items = [("Groth16 setup (once)", st["setup_s"]),
             ("Commit 50 clients (once)", pt["commit_50_clients_total_s"]),
             ("Prove one step", bm["prove_s_mean"]),
             ("Compile circuit", st["compile_s"]),
             ("Server round (50 verifies + I/O + FedAvg)", F["timing"]["server_round_ms_mean"] / 1000),
             ("Commit one client", pt["commit_one_client_go_ms"] / 1000),
             ("Merkle tree rebuild", bm["merkle_tree_build_ms"] / 1000),
             ("Load proving key", bm["pk_load_s"]),
             ("Verify one proof", bm["verify_ms_mean"] / 1000),
             ("Native fixed-point step", bm["native_step_ms_mean"] / 1000),
             ("Witness assembly", bm["witness_build_ms_mean"] / 1000)]
    items.sort(key=lambda t: t[1])
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    y = np.arange(len(items))
    ax.barh(y, vals, color=BLUE, height=0.62)
    for yi, v in zip(y, vals):
        s = f"{v*1000:.2f} ms" if v < 0.01 else (f"{v*1000:.0f} ms" if v < 1 else f"{v:.2f} s")
        ax.text(v * 1.25, yi, s, va="center", fontsize=6.3, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 2e2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Time (s, log scale), Apple M3, 8 cores")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_timing")


def audit():
    e0 = load("E0_noverify_fusion")
    m, R = e0["rounds_to_BA60"], e0["args"]["fu_rounds"]
    per = load("bench_real")["prove_s_mean"] + load("bench_real")["merkle_tree_build_ms"] / 1000
    p = np.linspace(0, 1, 201)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    a1.plot(p, 1 - (1 - p) ** 40, color=BLUE, label="paper's assumption, m = 40 rounds")
    a1.plot(p, 1 - (1 - p) ** m, color=ORANGE, ls="-", label=f"measured m = {m} rounds")
    a1.plot(p, 1 - (1 - p) ** R, color=AQUA, ls="--", label=f"any of the {R} FU rounds")
    pts = [(0.05, "A05_audit"), (0.10, "A10_audit"), (0.20, "A20_audit"), (1.0, "F_full_proofs")]
    xs = [x for x, _ in pts]
    ys = [load(n)["TP"] / len(load(n)["malicious"]) for _, n in pts]
    a1.plot(xs, ys, ls="none", marker="D", ms=5, color=YELLOW, markeredgecolor=INK, markeredgewidth=0.4,
            label="measured: fraction of 10 attackers caught")
    a1.set_xscale("log")
    a1.set_xlim(0.02, 1.05)
    a1.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0], ["2%", "5%", "10%", "20%", "50%", "100%"])
    a1.set_xlabel("Audit probability p")
    a1.set_ylabel("P(one attacker caught)")
    a1.set_title("(a) Detection probability")
    a1.legend(loc="lower right", fontsize=6.3)
    n_upd = e0["args"]["clients"] * (e0["args"]["fl_rounds"] + e0["args"]["fu_rounds"])
    a2.plot(p, n_upd * p * per / 60, color=BLUE, label=f"model: {n_upd} updates × p × {per:.2f} s")
    xs2 = [0.05, 0.10, 0.20]
    ys2 = [load(n)["timing"]["client_proving_total_s"] / 60 for n in ("A05_audit", "A10_audit", "A20_audit")]
    a2.plot(xs2, ys2, ls="none", marker="s", ms=5, color=ORANGE, label="measured (simulation)")
    a2.plot([1.0], [n_upd * per / 60], ls="none", marker="P", ms=6, color=GREEN,
            label="every update (extrapolated from p = 1 run)")
    a2.set_xlabel("Audit probability p")
    a2.set_ylabel("Total client proving time (min)")
    a2.set_title(f"(b) Proving cost, {e0['args']['clients']} clients × {e0['args']['fl_rounds'] + e0['args']['fu_rounds']} rounds")
    a2.legend(loc="upper left", fontsize=6.3)
    save(fig, "fig_audit")


def catches():
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    rows = [("F_full_proofs", "p = 100%", GREEN, "P"), ("A20_audit", "p = 20%", YELLOW, "D"),
            ("A10_audit", "p = 10%", AQUA, "^"), ("A05_audit", "p = 5%", ORANGE, "s")]
    for i, (name, lab, c, mk) in enumerate(rows):
        r = load(name)
        fl = r["args"]["fl_rounds"]
        rounds = sorted(v - fl + 1 for v in r["caught_round"].values())
        jit = np.linspace(-0.22, 0.22, max(len(rounds), 1))
        xs, ys, seen = [], [], {}
        for x in rounds:
            k = seen.get(x, 0)
            seen[x] = k + 1
            xs.append(x)
            ys.append(i + (k - (rounds.count(x) - 1) / 2) * 0.07)
        ax.plot(xs, ys, ls="none", marker=mk, color=c, ms=4.5, markeredgecolor="white", markeredgewidth=0.4)
        miss = len(r["malicious"]) - r["TP"]
        ax.text(21.3, i, f"{r['TP']}/10 caught" + (f", {miss} never" if miss else ""), va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.set_xlim(0.3, 20.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Unlearning round in which an attacker was caught")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_catch")


def hashes():
    h = load("hashcost")
    items = [("Poseidon2", h["two_to_one_hash"]["poseidon2"], h["projected_total"]["poseidon2"]),
             ("MiMC", h["two_to_one_hash"]["mimc"], h["projected_total"]["mimc"]),
             ("SHA-256", h["two_to_one_hash"]["sha256_64B"], h["projected_total"]["sha256"])]
    fig, ax = plt.subplots(figsize=(COL_W, 1.35))
    y = np.arange(len(items))
    ax.barh(y, [v for _, v, _ in items], color=BLUE, height=0.6)
    for yi, (_, v, tot) in zip(y, items):
        ax.text(v * 1.3, yi, f"{v:,}  (circuit ≈ {tot/1e6:.2f} M)", va="center", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(100, 3e7)
    ax.set_yticks(y, [n for n, _, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("R1CS constraints per 2-to-1 hash (log scale)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_hash")


if __name__ == "__main__":
    part1()
    part2_ba()
    part2_ma()
    constraints()
    timing()
    audit()
    catches()
    hashes()
