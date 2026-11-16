# FUsion Backdoor in Active Federated Unlearning — run & test guide

This project reproduces the attack **and** defense from *"Malicious Forgetting: Backdoor
Injection in Active Federated Unlearning and Countermeasure Design"* (IEEE INFOCOM 2026),
as a local simulation of **1 server + N clients**:

1. **FL training** — all clients train an MNIST CNN honestly (FedAvg).
2. **Backdoor preparation** — malicious clients secretly build a compact **backdoor subnetwork**
   (BC-layer identification + Grad-CAM trigger-sensitive units), Sec. III-B.
3. **Active FU** — clients request unlearning; malicious clients **fuse** the subnetwork into their
   unlearning updates (Sec. III-C) to implant the backdoor within the limited unlearning rounds.
4. **SubID defense** — the server detects malicious clients from directional subspace deviations
   (Sec. IV), reporting TPR / FPR / F1.
5. **ZK-compatibility hooks** — dataset commitments, forget bitmask, and a pluggable `Verifier`
   so you can later drop in the *Verifiable FU with ZKPs* paper (no ZK system is chosen yet).

Everything runs on CPU. No GPU needed.

---

## 0. One-time setup

```bash
cd /Users/neerajkumar/Downloads/btp
python3.12 -m venv .venv            # a venv already exists if you ran the setup; skip if so
.venv/bin/pip install -r requirements.txt
```

Verify the install:

```bash
.venv/bin/python -c "import torch, torchvision; print('torch', torch.__version__)"
```

---

## 1. Fastest sanity check (seconds, random data)

```bash
.venv/bin/python -m fubd --preset tiny --attack fusion --verifier none
```

`tiny` uses random toy data and ~6 clients. It only proves the plumbing works end-to-end
(you'll see `attack_success=True`). It is **not** meaningful for accuracy numbers — use `quick`
or `paper` below for real results.

Run the test suite (9 tests, ~6 s):

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## 2. The three experiments you actually want

All write a JSON summary to `runs/<name>.json` and print a live MA/BA trace.
`MA` = main-task accuracy (clean test set). `BA` = backdoor accuracy (triggered inputs → target
label). A **successful attack** is `BA > 0.60` (the paper's threshold).

### (A) Attack works when the server is naive — `verifier=none`

```bash
.venv/bin/python -m fubd --preset quick --attack fusion --verifier none \
    --name A_attack --threads 8
```

Expect: `MA` high (~0.95+), `BA` climbing past 0.60 during unlearning → `attack_success=True`.
This is the paper's threat: the backdoor is injected purely in the FU phase.

### (B) SubID detects the malicious clients

SubID is on by default. Look at the `[SubID]` lines in run (A)'s output (or its JSON):

```bash
cat runs/A_attack.json | .venv/bin/python -m json.tool | grep -A20 '"subid"'
```

Expect `TPR` high / `FPR` low — SubID names the malicious client IDs (`0 .. n_malicious-1`).

### (C) The ZKP hook blocks the attack — `verifier=replay`

```bash
.venv/bin/python -m fubd --preset quick --attack fusion --verifier replay \
    --name C_zk --threads 8
```

Expect: `rejected` grows every round and `BA` stays ~0 → `attack_success=False`.
The `replay` verifier is a **cleartext oracle**: it recomputes the fixed honest unlearning
procedure on each client's committed data + forget mask and accepts only a bit-identical update.
A fused (patched) update cannot match, so it is rejected — exactly the statement a zero-knowledge
proof would certify. **This is the slot where your ZKP paper plugs in** (see §5).

---

## 3. Reproducing the paper's default scenario (slow)

The paper default: 50 clients, 10 malicious, 500 training + 200 unlearning rounds, NoT unlearning,
FedAvg, IID MNIST. On a laptop CPU this is tens of minutes — run it in the background:

```bash
nohup .venv/bin/python -m fubd --preset paper --attack fusion --verifier none \
    --name paper_fusion --threads 8 > runs/paper_fusion.out 2>&1 &
tail -f runs/paper_fusion.out      # watch; Ctrl-C to stop watching (job keeps running)
```

Compare against the **baseline** attack (direct backdoor-data injection, no subnetwork — the paper
shows this fails under FedRR/ConFUSE):

```bash
.venv/bin/python -m fubd --preset quick --attack baseline --name baseline --threads 8
```

---

## 4. Knobs for ablations (reproduce the paper's figures)

Any `Config` field (see [fubd/config.py](fubd/config.py)) is a CLI flag: `--<field> <value>`.
Presets set sensible defaults; flags override them. Useful ones:

| Flag | Meaning | Paper reference |
|------|---------|-----------------|
| `--n_clients 50 --n_malicious 10` | client / attacker counts | Fig. 2–3 (vary ratio) |
| `--fu_method not` \| `ga` \| `retrain` | active FU method (NoT / grad-ascent / fine-tune) | NoT, FedRR, ConFUSE analogues |
| `--aggregator mean` \| `trim` \| `median` | server aggregation | Fig. 8–9 |
| `--rho 0.1` | fraction of trigger-sensitive units | Fig. 4 |
| `--beta 0.2` | subnetwork fusion strength | Fig. 6–7 |
| `--theta 0.6` | BC-layer contiguous-expansion threshold | Eq. (7) |
| `--iid false --dirichlet_alpha 0.5` | non-IID client data | Fig. 5, Fig. 13 |
| `--fu_benign_participate true` | all clients participate in FU (default; keeps malicious a minority so SubID's median/MAD stays benign-anchored). Set `false` to let only target clients participate. | Sec. IV |
| `--subid_kappa 7 --subid_phi 0.5` | SubID MAD threshold / flag fraction | Fig. 11–12 |
| `--attack signflip` | untargeted sign-flip (SubID robustness) | Table III |
| `--attack none` | no attack (SubID false-positive check) | Table III "w/o attack" |

Examples:

```bash
# malicious-client-ratio sweep (Fig. 2)
for m in 1 2 4 6 8; do
  .venv/bin/python -m fubd --preset quick --attack fusion --n_malicious $m \
      --name ratio_$m --threads 8
done

# aggregation robustness (Fig. 8-9)
for agg in mean trim median; do
  .venv/bin/python -m fubd --preset quick --attack fusion --aggregator $agg \
      --name agg_$agg --threads 8
done

# non-IID attack + detection (Fig. 5 / Fig. 13)
.venv/bin/python -m fubd --preset quick --attack fusion --iid false --name noniid --threads 8

# SubID against non-adversarial baselines (Table III)
.venv/bin/python -m fubd --preset quick --attack signflip --name signflip --threads 8
.venv/bin/python -m fubd --preset quick --attack none     --name noattack --threads 8
```

---

## 5. Where the ZKP paper plugs in (for your BTP)

You haven't picked a ZK system yet, so the code is written so you don't have to. The interfaces
live in [fubd/zk_hooks.py](fubd/zk_hooks.py):

- **`DatasetCommitment`** — `Com(A)` = Merkle root over each client's salted samples, built once
  before training (ZKP paper Step 2, "bind the data"). Already attached to every client.
- **forget bitmask `b`** — one bit per sample (`1` = use, `0` = forget), on `Client.mask`
  (Step 3, the bitmask idea from zkUnlearner). The honest FU procedure uses it.
- **`ClientMessage`** — what a client sends: update + `base_digest` + `commit_root` + `mask_hash`
  + a `proof` placeholder.
- **`Verifier`** — server-side check (Step 1, "prove the procedure").
  - `NullVerifier` (`--verifier none`): accepts everything = the attacked setting.
  - `ReplayVerifier` (`--verifier replay`): cleartext oracle described in §2(C).

**To add a real ZK proof later:** write a `ZKVerifier.verify(msg)` that checks `msg.proof` against
the public inputs (`base_digest`, update digest, `commit_root`, `mask_hash`). The statement the
circuit must encode is already implemented and fixed in `Client.honest_fl_round` /
`honest_fu_round` — those are **deterministic pure functions** of (global model, committed data,
mask), which is exactly what makes them provable. The test
`tests/test_pipeline.py::test_honest_round_deterministic` guards that property; keep it green when
you modify the procedure, or your circuit and your code will diverge.

Candidate backends to drop in: Groth16 (e.g. via `circom`/`snarkjs` or `arkworks`),
Halo2, or **EZKL** (proves PyTorch/ONNX graphs directly — the lowest-friction option since the
procedure is already a small torch model).

---

## 6. Output files

- `runs/<name>.json` — full config + `final_MA`, `final_BA`, `attack_success`, `rejected_updates`,
  `bc_layers`, `ts_units`, and the `subid` block (`predicted`, `truth`, `TPR`, `FPR`, `F1`,
  per-client `flag_fraction`).
- Console — per-round `MA` / `BA` trace and `[SubID]` / `[RESULT]` summaries.

## 7. Code map

| File | Role |
|------|------|
| [fubd/config.py](fubd/config.py) | all hyperparameters + `tiny`/`quick`/`paper` presets |
| [fubd/data.py](fubd/data.py) | MNIST / toy data, client partition (IID & Dirichlet), trigger |
| [fubd/models.py](fubd/models.py) | MNIST CNN + parameter-vector helpers |
| [fubd/client.py](fubd/client.py) | client state, commitment, forget mask, **deterministic** honest FL/FU rounds |
| [fubd/attack.py](fubd/attack.py) | **FUsion**: BC layers (Eq. 4–8), Grad-CAM TS units, subnetwork (Eq. 10), fusion (Eq. 11–12); baseline; sign-flip |
| [fubd/unlearning](fubd/client.py) | NoT / gradient-ascent / retrain FU methods (in `client.py`) |
| [fubd/aggregators.py](fubd/aggregators.py) | FedAvg / trimmed-mean / median |
| [fubd/subid.py](fubd/subid.py) | **SubID** detection (Eq. 13–15) + TPR/FPR/F1 |
| [fubd/zk_hooks.py](fubd/zk_hooks.py) | commitments, bitmask digest, `Verifier` interface (ZKP slot) |
| [fubd/runner.py](fubd/runner.py) | end-to-end orchestration |
| [tests/test_pipeline.py](tests/test_pipeline.py) | 9 fast correctness tests |

---
---

# Part 2 — ZKP prevention (Groth16) + random auditing: `zkfu/`

Implements *Verifiable Federated Unlearning with Zero-Knowledge Proofs*: every client update is
one fixed-point SGD step of a linear head on a frozen CNN, proved with Groth16 (gnark, BN254)
against the client's committed dataset (Merkle root) and its forget mask. The FUsion attacker's
patched update has no valid proof. Design choices and deviations: `zkfu/DESIGN_NOTES.md`.
Results and timing report: `zkfu/ZK_REPORT.md`.

## 2.0 Requirements (one time)

```bash
go version                      # Go >= 1.24; go.mod pins toolchain go1.26.8 (auto-downloaded)
cd /Users/neerajkumar/Downloads/btp/zkfu
go mod download                 # gnark v0.16.5, gnark-crypto (pinned in go.mod)
../.venv/bin/pip install -r sim/requirements.txt   # numpy/torch/torchvision (already in .venv)
```

## 2.1 Build, set up keys, run the tests (milestones 1–3)

```bash
cd /Users/neerajkumar/Downloads/btp/zkfu
for c in setup commit prove verify bench; do go build -o bin/$c ./cmd/$c; done
./bin/setup -keys keys -breakdown -json results/setup.json   # ~16 s: constraints, pk/vk
go test ./tests/ -count=1 -v                                # ~1 min, all must PASS
```

Tests: truncation gadget vs native (random/zero/negative/boundary), Poseidon2 native vs circuit,
Merkle native vs circuit, **1000 random steps native == circuit bit for bit**, completeness (honest
FL and FU steps verify), and soundness (weight +1 patch, wrong model, fresh/modified sample, wrong
index, flipped mask bit, swapped public inputs, out-of-range and field-wrapped witnesses).
Fewer equivalence steps while iterating: `ZKFU_EQUIV_STEPS=100 go test ./tests/`.

## 2.2 Benchmark proving / verification (spec §10)

```bash
/usr/bin/time -l ./bin/bench -keys keys -runs 20 -json results/bench_random.json
# "maximum resident set size" printed by /usr/bin/time = peak prover memory
```

## 2.3 Simulation: attack vs ZK defence (run from `zkfu/sim`)

```bash
cd /Users/neerajkumar/Downloads/btp/zkfu/sim
PY=../../.venv/bin/python
$PY fl_sim.py --audit_p 0    --name noverify          # FUsion succeeds (no proofs)   ~10 s
$PY fl_sim.py --audit_p 0.1  --name audit10           # random auditing, 10 %         ~5 min
$PY fl_sim.py --audit_p 1.0  --resume_fl work/ckpt_fl_noverify.json --fu_rounds 3 --name full   # every update proved
$PY float_baseline.py                                 # float head, accuracy comparison
$PY summarize.py                                      # tables -> ../results/summary.md
```

The first run pretrains the frozen extractor (~5 s), extracts features and commits all 50
datasets (cached in `sim/work/`). Main flags:

| flag | meaning |
|---|---|
| `--audit_p` | 0 = accept unverified (attacked setting); 0<p<1 = random auditing; 1 = prove every update |
| `--on_catch rollback\|exclude` | after catching a cheater in FU: revert to the FU-start model (default) or only exclude it |
| `--attack fusion\|none` | FUsion head attacker on the malicious clients, or all honest |
| `--clients 50 --malicious 10 --benign_unlearn 5` | population |
| `--fl_rounds 20 --fu_rounds 20` | schedule (small defaults so runs take minutes) |
| `--features cnn+grid\|cnn` | frozen extractor variant (`cnn` = ablation where head-only FUsion barely works) |
| `--beta --rho --lam --ft_steps` | FUsion parameters (defaults = paper: 0.2, 0.1, 0.5) |
| `--resume_fl ckpt.json` | start FU from a saved FL-final head (to run p = 1 cheaply) |

Results go to `zkfu/results/<name>.json` (MA/BA history, who was caught and when, every timing).

## 2.4 Everything in the report, one command (~30 min)

```bash
cd /Users/neerajkumar/Downloads/btp/zkfu && ./run_experiments.sh        # or: ./run_experiments.sh quick
../.venv/bin/python sim/summarize.py
```
