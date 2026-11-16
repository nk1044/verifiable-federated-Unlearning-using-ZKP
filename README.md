# Malicious Forgetting Meets Zero Knowledge

This project reproduces the **FUsion backdoor** attack on active federated unlearning (FU) together
with its statistical defence, **SubID**. It then **prevents** the attack cryptographically: every
client update must carry a **Groth16 zero-knowledge proof** that it is exactly one honest training
step on the client's own committed data, with **random auditing** to cut the proving cost.

It is based on two papers:

1. *Malicious Forgetting: Backdoor Injection in Active Federated Unlearning and Countermeasure
   Design* (IEEE INFOCOM 2026). This is the attack (FUsion) and the detector (SubID).
   [PDF](<Backdoor Injection in Federated Unlearning (INFOCOM 2026).pdf>)
2. *Verifiable Federated Unlearning with Zero-Knowledge Proofs*. This is the prevention mechanism:
   prove the procedure, bind the data, use a forget bitmask, and audit randomly.
   [PDF](<Verifiable Federated Unlearning with Zero-Knowledge Proofs explanation.pdf>)

**The full write-up is [`report.pdf`](report.pdf)**, a 10-page research-paper-style report with
graphs, tables, design rationale and measured timings.

---

## Results at a glance

**Part I: the attack and SubID** (MNIST CNN, 10 clients, 2 malicious, NoT unlearning)

| run | main accuracy (MA) | backdoor accuracy (BA) | outcome |
|---|---:|---:|---|
| FUsion, no defence | 0.975 | **0.986** | the backdoor is implanted during unlearning; MA is untouched |
| SubID detection | – | – | both attackers flagged, TPR 1.0 / FPR 0.0 / F1 1.0 |
| FUsion + replay oracle (cleartext stand-in for a ZK proof) | 0.972 | **0.005** | all 120 patched updates rejected |

**Part II: Groth16 prevention** (frozen CNN + proved linear head, 50 clients, 10 malicious,
20 FL + 20 FU rounds)

| setting | final BA | attack | attackers caught | proofs |
|---|---:|:-:|:-:|---:|
| no verification | **1.000** | succeeds | – | 0 |
| **every update proved** | **0.013** | blocked | 10/10, all in the first FU round | 130 (3 rounds) |
| random audit p = 5 % / 10 % / 20 %, with rollback | 0.034 / 0.019 / 0.013 | blocked | 7 / 9 / 10 of 10 | 97 / 186 / 363 |
| random audit p = 10 %, exclude only (no rollback) | **0.789** | succeeds | 9/10 | 186 |

| cost (Apple M3, 8 cores, CPU only) | value |
|---|---|
| circuit size | **290 129** R1CS constraints, 8 public inputs |
| Groth16 setup (once) | 16.0 s; proving key 117 MB, verifying key 1.5 KB |
| **proof generation per update** | **1.44 s ± 0.06** (peak RAM 0.93 GB) |
| **verification per proof** | **1.8 ms**; 0.12 s for 50 clients per round |
| proof size | 196 bytes |
| fixed-point vs. float accuracy | ≤ 0.34 points lower after FL, equal after FU |
| proving cost, 40 rounds × 50 clients | 52 min when every update is proved vs. 5.3 min with 10 % auditing |

**Three findings worth knowing:**
1. **Auditing only works with rollback.** A caught cheater's earlier, unaudited patches are already
   in the model, so the server must also revert the unlearning progress, not just exclude the
   client.
2. **FUsion needed only m = 6 patched rounds** in this setting, not the m = 40 that the ZK paper's
   auditing analysis assumes. This makes the probabilistic guarantee much weaker than the paper's
   table suggests (catch probability 0.47 vs. 0.985 at p = 10 %).
3. **A frozen, clean-pretrained feature extractor is itself a partial defence.** With pure CNN
   features, a head-only backdoor barely works (BA ≈ 0.08) even without proofs.

---

## Repository layout

```
btp/
├── README.md                  ← this file
├── report.pdf / report.tex    ← final research-paper report (IEEE two-column)
├── make_figures.py            ← draws every figure in figures/ from the raw results
├── figures/                   ← 8 PDF graphs used by the report
├── commands.md                ← command cheat-sheet for both parts
├── REPORT.md                  ← Part I write-up (markdown)
├── requirements.txt           ← pinned Python dependencies
│
├── fubd/                      ← PART I: FUsion backdoor + SubID (Python / PyTorch)
│   ├── config.py              ← every hyperparameter + presets (tiny / quick / paper)
│   ├── data.py, models.py     ← MNIST, IID / Dirichlet split, trigger, CNN
│   ├── client.py              ← client, forget mask, deterministic honest FL/FU rounds (NoT, GA, retrain)
│   ├── attack.py              ← FUsion: BC layers, Grad-CAM TS units, subnetwork, fusion; baseline; sign-flip
│   ├── subid.py               ← SubID detector + TPR / FPR / F1
│   ├── aggregators.py         ← FedAvg / trimmed mean / median
│   ├── zk_hooks.py            ← commitment, mask digest, Verifier interface, replay oracle
│   └── runner.py, __main__.py ← end-to-end experiment
├── tests/                     ← Part I tests (pytest)
├── runs/                      ← Part I results (JSON) + canonical log
│
└── zkfu/                      ← PART II: Groth16 prevention (Go + gnark, Python orchestration)
    ├── params/                ← all protocol constants in one file (D, C, B, N, FRAC, s, k)
    ├── native/                ← reference fixed-point trainer (bit-identical to the circuit)
    ├── commit/                ← Poseidon2 hash, value packing, Merkle tree, private dataset file
    ├── circuit/               ← ZkStepCircuit + gadgets (truncation, range checks, hashing)
    ├── protocol/              ← seed / indices / mask, public inputs, witness, key & proof I/O
    ├── cmd/                   ← CLIs: setup, commit, prove, verify, bench, hashcost
    ├── tests/                 ← Go tests: gadgets, hashing, 1000-step equivalence, soundness
    ├── sim/                   ← Python: frozen extractor, FL/FU simulation, auditing, FUsion attacker
    ├── results/               ← every Part II result (JSON) + summary.md
    ├── run_experiments.sh     ← reproduces every Part II number (~30 min)
    ├── ZK_REPORT.md           ← Part II write-up (markdown)
    └── DESIGN_NOTES.md        ← every design decision, the bound derivation, flagged deviations
```

---

## Requirements

| tool | version used | needed for |
|---|---|---|
| Python | 3.12 | everything in Python (PyTorch 2.15 has no wheels for 3.15) |
| Go | ≥ 1.24 (go.mod pins toolchain 1.26.8, downloaded automatically) | Part II |
| tectonic (or any LaTeX) | any | only to rebuild `report.pdf` |

The Go dependencies are pinned in `zkfu/go.mod`: gnark **v0.16.5** and the matching gnark-crypto.
The Python dependencies are pinned in `requirements.txt`.

## One-time setup

```bash
cd /Users/neerajkumar/Downloads/btp

# Python environment
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Go dependencies + binaries for Part II
cd zkfu
go mod download
for c in setup commit prove verify bench hashcost; do go build -o bin/$c ./cmd/$c; done

# Groth16 keys (~16 s, writes zkfu/keys/pk.bin and vk.bin)
./bin/setup -keys keys -breakdown -json results/setup.json
```

MNIST downloads automatically into `data/` on first use.

---

## Part I: reproduce FUsion and SubID (`fubd/`)

```bash
cd /Users/neerajkumar/Downloads/btp

# sanity check in seconds (random toy data)
.venv/bin/python -m fubd --preset tiny --attack fusion

# tests (9, ~6 s)
.venv/bin/python -m pytest tests/ -q

# the four Part I experiments (~1-2 min each)
.venv/bin/python -m fubd --preset quick --attack fusion   --verifier none   --name A_attack   # attack + SubID
.venv/bin/python -m fubd --preset quick --attack fusion   --verifier replay --name C_zk       # replay oracle blocks it
.venv/bin/python -m fubd --preset quick --attack baseline --verifier none   --name baseline   # naive attack fails
.venv/bin/python -m fubd --preset quick --attack none     --verifier none   --name noattack   # SubID false alarms
```

Each run prints a per-round MA/BA trace and writes `runs/<name>.json`. Any hyperparameter can be
overridden with `--<field> <value>`, for example `--n_malicious 4 --aggregator trim --iid false`.
The presets are `tiny` (seconds), `quick` (about 1–2 min, the reported setting) and `paper` (the
paper's 50-client / 500 + 200-round scale, slow). See `commands.md` §4 for the ablations behind
each figure of the paper.

Two modelling details matter here:
- **NoT negates the layer once, at the start of FU.** Negating it every round stalls learning.
- **All clients take part in FU.** This keeps attackers a minority, which SubID's median/MAD
  threshold needs.

---

## Part II: Groth16 prevention (`zkfu/`)

### Tests (milestones 1–3, ~1 min)

```bash
cd /Users/neerajkumar/Downloads/btp/zkfu
go test ./tests/ -count=1 -v
```

What the tests check:
- the truncation gadget against native truncation (random, zero, negative and boundary inputs);
- Poseidon2 and the Merkle tree, native against in-circuit;
- **1000 random training steps that are bit-identical between the native trainer and the
  circuit**;
- completeness of honest FL and FU steps;
- soundness: each of these must be rejected:
  - a weight patched by +1 unit;
  - the wrong starting model;
  - a fresh, modified or relabelled sample;
  - a sample at the wrong index;
  - a flipped mask bit;
  - public inputs swapped between clients or between rounds;
  - out-of-range or field-wrapped values.

Use `ZKFU_EQUIV_STEPS=100 go test ./tests/` for a faster equivalence check.

### Benchmarks

```bash
/usr/bin/time -l ./bin/bench -keys keys -runs 20 -json results/bench_random.json   # prove / verify timing
./bin/hashcost -json results/hashcost.json                                         # Poseidon2 vs MiMC vs SHA-256
```

### Simulation: 1 server, 50 clients, FL then FU

```bash
cd /Users/neerajkumar/Downloads/btp/zkfu/sim
PY=../../.venv/bin/python
$PY fl_sim.py --audit_p 0   --name noverify     # no verification: FUsion succeeds (~10 s)
$PY fl_sim.py --audit_p 0.1 --name audit10      # random auditing at 10 % (~5 min)
$PY fl_sim.py --audit_p 1.0 --resume_fl work/ckpt_fl_noverify.json --fu_rounds 3 --name full   # prove every update
$PY float_baseline.py                           # same head trained in float, for comparison
$PY summarize.py                                # tables -> ../results/summary.md
```

The first run pre-trains the frozen extractor (~2 s), extracts features and commits all 50
datasets (~7 s). All of this is cached in `zkfu/sim/work/`.

| flag | meaning |
|---|---|
| `--audit_p` | `0` = accept updates unverified; `0 < p < 1` = random auditing; `1` = prove every update |
| `--on_catch rollback\|exclude` | after catching a cheater in FU: revert to the FU-start model (default), or only exclude it |
| `--attack fusion\|none` | FUsion attacker on the malicious clients, or all honest |
| `--clients --malicious --benign_unlearn` | population (default 50 / 10 / 5) |
| `--fl_rounds --fu_rounds` | schedule (default 20 + 20) |
| `--features cnn+grid\|cnn` | extractor variant (`cnn` = ablation in which the head-only backdoor barely works) |
| `--beta --rho --lam --ft_steps` | FUsion parameters (defaults are the paper's 0.2 / 0.1 / 0.5) |
| `--resume_fl <ckpt>` | start FU from a saved FL-final model (makes `--audit_p 1` affordable) |

Each run writes `zkfu/results/<name>.json` with the MA/BA history, who was caught and when, and
every timing (native steps, proofs, verification, server round, wall clock).

---

## Reproduce everything (numbers, figures, PDF)

```bash
cd /Users/neerajkumar/Downloads/btp
(cd zkfu && ./run_experiments.sh)              # Part II: ~30 min (setup, benchmarks, 8 simulations)
(cd zkfu/sim && ../../.venv/bin/python time_pipeline.py && ../../.venv/bin/python summarize.py)
.venv/bin/python make_figures.py               # figures/*.pdf from the raw JSON
tectonic report.tex                            # -> report.pdf
```

The Part I numbers come from the four `fubd` runs above (`runs/*.json`, `runs/part1_canonical.log`).
`./run_experiments.sh quick` skips the two slowest Part II runs.

---

## How the prevention works (short version)

```
client                                                    server
  frozen CNN → features h → quantise (×2^16)
  Com(A) = Poseidon2 Merkle root over salted samples ───▶ store root_A, n_real
                                                     ◀─── global head (W, b), round r
  indices = H(H(W,b), r, id) → 32 samples
  native fixed-point SGD step → (W', b') ───────────────▶ range-check head
                                                          audit? (coin drawn AFTER arrival)
  if audited: rebuild witness, Groth16 proof (1.44 s) ──▶ recompute every public input, verify (1.8 ms)
                                                          reject + flag + roll back on failure
                                                          FedAvg over accepted heads
```

The circuit (`ZkStepCircuit`, 290k constraints) proves all of the following in one statement:
- **model binding:** H(W, b) equals what the server sent, and H(W′, b′) equals what came back;
- **batch binding:** the public indices are derived from the model, the round and the client;
- **data binding:** each of the 32 samples opens against Com(A) at exactly its index, and the same
  values are trained on;
- **mask binding:** a public forget bit multiplies each sample's gradient;
- **one exact step:** a fixed-point MSE step with every value range-checked.

A FUsion-patched update is not the output of this procedure, so no valid proof exists for it.

Why these choices (details in `report.pdf` §VI–VII and `zkfu/DESIGN_NOTES.md`):
- **Groth16:** constant 196-byte proofs and millisecond verification, which matters because the
  server checks up to 50 proofs every round. Its one-time per-circuit setup is fine because the
  bitmask design keeps one fixed circuit for both FL and FU.
- **Head-only proving:** proving the full CNN would take on the order of 10⁸ constraints.
- **MSE loss:** no exp, log or division in the circuit.
- **Poseidon2:** 372 constraints per hash, against 661 for MiMC and 68,411 for SHA-256.
- **Packing 9 range-checked values per field element:** about 2.5× fewer constraints.

**Deviations from the original specification** are all flagged in `zkfu/DESIGN_NOTES.md`:
- packed hash inputs;
- an extra in-circuit range check on W′ and b′ (needed for packing to stay sound);
- the value interval [−2²⁴, 2²⁴);
- an extractor with 16 region-intensity features, so the head-only attack works and the defence is
  tested against a real threat.

## Limitations

- **Setup:** the Groth16 setup is single-party (prototype only); a deployment needs an MPC
  ceremony.
- **What is proved:** aggregation and the frozen extractor are not proved.
- **Data poisoning:** poison committed into the dataset from the start is ordinary data poisoning
  and is not prevented.
- **Mask privacy:** the plain forget mask reveals which positions were forgotten.
- **Statistics:** results use one seed per configuration. Audit outcomes are random, so read them
  against the analytic probabilities in the report.

## Documents

| file | contents |
|---|---|
| [`report.pdf`](report.pdf) | the final report: both parts, design rationale, figures, all measurements |
| [`commands.md`](commands.md) | command reference for both parts |
| [`REPORT.md`](REPORT.md) | Part I write-up |
| [`zkfu/ZK_REPORT.md`](zkfu/ZK_REPORT.md) | Part II write-up |
| [`zkfu/DESIGN_NOTES.md`](zkfu/DESIGN_NOTES.md) | design decisions, the fixed-point bound derivation, flagged deviations |
| [`zkfu/results/summary.md`](zkfu/results/summary.md) | auto-generated Part II result tables |

## Troubleshooting

| problem | fix |
|---|---|
| `No module named torch` | use `.venv/bin/python` (Python 3.12), not the system Python |
| `open pk (run cmd/setup first)` | run `./bin/setup -keys keys` inside `zkfu/` |
| simulation says a dataset file is missing | delete `zkfu/sim/work/commit_*.json` to force a fresh commitment |
| changed `params.go` or the circuit | re-run `./bin/setup` (new keys) and `go test ./tests/` (the 1000-step equivalence must stay green) |
| proving feels slow | expected about 1.4 s per proof on 8 cores; lower `--audit_p` or use fewer rounds |
