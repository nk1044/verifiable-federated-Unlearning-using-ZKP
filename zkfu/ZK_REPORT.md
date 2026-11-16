# Verifiable Federated Unlearning with Groth16 Proofs vs. the FUsion Backdoor: Prototype Report

**Scope.** This prototype implements the design of *Verifiable Federated Unlearning with
Zero-Knowledge Proofs*. Every client update in federated learning (FL) and federated unlearning
(FU) is one fixed-point SGD step, proved with Groth16, starting from the server's global model, on
a server-chosen batch from the client's pre-committed dataset, with the forget mask applied. The
prototype also implements the paper's **random auditing** scheme, and is evaluated against the
FUsion backdoor from *Malicious Forgetting* (INFOCOM 2026), the attack reproduced in Part 1
(`../REPORT.md`).

All numbers come from `results/*.json`, produced by `./run_experiments.sh` and tabulated by
`sim/summarize.py` (`results/summary.md`). Hardware: Apple M3, 8 cores, 16 GB RAM, CPU only.
Design decisions and every deviation from the spec are in [DESIGN_NOTES.md](DESIGN_NOTES.md).

---

## 1. Summary

| question | answer |
|---|---|
| Does FUsion work when updates are not verified? | **Yes.** BA 0.011 → **1.000** in 20 FU rounds, MA unchanged (0.934 → 0.938). |
| Does proving every update stop it? | **Yes, deterministically.** All 10 attackers were rejected in the first FU round; BA stayed **0.013**; no honest client was rejected. |
| Does random auditing stop it? | **Yes, if a catch triggers a rollback.** BA ended at 0.034 / 0.019 / 0.013 for p = 5 / 10 / 20 %. **With exclusion only, the attack still succeeds** (BA 0.789 at p = 10 %). |
| Proving time per update | **1.44 s ± 0.06** (290 129 constraints, 8 cores); the paper estimated 2–10 s. |
| Verification time | **1.8 ms**; 50 clients take **0.12 s per round** on the server. The paper estimated "under 10 s". |
| Proof size / keys | **196 B** proof; 117 MB proving key; 1.5 KB verifying key; 16 s setup. |
| Total cost, 40 rounds × 50 clients | Every update: **2 000 proofs, 52 min of proving**. Auditing at p = 10 %: **186 proofs, 5.3 min** (−90 %). |
| Cost of fixed-point arithmetic | Accuracy loss **≤ 0.34 points** vs. float training of the same head (≈ 0 after FU). |
| Main caveat for the ZK paper | FUsion needed only **m = 6** patched rounds here, not the paper's assumed m = 40, so the paper's auditing probabilities are optimistic (§6). |

---

## 2. System

```
            client i (Go CLIs, private data)                          server (Go CLI + Python orchestration)
 ┌──────────────────────────────────────────────┐   once   ┌──────────────────────────────────────────────┐
 │ frozen CNN φ (not proved) → quantised h_i    │ ───────▶ │ store root_A,i , n_real,i                    │
 │ cmd/commit: Merkle tree over salted leaves    │ root_A  │                                              │
 ├──────────────────────────────────────────────┤          ├──────────────────────────────────────────────┤
 │ each round r:                                 │ ◀─────── │ send global head (W, b)                      │
 │  seed = H(H(W,b), r, i) → 32 indices          │          │                                              │
 │  native fixed-point SGD step  → (W', b')      │ ───────▶ │ range-check head; decide audit AFTER arrival │
 │  if audited: cmd/prove → Groth16 proof (196B) │ ───────▶ │ cmd/verify: recompute ALL public inputs,     │
 │                                               │          │   verify; reject + flag + rollback on fail   │
 │                                               │          │ FedAvg over accepted heads (deterministic)   │
 └──────────────────────────────────────────────┘          └──────────────────────────────────────────────┘
```

| component | implementation |
|---|---|
| proof system | Groth16 over BN254, gnark **v0.16.5** (single-party setup; a deployment needs an MPC ceremony) |
| hash H | **Poseidon2** (width 2) in a Merkle–Damgård chain; the same gnark package natively and in-circuit |
| frozen extractor φ (not proved) | 2 conv + pool + linear; 48 CNN features + 16 region intensities (D = 64); pre-trained once on 10 k **public** MNIST images |
| proved model | linear head z = W·h + b, W ∈ Z^{10×64}, b ∈ Z^{10}, MSE vs. one-hot (no softmax) |
| fixed point | FRAC = 16 fractional bits; all values in [−2^24, 2^24); η/B = 2^-8; batch B = 32; k = 1 step per round |
| commitment | Merkle tree, N = 2048 padded leaves (depth 11), leaf = H(salt, packed h, y) |
| FL / FU | the same circuit (C_FL = C_FU); FU differs only in the public mask bits |
| population | 50 clients × 1 000 IID samples; 10 malicious (20 %); 5 benign clients also request unlearning (2 % of data each) |
| schedule | 20 FL rounds, then 20 FU rounds (defaults chosen so every run finishes in minutes) |

Repository: `circuit/` (gadgets, ZkStepCircuit) · `native/` (reference trainer) · `commit/` (H, packing,
Merkle, dataset) · `protocol/` (seed/indices/mask, public inputs, key and proof I/O) ·
`cmd/{setup,commit,prove,verify,bench}` · `sim/` (Python FL/FU, attacker, auditing) · `tests/`.

---

## 3. Circuit design (`ZkStepCircuit`)

**Public inputs (8):** root_A, h_w = H(W,b), h_w_new = H(W′,b′), idx_packed[2] (32 × 11-bit indices),
mask_packed (32 bits), round, client_id.
**Private witness:** W, b; for each of the 32 batch slots: salt, h_j[64], y_j, Merkle path[11].

Constraints, in order:
1. Unpack indices and mask bits (`ToBinary`, so they are boolean by construction). The 11 bits of
   idx_j are the Merkle direction bits, so a sample must sit exactly at the public position.
2. Range-check W, b, and assert H(packed W, b) = h_w.
3. For each sample: range-check its 64 features, build the one-hot label (asserting y ∈ [0, 9]),
   hash the leaf, and walk 11 Merkle levels to root_A.
4. Forward pass, `z = trunc(Σ W·h, 16) + b`; error, `e = z − y_onehot·2^16`; masked error `m_j·e`.
5. Gradients `G_W = Σ m e h`, `G_b = Σ m e`; update `W′ = W − trunc(G_W, 24)`, `b′ = b − trunc(G_b, 8)`.
6. Range-check W′, b′, and assert H(packed W′, b′) = h_w_new.
7. Bind round and client_id with one multiplication constraint, so they cannot be swapped.

Truncation gadget: `x + 2^(n−1) = q·2^t + r` with range-checked `r < 2^t` and `q < 2^(n−t)`. The
widths n are 56, 70 and 46 bits, from the worst-case bound derivation in DESIGN_NOTES §3. The
hint outputs are therefore fully constrained.

### Constraint count

| item | value |
|---|---|
| **constraints (R1CS)** | **290 129** |
| public inputs | 8 |
| compile time | 0.34 s |
| Groth16 setup time | 16.0 s (peak RAM 0.6 GB) |
| proving key / verifying key | 117.1 MB / 1 488 B |

| component (marginal cost) | constraints | share |
|---|---:|---:|
| 32 samples: range checks, one-hot, leaf hash, 11-level Merkle path | 204 512 | 70.5 % |
| forward + gradient multiplications (2 × 20 480 + 320) | 41 280 | 14.2 % |
| 2 head hashes with range checks: H(W, b) and H(W′, b′) | 33 668 | 11.6 % |
| 970 truncations | 21 960 | 7.6 % |
| *range-check lookup amortisation in the full circuit* | −11 291 | |

One Poseidon2 compression costs 186 constraints, and one sample costs 6 391. The Merkle paths
(704 compressions) dominate. **Speed-up applied:** range-checked 25-bit values are packed 9 per field
element before hashing (DESIGN_NOTES §4, flagged). This cuts the leaf hash from 66 to 10 absorbs and
the head hash from 650 to 73. Without packing the circuit would be roughly 0.75 M constraints and
proving about 2.5× slower. Further optional speed-up: a single-compression Merkle node would save
about 65 k constraints (22 %), but the spec asks for node = H(l, r), so it is not applied.

---

## 4. Proof generation and verification time

20 consecutive real training steps on one client (alternating FL and FU masks), `cmd/bench`:

| metric | random-feature client | **real MNIST client** |
|---|---:|---:|
| proving time per step | 1.424 s ± 0.042 (1.35–1.53) | **1.439 s ± 0.061** (1.38–1.65) |
| verification time | 1.77 ms ± 0.06 | **1.83 ms ± 0.24** |
| proof size | 196 B | 196 B |
| native fixed-point step (no proof) | 1.32 ms | 1.48 ms |
| Merkle tree rebuild (2 048 leaves) | 110 ms | 109 ms |
| witness assembly | 0.37 ms | 0.37 ms |
| proving-key load (raw format) | 0.10 s | 0.10 s |
| peak prover memory (max RSS, `/usr/bin/time -l`) | 0.93 GB | 0.92 GB |

The client's cost per proved round is therefore about **1.55 s** (tree + proof); proving is about
1 000× the cost of the plain step. On the server side, the p = 1 run measured **0.116 s per round**
to recompute the public inputs and verify all 50 proofs (about 2.3 ms per client including
Poseidon2 recomputation).

### Against the ZK paper's estimates

| item | paper's estimate / measurement | this prototype |
|---|---|---|
| proof generation, MNIST head-only, per step | about 2–10 s (estimate) | **1.44 s** |
| verification | 0.2 s (zkUnlearner) | **0.0018 s** |
| proof size | 192 B | 196 B (+4 B from gnark's range-check commitment) |
| server time, 50 clients, per round | under about 10 s (estimate) | **0.12 s** |
| total proving per client, 500 FL + 200 FU rounds | about 1–2 h (estimate) | **≈ 18 min** (700 × 1.55 s) |

---

## 5. Attack vs. defence (50 clients, 10 malicious, 20 FL + 20 FU rounds)

FUsion attacker (`sim/fusion_head.py`), adapted to the head-only model:
- The head is the only trainable layer, so it is the backdoor-critical layer.
- Grad-CAM selects ρ = 10 % of input units (6 dims).
- The subnetwork is trained on backdoor data with the Eq. (10) clean regulariser.
- In each FU round the attacker takes the honest Go step, fuses β·S_bd (β = 0.2), fine-tunes on
  clean data plus λ = 0.5 × backdoor data, and submits.

The local backdoor model and the subnetwork both reached BA = 1.0 during preparation. The trigger is
the paper's 3×3 white square, with target label 0.

| run | setting | FL-end MA | final MA | **final BA** | attack succeeds | caught | false positives | rollbacks | proofs | proving time (s) | server verify (s) | wall (s) |
|---|---|---:|---:|---:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|
| E1 | no verification, no attack | 0.934 | 0.945 | 0.011 | – | – | 0 | 0 | 0 | 0 | 0 | 7 |
| **E0** | **no verification, FUsion** | 0.934 | 0.938 | **1.000** | **yes** | 0/10 | 0 | 0 | 0 | 0 | 0 | 13 |
| **F** | **every update proved (p = 1)**¹ | 0.934 | 0.937 | **0.013** | no | **10/10 (round 1)** | 0 | 1 | 130 | 203 | 0.35 | 207 |
| A05 | auditing p = 5 %, rollback | 0.934 | 0.937 | 0.034 | no | 7/10 | 0 | 4 | 97 | 155 | 0.27 | 181 |
| A10 | auditing p = 10 %, rollback | 0.934 | 0.938 | 0.019 | no | 9/10 | 0 | 6 | 186 | 318 | 0.54 | 347 |
| A20 | auditing p = 20 %, rollback | 0.934 | 0.941 | 0.013 | no | 10/10 | 0 | 5 | 363 | 619 | 1.05 | 648 |
| A10x | auditing p = 10 %, **exclude only** | 0.934 | 0.942 | **0.789** | **yes** | 9/10 | 0 | 0 | 186 | 308 | 0.55 | 336 |
| E2 | ablation: pure-CNN extractor, no verification | 0.945 | 0.892 | 0.077 | no | – | 0 | 0 | 0 | 0 | 0 | 20 |

¹ F resumes from E0's FL-final model and runs 3 FU rounds with all 50 clients proving. This is
enough to show the deterministic outcome and to measure per-round cost. The full 40-round p = 1
cost is extrapolated in §7. "Proving time" is the sum of the per-proof times (each proof uses all
8 cores).

**FUsion without verification (E0), per FU round:**

| FU round | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MA | 0.934 | 0.932 | 0.933 | 0.934 | 0.934 | 0.936 | 0.937 | 0.936 | 0.937 | 0.938 |
| BA | 0.091 | 0.337 | **0.657** | 0.915 | 0.972 | 0.992 | 0.998 | 0.999 | 1.000 | 1.000 |

The attack is fast and stealthy: it needs **m = 6 patched rounds** to pass BA 0.6, and MA never
drops.

**Under auditing with rollback (BA by FU round):**

| FU round | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | catches (FU round: clients) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A05 | 0.03 | 0.10 | **0.26** | 0.02 | 0.01 | 0.03 | 0.07 | 0.12 | 0.02 | 0.03 | 1: {1,2,3} · 7: {8,9} · 10: {5} · 17: {7} |
| A10 | 0.03 | 0.03 | 0.08 | **0.19** | 0.01 | 0.04 | 0.01 | 0.02 | 0.02 | 0.02 | 1: {1,2,3} · 3: {9} · 10: {7} · 14: {8} · 15: {5,6} · 17: {4} |
| A20 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | all 10 caught by round 7 |
| A10x | 0.06 | 0.15 | 0.30 | 0.48 | **0.63** | 0.74 | 0.84 | 0.85 | 0.81 | 0.79 | same catches as A10, no rollback |

### What these runs show

1. **Proving every update gives the deterministic guarantee the ZK paper claims.** A patched head
   is not the output of the fixed procedure, so its proof cannot verify. The soundness tests show
   this holds for a change of a single fixed-point unit. In run F, all 10 attackers were rejected
   the first time they patched, BA never moved, and no honest client was ever rejected across 130
   proofs.
2. **Auditing works only together with a rollback.** Before its first catch, an attacker's
   unaudited patched updates have already been aggregated. A10 and A10x used *identical* audit
   draws: excluding the caught clients alone left their backdoor in the model, and the uncaught
   attacker (client 0) pushed BA to 0.79. Rolling back to the FU-start model on each catch (the
   same "discard the unlearning result" response SubID uses) kept final BA at baseline for every p.
3. **Auditing leaves windows of exposure.** Between catches the global model was partially
   backdoored: peak BA was 0.26 (p = 5 %) and 0.19 (p = 10 %). At p = 5 % and 10 %, 3 and 1
   attackers respectively were never caught. They failed here only because each rollback reset
   their progress and, being few, they needed many uninterrupted rounds. A longer FU phase gives
   them more chances.
4. **Fixed point costs essentially nothing.** Float training of the same head under the same
   schedule reached 0.9379 / 0.9452 MA (after FL / after FU); the proved fixed-point head reached
   0.9345 / 0.9454 (−0.34 / +0.02 points).
5. **The ZK paper's own design choice already limits FUsion (E2).** With a purely clean-pretrained
   frozen CNN as the extractor, even a head trained on 10 % poisoned data reaches only BA ≈ 0.01.
   In E2 the FUsion attacker reached just 0.077 while costing 5 points of MA. The main experiments
   therefore use an extractor that keeps coarse spatial features (48 CNN + 16 region intensities),
   giving the attacker a working attack against which to test the proof system. How much the ZK
   layer adds in practice depends on how much backdoor capacity the proved part of the model has.

---

## 6. Random auditing against the paper's analysis

The paper gives `P(caught) = 1 − (1 − p)^m` with m = 40 patched rounds (read from the FUsion
paper's plot). In our head-only setting FUsion needs only **m = 6**:

| p | P(one attacker caught within m = 6) | paper's figure (m = 40) | P(one attacker caught in 20 FU rounds) | P(all 10 caught in 20 rounds) | proving saved |
|---|---:|---:|---:|---:|---:|
| 5 % | **0.265** | 0.871 | 0.642 | 0.012 | 95 % |
| 10 % | **0.469** | 0.985 | 0.878 | 0.274 | 90 % |
| 20 % | **0.738** | 1.000 | 0.988 | 0.891 | 80 % |
| 60 % | 0.996 | 1.000 | 1.000 | 1.000 | 40 % |

Measured catches (7/10, 9/10 and 10/10 for p = 5 / 10 / 20 %) agree with the per-attacker column
for 20 rounds. **Recommendation for the ZK paper:**
- State the guarantee in terms of the measured m and the number of colluding attackers.
- Make rollback-on-catch part of the protocol.
- Use p ≥ 20 % when m is small, or prove every FU round (FU is short: only 20 of 40 rounds here,
  and 200 of 700 in the paper's default) while auditing FL rounds at p = 5 %. This hybrid would give
  the deterministic guarantee where FUsion operates, at about 52 % of the full proving cost in our
  schedule (≈ 32 % in the paper's 500 + 200 schedule).

---

## 7. Cost: full verification vs. auditing (whole 40-round schedule, 50 clients)

| mode | proofs | total proving time | per client (clients prove in parallel) | server verification |
|---|---:|---:|---:|---:|
| every update (p = 1), extrapolated from F | 2 000 | 51.6 min | 62 s | 4.6 s total (0.12 s per round) |
| auditing p = 5 % (measured) | 97 | 2.6 min | 3 s | 0.27 s |
| auditing p = 10 % (measured) | 186 | 5.3 min | 6 s | 0.54 s |
| auditing p = 20 % (measured) | 363 | 10.3 min | 12 s | 1.05 s |

In this simulation one laptop proves for all 50 clients one after another, so wall time is the
total proving time. In a real deployment each client proves only its own updates, in parallel; the
"per client" column is that client's cost.

---

## 8. Comparison with Part 1 (the original FUsion reproduction)

| defence | kind | data private? | outcome against FUsion | cost |
|---|---|:-:|---|---|
| none (Part 1, full CNN) | – | – | BA 0.986, MA 0.975 | – |
| SubID (Part 1) | statistical detection after the fact | yes | 2/2 attackers flagged, F1 = 1.0; requires discarding the unlearning result | negligible (SVD per window) |
| replay oracle (Part 1) | recompute the procedure in the clear | **no** | BA 0.005, all malicious updates rejected | one extra local step per update; server sees the data |
| none (Part 2, head-only) | – | – | **BA 1.000**, MA 0.938 | – |
| **Groth16, every update (Part 2)** | cryptographic, deterministic | **yes** | **BA 0.013**, 10/10 attackers rejected at first patch | 1.44 s per client-round; 0.12 s per server-round |
| **Groth16 + auditing p = 10 %, rollback (Part 2)** | cryptographic, probabilistic | **yes** | BA 0.019 final (peak 0.19), 9/10 caught | −90 % proving |

The Groth16 system achieves the replay oracle's outcome (Part 1's stand-in for a ZK proof) while
keeping client data private, as the ZK paper argues. Unlike SubID, its guarantee does not depend
on attackers being a statistical minority. Caveat: Part 1 used a fully trained CNN with 10 clients;
Part 2 proves only a linear head with 50 clients, so the BA figures are not from the same model.

---

## 9. Correctness evidence (`go test ./tests/`, all pass, about 1 min)

| milestone | test | result |
|---|---|---|
| 1 | truncation gadget vs. native vs. floor division: random, 0, ±1, ±2^t ± 1, both width extremes; off-by-one and out-of-width inputs rejected | pass (3 widths) |
| 2 | Poseidon2: native vs. gnark-crypto hasher vs. circuit (input lengths 1, 2, 3, 10, 73); Merkle native vs. circuit; wrong index, sibling, feature, label or salt rejected | pass |
| 3 | **1000 random steps: native == circuit bit for bit** (4 feature scales, 4 weight scales, FL and FU masks) | **1000/1000** |
| 3 | completeness: honest FL step and honest FU step verify | pass |
| 3 | soundness: one weight +1 (backdoor patch); wrong starting model; fresh sample; modified feature; modified label; correct sample at wrong index; mask bit 0 → 1; proofs swapped between clients and between rounds; only round changed; only client_id changed; out-of-range feature (with in-range control), out-of-range weight, field-wrapped feature, label = 10 | all rejected |

---

## 10. Limitations

- Single-party Groth16 setup (prototype only). Aggregation and the extractor are not proved; the
  server is trusted for aggregation.
- The proof binds the procedure and the data, not the data's content. Poison committed from the
  start is ordinary data poisoning (the ZK paper's own limitation). The plain forget mask reveals
  which positions were forgotten, and a client chooses which of its samples to forget.
- Results are one seed per configuration. Audit outcomes are random; §6 gives the analytic
  probabilities that the single runs should be read against.
- Deviations from the spec, all flagged in DESIGN_NOTES.md:
  - packed hash inputs;
  - an extra range check on W′, b′;
  - the value interval [−2^24, 2^24);
  - a feature extractor with region-intensity features (chosen so the attack is possible).

## 11. Reproduce

```bash
cd /Users/neerajkumar/Downloads/btp/zkfu
./run_experiments.sh                 # ~30 min: setup, benchmarks, E0–E2, F, A05/A10/A20/A10x
../.venv/bin/python sim/summarize.py # tables -> results/summary.md
go test ./tests/ -count=1            # milestones 1–3
```

Individual runs and flags are listed in `../commands.md`, Part 2.
