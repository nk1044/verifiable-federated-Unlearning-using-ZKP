# FUsion Backdoor in Active Federated Unlearning — Reproduction Report

**Project:** reproduce the attack and defense of *"Malicious Forgetting: Backdoor Injection in
Active Federated Unlearning and Countermeasure Design"* (IEEE INFOCOM 2026), and expose the
integration points for *Verifiable Federated Unlearning with Zero-Knowledge Proofs*.

**Implementation:** `fubd/` (PyTorch, CPU). See [commands.md](commands.md) to reproduce every
number below. No official code exists for the paper; this is a from-scratch reimplementation.

---

## 1. Experimental setup

| | value |
|---|---|
| Dataset / model | MNIST, lightweight CNN (conv1-conv2-fc1-fc2, ~47k params) |
| Clients / server | 10 clients, 1 server, FedAvg aggregation |
| Malicious clients | 2 of 10 (20%) — a minority, as in the paper |
| FL training | 25 rounds, 15 local SGD steps/round (lr 0.01, momentum 0.9), IID |
| Active FU | 60 unlearning rounds, method = **NoT** (negate layer, fine-tune on retained data) |
| Trigger | 3×3 white square, bottom-right corner; target label 0 (paper's MNIST trigger) |
| Attack params | θ=0.6 (BC-layer threshold), ρ=0.10 (trigger-sensitive units), β=0.2 (fusion strength) |
| SubID params | k=5, T_r=5, τ=10, κ=7, φ=0.5 |

Reproduce with `--preset quick`. Each run takes ~1–2 min on a laptop CPU. The paper's full-scale
default (50 clients, 500+200 rounds) is available with `--preset paper`.

**Metrics.** `MA` = main-task accuracy on the clean test set. `BA` = backdoor accuracy (fraction of
triggered, non-target test inputs classified as the target label). A successful attack is `BA > 0.60`
(the paper's threshold). Detection is reported as TPR / FPR / F1 over the client set.

---

## 2. Headline results

| # | Experiment | Server verifier | Final MA | Final BA | Attack success | SubID TPR / FPR / F1 | Malicious updates rejected |
|---|------------|-----------------|:--------:|:--------:|:--------------:|:--------------------:|:--------------------------:|
| **A** | FUsion attack, naive server | none | **0.975** | **0.986** | ✅ **yes** | **1.00 / 0.00 / 1.00** | 0 |
| **C** | FUsion attack + ZKP hook | replay (ZK oracle) | 0.972 | **0.005** | ❌ no | 1.00 / 0.00 / 1.00 | **120** |
| **D** | Baseline attack (no subnetwork) | none | 0.971 | 0.012 | ❌ no | 1.00 / 0.00 / 1.00 | 0 |
| **E** | No attack (control) | none | 0.974 | 0.004 | ❌ no | — / **0.00 FPR** / — | 0 |

All four runs share the same seed and `quick` configuration; only the attack and verifier differ.
Raw JSON for each is in `runs/{A_attack,C_zk,baseline,noattack}.json`.

---

## 3. What each result shows

### A — The attack works, and it is fast and stealthy
During 25 rounds of honest FL the backdoor is dormant (BA ≈ 0.007). The malicious clients then
request unlearning and, each FU round, fuse their pre-built subnetwork into the unlearning update.
The backdoor is implanted within the limited unlearning window while the main task is untouched:

| FU round | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| MA | 0.951 | 0.960 | 0.966 | 0.971 | 0.974 | 0.975 |
| BA | 0.009 | 0.008 | 0.031 | **0.887** | 0.978 | **0.986** |

The backdoor crosses the 0.60 success bar between rounds 30 and 40 and reaches **BA = 0.986** by
round 60, with **MA = 0.975** (no main-task degradation). The prepared local backdoor model reached
BA = 0.998; the attack localized to a contiguous backdoor-critical block **conv2 (+ fc1)** with **9
trigger-sensitive units** total — i.e. the whole backdoor lives in a handful of channels. This
matches the paper's central claim: rapid, high-success, main-task-preserving injection during FU.

### C — The ZKP hook neutralizes the attack
With the `replay` verifier the server recomputes the fixed honest unlearning procedure on each
client's committed data + forget mask and accepts only a bit-identical update. A fused (patched)
update can never match, so **all 120 malicious updates (2 clients × 60 rounds) are rejected** and
**BA stays at 0.005** while honest unlearning proceeds normally (MA = 0.972). This is exactly the
statement a zero-knowledge proof would certify — demonstrated here without a concrete ZK backend.

### D — The naive baseline attack fails (as the paper reports)
Directly injecting backdoor data during unlearning, with no pre-built subnetwork, never takes hold
(**BA = 0.012**): the limited FU rounds plus NoT's retraining wash it out. This is the paper's
motivation for the subnetwork-fusion design — compare D (0.012) against A (0.986).

### E — SubID does not raise false alarms
With no attack, SubID flags **no clients (FPR = 0.00)**. (TPR is undefined/zero here because no
client is actually attacking, even though two are nominally designated — the meaningful quantity is
the zero false-positive rate.) This is the paper's Table III "w/o attack" behavior.

### SubID detection (A, C, D)
In every run that contains a real attack, SubID identifies **both** malicious clients with **TPR =
1.00, FPR = 0.00, F1 = 1.00**, matching the paper's "100% detection with no false positives." It
works by extracting the benign update subspace from the last training rounds (SVD), then, per
unlearning window, measuring each client's alignment with the coordinated trigger-direction
deviation subspace and thresholding with a median-absolute-deviation rule.

---

## 4. Two modeling points that mattered

Both were needed to make the attack *and* the defense behave as in the paper (recorded so they are
not lost):

1. **NoT negates the chosen layers once at the start of unlearning, then only fine-tunes.**
   Negating every round destabilizes training (BA stalls near 0.1). Negate-once gives the clean
   A-result above.
2. **All clients participate in FU aggregation every round** (`fu_benign_participate=True`,
   default), keeping malicious a 20% minority. SubID's median/MAD baseline assumes malicious are
   outliers against a benign majority; if only the few target clients participated, the malicious
   could become a majority of FU participants and MAD-based detection would return nothing. The
   attack itself succeeds either way.

---

## 5. Compatibility with the ZKP paper

The code is written so the undecided ZK system drops into one place without touching the FL/FU/attack
logic (details in [commands.md](commands.md) §5 and [fubd/zk_hooks.py](fubd/zk_hooks.py)):

- **Com(A)** — a Merkle commitment to each client's dataset is built before training (Step 2, "bind
  the data").
- **Forget bitmask b** — one bit per sample on `Client.mask` (Step 3, the zkUnlearner bitmask).
- **Fixed, deterministic procedure** — `Client.honest_fl_round` / `honest_fu_round` are pure
  functions of (global model, committed data, mask); this is precisely what a circuit must encode,
  and a unit test guards the property.
- **`Verifier` interface** — `NullVerifier` (attacked setting) and `ReplayVerifier` (the cleartext
  oracle used in experiment C) today; a `ZKVerifier.verify(msg)` checking `msg.proof` against the
  public inputs is the only new piece a real ZK backend (Groth16 / Halo2 / **EZKL**) needs.

**Bottom line.** Experiment C already proves, in the clear, that forcing clients to follow the exact
procedure reduces the FUsion backdoor from BA = 0.986 to BA = 0.005. Replacing the replay oracle
with a zero-knowledge proof preserves that guarantee while keeping client data private — the thesis
of the verifiable-FU paper.

---

## 6. Reproducibility

```bash
# the four runs in this report
.venv/bin/python -m fubd --preset quick --attack fusion   --verifier none   --name A_attack
.venv/bin/python -m fubd --preset quick --attack fusion   --verifier replay --name C_zk
.venv/bin/python -m fubd --preset quick --attack baseline --verifier none   --name baseline
.venv/bin/python -m fubd --preset quick --attack none     --verifier none   --name noattack

# correctness tests (9, ~6s)
.venv/bin/python -m pytest tests/ -q
```

Numbers in this report are from seed 0. Results are stable across the four runs (shared seed and
configuration); the `paper` preset scales to the full 50-client / 700-round setting.
