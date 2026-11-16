# DESIGN_NOTES — verifiable FL + FU with Groth16 (k = 1)

This file records every choice the spec left open, the bound derivation, and — marked
**⚠ FLAGGED** — every place where the implementation departs from the spec text or adds something
that touches soundness. Nothing that weakens soundness was changed; the flagged items are either
optimisations with an injectivity argument or *additional* checks.

Pinned versions: Go module `zkfu` with `github.com/consensys/gnark v0.16.5` and
`github.com/consensys/gnark-crypto v0.22.1-0.20261109214502-1b28c4e97311` (the commit gnark v0.16.5
requires), toolchain `go1.26.8` (go.mod); Python `numpy==2.5.5 torch==2.15.0 torchvision==0.29.2`
(`sim/requirements.txt`).

---

## 1. Stack and hash

* Circuit, commitment, native trainer, prover and verifier: Go + gnark, Groth16 over BN254.
* **Hash H = Poseidon2** (width 2, gnark's default BN254 parameters), used through
  `gnark-crypto/ecc/bn254/fr/poseidon2` natively and `gnark/std/permutation/poseidon2` in-circuit.
  H is Merkle–Damgård over the Poseidon2 compression with IV 0:
  `s = 0; for x in inputs: s = Compress(s, x)`. This is exactly `poseidon2.NewHasher()` natively and
  `std/hash/poseidon2.New(api)` in-circuit; `tests/hash_merkle_test.go` checks native H against
  both the library hasher and the circuit for every input length the protocol uses.
* MD here has no length padding. Every use has a **fixed length per role** (leaf: 10 inputs,
  node: 2, head: 73, seed: 3, index: 2), and the Merkle depth is fixed (11), so a collision across
  roles would need a Poseidon2 compression preimage.
* One proof = one SGD step. Since k = 1 and the mask is a public input, **C_FL and C_FU are the
  same circuit** (`ZkStepCircuit`): FL uses mask bits = 1 (except padding), FU uses the stored
  forget mask. One setup covers both phases, as in the ZK paper ("only the mask changes").

## 2. Model, features and learning-rate stability

* Proved head: `z = W h + b`, W ∈ Z^{10×64}, b ∈ Z^{10}, MSE against one-hot; constants D = 64,
  C = 10, B = 32, N = 2048 (depth 11), FRAC = 16, LR_SHIFT s = 8 (η/B = 2^-8), k = 1
  (`params/params.go`, the single config file used by circuit, native, commitment and server).
* Frozen extractor φ (Python, not proved), pre-trained once for 2 epochs on **public** MNIST data
  (the last 10 000 training images; clients get the other 50 000, 1 000 each, IID).
* Real-valued update: `ΔW = 2^-8 Σ_j m_j e_j h_jᵀ`, i.e. an effective learning rate of 0.125 on the
  batch-mean MSE gradient. Stability needs `0.125·λ_max(E[hhᵀ]) < 2`, so features are centred by the
  public mean and L2-normalised (‖h‖ ≈ 2.5–3). Then λ_max ≤ ‖h‖² ≈ 7–9 and class directions train
  in ≈ 10–20 rounds.
* **Two extractor variants** (`sim/extractor.py`):
  * `cnn` — 64 CNN features. **Finding:** the clean-pretrained CNN discards the paper's 3×3 corner
    trigger. Even a head trained on 10 % poisoned data reaches only BA ≈ 0.01 (corner 5×5: 0.013,
    8×8: 0.088). With this extractor, the frozen-feature design already defeats a head-only FUsion
    attack, so it is reported as an ablation (E2).
  * `cnn+grid` (default) — 48 CNN features + 16 fixed region intensities (mean of each 7×7 cell,
    centred, ×2). Coarse spatial information survives, so a head-only backdoor is possible
    (poisoned head: BA 0.82 at MA 0.947). This gives the attacker a fair chance, so the ZK defence is
    evaluated against an attack that actually works without it.
* Features are quantised by Python (`round(h·2^16)`, clipped). That is data preprocessing, not a
  weight update; the quantised integers are what the client commits to.

## 3. Fixed-point arithmetic and the bound derivation

Signed integers represent `value·2^16`; negative values are `p − |x|` in the field.

**Range of private values ⚠ FLAGGED (cosmetic).** The spec says `|v| < 2^24`. Power-of-two range
checks give the asymmetric interval `v ∈ [−2^24, 2^24)` (checked as `v + 2^24 ∈ [0, 2^25)`), so
`v = −2^24` is also allowed. The derivation below uses `|v| ≤ 2^24`, so all widths keep their
margin.

Truncation `trunc(x, t) = ⌊x / 2^t⌋` for `|x| < 2^(n−1)`: `x' = x + 2^(n−1) = q·2^t + r`, range-check
`r < 2^t` and `q < 2^(n−t)`, output `q − 2^(n−1−t)`. The equation holds over the integers because
`q·2^t + r < 2^n ≪ p`, so the hint outputs are unique (fully constrained), and an out-of-width `x`
has no witness. The native side (`native.Trunc`) performs the identical computation.

| quantity | bound | width n used |
|---|---|---|
| dot = Σ_d W·h (64 terms) | 64 · 2^24 · 2^24 = 2^54 | **n_fwd = 56** (`trunc(·,16)`, q: 40 bits) |
| z = trunc(dot,16) + b | 2^38 + 2^24 | — |
| e = z − y, y ∈ {0, 2^16} | < 2^39 | — |
| G_W = Σ_j m e h (32 terms) | 32 · 2^39 · 2^24 = 2^68 | **n_GW = 70** (`trunc(·,24)`, q: 46 bits) |
| G_b = Σ_j m e (32 terms) | 32 · 2^39 = 2^44 | **n_Gb = 46** (`trunc(·,8)`, q: 38 bits) |
| W' = W − trunc(G_W, 24) | < 2^45 | — |

All of these are far below p/2 ≈ 2^253, so the signed field representation is unambiguous. The
update divides by the constant B (via s), not by the number of ones in the mask, as specified.

**⚠ FLAGGED — extra in-circuit range check on W', b'.** W', b' are also range-checked to
`[−2^24, 2^24)`. The spec does not ask for this, but packing (§4) needs it for injectivity. It also
guarantees the next round's range check on W can pass. The cost to completeness is that an honest
step whose output leaves the range cannot be proved; the native trainer reports this as an error.
It never happened in any run (real weights are ≪ 256).

## 4. Commitment, packing and the head hash

**⚠ FLAGGED — packed hash inputs (optimisation, soundness argued).** The spec defines
`leaf_i = H(salt_i, h_i[0..63], y_i)`, which is 66 absorbs. Instead, each range-checked value is
mapped to `u = v + 2^24 ∈ [0, 2^25)` and 9 values are packed per field element (225 < 253 bits), so
`leaf_i = H(salt_i, P_0, …, P_7, y_i)` takes 10 absorbs, and `H(W, b)` takes 73 absorbs instead of
650. Packing is linear (free in R1CS). It is injective because every packed value is range-checked
in-circuit (features, W, b, and W', b' via the extra check in §3), and the server only hashes heads
it has range-checked itself. This cuts the circuit from about 0.75 M to **290 129 constraints**,
roughly 2.5× faster proving. Revert to unpacked hashing only if exact spec conformance matters more
than speed.

* Merkle tree: binary, N = 2048 leaves, `node = H(left, right)`, as the spec says. The direction bit
  at level ℓ is bit ℓ (LSB first) of the public index, taken from the in-circuit `ToBinary` of
  `idx_packed`, so the sample must sit at exactly that position. The same `h_j, y_j` wires feed the
  leaf hash and the training step.
* Padding positions are `[n_real, N)`, with zero features, label 0 and a fresh salt. The client
  publishes `n_real` together with `root_A`, and the server forces those mask bits to 0 forever.
* Salts are random field elements (`fr.Element.SetRandom`, crypto/rand) kept in the client's
  private dataset file.

## 5. Public inputs, seed, indices and mask

* Public inputs (8): `root_A, h_w, h_w_new, idx_packed[2] (16 × 11 bits each), mask_packed (32 bits),
  round, client_id`.
* `round` and `client_id` are bound by the constraint `(round + 1)·(client_id + 1) = t`. A public
  input used by no constraint would have a zero Groth16 verification-key term and could be changed
  freely. The soundness test checks that changing only `round`, or only `client_id`, is rejected.
* `seed = H(h_w, r, client_id)`; `idx_j` = low 11 bits of the canonical integer of `H(seed, j)`.
  **Duplicates are allowed** (sampling with replacement); a duplicated sample simply counts twice.
  Because N = 2^11, there is no modulo bias beyond the negligible bias from p not being a power of
  two (≈ 2^11/p).
* Mask bit j = 1 iff `idx_j < n_real` and `idx_j` has not been forgotten.
* Unlearning request: the client sends its cleared positions in plain. The server checks that every
  position is in `[0, N)`, that at least one is a non-padding position, and that bits are only
  cleared, never restored. It then stores the mask permanently.

## 6. Server, aggregation and auditing

* The server recomputes `h_w`, seed, indices, `mask_packed` and `h_w_new` (from the head the client
  returned) itself. Clients never supply public inputs. A received head with any value out of range
  is rejected before hashing. Only the head is ever sent, so the frozen extractor cannot be changed.
* Aggregation (not proved): FedAvg over accepted heads with exact integer arithmetic and
  deterministic re-quantisation `q = ⌊(Σ + ⌊n/2⌋)/n⌋` (round half up), clipped to the value range.
* **Random auditing** (ZK paper §"Random auditing", milestone 6):
  * Each update is audited independently with probability p. The coin is drawn by the server
    **after** the update arrives, from a server-only RNG, so clients cannot predict audited rounds.
  * Clients keep each round's inputs (global head, round, mask). Because the procedure is
    deterministic, the client can rebuild the exact witness and prove later.
  * A failed or missing proof flags the client; it is excluded from that round and every later one.
  * On a catch during FU, `on_catch=rollback` (default) reverts the global head to the FU-start
    checkpoint, because un-audited patched updates may already be in it (the same "discard the
    unlearning result" response SubID uses). `on_catch=exclude` keeps the model and only removes the
    client; it is reported for comparison.
* p = 0 is the unverified baseline that FUsion attacks; p = 1 proves every update.

## 7. What Python is allowed to do

Python never computes an honest weight update: every honest step comes from the Go native trainer,
which is bit-identical to the circuit (1000-step test). Python does extract and quantise features,
aggregates heads (not proved), evaluates accuracy, and runs the **attacker** (`sim/fusion_head.py`).
The attacker's patched update is, by construction, not the output of the fixed procedure; that is
exactly what the proof system must reject.

## 8. Trusted setup and gnark specifics

* `groth16.Setup` is a single-party setup, acceptable for a prototype. A deployment needs an MPC
  ceremony (powers of tau plus a circuit-specific phase 2), because whoever holds the toxic waste
  can forge proofs.
* gnark's `rangecheck` uses log-derivative lookups backed by a BSB22 (Pedersen) commitment inside
  Groth16. The proof therefore carries one extra commitment and proof of knowledge, which is why it
  is 196 bytes rather than 128.

## 9. Known limitations (same as the ZK paper, plus two observed here)

1. The proof checks the procedure, not the content of the data. Poisoned samples committed from the
   start are ordinary data poisoning (trained on openly in FL through random batches).
2. The plain mask reveals which positions are forgotten (not their content). A client may also
   choose *which* of its samples to forget, which a data-poisoning client could use to amplify
   committed poison.
3. Auditing is probabilistic. Under it, un-audited patched updates are aggregated until the first
   catch, which is why the rollback policy matters (results E/A10 vs A10x).
4. **Observed:** in our head-only setting FUsion needs only m ≈ 6 patched FU rounds to pass
   BA 0.6, not the m ≈ 40 the ZK paper assumes, so `P(caught) = 1 − (1 − p)^m` per attacker is much
   lower than the paper's table (see ZK_REPORT.md §6).
5. **Observed:** with a clean-pretrained CNN extractor, head-only FUsion barely works at all
   (ablation E2). How much the ZK defence adds therefore depends on how much backdoor capacity the
   proved part of the model has.
6. Aggregation and the extractor are not proved; the server is trusted for aggregation.
