# Measured Statistics: Proof Generation, Verification and Costs

All numbers were measured on one **Apple M3 laptop (8 cores, 16 GB RAM, CPU only)**.
Raw data is in `zkfu/results/`: `bench_real.json`, `setup.json`, `pipeline_timing.json`,
`hashcost.json`, and the simulation JSONs. In the report (`report.pdf`) they appear in Tables
VI–X and Fig. 6.

---

## Client side

| What | Time |
|---|---:|
| **Proof generation per update** | **1.44 s** (± 0.06 s, range 1.38–1.65 s over 20 runs) |
| Rebuilding the Merkle tree before proving | 109 ms |
| Assembling the witness (proof inputs) | 0.37 ms |
| Normal training step without a proof | 1.48 ms |
| Loading the proving key (117 MB) | 0.10 s |
| Committing its dataset (once, at the start) | 112 ms |
| Peak memory while proving | 0.93 GB |

So a client that has to prove spends about **1.55 s per round**, roughly 1,000× a plain training
step.

## Server side

| What | Time |
|---|---:|
| **Verifying one proof** | **1.8 ms** |
| Verifying all 50 clients in a round | 0.12 s |
| Full server round (50 verifications, file I/O, FedAvg) | 0.33 s |

## One-time setup

| What | Value |
|---|---:|
| Pre-training the frozen CNN (10k public images) | 2.1 s |
| Extracting features (50k images) | 1.25 s |
| Compiling the circuit | 0.34 s |
| **Groth16 setup** | **16 s** (0.6 GB RAM) |
| Committing all 50 clients' datasets | 7.3 s |

## Proof and key sizes

| What | Size |
|---|---:|
| **Proof** | **196 bytes** |
| Proving key | 117 MB |
| Verifying key | 1.5 KB |
| Circuit | 290,129 constraints, 8 public inputs |

## Whole run (50 clients, 40 rounds)

| Mode | Proofs | Total proving time | Per client |
|---|---:|---:|---:|
| Prove every update | 2,000 | 52 min | 62 s |
| Audit 20 % | 363 | 10.3 min | 12 s |
| Audit 10 % | 186 | 5.3 min | 6 s |
| Audit 5 % | 97 | 2.6 min | 3 s |

"Per client" is what each client actually pays, since in a real system clients prove in parallel
on their own machines.

## Compared with the ZK paper's estimates

| Item | Paper's estimate | Measured |
|---|---|---|
| Proof time per step | 2–10 s | **1.44 s** |
| Verification | 0.2 s | **1.8 ms** |
| Proof size | 192 B | 196 B |
| Server time per round (50 clients) | under 10 s | **0.12 s** |
| Proving per client for 500 + 200 rounds | 1–2 hours | **about 18 min** |

## Accuracy cost of the integer maths

| | After training | After unlearning |
|---|---:|---:|
| Normal float training | 0.9379 | 0.9452 |
| Proved fixed-point training | 0.9345 | 0.9454 |

That is at most a 0.34-point drop, which disappears after unlearning.

## Hash cost (why Poseidon2)

| Hash | Constraints per hash | Whole circuit if used |
|---|---:|---:|
| **Poseidon2** | **372** | **0.29 M** |
| MiMC | 661 | 0.46 M |
| SHA-256 | 68,411 | about 40 M |

---

## Auditing results: attacker detection

Setup: 50 clients (10 malicious FUsion attackers, 40 honest), 20 training + 20 unlearning rounds.
An attacker counts as **detected** when one of its updates is audited and its proof fails; the
server then flags it and excludes it from every later round.

### Detection accuracy

| Mode | Detected (TP) | Missed (FN) | False alarms (FP) | Detection rate (TPR) | False-alarm rate (FPR) | Precision | F1 | Accuracy (50 clients) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Prove every update (p = 100 %)** | **10** | 0 | 0 | **1.00** | 0.00 | 1.00 | **1.00** | **1.00** |
| Audit 20 % | 10 | 0 | 0 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| Audit 10 % | 9 | 1 | 0 | 0.90 | 0.00 | 1.00 | 0.95 | 0.98 |
| Audit 5 % | 7 | 3 | 0 | 0.70 | 0.00 | 1.00 | 0.82 | 0.94 |
| No verification | 0 | 10 | 0 | 0.00 | 0.00 | – | – | – |

**Zero false positives in every run:** no honest client was ever flagged, and every honest proof
verified. Precision is always 1.0, because a failed proof is cryptographic evidence of cheating,
not a statistical guess.

### When attackers were caught, and what reached the model

| Mode | First catch (FU round) | Last catch | Never caught | Rollbacks | Peak BA during FU | Final BA | Attack succeeded? | Proofs |
|---|---:|---:|---|---:|---:|---:|:-:|---:|
| **Prove every update** | 1 | 1 (all 10 at once) | none | 1 | 0.014 | **0.013** | no | 130 (3 FU rounds) |
| Audit 20 %, rollback | 1 | 7 | none | 5 | 0.014 | **0.013** | no | 363 |
| Audit 10 %, rollback | 1 | 17 | client 0 | 6 | 0.186 | **0.019** | no | 186 |
| Audit 5 %, rollback | 1 | 17 | clients 0, 4, 6 | 4 | 0.259 | **0.034** | no | 97 |
| Audit 10 %, **exclude only** | 1 | 17 | client 0 | 0 | 0.847 | **0.789** | **yes** | 186 |
| No verification | – | – | all 10 | 0 | 1.000 | **1.000** | **yes** | 0 |

Backdoor accuracy (BA) is the fraction of triggered test images classified as the attacker's
target; the attack counts as successful when BA > 0.6. The clean baseline BA is about 0.01.

**What this shows:**
- **Proving every update** catches all 10 attackers the first time they patch an update, so the
  backdoor never enters the model.
- **Auditing with rollback** stops the backdoor at every audit rate. At low rates some attackers
  are never caught, and the model is briefly partly backdoored between catches (peak BA 0.26 at
  5 %, 0.19 at 10 %). Each catch rolls the model back, which wipes the uncaught attackers'
  progress too.
- **Auditing without rollback fails.** It used the exact same audit draws as the 10 % rollback
  run and caught the same 9 attackers, yet the backdoor reached BA 0.789, because patches
  aggregated before a catch stayed in the model.

### Expected catch probability (formula P = 1 − (1 − p)^m)

The attack needed **m = 6** patched rounds to succeed here; the ZK paper assumed m = 40.

| Audit rate p | Catch one attacker within 6 rounds | Paper's figure (m = 40) | Catch one attacker within 20 FU rounds | Catch all 10 within 20 rounds | Proving saved |
|---:|---:|---:|---:|---:|---:|
| 5 % | 0.265 | 0.871 | 0.642 | 0.012 | 95 % |
| 10 % | 0.469 | 0.985 | 0.878 | 0.274 | 90 % |
| 20 % | 0.738 | 1.000 | 0.988 | 0.891 | 80 % |
| 60 % | 0.996 | 1.000 | 1.000 | 1.000 | 40 % |
| 100 % | 1.000 | 1.000 | 1.000 | 1.000 | 0 % |

The measured detection rates (0.70, 0.90 and 1.00 at 5 %, 10 % and 20 %) match the
"within 20 FU rounds" column.

### Comparison with SubID (Part I, statistical detection)

| Defence | Detection rate | False alarms | F1 | Basis |
|---|---:|---:|---:|---|
| SubID (Part I, 2 attackers of 10 clients) | 1.00 | 0.00 | 1.00 | statistical; needs attackers to be a minority |
| Groth16, every update proved | 1.00 | 0.00 | 1.00 | cryptographic; certain |
| Groth16 + 10 % audit | 0.90 | 0.00 | 0.95 | cryptographic; probabilistic coverage |
