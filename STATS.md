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
