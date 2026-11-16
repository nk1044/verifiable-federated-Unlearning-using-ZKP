## Circuit and keys

| item | value |
|---|---|
| constraints (R1CS) | 290,129 |
| public inputs | 8 |
| compile time | 0.34 s |
| Groth16 setup time | 16.0 s |
| proving key | 117.1 MB |
| verifying key | 1,488 B |

### Constraint breakdown (marginal cost per component)

| component | constraints | share |
|---|---:|---:|
| 32 samples: range checks, one-hot, leaf hash, Merkle path | 204,512 | 70.5% |
| 2 head hashes: range checks + H(W,b), H(W',b') | 33,668 | 11.6% |
| forward + gradient multiplications | 41,280 | 14.2% |
| 970 truncations (hint + 2 range checks each) | 21,960 | 7.6% |
| *(range-check lookup amortisation difference)* | -11,291 | |

Poseidon2 compression = 186 constraints; one sample (leaf + 11-level path + 64 range checks + one-hot) = 6,391.

## Proving benchmark (random features, 20 proofs, 8 cores)

| metric | value |
|---|---|
| proving time per step | **1.424 s ± 0.042** (min 1.347, max 1.527) |
| verification time | **1.77 ms ± 0.06** |
| proof size | 196 B |
| native fixed-point step | 1.32 ms |
| Merkle tree rebuild (2048 leaves) | 110 ms |
| witness assembly | 0.37 ms |
| proving-key load | 0.10 s |
| Go runtime peak memory | 933 MB |

## Proving benchmark (real client dataset, 20 proofs, 8 cores)

| metric | value |
|---|---|
| proving time per step | **1.439 s ± 0.061** (min 1.384, max 1.651) |
| verification time | **1.83 ms ± 0.24** |
| proof size | 196 B |
| native fixed-point step | 1.48 ms |
| Merkle tree rebuild (2048 leaves) | 109 ms |
| witness assembly | 0.37 ms |
| proving-key load | 0.10 s |
| Go runtime peak memory | 916 MB |

## Attack vs. defence (50 clients, 10 malicious, 5 benign unlearners)

| run | setting | FL-end MA | final MA | final BA | attack OK | caught | FP | rollbacks | proofs | client proving (CPU s) | server verify (s) | wall (s) |
|---|---|---:|---:|---:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|
| E1_noverify_noattack | no verification, no attack | 0.934 | 0.945 | **0.011** | no | 0/10 | 0 | 0 | 0 | 0 | 0.00 | 7 |
| E0_noverify_fusion | no verification, FUsion | 0.934 | 0.938 | **1.000** | yes | 0/10 | 0 | 0 | 0 | 0 | 0.00 | 13 |
| E2_cnn_features_fusion | ablation: pure-CNN features, FUsion | 0.945 | 0.892 | **0.077** | no | 0/10 | 0 | 0 | 0 | 0 | 0.00 | 20 |
| A05_audit | auditing p = 5 % | 0.934 | 0.937 | **0.034** | no | 7/10 | 0 | 4 | 97 | 155 | 0.27 | 181 |
| A10_audit | auditing p = 10 % | 0.934 | 0.938 | **0.019** | no | 9/10 | 0 | 6 | 186 | 318 | 0.54 | 347 |
| A20_audit | auditing p = 20 % | 0.934 | 0.941 | **0.013** | no | 10/10 | 0 | 5 | 363 | 619 | 1.05 | 648 |
| A10_exclude | auditing p = 10 %, exclude only | 0.934 | 0.942 | **0.789** | yes | 9/10 | 0 | 0 | 186 | 308 | 0.55 | 336 |
| F_full_proofs | every update proved (p = 1) | 0.934 | 0.937 | **0.013** | no | 10/10 | 0 | 1 | 130 | 203 | 0.35 | 207 |

## Fixed point vs float (same head, same schedule, no attack)

| | after FL | after FU |
|---|---:|---:|
| float | 0.9379 | 0.9452 |
| fixed point (proved procedure) | 0.9345 | 0.9454 |
| difference | -0.0034 | +0.0002 |

## FUsion trajectory without verification (E0)

| FU round | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MA | 0.934 | 0.932 | 0.933 | 0.934 | 0.934 | 0.936 | 0.937 | 0.936 | 0.937 | 0.938 |
| BA | 0.091 | 0.337 | 0.657 | 0.915 | 0.972 | 0.992 | 0.998 | 0.999 | 1.000 | 1.000 |

Rounds of patching to reach BA > 0.6: **m = 6**. Attack prep: {'ts_units': [2, 21, 25, 42, 54, 63], 'gbd_BA': 1.0, 'subnet_BA': 1.0, 'S_bd_l2': 2.652540700925452}

## Auditing: analytic catch probability with the measured m

| p | P(one attacker caught within m=6) | paper's figure (m=40) | P(one attacker caught within 20 FU rounds) | P(all 10 caught within 20) | proving saved |
|---|---:|---:|---:|---:|---:|
| 0.05 | 0.265 | 0.871 | 0.642 | 0.012 | 95% |
| 0.10 | 0.469 | 0.985 | 0.878 | 0.274 | 90% |
| 0.20 | 0.738 | 1.000 | 0.988 | 0.891 | 80% |
| 0.60 | 0.996 | 1.000 | 1.000 | 1.000 | 40% |
| 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0% |

## Cost of full verification vs auditing (whole 40-round schedule, 50 clients)

Measured with p = 1 (F_full_proofs): server time per round = verify 50 proofs = 0.116 s; mean proving time 1.446 s per client per round.

| mode | proofs | client proving, total CPU | per client (parallel clients) |
|---|---:|---:|---:|
| every update (p = 1) | 2000 | 51.6 min | 62 s |
| A05_audit (measured) | 97 | 2.6 min | 3 s |
| A10_audit (measured) | 186 | 5.3 min | 6 s |
| A20_audit (measured) | 363 | 10.3 min | 12 s |
