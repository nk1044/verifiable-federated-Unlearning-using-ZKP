#!/usr/bin/env bash
# Reproduces every number in ZK_REPORT.md. Total ~30 min on an Apple M3 (8 cores).
# Usage (from zkfu/):  ./run_experiments.sh            or   ./run_experiments.sh quick   (skips p=0.2 and exclude)
set -euo pipefail
cd "$(dirname "$0")"
PY="$(cd .. && pwd)/.venv/bin/python"
mkdir -p results bin

echo "== build Go binaries"
for c in setup commit prove verify bench; do go build -o bin/$c ./cmd/$c; done

echo "== setup (compile + Groth16 keys)"
/usr/bin/time -l ./bin/setup -keys keys -breakdown -json results/setup.json 2> results/setup.time >/dev/null
grep "maximum resident" results/setup.time

echo "== benchmark: 20 proofs on a random client"
/usr/bin/time -l ./bin/bench -keys keys -runs 20 -json results/bench_random.json 2> results/bench.time >/dev/null
grep "maximum resident" results/bench.time

cd sim
echo "== E0: FUsion attack, no verification";        $PY fl_sim.py --audit_p 0   --name E0_noverify_fusion | tail -1
echo "== E1: no attack, no verification (fixed point)"; $PY fl_sim.py --audit_p 0 --attack none --name E1_noverify_noattack | tail -1
echo "== float baseline";                             $PY float_baseline.py --name float_baseline | tail -1
echo "== E2: ablation, pure-CNN features";            $PY fl_sim.py --audit_p 0 --features cnn --name E2_cnn_features_fusion | tail -1
DS=$(ls -d work/datasets_cnn_grid_* | head -1)/client_000.gob
echo "== benchmark on a real client dataset"
(cd .. && ./bin/bench -keys keys -runs 20 -dataset "sim/$DS" -json results/bench_real.json > /dev/null)
echo "== F: full verification (p=1), resumed after FL, 3 FU rounds"
$PY fl_sim.py --audit_p 1.0 --resume_fl work/ckpt_fl_E0_noverify_fusion.json --fu_rounds 3 --eval_every 1 --name F_full_proofs | tail -1
echo "== A05: random auditing p=0.05";               $PY fl_sim.py --audit_p 0.05 --name A05_audit | tail -1
echo "== A10: random auditing p=0.10";               $PY fl_sim.py --audit_p 0.10 --name A10_audit | tail -1
if [ "${1:-}" != "quick" ]; then
  echo "== A20: random auditing p=0.20";             $PY fl_sim.py --audit_p 0.20 --name A20_audit | tail -1
  echo "== A10x: p=0.10, exclude-only on catch";     $PY fl_sim.py --audit_p 0.10 --on_catch exclude --name A10_exclude | tail -1
fi
echo "== done"
