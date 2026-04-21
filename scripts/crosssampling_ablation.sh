#!/bin/bash
# Cross-sampling ablation suite — seq-cifar10, 3 epochs
# Uses derpp_twotier_crosssampling_direct (no consolidation, direct writes to both buffers)
# Usage: bash scripts/crosssampling_ablation.sh
# Run from the mammoth/ root directory.

set -e
mkdir -p logs/crosssampling_direct

COMMON=(
  --dataset seq-cifar10
  --buffer_size 200
  --stm_size 50
  --alpha 0.5
  --beta 0.5
  --n_epochs 3
  --lr 0.001
  --enable_other_metrics 1
)

# ── BASELINES ──────────────────────────────────────────────────────────────

echo "=== B1: DER++ baseline ==="
python main.py --model derpp --dataset seq-cifar10 \
  --buffer_size 200 --alpha 0.5 --beta 0.5 \
  --n_epochs 3 --lr 0.001 --seed 0 \
  --enable_other_metrics 1 \
  2>&1 | tee logs/crosssampling_direct/B1_derpp.log

echo "=== B2: Two-tier baseline (original, no cross-sampling) ==="
python main.py --model derpp_twotier "${COMMON[@]}" \
  --seed 0 --ce_replay_mode mixed --ce_stm_ratio 0.5 \
  2>&1 | tee logs/crosssampling_direct/B2_twotier.log

echo "=== B3: Cross-sampling direct — pure routing (mse_ltm=1.0, ce_stm=1.0) ==="
python main.py --model derpp_twotier_crosssampling_direct "${COMMON[@]}" \
  --seed 0 --mse_ltm_ratio 1.0 --ce_stm_ratio 1.0 \
  2>&1 | tee logs/crosssampling_direct/B3_pure_routing.log

# ── SWEEP 1: vary mse_ltm_ratio, fix ce_stm=0.5 ───────────────────────────
# Q: Does distillation benefit from seeing recent STM logits?

echo "=== Sweep 1: mse_ltm_ratio sweep (ce_stm fixed at 0.5) ==="
for MSE in 0.0 0.2 0.4 0.6 0.8 1.0; do
  echo "--- S1: mse_ltm=${MSE} ce_stm=0.5 ---"
  python main.py --model derpp_twotier_crosssampling_direct "${COMMON[@]}" \
    --seed 0 --mse_ltm_ratio $MSE --ce_stm_ratio 0.5 \
    2>&1 | tee logs/crosssampling_direct/S1_mse${MSE}_ce0.5.log
done

# ── SWEEP 2: vary ce_stm_ratio, fix mse_ltm=0.8 ───────────────────────────
# Q: Does CE replay benefit from seeing more LTM (diverse/old) samples?

echo "=== Sweep 2: ce_stm_ratio sweep (mse_ltm fixed at 0.8) ==="
for CE in 0.0 0.2 0.4 0.6 0.8 1.0; do
  echo "--- S2: mse_ltm=0.8 ce_stm=${CE} ---"
  python main.py --model derpp_twotier_crosssampling_direct "${COMMON[@]}" \
    --seed 0 --mse_ltm_ratio 0.8 --ce_stm_ratio $CE \
    2>&1 | tee logs/crosssampling_direct/S2_mse0.8_ce${CE}.log
done

# ── GRID: 3x3 interaction sweep ────────────────────────────────────────────
# Q: Is there an interaction between the two ratios?

echo "=== Grid: 3x3 interaction sweep ==="
for MSE in 0.6 0.8 1.0; do
  for CE in 0.2 0.4 0.6; do
    echo "--- GRID: mse_ltm=${MSE} ce_stm=${CE} ---"
    python main.py --model derpp_twotier_crosssampling_direct "${COMMON[@]}" \
      --seed 0 --mse_ltm_ratio $MSE --ce_stm_ratio $CE \
      2>&1 | tee logs/crosssampling_direct/GRID_mse${MSE}_ce${CE}.log
  done
done

# ── SEEDED RUNS ────────────────────────────────────────────────────────────
# Update BEST_MSE and BEST_CE after reviewing sweep results above.

BEST_MSE=0.8
BEST_CE=0.2

echo "=== Seeded runs: mse_ltm=${BEST_MSE} ce_stm=${BEST_CE} ==="
for SEED in 0 1 2; do
  echo "--- FINAL seed=${SEED} crosssampling_direct ---"
  python main.py --model derpp_twotier_crosssampling_direct "${COMMON[@]}" \
    --seed $SEED --mse_ltm_ratio $BEST_MSE --ce_stm_ratio $BEST_CE \
    2>&1 | tee logs/crosssampling_direct/FINAL_seed${SEED}_mse${BEST_MSE}_ce${BEST_CE}.log

  echo "--- FINAL seed=${SEED} derpp baseline ---"
  python main.py --model derpp --dataset seq-cifar10 \
    --buffer_size 200 --alpha 0.5 --beta 0.5 \
    --n_epochs 3 --lr 0.001 --seed $SEED \
    --enable_other_metrics 1 \
    2>&1 | tee logs/crosssampling_direct/FINAL_seed${SEED}_derpp.log
done

echo "=== All runs complete. Logs in logs/crosssampling_direct/ ==="
