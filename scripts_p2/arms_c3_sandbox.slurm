#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/arms_c3_sandbox_%j.log
# Arm C3 sandbox leg: patched-energy-score direct sampler (single variable vs C1).
# Pre-registered log/2026-07-17-prereg-c3-sandbox.md; approved R10 with conditions
# (validation gates in tests_p2/test_energy_score.py green before this submission).
# MIG + <=3h => b1 pool. ~8x C1 step cost (m=8 forwards); C1 took 10 min total.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/run_c3_arms.py --field sandbox --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 20000 --batch 32 --n-heldout 64 \
    --m-samples 8 --patch 8 --stride 4 --augment \
    --ckpt-steps 1000 2000 4000 6000 8000 12000 16000 \
    --data "$REPO/data_cache/tiles_sandbox.npz" \
    --coords-file "$REPO/data_cache/running_couplings_sandbox.json" \
    --out "$REPO/results_p2/arms_c3_sandbox.npz" \
    --ckpt-dir "$REPO/data_cache/ckpt_c3_sandbox" || exit 1
$PY scripts_p2/c3_conditional_sweep.py \
    --ckpt-dir "$REPO/data_cache/ckpt_c3_sandbox" \
    --out "$REPO/results_p2/c3_conditional_sandbox.json"
