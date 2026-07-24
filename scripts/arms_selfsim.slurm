#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results/joblogs/arms_selfsim_%j.log
# Step 4: the exactly scale-invariant control (no drift, in-class law). Frozen config.
# Pre-registered log/2026-07-16-prereg-step4-selfsim.md, config_hash 3c03bded83.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_two_arms.py --field selfsim --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 20000 --batch 32 --n-heldout 64 \
    --sample-steps 80 --nll-head --augment \
    --data "$REPO/data_cache/tiles_selfsim.npz" \
    --coords-file "$REPO/data_cache/running_couplings.json" \
    --out "$REPO/results/npz/arms_selfsim.npz" \
    --ckpt-dir "$REPO/data_cache/ckpt_selfsim"
