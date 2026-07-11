#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/pnull_nll_%j.log
# Phase-1c step 1: GRF null with the Gaussian-NLL detail head (binding condition).
# Pre-registered log/2026-07-11-prereg-1c-nllhead.md, config_hash dc676a1f01.
# Paths are absolute so the runtime hash matches the pre-registered one exactly.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_pnull.py --field grf --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --steps 8000 --batch 32 --n-heldout 64 --sample-steps 80 \
    --nll-head \
    --data "$REPO/data_cache/tiles_pnull.npz" \
    --out "$REPO/results/pnull_nll.npz"
