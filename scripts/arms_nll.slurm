#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=results/joblogs/arms_nll_%j.log
# Phase-1c step 1 (load-bearing): gowerstreet arms A/B with the Gaussian-NLL detail
# head, FiLM conditioning, 2k-granularity checkpoints. Full H100, <=2:59 => b1 pool.
# Pre-registered log/2026-07-11-prereg-1c-nllhead.md, config_hash 4866f6e236.
# Paths are absolute so the runtime hash matches the pre-registered one exactly.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_two_arms.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 10000 --batch 32 --n-heldout 64 \
    --sample-steps 80 --nll-head --ckpt-steps 2000 4000 6000 8000 \
    --data "$REPO/data_cache/tiles_pnull.npz" \
    --coords-file "$REPO/data_cache/running_couplings.json" \
    --out "$REPO/results/npz/arms_nll.npz" \
    --ckpt-dir "$REPO/data_cache/ckpt_nll"
