#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results/joblogs/arms_aug_%j.log
# Attempt 4a: D4 augmentation ALONE (the diagnosis falsification test). Pre-registered
# log/2026-07-11-prereg-4a-augment.md, config_hash 1e61bd812a. MIG + <=3h => b1 pool.
# Paths absolute so the runtime hash matches the pre-registered one exactly.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_two_arms.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 20000 --batch 32 --n-heldout 64 \
    --sample-steps 80 --nll-head --augment \
    --ckpt-steps 1000 2000 4000 6000 8000 12000 16000 \
    --data "$REPO/data_cache/tiles_pnull.npz" \
    --coords-file "$REPO/data_cache/running_couplings.json" \
    --out "$REPO/results/npz/arms_aug.npz" \
    --ckpt-dir "$REPO/data_cache/ckpt_aug"
