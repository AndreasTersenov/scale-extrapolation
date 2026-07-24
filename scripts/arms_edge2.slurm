#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results/joblogs/arms_edge2_%j.log
# Step 2 pilot, component 1: slide-the-edge at a second position (train 3-4, test into
# octave 2; amended -- the frozen U-Net cannot ingest 4x4 octave-5 pairs). Pre-registered
# log/2026-07-16-prereg-step2-pilot.md + amendment, config_hash f77178e6cf.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_two_arms.py --field gowerstreet --train-octaves 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 20000 --batch 32 --n-heldout 64 \
    --sample-steps 80 --nll-head --augment \
    --data "$REPO/data_cache/tiles_pnull.npz" \
    --coords-file "$REPO/data_cache/running_couplings.json" \
    --out "$REPO/results/npz/arms_edge2.npz" \
    --ckpt-dir "$REPO/data_cache/ckpt_edge2"
