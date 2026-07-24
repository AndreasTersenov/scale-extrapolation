#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/joblogs/sweep_%j.log
# Step (b): checkpoint sweep on gowerstreet (MIG). Snapshots arm A at 2k/4k/6k/8k/12k and
# generates each -> the var_slope-vs-steps (dispersion-collapse) curve. Harvest:
#   source env.sh; for s in 2000 4000 6000 8000 12000; do
#     python scripts/measure_generated.py --npz results/npz/sweep_step$s.npz ; done
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/ckpt_sweep.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --ckpt-steps 2000 4000 6000 8000 12000 --batch 32 \
    --n-heldout 64 --sample-steps 80 --out-prefix "$SLURM_SUBMIT_DIR/results/npz/sweep"
