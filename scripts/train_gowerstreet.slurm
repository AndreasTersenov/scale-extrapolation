#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=1
#SBATCH --time=02:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/arms_%j.log
# Rung (iv) break & repair on gowerstreet (full H100, <3h => fast b1 pool). Load-bearing
# run. rrg-lplevass GPU account. Submit from repo root:  sbatch scripts/train_gowerstreet.slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_two_arms.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --steps 10000 --batch 32 --n-heldout 64 --sample-steps 80 \
    --out results/arms_generated.npz
