#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/train_%j.log
# Rung (iii) P-null: train arms A/B on GRF and generate (GPU). MIG slice + <3h => the fast
# b1 pool (rorqual-jobs). rrg-lplevass is the GPU account. Submit from repo root:
#   sbatch scripts/train_gpu.slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_pnull.py --field grf --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --steps 8000 --batch 32 --n-heldout 64 --sample-steps 80 \
    --out results/pnull_generated.npz
