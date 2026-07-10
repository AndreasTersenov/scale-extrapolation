#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/arms_2k_%j.log
# Var-faithful (a)+(b): train arms A/B (FiLM) to the ~2k dispersion PEAK, save checkpoints,
# then generate at churn {0,4,8} (SDE) for the combined test. MIG slice.
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
R="$SLURM_SUBMIT_DIR"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_two_arms.py --field gowerstreet --cond-mode film --steps 2000 \
    --channels 32 64 128 --batch 32 --n-heldout 64 --sample-steps 80 \
    --ckpt-dir "$R/data_cache/ckpt_2k" --out "$R/results/arms_2k.npz"
for c in 4.0 8.0; do
  $PY scripts/run_transfer.py --field gowerstreet --src gowerstreet \
      --ckpt-dir "$R/data_cache/ckpt_2k" --churn $c --n-heldout 64 --sample-steps 80 \
      --out "$R/results/arms_2k_c${c}.npz"
done
echo "DONE 2k+churn"
