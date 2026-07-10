#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/disp_sweep_%j.log
# Step (c): sweep the dispersion-regularizer weight lambda in {0.1,0.3,1.0} on gowerstreet
# (FiLM arm B), early stopping at 3000 steps (~ the 2-4k dispersion window). Saves per-lambda
# gen npz + checkpoints. Harvest: measure_generated.py per npz -> trained-octave var_slope
# (fidelity bar) + octave-1 P6.
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"; R="$SLURM_SUBMIT_DIR"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
for lam in 0.1 0.3 1.0; do
  echo "=== lambda=$lam ==="
  $PY scripts/run_two_arms.py --field gowerstreet --cond-mode film --lambda-disp $lam \
      --steps 3000 --channels 32 64 128 --batch 32 --n-heldout 64 --sample-steps 80 \
      --ckpt-dir "$R/data_cache/ckpt_disp_lam${lam}" \
      --out "$R/results/arms_disp_lam${lam}.npz"
done
echo "DONE disp sweep"
