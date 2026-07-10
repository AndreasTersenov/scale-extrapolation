#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/disp_opt1_%j.log
# Step (c') option 1: t-consistent (late-t, t_lo=0.6) dispersion penalty. Sweep lambda in
# {0.3,1.0,3.0} on gowerstreet (FiLM arm B), early stop 3000 steps. Per-lambda gen npz +
# checkpoints. Harvest: measure_generated.py per npz -> trained-octave var_slope + octave-1.
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"; R="$SLURM_SUBMIT_DIR"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
for lam in 0.3 1.0 3.0; do
  echo "=== lambda=$lam t_lo=0.6 ==="
  $PY scripts/run_two_arms.py --field gowerstreet --cond-mode film --lambda-disp $lam \
      --disp-t-lo 0.6 --steps 3000 --channels 32 64 128 --batch 32 --n-heldout 64 \
      --sample-steps 80 --ckpt-dir "$R/data_cache/ckpt_opt1_lam${lam}" \
      --out "$R/results/arms_opt1_lam${lam}.npz"
done
echo "DONE opt1 sweep"
