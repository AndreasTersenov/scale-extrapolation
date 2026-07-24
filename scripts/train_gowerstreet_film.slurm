#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/joblogs/arms_film_%j.log
# Rung (iv) attempt 2: arm B with FiLM conditioning (honest P6 tuning). Separate out/ckpt
# so attempt-1 (additive) artifacts are preserved for comparison. config_hash ee62f09bb1.
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_two_arms.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 10000 --batch 32 --n-heldout 64 \
    --sample-steps 80 \
    --out "$SLURM_SUBMIT_DIR/results/npz/arms_film.npz" \
    --ckpt-dir "$SLURM_SUBMIT_DIR/data_cache/ckpt_film"
