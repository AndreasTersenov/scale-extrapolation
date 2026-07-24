#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/joblogs/arms_big_%j.log
# Rung (iv) attempt 3: higher-capacity + longer FiLM run to lift the generator-fidelity
# ceiling (attempts 1-2 under-shot var_slope ~25% at trained octaves, capping P6).
# config_hash 63423d9af3.
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts/run_two_arms.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 48 96 192 --cond-mode film --steps 25000 --batch 32 --n-heldout 64 \
    --sample-steps 120 \
    --out "$SLURM_SUBMIT_DIR/results/npz/arms_big.npz" \
    --ckpt-dir "$SLURM_SUBMIT_DIR/data_cache/ckpt_big"
