#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/joblogs/arms_%j.log
# Rung (iv) break & repair on gowerstreet. MIG h100_20gb slice (near-instant start; the
# model is small — same sizes as the rung-iii run that finished in 217s). <3h => fast b1
# pool. rrg-lplevass GPU account. Submit from repo root:  sbatch scripts/train_gowerstreet.slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
# args left at their (absolute-path) defaults so the runtime config_hash matches the
# pre-registered 71a9fd7e6d (do not pass a relative --out — it changes the hash).
$PY scripts/run_two_arms.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --steps 10000 --batch 32 --n-heldout 64 --sample-steps 80
