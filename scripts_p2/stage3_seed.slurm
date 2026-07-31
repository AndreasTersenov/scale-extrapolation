#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=01:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/stage3_seed_%j.log
# Stage-3 (a) seed-ensemble training (prereg af34d93, R42 cleared):
# production recipe, arm A only, SEED passed via --export.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/run_c1t_arms.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 20000 --ckpt-every 500 \
    --batch 32 --n-heldout 64 --sample-steps 80 --sel-K 4 --arms A \
    --seed "$SEED" \
    --data "$REPO/data_cache/tiles_pnull.npz" \
    --coords-file "$REPO/data_cache/running_couplings.json" \
    --out "$REPO/results_p2/stage3_seed${SEED}_arms.npz" \
    --sel-out "$REPO/results_p2/stage3_seed${SEED}_selection.json" \
    --ckpt-dir "$REPO/data_cache/ckpt_stage3_seed${SEED}"
