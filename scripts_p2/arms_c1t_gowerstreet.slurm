#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/arms_c1t_gowerstreet_%j.log
# Arm C1-t gowerstreet leg: t(5)-base CFM + pre-registered validation-selected early
# stop (R17). Prereg log/2026-07-18-prereg-c1t-sandbox.md (+ dated gowerstreet append) committed BEFORE this
# submission. Training data identical to C1 (322 tiles + 8x D4); held-out split
# val-32 (selection only) / test-32 (adjudication only). MIG + <=3h => b1 pool.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/run_c1t_arms.py --field gowerstreet --train-octaves 2 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 20000 --ckpt-every 500 \
    --batch 32 --n-heldout 64 --sample-steps 80 --sel-K 4 \
    --data "$REPO/data_cache/tiles_pnull.npz" \
    --coords-file "$REPO/data_cache/running_couplings.json" \
    --truth "$REPO/results_p2/gowerstreet_val_ref.json" \
    --out "$REPO/results_p2/arms_c1t_gowerstreet.npz" \
    --sel-out "$REPO/results_p2/c1t_selection_gowerstreet.json" \
    --ckpt-dir "$REPO/data_cache/ckpt_c1t_gowerstreet"
