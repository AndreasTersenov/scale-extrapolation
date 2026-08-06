#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=01:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/night3_aug_train_%j.log
# NIGHT-3 AUG arm training (prereg 2026-08-05-night3): fractional
# band-limited copies (g96 {1,2} + g80 {1} + originals {2,3,4}; A-N3-4),
# 40k steps (per-slot parity, 6 slots), seed via --export=SEED=11|12.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/run_aug_arm.py --field gowerstreet --steps 40000 \
    --ckpt-every 1000 --seed "$SEED" \
    --data "$REPO/data_cache/tiles_pnull.npz" \
    --coords-file "$REPO/data_cache/running_couplings.json" \
    --truth "$REPO/results_p2/gowerstreet_val_ref.json" \
    --out "$REPO/results_p2/night3_aug${SEED}_arms.npz" \
    --sel-out "$REPO/results_p2/night3_aug${SEED}_selection.json" \
    --ckpt-dir "$REPO/data_cache/ckpt_night3_aug${SEED}"
