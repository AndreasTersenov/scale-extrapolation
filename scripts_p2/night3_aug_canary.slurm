#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/night3_aug_canary_%j.log
# NIGHT-3 AUG sandbox canary (prereg 2026-08-05-night3): fractional-copy
# trainer on sandbox, 6k steps; kill rule = conditional-dispersion
# deficit > 40% at ckpt >= 4k (night3_canary_check.py on s5000).
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/run_aug_arm.py --field sandbox --steps 6000 --ckpt-every 1000 \
    --seed 11 \
    --data "$REPO/data_cache/tiles_sandbox.npz" \
    --coords-file "$REPO/data_cache/running_couplings_sandbox.json" \
    --truth "$REPO/results_p2/sandbox_truth_normconv.json" \
    --out "$REPO/results_p2/night3_aug_canary_arms.npz" \
    --sel-out "$REPO/results_p2/night3_aug_canary_selection.json" \
    --ckpt-dir "$REPO/data_cache/ckpt_night3_aug_canary" || exit 1
$PY scripts_p2/night3_canary_check.py \
    "$REPO/data_cache/ckpt_night3_aug_canary/armA_sandbox_s5000.pkl" \
    "$REPO/data_cache/tiles_sandbox.npz" sandbox 2 \
    "$REPO/results_p2/night3_aug_canary_check.json"
