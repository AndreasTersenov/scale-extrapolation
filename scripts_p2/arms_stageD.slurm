#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/arms_stageD_%j.log
# STAGE D (PLAN-phase2 §4, one run): slide-the-edge on gowerstreet, C1-t substrate.
# Train octaves {3,4}; octave 2 is the HELD-OUT edge. Arm-B dial = coupling-curve
# extrapolation (running_couplings_stageD.json — deployment protocol, no measured
# target-octave couplings). Selection caged at octave 3 (a TRAINED octave) on
# val-32. Prereg log/2026-07-18-prereg-stageD.md committed BEFORE this submission.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
echo "=== node $(hostname) GPU check ==="
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/run_c1t_arms.py --field gowerstreet --train-octaves 3 4 --gen-from 4 \
    --channels 32 64 128 --cond-mode film --steps 20000 --ckpt-every 500 \
    --batch 32 --n-heldout 64 --sample-steps 80 --sel-K 4 --sel-octave 3 \
    --data "$REPO/data_cache/tiles_pnull.npz" \
    --coords-file "$REPO/data_cache/running_couplings_stageD.json" \
    --truth "$REPO/results_p2/gowerstreet_val_ref_oct3.json" \
    --out "$REPO/results_p2/arms_stageD.npz" \
    --sel-out "$REPO/results_p2/c1t_selection_stageD.json" \
    --ckpt-dir "$REPO/data_cache/ckpt_stageD"
