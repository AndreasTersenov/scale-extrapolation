#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/d1_finalhc_%j.log
# D1 finals + head-conditional eval at a corrected pick (R48 order 1).
# --export=ALL,TAG=<tag>,CKPT=<ckpt>,LO=<lo>
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/stage3_b_final.py "$TAG" "$CKPT" "$LO" || exit 1
$PY scripts_p2/d1_hc_eval.py "$TAG" "$CKPT" "$LO"
