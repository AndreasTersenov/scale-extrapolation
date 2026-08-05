#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/night3_ckpt_sweep_%j.log
# NIGHT-3 CKPT-SWEEP (prereg 2026-08-05-night3; grant-2 probe): 24 e2e
# recursions from committed ckpt dirs (oracle/prod/blind x 8 steps).
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/night3_ckpt_sweep.py
