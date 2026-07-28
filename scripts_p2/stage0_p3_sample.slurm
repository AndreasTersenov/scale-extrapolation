#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/stage0_p3_sample_%j.log
# Phase-3 Stage-0 SAMPLE (prereg 62db5f0, APPROVED + A1 in
# log/2026-07-28-reconvene-stage0-review.md): F2 group-averaged e2e from the
# committed Stage-D checkpoints, zero training. MIG + <=3h => b1 pool.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/stage0_p3_sample.py
