#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/parity_ckpt_sample_%j.log
# Phase A' M5 SAMPLE phase (R31 order 1): e2e generations from the committed
# c1t sandbox dense checkpoints (inference only, zero training). Authorized by
# log/2026-07-25-reconvene-phaseA.md; plan logged in the A' readout before
# submission. MIG + <=3h => b1 pool (rorqual-jobs).
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/parity_ckpt_sample.py
