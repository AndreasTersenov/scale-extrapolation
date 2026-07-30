#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/l1pp_decision_edge_%j.log
# R41 decision package: the missing deconvolved edge stream. Zero training.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/l1pp_decision_edge_sample.py
