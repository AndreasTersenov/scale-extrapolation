#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=results_p2/l1p_main_%j.log
# L1' main sample (pre-statement 67d7304): gow replay gate + 3 adjudicating
# + 1 ablation + edge continuity streams. Submitted ONLY after CANARY PASS.
REPO=/home/tersenov/software/scale-extrapolation
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1
$PY scripts_p2/l1p_main_sample.py
