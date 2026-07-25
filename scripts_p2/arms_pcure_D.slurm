#!/bin/bash
#SBATCH --account=rrg-lplevass
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=results_p2/arms_pcure_%x_%j.log
# Arm D of the parity-cure prereg (R32 approved, R33/R34 sequencing): one data
# arm per submission via --export=ARM=n1s1|n8|n32|cens. UNMODIFIED runner
# (raw-mode estimands, comparable to the committed 1x baseline); the F2-mode
# production texture is a separate inference pass at the picks (post-training).
REPO=/home/tersenov/software/scale-extrapolation
SCRATCH=$HOME/links/scratch/scale-extrap-p2
cd "$REPO" || exit 1
module load cuda/12.6 2>/dev/null || true
PY="$HOME/wl-challenge-env/bin/python"
$PY -c "import jax; print('jax devices:', jax.devices())" || exit 1

case "$ARM" in
  n1s1) DATA="$REPO/data_cache/tiles_sandbox.npz"; SEED=1 ;;
  n8)   DATA="$SCRATCH/tiles_placement_n8.npz"; SEED=0 ;;
  n32)  DATA="$SCRATCH/tiles_placement_n32.npz"; SEED=0 ;;
  cens) DATA="$SCRATCH/tiles_placement_cens.npz"; SEED=0 ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

$PY scripts_p2/run_c1t_arms.py --field sandbox --train-octaves 2 3 4 \
    --gen-from 4 --channels 32 64 128 --cond-mode film --steps 20000 \
    --ckpt-every 500 --batch 32 --n-heldout 64 --sample-steps 80 --sel-K 4 \
    --seed $SEED --data "$DATA" \
    --out "$REPO/results_p2/arms_pcure_${ARM}.npz" \
    --sel-out "$REPO/results_p2/pcure_selection_${ARM}.json" \
    --ckpt-dir "$REPO/data_cache/ckpt_pcure_${ARM}"
