#!/bin/bash
#SBATCH --account=def-lplevass   # CPU jobs use def-lplevass; rrg-lplevass is GPU-only
#SBATCH --job-name=scaledrift-M1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/joblogs/measure_%j.log
# CPU-only Measurement M1 on a compute node (dedicated cores + guaranteed /project
# data access; avoids the login-node arbiter). Submit from the repo root:
#   sbatch scripts/measure.slurm
cd "$SLURM_SUBMIT_DIR" || exit 1
source env.sh                     # modules (incl. arrow) + venv; needed even in a job
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/measure.py --n-parents 18 --max-shards 10 --n-boot 150 --out measurement
