# Environment setup for scale-extrapolation on Rorqual (Compute Canada).
# Source this (do not execute) before running python/pytest:  `source env.sh`
# numpy/scipy/matplotlib come from the scipy-stack module; pywt/pytest live in the venv.
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh 2>/dev/null
module load python/3.11 scipy-stack/2025a >/dev/null 2>&1
module load arrow/19.0.1 >/dev/null 2>&1   # provides pyarrow for reading the .arrow data
source "$HOME/venvs/scale-extrap/bin/activate"
export MPLBACKEND=Agg   # headless plotting on login/compute nodes
