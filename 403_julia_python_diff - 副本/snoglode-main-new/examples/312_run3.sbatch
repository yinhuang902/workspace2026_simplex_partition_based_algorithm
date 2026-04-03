#!/bin/bash
#SBATCH --job-name=sngo12h
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-11
#SBATCH --output=sngo12h-%A_%a.out
#SBATCH --error=sngo12h-%A_%a.err

set -euo pipefail

BASE="/home/yinhuang/312_run3_snog_1e3_TL15/snoglode-main-2-ori-record3/examples"

DIRS=(
  "problem_2_1_7"
  "problem_2_1_8"
  "problem_st_fp7a"
  "problem_st_fp7b"
  "problem_st_fp7c"
  "problem_st_fp7d"
  "problem_st_fp7e"
  "problem_st_fp8"
  "problem_st_rv2"
  "problem_st_rv3"
  "problem_st_rv7"
  "problem_st_rv8"
)

FILES=(
  "problem_2_1_7.py"
  "problem_2_1_8.py"
  "problem_st_fp7a.py"
  "problem_st_fp7b.py"
  "problem_st_fp7c.py"
  "problem_st_fp7d.py"
  "problem_st_fp7e.py"
  "problem_st_fp8.py"
  "problem_st_rv2.py"
  "problem_st_rv3.py"
  "problem_st_rv7.py"
  "problem_st_rv8.py"
)

IDX=${SLURM_ARRAY_TASK_ID}
WORKDIR="${BASE}/${DIRS[$IDX]}"
PY="${WORKDIR}/${FILES[$IDX]}"

echo "JobID:    $SLURM_JOB_ID"
echo "ArrayID:  $SLURM_ARRAY_TASK_ID"
echo "Node:     $(hostname)"
echo "Start:    $(date)"
echo "Workdir:  $WORKDIR"
echo "Script:   $PY"

cd "$WORKDIR"
ls -lah "$PY"

module purge
module load gurobi/13.0.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($HOME/bin/bin/micromamba shell hook -s bash)"
micromamba activate pidsp-sim-fir

echo "Python: $(which python)"
python -V

python "$PY"

echo "End: $(date)"