#!/bin/bash
# kinextract-specific SLURM job template (used only by kinematicextraction's
# jobsplitter call, via the local ./template.slurm override -- see jobsplitter's
# "1) local template.slurm in current directory" resolution rule). Kept
# separate from the generic templates.slurm so other parts of the modeling
# framework are unaffected by kinextract-specific setup.
#
# Edit --account, --partition, --mail-user, and the conda section below
# to match your cluster environment before submitting jobs.
#
# Placeholder tokens ({BATCH_NUM}, {CPUS_PER_TASK}, etc.) are filled in
# automatically by the jobsplitter script — do not edit those tokens.
#SBATCH --job-name={BATCH_NUM}
#SBATCH --account=kayhan0
#SBATCH --partition=standard
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem-per-cpu=4G
#SBATCH --time={TIME_LIMIT}
#SBATCH --output={JOB_OUTPUT_LOGS}/%x_%A_%a.out
#SBATCH --error={JOB_OUTPUT_LOGS}/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=waterstk@umich.edu
#SBATCH --array=1-{N_TASKS}


set -euo pipefail
set +o noclobber

echo "JobID=$SLURM_JOB_ID ArrayTaskID=${SLURM_ARRAY_TASK_ID-} Node=$SLURMD_NODENAME"
echo "Start: $(date)"
echo "PWD: $(pwd)"

# Activate your Python environment.
# Adjust the path and environment name for your cluster setup.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
echo "python: $(which python)"
python -c "import sys; print(sys.executable, sys.version)"

# Numba cache (fast local)
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"
echo "NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR"

# Match thread pools to the CPUs assigned by Slurm.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- JAX/XLA thread cap -----------------------------------------------------
# kinextract's bootstrap error estimation re-enables JAX inside a
# ThreadPoolExecutor (n_jobs worker threads sharing one process, see
# residual_bootstrap in errors.py). threadpoolctl only constrains
# OpenBLAS/MKL/OpenMP ("blas" user_api) to 1 thread per worker -- it does not
# see or limit XLA's own CPU (Eigen) thread pool, which by default sizes
# itself off the node's total core count rather than the cpuset Slurm
# actually granted. With n_jobs concurrent threads each independently able to
# spawn XLA-internal threads, this oversubscribes the {CPUS_PER_TASK}
# allocated CPUs -- observed on Greatlakes as 5-6x wall-time variance between
# otherwise-identical array tasks depending on node load.
#
# Force XLA:CPU to run every op single-threaded, so n_jobs threads x 1 XLA
# thread each = n_jobs CPUs used, matching --cpus-per-task.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

COMMANDS_FILE="{COMMANDS_FILE}"
if [[ -z "${SLURM_ARRAY_TASK_ID-}" ]]; then
  echo "Error: this template expects an array job (--array)."
  exit 2
fi
if [[ ! -f "$COMMANDS_FILE" ]]; then
  echo "Error: Commands file not found: $COMMANDS_FILE" >&2
  exit 1
fi

COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMANDS_FILE")
if [[ -z "${COMMAND// }" ]]; then
  echo "Error: No command found for task ID $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

echo "Running: $COMMAND"
bash -lc "$COMMAND"

echo "End: $(date)"
