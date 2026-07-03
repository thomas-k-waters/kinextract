#!/bin/bash
#SBATCH --job-name={BATCH_NUM}
#SBATCH --account=kayhan0
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --time={TIME_LIMIT}
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=waterstk@umich.edu

set +o noclobber

current_dir=$(pwd)

LAUNCHER_JOB_FILE=template
chmod a+rx ${LAUNCHER_JOB_FILE}
${current_dir}/${LAUNCHER_JOB_FILE}
