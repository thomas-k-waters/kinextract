#!/bin/bash
#SBATCH --job-name={BATCH_NUM}
#SBATCH --account=kayhan0
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --time={TIME_LIMIT}
#SBATCH --output={JOB_OUTPUT_LOGS}/%x_%A_%a.out
#SBATCH --error={JOB_OUTPUT_LOGS}/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=waterstk@umich.edu

set +o noclobber

current_dir=$(pwd)

# For job arrays: read command from the commands file based on array task ID
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    COMMANDS_FILE="{COMMANDS_FILE}"
    if [ ! -f "$COMMANDS_FILE" ]; then
        echo "Error: Commands file not found: $COMMANDS_FILE" >&2
        exit 1
    fi

    # Read the command for this array task ID
    COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMANDS_FILE")

    if [ -z "$COMMAND" ]; then
        echo "Error: No command found for task ID $SLURM_ARRAY_TASK_ID" >&2
        exit 1
    fi

    # Execute the command
    eval "$COMMAND"
else
    # Fallback for non-array jobs (backward compatibility)
    LAUNCHER_JOB_FILE=template
    if [ -f "${LAUNCHER_JOB_FILE}" ]; then
        chmod a+rx ${LAUNCHER_JOB_FILE}
        ${current_dir}/${LAUNCHER_JOB_FILE}
    else
        echo "Error: No array task ID and no template file found" >&2
        exit 1
    fi
fi
