#!/bin/bash
# sco_env.sh - portable SCO Framework Environment Configuration

# Utility function to get analysis path for a galaxy.
sco_get_galaxy_path() {
    local galaxy_name=$1
    if [ -z "$galaxy_name" ]; then
        echo "Error: galaxy_name required" >&2
        return 1
    fi
    echo "${SCO_ANALYSIS_ROOT}/${galaxy_name}"
}

# Utility function to get kinematics path for a galaxy and spectrograph.
sco_get_kinematics_path() {
    local galaxy_name=$1
    local spectrograph=$2
    if [ -z "$galaxy_name" ] || [ -z "$spectrograph" ]; then
        echo "Error: galaxy_name and spectrograph required" >&2
        return 1
    fi
    echo "${SCO_ANALYSIS_ROOT}/${galaxy_name}/kinematics/${spectrograph}/kinematic_extraction"
}

# Utility function to get modeling path for a galaxy.
sco_get_modeling_path() {
    local galaxy_name=$1
    if [ -z "$galaxy_name" ]; then
        echo "Error: galaxy_name required" >&2
        return 1
    fi
    echo "${SCO_ANALYSIS_ROOT}/${galaxy_name}/modeling"
}

# Utility function to get sb_analysis path for a galaxy.
sco_get_sb_analysis_path() {
    local galaxy_name=$1
    if [ -z "$galaxy_name" ]; then
        echo "Error: galaxy_name required" >&2
        return 1
    fi
    echo "${SCO_ANALYSIS_ROOT}/${galaxy_name}/sb_analysis"
}

if [ -n "${SCO_FRAMEWORK_ROOT:-}" ] && [ -n "${SCO_ANALYSIS_ROOT:-}" ]; then
    return 0 2>/dev/null || :
fi

if [ -z "${SCO_FRAMEWORK_ROOT:-}" ] && [ -n "${SCO_ROOT:-}" ]; then
    export SCO_FRAMEWORK_ROOT="$SCO_ROOT"
fi

SCO_FRAMEWORK_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${SCO_FRAMEWORK_ROOT:-}" ]; then
    export SCO_FRAMEWORK_ROOT="$(dirname "$SCO_FRAMEWORK_BIN")"
fi

SCO_CONFIG_FILE="${SCO_FRAMEWORK_ROOT}/.sco_config"
if [ -f "$SCO_CONFIG_FILE" ]; then
    source "$SCO_CONFIG_FILE"
else
    export SCO_ANALYSIS_ROOT="$(pwd)"
    export SCO_PROJECT_ROOT="$(dirname "$SCO_ANALYSIS_ROOT")"
    export SCO_GALAXY_DATA_ROOT="${HOME}/galaxy_data"
    export JOB_OUTPUT_LOGS="${HOME}/job_output_logs"
fi

mkdir -p "$JOB_OUTPUT_LOGS" 2>/dev/null || true

export SCO_FRAMEWORK_BIN
export SCO_FRAMEWORK_ROOT

for SCO_FRAMEWORK_LIB in "${SCO_FRAMEWORK_ROOT}/lib" "${SCO_FRAMEWORK_ROOT}/lib/python3.9/site-packages"; do
    if [ -d "$SCO_FRAMEWORK_LIB" ] && [[ ":${PYTHONPATH:-}:" != *":${SCO_FRAMEWORK_LIB}:"* ]]; then
        export PYTHONPATH="${SCO_FRAMEWORK_LIB}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
done

if [ -z "${SCO_PROJECT_ROOT:-}" ]; then
    export SCO_PROJECT_ROOT="$(dirname "$SCO_ANALYSIS_ROOT")"
fi
export SCO_PROJECT_ROOT

if [ -z "${__SCO_PATH_INITIALIZED__:-}" ]; then
    if [[ ":$PATH:" != *":${SCO_FRAMEWORK_BIN}:"* ]]; then
        export PATH="${SCO_FRAMEWORK_BIN}:${PATH}"
    fi
    export __SCO_PATH_INITIALIZED__=1
fi

