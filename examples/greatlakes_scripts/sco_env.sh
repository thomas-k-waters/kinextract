#!/bin/bash
# sco_env.sh - SCO Framework Environment Configuration
# 
# This script should be sourced by all SCO framework scripts to ensure
# consistent path resolution across the entire framework.
#
# OPTIMIZED: Early exit if already sourced, minimal disk I/O
#
# Usage in scripts: source "$(dirname "${BASH_SOURCE[0]}")/sco_env.sh"
#

# OPTIMIZATION: Skip initialization if already done in this shell session
if [ -n "${SCO_FRAMEWORK_ROOT:-}" ] && [ -n "${SCO_ANALYSIS_ROOT:-}" ]; then
    return 0 2>/dev/null || :
fi


# Determine the framework root directory (only done once per session)
# SCO_FRAMEWORK_BIN is the directory containing this script
SCO_FRAMEWORK_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCO_FRAMEWORK_ROOT="$(dirname "$SCO_FRAMEWORK_BIN")"

# Compiled Fortran binaries (pallmc, rtransvd, transvd, pfitlove, jobsplitter,
# etc.) live in storage/, separate from the bin/ shell scripts.
SCO_FRAMEWORK_STORAGE="${SCO_FRAMEWORK_ROOT}/storage"

# Configuration file location
SCO_CONFIG_FILE="${SCO_FRAMEWORK_ROOT}/.sco_config"

# Initialize or load configuration
if [ -f "$SCO_CONFIG_FILE" ]; then
    # OPTIMIZATION: Use source to load config directly (most common case)
    source "$SCO_CONFIG_FILE"
else
    # Set defaults if config doesn't exist
    # These will be updated by initialize_directory_structure
    export SCO_ANALYSIS_ROOT="$(pwd)"
    export SCO_PROJECT_ROOT="$(dirname "$SCO_ANALYSIS_ROOT")"
    export SCO_GALAXY_DATA_ROOT="${HOME}/galaxy_data"
    export JOB_OUTPUT_LOGS="${HOME}/job_output_logs"
    
    # Create default config file
    cat > "$SCO_CONFIG_FILE" << EOF
# SCO Framework Configuration
# Auto-generated on first use. Edit as needed.

# Root analysis directory (typically the directory where galaxy workspaces are created)
export SCO_ANALYSIS_ROOT="${SCO_ANALYSIS_ROOT}"

# Project root (contains analysis, model_files, templates, psf directories)
export SCO_PROJECT_ROOT="${SCO_PROJECT_ROOT}"

# External galaxy data directory (optional, for spectral data lookups)
export SCO_GALAXY_DATA_ROOT="${SCO_GALAXY_DATA_ROOT}"

# Job output logs directory for SLURM submissions
export JOB_OUTPUT_LOGS="${JOB_OUTPUT_LOGS}"

# Framework bin directory (automatically set)
export SCO_FRAMEWORK_BIN="${SCO_FRAMEWORK_BIN}"
export SCO_FRAMEWORK_ROOT="${SCO_FRAMEWORK_ROOT}"
EOF
fi

# Ensure required directories exist (silent fail if permission denied)
mkdir -p "$JOB_OUTPUT_LOGS" 2>/dev/null || true

# Export framework paths
export SCO_FRAMEWORK_BIN
export SCO_FRAMEWORK_ROOT
export SCO_FRAMEWORK_STORAGE

# Backfill project root if older config files do not define it.
if [ -z "$SCO_PROJECT_ROOT" ]; then
    export SCO_PROJECT_ROOT="$(dirname "$SCO_ANALYSIS_ROOT")"
fi
export SCO_PROJECT_ROOT

# OPTIMIZATION: Add framework bin and storage dirs to PATH only once per session
if [ -z "${__SCO_PATH_INITIALIZED__:-}" ]; then
    if [[ ":$PATH:" != *":${SCO_FRAMEWORK_BIN}:"* ]]; then
        export PATH="${SCO_FRAMEWORK_BIN}:${PATH}"
    fi
    if [ -d "$SCO_FRAMEWORK_STORAGE" ] && [[ ":$PATH:" != *":${SCO_FRAMEWORK_STORAGE}:"* ]]; then
        export PATH="${SCO_FRAMEWORK_STORAGE}:${PATH}"
    fi
    export __SCO_PATH_INITIALIZED__=1
fi

# Utility function to get analysis path for a galaxy
sco_get_galaxy_path() {
    local galaxy_name=$1
    if [ -z "$galaxy_name" ]; then
        echo "Error: galaxy_name required" >&2
        return 1
    fi
    echo "${SCO_ANALYSIS_ROOT}/${galaxy_name}"
}

# Utility function to get kinematics path for a galaxy and spectrograph
sco_get_kinematics_path() {
    local galaxy_name=$1
    local spectrograph=$2
    if [ -z "$galaxy_name" ] || [ -z "$spectrograph" ]; then
        echo "Error: galaxy_name and spectrograph required" >&2
        return 1
    fi
    echo "${SCO_ANALYSIS_ROOT}/${galaxy_name}/kinematics/${spectrograph}/kinematic_extraction"
}

# Utility function to get modeling path for a galaxy
sco_get_modeling_path() {
    local galaxy_name=$1
    if [ -z "$galaxy_name" ]; then
        echo "Error: galaxy_name required" >&2
        return 1
    fi
    echo "${SCO_ANALYSIS_ROOT}/${galaxy_name}/modeling"
}

# Utility function to get sb_analysis path for a galaxy
sco_get_sb_analysis_path() {
    local galaxy_name=$1
    if [ -z "$galaxy_name" ]; then
        echo "Error: galaxy_name required" >&2
        return 1
    fi
    echo "${SCO_ANALYSIS_ROOT}/${galaxy_name}/sb_analysis"
}

# OPTIONAL PERFORMANCE OPTIMIZATION:
# For environments that run many SCO scripts in sequence (e.g., batch processing),
# you can pre-activate the SCO environment in your shell login script (.bashrc/.zshrc):
#
#   # In ~/.bashrc or ~/.zshrc, add:
#   export SCO_FRAMEWORK_ROOT="/path/to/sco_framework"
#   source "${SCO_FRAMEWORK_ROOT}/bin/sco_env.sh"
#
# Then every subsequent script sourcing sco_env.sh will skip initialization
# (due to the early exit check) resulting in zero overhead.
#
# This approach is ideal for:
# - Interactive shell sessions with many manual script calls
# - Batch job submission workflows
# - Development environments
#

