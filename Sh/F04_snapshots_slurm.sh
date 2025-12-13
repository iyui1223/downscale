#!/bin/bash
#SBATCH --job-name=snapshots
#SBATCH --account=cranmer-sl3-cpu
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=../Log/snapshots.out
#SBATCH --error=../Log/snapshots.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yi260@cam.ac.uk

################################################################################
# F04: Extreme Event Snapshot Generation (SLURM Job)
#
# This script generates temperature snapshots for UK extreme temperature events.
# It creates temperature maps, anomaly maps, and difference maps for each date
# within each event period.
#
# Input:
#   - Binary data files from F03 inference/evaluation
#   - Event definitions from Const/snapshot_events.yaml
#
# Output:
#   - PNG images organized by event in Figs/F04_snapshots/{model_name}/
#   - Interactive HTML viewer for comparing datasets
#
# Configuration:
#   - Model name: Const/env_setting.sh (MODEL_NAME variable)
#   - Events: Const/snapshot_events.yaml
#
# Usage:
#   sbatch Sh/F04_snapshots_slurm.sh
#
# Or with options:
#   sbatch Sh/F04_snapshots_slurm.sh --event aug2003_heatwave
#
# To check job status:
#   squeue -u $USER
#   tail -f Log/snapshots.out
################################################################################

# Load environment settings FIRST
source ../Const/env_setting.sh

echo "================================================================================"
echo "F04: Extreme Event Snapshot Generation"
echo "================================================================================"
echo "Model name:          ${MODEL_NAME}"
echo "Job ID:              $SLURM_JOB_ID"
echo "Job name:            $SLURM_JOB_NAME"
echo "Node:                $SLURM_NODELIST"
echo "CPUs per task:       $SLURM_CPUS_PER_TASK"
echo "Memory:              $SLURM_MEM_PER_NODE MB"
echo "Working directory:   $(pwd)"
echo "Started:             $(date)"
echo "================================================================================"
echo ""

# Change to root directory
cd ${ROOT_DIR}

################################################################################
# Configuration
################################################################################

# Input: Data from F03 inference/evaluation
DATA_DIR="${ROOT_DIR}/Data/F03_inference_evaluate/${MODEL_NAME}"

# Output: Figures directory
OUTPUT_DIR="${FIGS_DIR}/F04_snapshots/${MODEL_NAME}"

# Events configuration
EVENTS_FILE="${ROOT_DIR}/Const/snapshot_events.yaml"

# GrADS configuration
GRADS_CMD="/home/yi260/rds/hpc-work/lib/opengrads/opengrads-2.2.1.oga.1/Contents/grads"
GRADS_DIR="${ROOT_DIR}/GrADS"

echo "Configuration:"
echo "  Data directory:   ${DATA_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Events file:      ${EVENTS_FILE}"
echo "  GrADS command:    ${GRADS_CMD}"
echo ""

################################################################################
# Validation
################################################################################

echo "Validating inputs..."

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please run F03 inference/evaluation first."
    exit 1
fi

if [ ! -f "${DATA_DIR}/predictions.ctl" ]; then
    echo "ERROR: predictions.ctl not found in $DATA_DIR"
    echo "Binary data files may not have been generated."
    exit 1
fi

if [ ! -f "$EVENTS_FILE" ]; then
    echo "ERROR: Events file not found: $EVENTS_FILE"
    exit 1
fi

if [ ! -x "$(command -v $GRADS_CMD)" ] && [ ! -f "$GRADS_CMD" ]; then
    echo "ERROR: GrADS not found at: $GRADS_CMD"
    exit 1
fi

echo "  ✓ Data directory exists"
echo "  ✓ Binary data files found"
echo "  ✓ Events configuration found"
echo "  ✓ GrADS available"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

################################################################################
# System Information
################################################################################

echo "System information:"
echo "  Hostname:    $(hostname)"
echo "  CPU info:    $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  Total CPUs:  $(nproc)"
echo "  Memory:      $(free -h | grep Mem | awk '{print $2}')"
echo ""

echo "Python environment:"
poetry run python3 --version 2>/dev/null || python3 --version
echo ""

################################################################################
# Run Snapshot Generation
################################################################################

echo "================================================================================"
echo "Starting snapshot generation..."
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Export variables for the script
export ROOT_DIR
export GRADS_CMD
export FIGS_DIR

# Run the generate_snapshots script with any passed arguments
bash Sh/generate_snapshots.sh \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --events-file "$EVENTS_FILE" \
    --grads-cmd "$GRADS_CMD" \
    "$@"

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

################################################################################
# Copy Viewer HTML
################################################################################

echo ""
echo "Setting up interactive viewer..."

VIEWER_DIR="${OUTPUT_DIR}/viewer"
mkdir -p "$VIEWER_DIR"

# Copy viewer HTML if it exists
if [ -f "${ROOT_DIR}/Viewer/snapshot_viewer.html" ]; then
    cp "${ROOT_DIR}/Viewer/snapshot_viewer.html" "$VIEWER_DIR/"
    echo "  ✓ Viewer copied to: $VIEWER_DIR/snapshot_viewer.html"
else
    echo "  ⚠ Viewer HTML not found (create Viewer/snapshot_viewer.html)"
fi

################################################################################
# Summary
################################################################################

echo ""
echo "================================================================================"
echo "Job Summary"
echo "================================================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Exit code:       $EXIT_CODE"
echo "Total time:      ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "Finished:        $(date)"
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Snapshot generation completed successfully!"
    echo ""
    echo "Output location: ${OUTPUT_DIR}"
    echo ""
    echo "Generated images:"
    find "$OUTPUT_DIR" -name "*.png" -type f | wc -l
    echo " PNG files"
    echo ""
    echo "Events processed:"
    ls -d "${OUTPUT_DIR}"/*/ 2>/dev/null | grep -v viewer | wc -l
    echo " events"
    echo ""
    echo "Interactive viewer:"
    echo "  ${OUTPUT_DIR}/viewer/snapshot_viewer.html"
    echo ""
else
    echo ""
    echo "✗ Snapshot generation failed with exit code $EXIT_CODE"
    echo "Check log files for details."
    echo ""
fi

exit $EXIT_CODE

