#!/bin/bash
################################################################################
# Downscaling Model Evaluation Script
#
# This script evaluates downscaling model predictions (model-agnostic):
#   1. Creates a Work directory for analysis
#   2. Links input files (predictions and ground truth)
#   3. Computes evaluation metrics and creates visualizations
#   4. Moves outputs to long-term storage
#
# NOTE: This script is designed to be called from inference_*.sh scripts
#       or F03_inference_evaluate_slurm.sh and inherits environment variables
#       (ROOT_DIR, PROCESSED_DIR, MODEL_NAME, etc.)
#
# Usage:
#   bash Sh/evaluate_model.sh --predictions FILENAME [OPTIONS]
#
# Options:
#   --predictions FILENAME   Predictions file (required)
#   --ground-truth FILENAME  Ground truth file (default: target_mswx_tmax.npz)
#   --output-name NAME       Output name base (default: evaluation)
#   --model-name NAME        Model name for output organization (required)
#
# Requirements:
#   - Downscaled predictions file
#   - Ground truth MSWX data
#   - ROOT_DIR environment variable set
################################################################################

set -e  # Exit on error
set -o pipefail

################################################################################
# NOTE: This script is designed to be called from F03_inference_evaluate_slurm.sh
#       Parameters are passed via command line arguments or environment variables.
#       DO NOT edit default parameters here - configure them in F03 instead.
################################################################################

################################################################################
# Parse command line arguments
################################################################################

# Initialize from environment variables (set by F03 wrapper)
PREDICTIONS_FILE="${PREDICTIONS_FILE:-}"
GROUND_TRUTH_FILE="${GROUND_TRUTH_FILE:-target_mswx_tmax.npz}"
ERA5_INPUT_FILE="${ERA5_INPUT_FILE:-training_era5_tmax.npz}"
OUTPUT_NAME="${OUTPUT_NAME:-evaluation}"
MODEL_NAME="${MODEL_NAME:-xgboost_downscale_tmax}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --predictions)
            PREDICTIONS_FILE="$2"
            shift 2
            ;;
        --ground-truth)
            GROUND_TRUTH_FILE="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --era5-input)
            ERA5_INPUT_FILE="$2"
            shift 2
            ;;
        --help)
            grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$PREDICTIONS_FILE" ]; then
    echo "ERROR: --predictions argument is required"
    echo "Usage: bash Sh/evaluate_model.sh --predictions FILENAME --model-name MODEL_NAME"
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: --model-name argument is required"
    echo "Usage: bash Sh/evaluate_model.sh --predictions FILENAME --model-name MODEL_NAME"
    exit 1
fi

################################################################################
# Setup environment and directories
################################################################################

# Verify environment is set by parent script
if [ -z "${ROOT_DIR}" ]; then
    echo "ERROR: ROOT_DIR environment variable is not set."
    echo "This script must be called with ROOT_DIR defined, typically from F03 wrapper."
    exit 1
fi

if [ -z "${PROCESSED_DIR}" ]; then
    echo "ERROR: PROCESSED_DIR environment variable is not set."
    echo "This script must be called with PROCESSED_DIR defined, typically from F03 wrapper."
    exit 1
fi

# Ensure we're in root directory
cd "${ROOT_DIR}"

# Create Work directory for this evaluation
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${ROOT_DIR}/Work/evaluate_${TIMESTAMP}"
mkdir -p "${WORK_DIR}"

# Create model-specific downscaled data directory
DOWNSCALED_DIR="${ROOT_DIR}/Data/Downscaled/${MODEL_NAME}"
mkdir -p "${DOWNSCALED_DIR}"

echo "================================================================================"
echo "Downscaling Model Evaluation"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Root directory:      ${ROOT_DIR}"
echo "  Work directory:      ${WORK_DIR}"
echo "  Model name:          ${MODEL_NAME}"
echo ""
echo "Input Data:"
echo "  Predictions:         Data/Downscaled/${MODEL_NAME}/${PREDICTIONS_FILE}"
echo "  Ground truth:        Data/Processed/${GROUND_TRUTH_FILE}"
echo "  ERA5 input:          Data/Processed/${ERA5_INPUT_FILE}"
echo ""
echo "Output:"
echo "  Output name:         ${OUTPUT_NAME}"
echo "  Downscaled dir:      Data/Downscaled/${MODEL_NAME}/"
echo "  Metrics file:        ${OUTPUT_NAME}_metrics.yaml"
echo "  Figures directory:   Figs/F03_inference_evaluate/${MODEL_NAME}/"
echo ""
echo "================================================================================"
echo ""

################################################################################
# Validate inputs
################################################################################

echo "Validating inputs..."

# Check predictions file (first try model-specific directory, then fall back to Processed)
PRED_PATH="${DOWNSCALED_DIR}/${PREDICTIONS_FILE}"
if [ ! -f "${PRED_PATH}" ]; then
    # Fall back to old location for backward compatibility
    PRED_PATH="${PROCESSED_DIR}/${PREDICTIONS_FILE}"
    if [ ! -f "${PRED_PATH}" ]; then
        echo "ERROR: Predictions file not found in either location:"
        echo "  - Data/Downscaled/${MODEL_NAME}/${PREDICTIONS_FILE}"
        echo "  - Data/Processed/${PREDICTIONS_FILE}"
        echo "Please run inference first"
        exit 1
    fi
    echo "  ⚠ Predictions found in old location: ${PRED_PATH}"
else
    echo "  ✓ Predictions found: ${PRED_PATH}"
fi

# Check ground truth file
TRUTH_PATH="${PROCESSED_DIR}/${GROUND_TRUTH_FILE}"
if [ ! -f "${TRUTH_PATH}" ]; then
    echo "ERROR: Ground truth file not found: ${TRUTH_PATH}"
    exit 1
fi

echo "  ✓ Ground truth found: ${TRUTH_PATH}"
echo ""

################################################################################
# Link input files to Work directory
################################################################################

echo "Setting up Work directory..."

ln -sf "${PRED_PATH}" "${WORK_DIR}/predictions.npz"
ln -sf "${TRUTH_PATH}" "${WORK_DIR}/ground_truth.npz"

# Link ERA5 input if it exists
ERA5_PATH="${PROCESSED_DIR}/${ERA5_INPUT_FILE}"
if [ -f "${ERA5_PATH}" ]; then
    ln -sf "${ERA5_PATH}" "${WORK_DIR}/era5_input.npz"
    echo "  ✓ ERA5 input found: ${ERA5_PATH}"
fi

mkdir -p "${WORK_DIR}/output"

echo "  ✓ Input files linked to Work directory"
echo ""

################################################################################
# Display data information
################################################################################

echo "Data information:"

poetry run python3 << EOF
import numpy as np

# Load predictions
pred = np.load('${WORK_DIR}/predictions.npz', allow_pickle=True)
pred_data = pred['air_temperature']
pred_times = pred['coord_time']

# Load ground truth
truth = np.load('${WORK_DIR}/ground_truth.npz', allow_pickle=True)
truth_data = truth['air_temperature']
truth_times = truth['coord_time']

print(f"  Predictions shape: {pred_data.shape}")
print(f"  Ground truth shape: {truth_data.shape}")
print(f"  Time range: {str(pred_times[0])} to {str(pred_times[-1])}")
print(f"  Total samples to evaluate: {int(pred_data.size):,}")

# Quick preview
print(f"\n  Predictions range: [{float(pred_data.min()):.2f}, {float(pred_data.max()):.2f}] °C")
print(f"  Ground truth range: [{float(truth_data.min()):.2f}, {float(truth_data.max()):.2f}] °C")
EOF

echo ""

################################################################################
# Run evaluation
################################################################################

echo "================================================================================"
echo "Starting Evaluation"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Run Python evaluation script
ERA5_ARG=""
if [ -f "${WORK_DIR}/era5_input.npz" ]; then
    ERA5_ARG="--era5-input ${WORK_DIR}/era5_input.npz"
fi

poetry run python3 "${ROOT_DIR}/Python/evaluate_model.py" \
    --predictions "${WORK_DIR}/predictions.npz" \
    --ground-truth "${WORK_DIR}/ground_truth.npz" \
    ${ERA5_ARG} \
    --output-dir "${WORK_DIR}/output" \
    --output-name "${OUTPUT_NAME}"

EVAL_EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "================================================================================"

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code ${EVAL_EXIT_CODE}"
    echo "Work directory preserved for debugging: ${WORK_DIR}"
    echo "================================================================================"
    exit $EVAL_EXIT_CODE
fi

echo "Evaluation completed successfully in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "================================================================================"
echo ""

################################################################################
# Move outputs to long-term storage
################################################################################

echo "Moving outputs to long-term storage..."

# Create figures directory organized by model
FIGS_DIR_MODEL="${FIGS_DIR}/F03_inference_evaluate/${MODEL_NAME}"
mkdir -p "${FIGS_DIR_MODEL}"

# Copy metrics file to model-specific downscaled directory
METRICS_FILE="${DOWNSCALED_DIR}/${OUTPUT_NAME}_metrics.yaml"
cp "${WORK_DIR}/output/${OUTPUT_NAME}_metrics.yaml" "${METRICS_FILE}"
echo "  ✓ Metrics saved to: ${METRICS_FILE}"

# Copy plots to Figs directory
cp "${WORK_DIR}/output/"*.png "${FIGS_DIR_MODEL}/"
echo "  ✓ Figures saved to: ${FIGS_DIR_MODEL}/"

# Create evaluation summary in model-specific directory
SUMMARY_FILE="${DOWNSCALED_DIR}/${OUTPUT_NAME}_summary.txt"
cat > "${SUMMARY_FILE}" << SUMEOF
Evaluation Summary
==================

Completed: $(date)
Evaluation time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s
Work directory: ${WORK_DIR}

Input Files:
  Predictions: ${PREDICTIONS_FILE}
  Ground truth: ${GROUND_TRUTH_FILE}

Output Files:
  Metrics: ${METRICS_FILE}
  Figures: ${FIGS_DIR_MODEL}/

Metrics:
--------
SUMEOF

cat "${METRICS_FILE}" >> "${SUMMARY_FILE}"

echo "  ✓ Summary saved to: ${SUMMARY_FILE}"
echo ""

################################################################################
# Display results
################################################################################

echo "================================================================================"
echo "Evaluation Results"
echo "================================================================================"
echo ""

cat "${METRICS_FILE}"

echo ""
echo "================================================================================"
echo "Output Files"
echo "================================================================================"
echo ""
echo "Metrics file:"
echo "  ${METRICS_FILE}"
echo ""
echo "Figures:"
ls -lh "${FIGS_DIR_MODEL}/"
echo ""
echo "Summary:"
echo "  ${SUMMARY_FILE}"
echo ""
echo "================================================================================"
echo "Work directory: ${WORK_DIR}"
echo "  (You can remove this directory after verification)"
echo "================================================================================"
echo ""
echo "Evaluation completed at: $(date)"
echo "================================================================================"

exit 0

