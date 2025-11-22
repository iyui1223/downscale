#!/bin/bash
################################################################################
# XGBoost Evaluation Script
#
# This script evaluates XGBoost downscaling predictions:
#   1. Creates a Work directory for analysis
#   2. Links input files (predictions and ground truth)
#   3. Computes evaluation metrics and creates visualizations
#   4. Moves outputs to long-term storage
#
# NOTE: This script is designed to be called from inference_xgboost.sh
#       and inherits its environment (ROOT_DIR, PROCESSED_DIR, etc.)
#
# Usage:
#   Called by: inference_xgboost.sh (automatic, default behavior)
#
# Options:
#   --predictions FILENAME   Predictions file (required)
#   --ground-truth FILENAME  Ground truth file (default: target_mswx_tmax.npz)
#   --output-name NAME       Output name base (default: evaluation)
#
# Requirements:
#   - Must be called from inference_xgboost.sh (or similar script that sets ROOT_DIR)
#   - Downscaled predictions file
#   - Ground truth MSWX data
################################################################################

set -e  # Exit on error
set -o pipefail

################################################################################
# EDITABLE PARAMETERS
################################################################################

# Input files (can be overridden by command line)
PREDICTIONS_FILE=""                      # Must be specified
GROUND_TRUTH_FILE="target_mswx_tmax.npz"  # Default ground truth

# Output configuration
OUTPUT_NAME="evaluation"                  # Base name for output files

################################################################################
# Parse command line arguments
################################################################################

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
    echo "Usage: bash Sh/evaluate_xgboost.sh --predictions FILENAME"
    exit 1
fi

################################################################################
# Setup environment and directories
################################################################################

# Verify environment is set by parent script
if [ -z "${ROOT_DIR}" ]; then
    echo "ERROR: This script must be called from inference_xgboost.sh"
    echo "ROOT_DIR environment variable is not set."
    exit 1
fi

if [ -z "${PROCESSED_DIR}" ]; then
    echo "ERROR: This script must be called from inference_xgboost.sh"
    echo "PROCESSED_DIR environment variable is not set."
    exit 1
fi

# Ensure we're in root directory
cd "${ROOT_DIR}"

# Create Work directory for this evaluation
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${ROOT_DIR}/Work/evaluate_${TIMESTAMP}"
mkdir -p "${WORK_DIR}"

echo "================================================================================"
echo "XGBoost Downscaling Evaluation"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Root directory:      ${ROOT_DIR}"
echo "  Work directory:      ${WORK_DIR}"
echo ""
echo "Input Data:"
echo "  Predictions:         Data/Processed/${PREDICTIONS_FILE}"
echo "  Ground truth:        Data/Processed/${GROUND_TRUTH_FILE}"
echo ""
echo "Output:"
echo "  Output name:         ${OUTPUT_NAME}"
echo "  Metrics file:        Data/Processed/${OUTPUT_NAME}_metrics.yaml"
echo "  Plots directory:     Data/Processed/${OUTPUT_NAME}_plots/"
echo ""
echo "================================================================================"
echo ""

################################################################################
# Validate inputs
################################################################################

echo "Validating inputs..."

# Check predictions file
PRED_PATH="${PROCESSED_DIR}/${PREDICTIONS_FILE}"
if [ ! -f "${PRED_PATH}" ]; then
    echo "ERROR: Predictions file not found: ${PRED_PATH}"
    echo "Please run inference first: bash Sh/inference_xgboost.sh"
    exit 1
fi

echo "  ✓ Predictions found: ${PRED_PATH}"

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
poetry run python3 "${ROOT_DIR}/Python/evaluate_xgboost.py" \
    --predictions "${WORK_DIR}/predictions.npz" \
    --ground-truth "${WORK_DIR}/ground_truth.npz" \
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

# Create plots directory
PLOTS_DIR="${PROCESSED_DIR}/${OUTPUT_NAME}_plots"
mkdir -p "${PLOTS_DIR}"

# Copy metrics file
METRICS_FILE="${PROCESSED_DIR}/${OUTPUT_NAME}_metrics.yaml"
cp "${WORK_DIR}/output/${OUTPUT_NAME}_metrics.yaml" "${METRICS_FILE}"
echo "  ✓ Metrics saved to: ${METRICS_FILE}"

# Copy plots
cp "${WORK_DIR}/output/"*.png "${PLOTS_DIR}/"
echo "  ✓ Plots saved to: ${PLOTS_DIR}/"

# Create evaluation summary
SUMMARY_FILE="${PROCESSED_DIR}/${OUTPUT_NAME}_summary.txt"
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
  Plots: ${PLOTS_DIR}/

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
echo "Plots:"
ls -lh "${PLOTS_DIR}/"
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

