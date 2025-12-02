#!/bin/bash
################################################################################
# Downscaling Model Evaluation Script (Streaming Pipeline)
#
# This script evaluates downscaling model predictions using a memory-efficient
# streaming approach:
#   1. Converts .npz files to GrADS binary format (year-by-year streaming)
#   2. Computes climatology during streaming
#   3. Calls GrADS scripts for evaluation and visualization
#
# Memory usage: ~2x single year of data (much better than loading full dataset)
#
# Usage:
#   bash Sh/evaluate_model.sh --predictions FILENAME [OPTIONS]
#
# Options:
#   --predictions FILENAME   Predictions file (required)
#   --ground-truth FILENAME  Ground truth file (default: target_mswx_tmax.npz)
#   --era5-input FILENAME    ERA5 input file (optional)
#   --output-name NAME       Output name base (default: evaluation)
#   --model-name NAME        Model name for output organization (required)
#
# Requirements:
#   - Downscaled predictions file
#   - Ground truth MSWX data
#   - ROOT_DIR environment variable set
################################################################################

set -e
set -o pipefail

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
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: --model-name argument is required"
    exit 1
fi

################################################################################
# Setup environment and directories
################################################################################

# Verify environment is set by parent script
if [ -z "${ROOT_DIR}" ]; then
    echo "ERROR: ROOT_DIR environment variable is not set."
    exit 1
fi

if [ -z "${PROCESSED_DIR}" ]; then
    echo "ERROR: PROCESSED_DIR environment variable is not set."
    exit 1
fi

# Ensure we're in root directory
cd "${ROOT_DIR}"

# Create evaluation data directory (NEW STRUCTURE)
EVAL_DATA_DIR="${ROOT_DIR}/Data/F03_inference_evaluate/${MODEL_NAME}"
mkdir -p "${EVAL_DATA_DIR}"

# Create figures directory
FIGS_DIR_MODEL="${ROOT_DIR}/Figs/F03_inference_evaluate/${MODEL_NAME}"
mkdir -p "${FIGS_DIR_MODEL}"

echo "================================================================================"
echo "Downscaling Model Evaluation (Streaming Pipeline)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Root directory:      ${ROOT_DIR}"
echo "  Model name:          ${MODEL_NAME}"
echo ""
echo "Input Data:"
echo "  Predictions:         ${PREDICTIONS_FILE}"
echo "  Ground truth:        ${GROUND_TRUTH_FILE}"
echo "  ERA5 input:          ${ERA5_INPUT_FILE}"
echo ""
echo "Output:"
echo "  Data directory:      ${EVAL_DATA_DIR}"
echo "  Figures directory:   ${FIGS_DIR_MODEL}"
echo ""
echo "================================================================================"
echo ""

################################################################################
# Validate inputs
################################################################################

echo "Validating inputs..."

# Check predictions file
DOWNSCALED_DIR="${ROOT_DIR}/Data/Downscaled/${MODEL_NAME}"
PRED_PATH="${DOWNSCALED_DIR}/${PREDICTIONS_FILE}"
if [ ! -f "${PRED_PATH}" ]; then
    # Fall back to old location
    PRED_PATH="${PROCESSED_DIR}/${PREDICTIONS_FILE}"
    if [ ! -f "${PRED_PATH}" ]; then
        echo "ERROR: Predictions file not found"
        exit 1
    fi
fi
echo "  ✓ Predictions found: ${PRED_PATH}"

# Check ground truth file
TRUTH_PATH="${PROCESSED_DIR}/${GROUND_TRUTH_FILE}"
if [ ! -f "${TRUTH_PATH}" ]; then
    echo "ERROR: Ground truth file not found: ${TRUTH_PATH}"
    exit 1
fi
echo "  ✓ Ground truth found: ${TRUTH_PATH}"

# Check ERA5 file (optional)
ERA5_PATH="${PROCESSED_DIR}/${ERA5_INPUT_FILE}"
if [ -f "${ERA5_PATH}" ]; then
    echo "  ✓ ERA5 input found: ${ERA5_PATH}"
    HAS_ERA5=true
else
    echo "  ⚠ No ERA5 input (skipping difference-based metrics)"
    HAS_ERA5=false
fi

echo ""

################################################################################
# Step 1: Convert to GrADS binary format (streaming)
################################################################################

echo "================================================================================"
echo "Step 1: Converting to GrADS Binary Format (Streaming)"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Convert predictions
echo "Converting predictions..."
poetry run python3 "${ROOT_DIR}/Python/write_binary.py" \
    --input "${PRED_PATH}" \
    --output "${EVAL_DATA_DIR}/predictions" \
    --varname air_temperature

echo ""

# Convert ground truth
echo "Converting ground truth..."
poetry run python3 "${ROOT_DIR}/Python/write_binary.py" \
    --input "${TRUTH_PATH}" \
    --output "${EVAL_DATA_DIR}/ground_truth" \
    --varname air_temperature

echo ""

# Convert and interpolate ERA5 if available
if [ "$HAS_ERA5" = true ]; then
    echo "Converting and interpolating ERA5..."
    poetry run python3 "${ROOT_DIR}/Python/write_binary.py" \
        --input "${ERA5_PATH}" \
        --output "${EVAL_DATA_DIR}/era5_input" \
        --varname t2m \
        --era5-input "${ERA5_PATH}" \
        --interpolate
    
    echo ""
fi

CONV_TIME=$(date +%s)
CONV_ELAPSED=$((CONV_TIME - START_TIME))
echo "✓ Conversion completed in ${CONV_ELAPSED}s"
echo ""

################################################################################
# Step 2: Run GrADS evaluation scripts
################################################################################

echo "================================================================================"
echo "Step 2: Running GrADS Evaluation"
echo "================================================================================"
echo ""

bash "${ROOT_DIR}/Sh/gsscripts.sh" \
    --data-dir "${EVAL_DATA_DIR}" \
    --output-dir "${FIGS_DIR_MODEL}"

EVAL_EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
EVAL_TIME=$((END_TIME - CONV_TIME))

echo ""
echo "================================================================================"

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code ${EVAL_EXIT_CODE}"
    echo "================================================================================"
    exit $EVAL_EXIT_CODE
fi

echo "Evaluation completed successfully"
echo "  Conversion time: ${CONV_ELAPSED}s"
echo "  Evaluation time: ${EVAL_TIME}s"
echo "  Total time:      ${ELAPSED}s"
echo "================================================================================"
echo ""

################################################################################
# Summary
################################################################################

echo "================================================================================"
echo "Evaluation Results"
echo "================================================================================"
echo ""
echo "Data files (binary + .ctl):"
ls -lh "${EVAL_DATA_DIR}"/*.ctl 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Figures:"
ls -lh "${FIGS_DIR_MODEL}"/*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "================================================================================"
echo "Data directory: ${EVAL_DATA_DIR}"
echo "Figures directory: ${FIGS_DIR_MODEL}"
echo "================================================================================"
echo ""
echo "Evaluation completed at: $(date)"
echo "================================================================================"

exit 0

