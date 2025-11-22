#!/bin/bash
################################################################################
# XGBoost Training Wrapper Script
#
# This script manages the XGBoost training workflow:
#   1. Creates a Work directory for analysis
#   2. Links input data files
#   3. Runs the training with visible parameters
#   4. Moves successful outputs to long-term storage
#
# Usage:
#   bash Sh/train_xgboost.sh [options]
#
# Options:
#   --sample-ratio RATIO    Fraction of data to use (default: 0.1)
#   --max-depth DEPTH       Max tree depth (default: 8)
#   --learning-rate LR      Learning rate (default: 0.1)
#   --n-estimators N        Number of boosting rounds (default: 1000)
#   --help                  Show this help message
#
# Requirements:
#   - Preprocessed ERA5 and MSWX data in Data/Processed/
#   - Python environment with XGBoost
################################################################################

set -e  # Exit on error
set -o pipefail

################################################################################
# EDITABLE PARAMETERS
################################################################################

# Input data files (relative to ROOT_DIR/Data/Processed/)
ERA5_DATA_FILE="training_era5_tmax.npz"
MSWX_DATA_FILE="target_mswx_tmax.npz"

# Model configuration
MODEL_NAME="xgboost_downscale_tmax"

# Training parameters
SAMPLE_RATIO="0.7"          # Use 10% of data for faster training (1.0 = use all data)
VAL_SPLIT="0.15"            # 15% for validation
TEST_SPLIT="0.15"           # 15% for test
RANDOM_SEED="42"

# XGBoost hyperparameters
MAX_DEPTH="8"               # Maximum tree depth
LEARNING_RATE="0.1"         # Learning rate (eta)
N_ESTIMATORS="1000"         # Number of boosting rounds
SUBSAMPLE="0.8"             # Row subsample ratio
COLSAMPLE_BYTREE="0.8"      # Column subsample ratio
EARLY_STOPPING="50"         # Early stopping rounds
N_JOBS="-1"                 # Number of threads (-1 = use all)

################################################################################
# Parse command line arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample-ratio)
            SAMPLE_RATIO="$2"
            shift 2
            ;;
        --max-depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --n-estimators)
            N_ESTIMATORS="$2"
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

################################################################################
# Setup environment and directories
################################################################################

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment settings
if [ -f "${SCRIPT_DIR}/../Const/env_setting.sh" ]; then
    source "${SCRIPT_DIR}/../Const/env_setting.sh"
else
    echo "ERROR: env_setting.sh not found!"
    exit 1
fi

# Change to root directory
cd "${ROOT_DIR}"

# Create Work directory for this training run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${ROOT_DIR}/Work/train_${TIMESTAMP}"
mkdir -p "${WORK_DIR}"

echo "================================================================================"
echo "XGBoost Downscaling Training"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Root directory:      ${ROOT_DIR}"
echo "  Work directory:      ${WORK_DIR}"
echo "  Model name:          ${MODEL_NAME}"
echo ""
echo "Input Data:"
echo "  ERA5 data:           Data/Processed/${ERA5_DATA_FILE}"
echo "  MSWX data:           Data/Processed/${MSWX_DATA_FILE}"
echo ""
echo "Training Parameters:"
echo "  Sample ratio:        ${SAMPLE_RATIO} (fraction of data to use)"
echo "  Validation split:    ${VAL_SPLIT}"
echo "  Test split:          ${TEST_SPLIT}"
echo "  Random seed:         ${RANDOM_SEED}"
echo ""
echo "XGBoost Hyperparameters:"
echo "  Max depth:           ${MAX_DEPTH}"
echo "  Learning rate:       ${LEARNING_RATE}"
echo "  N estimators:        ${N_ESTIMATORS}"
echo "  Subsample:           ${SUBSAMPLE}"
echo "  Colsample bytree:    ${COLSAMPLE_BYTREE}"
echo "  Early stopping:      ${EARLY_STOPPING}"
echo "  N jobs:              ${N_JOBS}"
echo ""
echo "Output Locations:"
echo "  Work directory:      ${WORK_DIR}"
echo "  Final model dir:     Models/${MODEL_NAME}"
echo ""
echo "================================================================================"
echo ""

################################################################################
# Validate input files
################################################################################

echo "Validating input files..."

ERA5_PATH="${PROCESSED_DIR}/${ERA5_DATA_FILE}"
MSWX_PATH="${PROCESSED_DIR}/${MSWX_DATA_FILE}"

if [ ! -f "${ERA5_PATH}" ]; then
    echo "ERROR: ERA5 data file not found: ${ERA5_PATH}"
    echo "Please run preprocessing first: bash Sh/cdo_preprocess.sh"
    exit 1
fi

if [ ! -f "${MSWX_PATH}" ]; then
    echo "ERROR: MSWX data file not found: ${MSWX_PATH}"
    echo "Please run preprocessing first: bash Sh/cdo_preprocess.sh"
    exit 1
fi

echo "  ✓ ERA5 data found: ${ERA5_PATH}"
echo "  ✓ MSWX data found: ${MSWX_PATH}"
echo ""

################################################################################
# Link input files to Work directory
################################################################################

echo "Setting up Work directory..."

# Create symbolic links to input data
ln -sf "${ERA5_PATH}" "${WORK_DIR}/era5_input.npz"
ln -sf "${MSWX_PATH}" "${WORK_DIR}/mswx_target.npz"

# Create output directory in Work
mkdir -p "${WORK_DIR}/model_output"

echo "  ✓ Input data linked to Work directory"
echo "  ✓ Output directory created: ${WORK_DIR}/model_output"
echo ""

################################################################################
# Display data information
################################################################################

echo "Data information:"

# Activate environment
source ~/venvs/c1coursework/bin/activate

poetry run python3 << EOF
import numpy as np
import yaml

# Load ERA5
era5 = np.load('${WORK_DIR}/era5_input.npz', allow_pickle=True)
era5_shape = era5['t2m'].shape
era5_size_mb = era5['t2m'].nbytes / (1024**2)

# Load MSWX
mswx = np.load('${WORK_DIR}/mswx_target.npz', allow_pickle=True)
mswx_shape = mswx['air_temperature'].shape
mswx_size_mb = mswx['air_temperature'].nbytes / (1024**2)

print(f"  ERA5:  shape={era5_shape}, size={era5_size_mb:.1f} MB")
print(f"  MSWX:  shape={mswx_shape}, size={mswx_size_mb:.1f} MB")

# Estimate training data size
n_samples_full = mswx_shape[0] * mswx_shape[1] * mswx_shape[2]
n_samples_used = int(n_samples_full * ${SAMPLE_RATIO})
print(f"  Total samples (MSWX grid * time): {n_samples_full:,}")
print(f"  Samples to use (${SAMPLE_RATIO} ratio): {n_samples_used:,}")
EOF

echo ""

################################################################################
# Run training
################################################################################

echo "================================================================================"
echo "Starting XGBoost Training"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Run Python training script
poetry run python3 "${ROOT_DIR}/Python/train_xgboost.py" \
    --era5-data "${WORK_DIR}/era5_input.npz" \
    --mswx-data "${WORK_DIR}/mswx_target.npz" \
    --output-dir "${WORK_DIR}/model_output" \
    --model-name "${MODEL_NAME}" \
    --sample-ratio "${SAMPLE_RATIO}" \
    --val-split "${VAL_SPLIT}" \
    --test-split "${TEST_SPLIT}" \
    --random-seed "${RANDOM_SEED}" \
    --max-depth "${MAX_DEPTH}" \
    --learning-rate "${LEARNING_RATE}" \
    --n-estimators "${N_ESTIMATORS}" \
    --subsample "${SUBSAMPLE}" \
    --colsample-bytree "${COLSAMPLE_BYTREE}" \
    --early-stopping "${EARLY_STOPPING}" \
    --n-jobs "${N_JOBS}"

TRAIN_EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "================================================================================"

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT_CODE}"
    echo "Work directory preserved for debugging: ${WORK_DIR}"
    echo "================================================================================"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed successfully in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "================================================================================"
echo ""

################################################################################
# Move outputs to long-term storage
################################################################################

echo "Moving outputs to long-term storage..."

# Create final model directory
FINAL_MODEL_DIR="${ROOT_DIR}/Models/${MODEL_NAME}"
mkdir -p "${FINAL_MODEL_DIR}"

# Copy model files
echo "  Copying model files..."
cp "${WORK_DIR}/model_output/${MODEL_NAME}.json" "${FINAL_MODEL_DIR}/"
cp "${WORK_DIR}/model_output/${MODEL_NAME}_metadata.yaml" "${FINAL_MODEL_DIR}/"
cp "${WORK_DIR}/model_output/${MODEL_NAME}_feature_importance.yaml" "${FINAL_MODEL_DIR}/"

# Create a training log
echo "  Creating training log..."
cat > "${FINAL_MODEL_DIR}/training_log.txt" << LOGEOF
Training completed: $(date)
Training time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s
Work directory: ${WORK_DIR}

Configuration:
  ERA5 data: ${ERA5_DATA_FILE}
  MSWX data: ${MSWX_DATA_FILE}
  Sample ratio: ${SAMPLE_RATIO}
  Validation split: ${VAL_SPLIT}
  Test split: ${TEST_SPLIT}
  Random seed: ${RANDOM_SEED}

XGBoost Hyperparameters:
  Max depth: ${MAX_DEPTH}
  Learning rate: ${LEARNING_RATE}
  N estimators: ${N_ESTIMATORS}
  Subsample: ${SUBSAMPLE}
  Colsample bytree: ${COLSAMPLE_BYTREE}
  Early stopping: ${EARLY_STOPPING}

Model files:
$(ls -lh "${FINAL_MODEL_DIR}")
LOGEOF

echo ""
echo "================================================================================"
echo "Success! Model saved to: ${FINAL_MODEL_DIR}"
echo "================================================================================"
echo ""
echo "Model files:"
ls -lh "${FINAL_MODEL_DIR}"
echo ""
echo "Metrics:"
cat "${FINAL_MODEL_DIR}/${MODEL_NAME}_metadata.yaml" | grep -A 10 "metrics:"
echo ""
echo "Feature importance:"
cat "${FINAL_MODEL_DIR}/${MODEL_NAME}_feature_importance.yaml"
echo ""
echo "================================================================================"
echo "Work directory: ${WORK_DIR}"
echo "  (You can remove this directory after verification)"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review model metrics above"
echo "  2. Run inference: bash Sh/inference_xgboost.sh"
echo "  3. Evaluate results: bash Sh/evaluate_xgboost.sh"
echo ""
echo "Training completed at: $(date)"
echo "================================================================================"

exit 0

