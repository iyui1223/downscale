#!/bin/bash
#SBATCH --job-name=xgb_inference
#SBATCH --account=cranmer-sl3-cpu
#SBATCH --partition=sapphire
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=../Log/inference.out
#SBATCH --error=../Log/inference.err
#SBATCH --mail-type=BEGIN,END,FAIL

################################################################################
# XGBoost Inference Wrapper Script
#
# This script manages the XGBoost inference workflow:
#   1. Creates a Work directory for analysis
#   2. Links input files (model and ERA5 data)
#   3. Runs inference to produce high-resolution predictions
#   4. Moves successful outputs to long-term storage
#   5. Optionally runs evaluation on the predictions
#
# Usage:
#   sbatch Sh/inference_xgboost.sh [OPTIONS]
#
# Options:
#   --era5-data FILENAME     ERA5 input file (default: training_era5_tmax.npz)
#   --output-name NAME       Output filename base (default: downscaled_tmax)
#   --chunk-size N           Timesteps per chunk (default: 100)
#   --ground-truth FILENAME  Ground truth for evaluation (default: target_mswx_tmax.npz)
#   --skip-evaluation        Skip automatic evaluation after inference
#
# Requirements:
#   - Trained model in Models/
#   - ERA5 data to downscale
#   - Reference grid (MSWX data) for target resolution
################################################################################

set -e  # Exit on error
set -o pipefail

################################################################################
# NOTE: This script is designed to be called from F03_inference_evaluate_slurm.sh
#       Parameters are passed via command line arguments or environment variables.
#       If running standalone, env_setting.sh is sourced for defaults.
################################################################################

# Source env_setting.sh if not already loaded (for standalone use)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${ROOT_DIR}" ] && [ -f "${SCRIPT_DIR}/../Const/env_setting.sh" ]; then
    source "${SCRIPT_DIR}/../Const/env_setting.sh"
fi

################################################################################
# Parse command line arguments
################################################################################

# Initialize from environment variables (from env_setting.sh or F03 wrapper)
MODEL_NAME="${MODEL_NAME:-xgboost_downscale_tmax}"
MODEL_DIR="Models/${MODEL_NAME}"
ERA5_DATA_FILE="${ERA5_DATA_FILE:-training_era5_tmax.npz}"
TARGET_GRID_FILE="${MSWX_DATA_FILE:-target_mswx_tmax.npz}"
OUTPUT_NAME="${OUTPUT_NAME:-downscaled_tmax}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
RUN_EVALUATION="${RUN_EVALUATION:-true}"
GROUND_TRUTH_FILE="${MSWX_DATA_FILE:-target_mswx_tmax.npz}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --era5-data)
            ERA5_DATA_FILE="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --ground-truth)
            GROUND_TRUTH_FILE="$2"
            shift 2
            ;;
        --skip-evaluation)
            RUN_EVALUATION="false"
            shift
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
# Setup environment and directories (env_setting.sh already sourced at top)
################################################################################

# Change to root directory
cd "${ROOT_DIR}"

# Set TMPDIR for heredocuments (required in some SLURM environments)
export TMPDIR="${ROOT_DIR}/Work/tmp"
mkdir -p "${TMPDIR}"

# Create model-specific downscaled data directory
DOWNSCALED_DIR="${ROOT_DIR}/Data/Downscaled/${MODEL_NAME}"
mkdir -p "${DOWNSCALED_DIR}"

# Create Work directory for this inference run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${ROOT_DIR}/Work/inference_${TIMESTAMP}"
mkdir -p "${WORK_DIR}"

echo "================================================================================"
echo "XGBoost Downscaling Inference"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Root directory:      ${ROOT_DIR}"
echo "  Work directory:      ${WORK_DIR}"
echo "  Model:               ${MODEL_DIR}"
echo ""
echo "Input Data:"
echo "  ERA5 data:           Data/Processed/${ERA5_DATA_FILE}"
echo "  Target grid ref:     Data/Processed/${TARGET_GRID_FILE}"
echo ""
echo "Output:"
echo "  Output name:         ${OUTPUT_NAME}"
echo "  Output location:     Data/Downscaled/${MODEL_NAME}/${OUTPUT_NAME}.npz"
echo ""
echo "Processing Parameters:"
echo "  Chunk size:          ${CHUNK_SIZE} timesteps"
echo "  Run evaluation:      ${RUN_EVALUATION}"
if [ "${RUN_EVALUATION}" = "true" ]; then
    echo "  Ground truth:        Data/Processed/${GROUND_TRUTH_FILE}"
fi
echo ""
echo "================================================================================"
echo ""

################################################################################
# Validate inputs
################################################################################

echo "Validating inputs..."

# Check model directory
if [ ! -d "${ROOT_DIR}/${MODEL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${ROOT_DIR}/${MODEL_DIR}"
    echo "Please train the model first: bash Sh/train_xgboost.sh"
    exit 1
fi

if [ ! -f "${ROOT_DIR}/${MODEL_DIR}/${MODEL_NAME}.json" ]; then
    echo "ERROR: Model file not found: ${ROOT_DIR}/${MODEL_DIR}/${MODEL_NAME}.json"
    exit 1
fi

echo "  ✓ Model found: ${MODEL_DIR}"

# Check ERA5 data
ERA5_PATH="${PROCESSED_DIR}/${ERA5_DATA_FILE}"
if [ ! -f "${ERA5_PATH}" ]; then
    echo "ERROR: ERA5 data file not found: ${ERA5_PATH}"
    exit 1
fi

echo "  ✓ ERA5 data found: ${ERA5_PATH}"

# Check target grid reference
TARGET_GRID_PATH="${PROCESSED_DIR}/${TARGET_GRID_FILE}"
if [ ! -f "${TARGET_GRID_PATH}" ]; then
    echo "ERROR: Target grid reference not found: ${TARGET_GRID_PATH}"
    exit 1
fi

echo "  ✓ Target grid reference found: ${TARGET_GRID_PATH}"
echo ""

################################################################################
# Link input files to Work directory
################################################################################

echo "Setting up Work directory..."

# Link model directory
ln -sf "${ROOT_DIR}/${MODEL_DIR}" "${WORK_DIR}/model"

# Link input data
ln -sf "${ERA5_PATH}" "${WORK_DIR}/era5_input.npz"
ln -sf "${TARGET_GRID_PATH}" "${WORK_DIR}/target_grid.npz"

echo "  ✓ Input files linked to Work directory"
echo ""

################################################################################
# Display model information
################################################################################

echo "Model information:"

# Activate environment
source ~/venvs/c1coursework/bin/activate

cat "${WORK_DIR}/model/${MODEL_NAME}_metadata.yaml"

echo ""

################################################################################
# Display data information
################################################################################

echo "Data information:"

poetry run python3 << EOF
import numpy as np

# Load ERA5
era5 = np.load('${WORK_DIR}/era5_input.npz', allow_pickle=True)
era5_shape = era5['t2m'].shape
era5_times = era5['coord_valid_time']

# Load target grid
target = np.load('${WORK_DIR}/target_grid.npz', allow_pickle=True)
if 'coord_lat' in target:
    target_lat = target['coord_lat']
    target_lon = target['coord_lon']
else:
    target_lat = target['coord_latitude']
    target_lon = target['coord_longitude']

target_grid_shape = (len(target_lat), len(target_lon))

print(f"  ERA5 shape: {era5_shape}")
print(f"  Time range: {str(era5_times[0])} to {str(era5_times[-1])}")
print(f"  Target grid: {target_grid_shape}")
print(f"  Output shape: ({era5_shape[0]}, {target_grid_shape[0]}, {target_grid_shape[1]})")
print(f"  Total pixels to predict: {int(era5_shape[0] * target_grid_shape[0] * target_grid_shape[1]):,}")
EOF

echo ""

################################################################################
# Run inference
################################################################################

echo "================================================================================"
echo "Starting Inference"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Run Python inference script
poetry run python3 "${ROOT_DIR}/Python/inference_xgboost.py" \
    --model-dir "${WORK_DIR}/model" \
    --model-name "${MODEL_NAME}" \
    --era5-data "${WORK_DIR}/era5_input.npz" \
    --target-grid "${WORK_DIR}/target_grid.npz" \
    --output "${WORK_DIR}/${OUTPUT_NAME}.npz" \
    --chunk-size "${CHUNK_SIZE}"

INFERENCE_EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "================================================================================"

if [ $INFERENCE_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Inference failed with exit code ${INFERENCE_EXIT_CODE}"
    echo "Work directory preserved for debugging: ${WORK_DIR}"
    echo "================================================================================"
    exit $INFERENCE_EXIT_CODE
fi

echo "Inference completed successfully in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "================================================================================"
echo ""

################################################################################
# Move outputs to long-term storage
################################################################################

echo "Moving outputs to long-term storage..."

# Copy predictions to model-specific downscaled directory
OUTPUT_FINAL="${DOWNSCALED_DIR}/${OUTPUT_NAME}.npz"
cp "${WORK_DIR}/${OUTPUT_NAME}.npz" "${OUTPUT_FINAL}"

echo "  ✓ Predictions saved to: ${OUTPUT_FINAL}"

# Create inference log in model-specific directory
LOG_FILE="${DOWNSCALED_DIR}/${OUTPUT_NAME}_inference_log.txt"
cat > "${LOG_FILE}" << LOGEOF
Inference completed: $(date)
Inference time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s
Work directory: ${WORK_DIR}

Configuration:
  Model: ${MODEL_DIR}
  ERA5 data: ${ERA5_DATA_FILE}
  Target grid: ${TARGET_GRID_FILE}
  Output: ${OUTPUT_NAME}.npz
  Chunk size: ${CHUNK_SIZE}

Output file:
$(ls -lh "${OUTPUT_FINAL}")

LOGEOF

echo "  ✓ Inference log saved to: ${LOG_FILE}"
echo ""

################################################################################
# Display results
################################################################################

echo "================================================================================"
echo "Success! Predictions saved to: ${OUTPUT_FINAL}"
echo "================================================================================"
echo ""

# Display output info
poetry run python3 << EOF
import numpy as np

data = np.load('${OUTPUT_FINAL}', allow_pickle=True)
print("Output file contents:")
for key in data.keys():
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

pred = data['air_temperature']
print(f"\nPrediction statistics:")
print(f"  Min:  {float(pred.min()):.2f} °C")
print(f"  Max:  {float(pred.max()):.2f} °C")
print(f"  Mean: {float(pred.mean()):.2f} °C")
print(f"  Std:  {float(pred.std()):.2f} °C")
EOF

echo ""
echo "================================================================================"
echo "Work directory: ${WORK_DIR}"
echo "  (You can remove this directory after verification)"
echo "================================================================================"

################################################################################
# Run evaluation (if requested)
################################################################################

if [ "${RUN_EVALUATION}" = "true" ]; then
    echo ""
    echo "================================================================================"
    echo "Running Evaluation"
    echo "================================================================================"
    echo ""
    
    # Export variables for evaluation script
    export FIGS_DIR
    
    # Call evaluation script with same job
    bash "${ROOT_DIR}/Sh/evaluate_model.sh" \
        --predictions "${OUTPUT_NAME}.npz" \
        --ground-truth "${GROUND_TRUTH_FILE}" \
        --output-name "${OUTPUT_NAME}_eval" \
        --model-name "${MODEL_NAME}"
    
    EVAL_EXIT_CODE=$?
    
    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "WARNING: Evaluation failed with exit code ${EVAL_EXIT_CODE}"
        echo "Inference was successful but evaluation encountered an error."
        echo "You can run evaluation manually: bash Sh/evaluate_model.sh --predictions ${OUTPUT_NAME}.npz --model-name ${MODEL_NAME}"
    fi
else
    echo ""
    echo "Skipping evaluation (use --skip-evaluation flag to control this)"
    echo ""
    echo "To evaluate later, run:"
    echo "  bash Sh/evaluate_model.sh --predictions ${OUTPUT_NAME}.npz --model-name ${MODEL_NAME}"
fi

echo ""
echo "Inference completed at: $(date)"
echo "================================================================================"

exit 0

