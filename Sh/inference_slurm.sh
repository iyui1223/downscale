#!/bin/bash
#SBATCH --job-name=xgb_inference
#SBATCH --account=cranmer-sl3-cpu
#SBATCH --partition=sapphire
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=../Log/inference.out
#SBATCH --error=../Log/inference.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yi260@cam.ac.uk

################################################################################
# F03a: Downscaling Model Inference (SLURM Job)
#
# This script runs inference only. It generates downscaled predictions.
# Submitted as part of the F03 pipeline via F03_inference_evaluate.sh
#
# Can also be run standalone:
#   sbatch Sh/F03a_inference_slurm.sh
################################################################################

set -e  # Exit on error
set -o pipefail

# Load environment settings
source ../Const/env_setting.sh

################################################################################
# CONFIGURATION (can be overridden via environment variables)
################################################################################

# Model configuration
MODEL_NAME="${MODEL_NAME:-xgboost_downscale_tmax}"

# Input data for inference
ERA5_DATA_FILE="${ERA5_DATA_FILE:-training_era5_tmax.npz}"
TARGET_GRID_FILE="${MSWX_DATA_FILE:-target_mswx_tmax.npz}"

# Output configuration
OUTPUT_NAME="${OUTPUT_NAME:-downscaled_tmax}"

# Processing parameters
CHUNK_SIZE="${CHUNK_SIZE:-100}"

# Ground truth for reference
GROUND_TRUTH_FILE="${MSWX_DATA_FILE:-target_mswx_tmax.npz}"

################################################################################
# MAIN EXECUTION
################################################################################

echo "================================================================================"
echo "F03a: Downscaling Model Inference"
echo "================================================================================"
echo "Job ID:              $SLURM_JOB_ID"
echo "Job name:            $SLURM_JOB_NAME"
echo "Node:                $SLURM_NODELIST"
echo "CPUs per task:       $SLURM_CPUS_PER_TASK"
echo "Memory:              $SLURM_MEM_PER_NODE MB"
echo "Working directory:   $(pwd)"
echo "Started:             $(date)"
echo "================================================================================"
echo ""

echo "Configuration:"
echo "  Model:               ${MODEL_NAME}"
echo "  ERA5 data:           ${ERA5_DATA_FILE}"
echo "  Target grid:         ${TARGET_GRID_FILE}"
echo "  Output:              ${OUTPUT_NAME}.npz"
echo "  Chunk size:          ${CHUNK_SIZE}"
echo ""

# Change to root directory
cd ${ROOT_DIR}

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"
echo ""

echo "================================================================================"
echo "Running Inference"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Export inference parameters
export MODEL_NAME
export ERA5_DATA_FILE
export TARGET_GRID_FILE
export OUTPUT_NAME
export CHUNK_SIZE
export GROUND_TRUTH_FILE
export RUN_EVALUATION="false"

# Run inference script (without evaluation)
bash Sh/inference_xgboost.sh \
    --era5-data "${ERA5_DATA_FILE}" \
    --output-name "${OUTPUT_NAME}" \
    --chunk-size "${CHUNK_SIZE}" \
    --ground-truth "${GROUND_TRUTH_FILE}" \
    --skip-evaluation

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "================================================================================"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Inference failed with exit code ${EXIT_CODE}"
    echo "================================================================================"
    exit $EXIT_CODE
fi

echo "✓ Inference completed successfully in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Output: Data/Downscaled/${MODEL_NAME}/${OUTPUT_NAME}.npz"
echo "Finished: $(date)"
echo "================================================================================"

exit 0

