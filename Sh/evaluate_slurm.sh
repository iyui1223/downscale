#!/bin/bash
#SBATCH --job-name=xgb_evaluate
#SBATCH --account=cranmer-sl3-cpu
#SBATCH --partition=sapphire
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=../Log/evaluate.out
#SBATCH --error=../Log/evaluate.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yi260@cam.ac.uk

################################################################################
# F03b: Downscaling Model Evaluation (SLURM Job)
#
# This script runs evaluation only. It evaluates existing predictions against
# ground truth data and generates metrics and visualizations.
# Submitted as part of the F03 pipeline via F03_inference_evaluate.sh
#
# Can also be run standalone:
#   sbatch Sh/F03b_evaluate_slurm.sh
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

# Input data for evaluation
ERA5_DATA_FILE="${ERA5_DATA_FILE:-training_era5_tmax.npz}"
PREDICTIONS_FILE="${PREDICTIONS_FILE:-downscaled_tmax.npz}"
GROUND_TRUTH_FILE="${MSWX_DATA_FILE:-target_mswx_tmax.npz}"

# Output configuration
EVAL_OUTPUT_NAME="${EVAL_OUTPUT_NAME:-downscaled_tmax_eval}"

################################################################################
# MAIN EXECUTION
################################################################################

echo "================================================================================"
echo "F03b: Downscaling Model Evaluation"
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
echo "  Predictions:         ${PREDICTIONS_FILE}"
echo "  Ground truth:        ${GROUND_TRUTH_FILE}"
echo "  ERA5 input:          ${ERA5_DATA_FILE}"
echo "  Output name:         ${EVAL_OUTPUT_NAME}"
echo ""

# Change to root directory
cd ${ROOT_DIR}

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"
echo ""

echo "================================================================================"
echo "Running Evaluation"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Export environment variables for evaluate script
export ROOT_DIR
export PROCESSED_DIR
export INTERMEDIATE_DIR
export TRAINING_DATA_DIR
export TARGET_DATA_DIR
export FIGS_DIR

# Export evaluation parameters
export MODEL_NAME
export PREDICTIONS_FILE
export GROUND_TRUTH_FILE
export ERA5_INPUT_FILE="${ERA5_DATA_FILE}"
export OUTPUT_NAME="${EVAL_OUTPUT_NAME}"

# Enable memory profiling
export MEMORY_PROFILE=1

# Run evaluation script
bash Sh/evaluate_model.sh \
    --predictions "${PREDICTIONS_FILE}" \
    --ground-truth "${GROUND_TRUTH_FILE}" \
    --era5-input "${ERA5_DATA_FILE}" \
    --model-name "${MODEL_NAME}" \
    --output-name "${EVAL_OUTPUT_NAME}"

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "================================================================================"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code ${EXIT_CODE}"
    echo "================================================================================"
    exit $EXIT_CODE
fi

echo "✓ Evaluation completed successfully in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Metrics:   Data/Downscaled/${MODEL_NAME}/${EVAL_OUTPUT_NAME}_metrics.yaml"
echo "  Summary:   Data/Downscaled/${MODEL_NAME}/${EVAL_OUTPUT_NAME}_summary.txt"
echo "  Figures:   Figs/F03_inference_evaluate/${MODEL_NAME}/"
echo "Finished: $(date)"
echo "================================================================================"

exit 0

