#!/bin/bash
#SBATCH --job-name=xgb_inference_eval
#SBATCH --account=cranmer-sl3-cpu
#SBATCH --partition=sapphire
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=../Log/inference_eval.out
#SBATCH --error=../Log/inference_eval.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yi260@cam.ac.uk

################################################################################
# F03: Downscaling Model Inference and Evaluation Wrapper Script (SLURM)
#
# This is the main SLURM wrapper script for running inference and evaluation.
# It provides flexible control over which steps to run:
#   - Inference only (generate downscaled predictions)
#   - Evaluation only (evaluate existing predictions)
#   - Both inference and evaluation (full pipeline)
#
# Works with any downscaling model type. Currently supports:
#   - XGBoost models (via inference_xgboost.sh)
#
# Usage:
#   sbatch Sh/F03_inference_evaluate_slurm.sh
#
# To check job status:
#   squeue -u $USER
#   tail -f Log/inference_eval_JOBID.out
################################################################################

set -e  # Exit on error
set -o pipefail

# Load environment settings FIRST (before configuration)
source ../Const/env_setting.sh

################################################################################
# USER CONFIGURATION SECTION - EDIT THESE PARAMETERS
################################################################################

# ============================================================================
# STEP CONTROL: Choose which steps to execute
# ============================================================================
RUN_INFERENCE="false"     # Set to "true" to run inference, "false" to skip
RUN_EVALUATION="true"    # Set to "true" to run evaluation, "false" to skip

# Examples:
#   RUN_INFERENCE="true"  RUN_EVALUATION="true"   → Run both (default)
#   RUN_INFERENCE="true"  RUN_EVALUATION="false"  → Inference only
#   RUN_INFERENCE="false" RUN_EVALUATION="true"   → Evaluation only

# ============================================================================
# INFERENCE CONFIGURATION (only used if RUN_INFERENCE="true")
# ============================================================================

# Model configuration (from env_setting.sh)
MODEL_NAME="${MODEL_NAME:-xgboost_downscale_tmax}"

# Input data for inference (from env_setting.sh, with fallbacks)
ERA5_DATA_FILE="${ERA5_DATA_FILE:-training_era5_tmax.npz}"
TARGET_GRID_FILE="${MSWX_DATA_FILE:-target_mswx_tmax.npz}"

# Output configuration
OUTPUT_NAME="downscaled_tmax"               # Base name for output predictions

# Processing parameters
CHUNK_SIZE="100"                            # Timesteps to process at once

# ============================================================================
# EVALUATION CONFIGURATION (only used if RUN_EVALUATION="true")
# ============================================================================

# If RUN_INFERENCE="true": predictions will be OUTPUT_NAME.npz (generated above)
# If RUN_INFERENCE="false": specify the existing predictions file to evaluate
PREDICTIONS_FILE="downscaled_tmax.npz"      # Predictions file to evaluate

# Ground truth data for evaluation (from env_setting.sh, with fallback)
GROUND_TRUTH_FILE="${MSWX_DATA_FILE:-target_mswx_tmax.npz}"

# Evaluation output name
EVAL_OUTPUT_NAME="downscaled_tmax_eval"     # Base name for evaluation outputs

################################################################################
# END OF USER CONFIGURATION SECTION
################################################################################

echo "================================================================================"
echo "F03: Downscaling Model Inference and Evaluation Pipeline"
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

# Display configuration
echo "Pipeline Configuration:"
echo "  Run Inference:       ${RUN_INFERENCE}"
echo "  Run Evaluation:      ${RUN_EVALUATION}"
echo ""

# Validate configuration
if [ "${RUN_INFERENCE}" != "true" ] && [ "${RUN_EVALUATION}" != "true" ]; then
    echo "ERROR: Both RUN_INFERENCE and RUN_EVALUATION are set to false."
    echo "At least one step must be enabled. Please edit the configuration section."
    exit 1
fi

# Change to root directory (env_setting.sh already sourced at top)
cd ${ROOT_DIR}

# Set OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"
echo ""

################################################################################
# STEP 1: INFERENCE (if enabled)
################################################################################

if [ "${RUN_INFERENCE}" = "true" ]; then
    echo "================================================================================"
    echo "STEP 1/2: Running Inference"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  Model:               ${MODEL_NAME}"
    echo "  ERA5 data:           ${ERA5_DATA_FILE}"
    echo "  Target grid:         ${TARGET_GRID_FILE}"
    echo "  Output:              ${OUTPUT_NAME}.npz"
    echo "  Chunk size:          ${CHUNK_SIZE}"
    echo ""
    echo "================================================================================"
    echo ""
    
    START_TIME_INFERENCE=$(date +%s)
    
    # Export inference parameters
    export MODEL_NAME
    export ERA5_DATA_FILE
    export TARGET_GRID_FILE
    export OUTPUT_NAME
    export CHUNK_SIZE
    export GROUND_TRUTH_FILE
    export RUN_EVALUATION="false"  # We handle evaluation separately in F03
    
    # Run inference script (without evaluation - we'll handle it separately)
    bash Sh/inference_xgboost.sh \
        --era5-data "${ERA5_DATA_FILE}" \
        --output-name "${OUTPUT_NAME}" \
        --chunk-size "${CHUNK_SIZE}" \
        --ground-truth "${GROUND_TRUTH_FILE}" \
        --skip-evaluation
    
    INFERENCE_EXIT_CODE=$?
    
    END_TIME_INFERENCE=$(date +%s)
    ELAPSED_INFERENCE=$((END_TIME_INFERENCE - START_TIME_INFERENCE))
    ELAPSED_MIN_INFERENCE=$((ELAPSED_INFERENCE / 60))
    ELAPSED_SEC_INFERENCE=$((ELAPSED_INFERENCE % 60))
    
    echo ""
    echo "================================================================================"
    
    if [ $INFERENCE_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Inference failed with exit code ${INFERENCE_EXIT_CODE}"
        echo "================================================================================"
        exit $INFERENCE_EXIT_CODE
    fi
    
    echo "✓ Inference completed successfully in ${ELAPSED_MIN_INFERENCE}m ${ELAPSED_SEC_INFERENCE}s"
    echo "================================================================================"
    echo ""
    
    # Update predictions file for evaluation (if needed)
    if [ "${RUN_EVALUATION}" = "true" ]; then
        PREDICTIONS_FILE="${OUTPUT_NAME}.npz"
    fi
else
    echo "================================================================================"
    echo "STEP 1/2: Skipping Inference (disabled in configuration)"
    echo "================================================================================"
    echo ""
fi

################################################################################
# STEP 2: EVALUATION (if enabled)
################################################################################

if [ "${RUN_EVALUATION}" = "true" ]; then
    echo "================================================================================"
    echo "STEP 2/2: Running Evaluation"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  Predictions:         ${PREDICTIONS_FILE}"
    echo "  Ground truth:        ${GROUND_TRUTH_FILE}"
    echo "  Output name:         ${EVAL_OUTPUT_NAME}"
    echo ""
    echo "================================================================================"
    echo ""
    
    START_TIME_EVAL=$(date +%s)
    
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
    
    # Run evaluation script (now model-agnostic)
    bash Sh/evaluate_model.sh \
        --predictions "${PREDICTIONS_FILE}" \
        --ground-truth "${GROUND_TRUTH_FILE}" \
        --era5-input "${ERA5_DATA_FILE}" \
        --model-name "${MODEL_NAME}" \
        --output-name "${EVAL_OUTPUT_NAME}"
    
    EVAL_EXIT_CODE=$?
    
    END_TIME_EVAL=$(date +%s)
    ELAPSED_EVAL=$((END_TIME_EVAL - START_TIME_EVAL))
    ELAPSED_MIN_EVAL=$((ELAPSED_EVAL / 60))
    ELAPSED_SEC_EVAL=$((ELAPSED_EVAL % 60))
    
    echo ""
    echo "================================================================================"
    
    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Evaluation failed with exit code ${EVAL_EXIT_CODE}"
        echo "================================================================================"
        exit $EVAL_EXIT_CODE
    fi
    
    echo "✓ Evaluation completed successfully in ${ELAPSED_MIN_EVAL}m ${ELAPSED_SEC_EVAL}s"
    echo "================================================================================"
    echo ""
else
    echo "================================================================================"
    echo "STEP 2/2: Skipping Evaluation (disabled in configuration)"
    echo "================================================================================"
    echo ""
fi

################################################################################
# SUMMARY
################################################################################

echo ""
echo "================================================================================"
echo "Pipeline Summary"
echo "================================================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo ""

if [ "${RUN_INFERENCE}" = "true" ]; then
    echo "Inference:"
    echo "  Status:        ✓ Completed"
    echo "  Exit code:     ${INFERENCE_EXIT_CODE}"
    echo "  Time:          ${ELAPSED_MIN_INFERENCE}m ${ELAPSED_SEC_INFERENCE}s"
    echo "  Output:        Data/Downscaled/${MODEL_NAME}/${OUTPUT_NAME}.npz"
    echo ""
fi

if [ "${RUN_EVALUATION}" = "true" ]; then
    echo "Evaluation:"
    echo "  Status:        ✓ Completed"
    echo "  Exit code:     ${EVAL_EXIT_CODE}"
    echo "  Time:          ${ELAPSED_MIN_EVAL}m ${ELAPSED_SEC_EVAL}s"
    echo "  Metrics:       Data/Downscaled/${MODEL_NAME}/${EVAL_OUTPUT_NAME}_metrics.yaml"
    echo "  Summary:       Data/Downscaled/${MODEL_NAME}/${EVAL_OUTPUT_NAME}_summary.txt"
    echo "  Figures:       Figs/F03_inference_evaluate/${MODEL_NAME}/"
    echo ""
fi

# Calculate total time
TOTAL_ELAPSED=0
if [ "${RUN_INFERENCE}" = "true" ]; then
    TOTAL_ELAPSED=$((TOTAL_ELAPSED + ELAPSED_INFERENCE))
fi
if [ "${RUN_EVALUATION}" = "true" ]; then
    TOTAL_ELAPSED=$((TOTAL_ELAPSED + ELAPSED_EVAL))
fi
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo "Total time:      ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "Finished:        $(date)"
echo "================================================================================"
echo ""

# Final status
if [ "${RUN_INFERENCE}" = "true" ] && [ "${RUN_EVALUATION}" = "true" ]; then
    echo "✓ Full pipeline completed successfully!"
elif [ "${RUN_INFERENCE}" = "true" ]; then
    echo "✓ Inference completed successfully!"
    echo "To evaluate: Edit F03_inference_evaluate_slurm.sh and set RUN_INFERENCE=\"false\", RUN_EVALUATION=\"true\""
elif [ "${RUN_EVALUATION}" = "true" ]; then
    echo "✓ Evaluation completed successfully!"
fi

echo ""
echo "================================================================================"

exit 0

