#!/bin/bash
#SBATCH --job-name=ds_train
#SBATCH --account=cranmer-sl3-cpu
#SBATCH --partition=sapphire
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=../Log/train.out
#SBATCH --error=../Log/train.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yi260@cam.ac.uk

################################################################################
# SLURM Job Script for Downscaling Model Training
#
# This script submits the model training to the HPC cluster.
# Model type is configured in Const/env_setting.sh (MODEL_TYPE variable).
#
# Supported model types:
#   - xgboost: XGBoost gradient boosting (default)
#   - cnn: Convolutional Neural Network (future)
#   - unet: U-Net architecture (future)
#
# Usage:
#   sbatch Sh/F02_train_slurm.sh
#
# Or with custom parameters:
#   sbatch Sh/F02_train_slurm.sh --sample-ratio 0.2 --max-depth 10
#
# To check job status:
#   squeue -u $USER
#   tail -f Log/train_JOBID.out
################################################################################

# Load environment settings FIRST
source ../Const/env_setting.sh

echo "================================================================================"
echo "Downscaling Model Training Job"
echo "================================================================================"
echo "Model type:          ${MODEL_TYPE}"
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

# Change to root directory (env_setting.sh already sourced at top)
cd ${ROOT_DIR}

# Set OpenMP threads for XGBoost
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"
echo ""

# Display system information
echo "System information:"
echo "  Hostname:    $(hostname)"
echo "  CPU info:    $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  Total CPUs:  $(nproc)"
echo "  Memory:      $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Display Python environment
echo "Python environment:"
poetry run python3 --version
echo ""

echo "Key package versions:"
poetry run python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>/dev/null || echo "  numpy: not available"
poetry run python3 -c "import scipy; print(f'  scipy: {scipy.__version__}')" 2>/dev/null || echo "  scipy: not available"
poetry run python3 -c "import xgboost; print(f'  xgboost: {xgboost.__version__}')" 2>/dev/null || echo "  xgboost: not available"
poetry run python3 -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "  scikit-learn: not available"
echo ""

################################################################################
# Run training based on MODEL_TYPE
################################################################################

echo "================================================================================"
echo "Starting ${MODEL_TYPE} training script..."
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Execute the appropriate training script based on MODEL_TYPE
case "${MODEL_TYPE}" in
    xgboost)
        echo "Running XGBoost training..."
        bash Sh/train_xgboost.sh "$@"
        ;;
    cnn)
        echo "Running CNN training..."
        # bash Sh/train_cnn.sh "$@"  # Uncomment when implemented
        echo "ERROR: CNN training not yet implemented"
        exit 1
        ;;
    unet)
        echo "Running U-Net training..."
        # bash Sh/train_unet.sh "$@"  # Uncomment when implemented
        echo "ERROR: U-Net training not yet implemented"
        exit 1
        ;;
    *)
        echo "ERROR: Unknown model type '${MODEL_TYPE}'"
        echo "Supported types: xgboost, cnn, unet"
        exit 1
        ;;
esac

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

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
    echo "✓ Training completed successfully!"
    echo ""
    echo "Model type:     ${MODEL_TYPE}"
    echo "Model location: ${ROOT_DIR}/Models/${MODEL_NAME}/"
    echo ""
else
    echo ""
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "Check log files for details."
    echo ""
fi

exit $EXIT_CODE

