#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=cranmer-sl3-cpu
#SBATCH --partition=sapphire
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=../Log/preprocess.out
#SBATCH --error=../Log/preprocess.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yi260@cam.ac.uk

################################################################################
# Memory-Safe SLURM Job Script for Preprocessing Downscaling Data
#
# This script uses a two-stage approach:
#   1. CDO to merge and slice NetCDF files
#   2. Python to convert to compressed format (working with single files)
#
# Usage:
#   sbatch Sh/F01_preprocess_slurm.sh
################################################################################

echo "=========================================="
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load environment settings
source ../Const/env_setting.sh

# Change to root directory
cd ${ROOT_DIR}

# Note: CDO module is loaded within cdo_preprocess.sh for self-contained execution

################################################################################
# STAGE 1: CDO Preprocessing
################################################################################
echo ""
echo "=========================================="
echo "STAGE 1: CDO Preprocessing"
echo "=========================================="
echo ""

START_TIME_CDO=$(date +%s)

bash Sh/cdo_preprocess.sh

EXIT_CODE_CDO=$?

END_TIME_CDO=$(date +%s)
ELAPSED_CDO=$((END_TIME_CDO - START_TIME_CDO))
ELAPSED_MIN_CDO=$((ELAPSED_CDO / 60))
ELAPSED_SEC_CDO=$((ELAPSED_CDO % 60))

if [ $EXIT_CODE_CDO -ne 0 ]; then
    echo "ERROR: CDO preprocessing failed with exit code $EXIT_CODE_CDO"
    exit $EXIT_CODE_CDO
fi

echo ""
echo "CDO preprocessing completed in ${ELAPSED_MIN_CDO}m ${ELAPSED_SEC_CDO}s"

################################################################################
# STAGE 2: Python Preprocessing
################################################################################
echo ""
echo "=========================================="
echo "STAGE 2: Python Preprocessing"
echo "=========================================="
echo ""

# Activate Poetry environment
echo "Activating Poetry environment..."
poetry install --no-interaction --quiet

# Set OpenMP threads for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"

# Reduce Dask memory usage
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.6  # 60% of memory
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.75  # Spill at 75%
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.85  # Pause at 85%
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95  # Terminate at 95%

echo ""
echo "Python version:"
poetry run python --version

echo ""
echo "Key package versions:"
poetry run python -c "import xarray; print(f'xarray: {xarray.__version__}')" || echo "xarray not available"
poetry run python -c "import numpy; print(f'numpy: {numpy.__version__}')" || echo "numpy not available"
poetry run python -c "import netCDF4; print(f'netCDF4: {netCDF4.__version__}')" || echo "netCDF4 not available"

echo ""
echo "Starting Python conversion..."

if [ ! -f "$PREPROCESS_CONFIG" ]; then
    echo "ERROR: Configuration file not found: $PREPROCESS_CONFIG"
    exit 1
fi

START_TIME_PY=$(date +%s)

poetry run python Python/preprocess_data.py \
    --config "$PREPROCESS_CONFIG"

EXIT_CODE_PY=$?

END_TIME_PY=$(date +%s)
ELAPSED_PY=$((END_TIME_PY - START_TIME_PY))
ELAPSED_MIN_PY=$((ELAPSED_PY / 60))
ELAPSED_SEC_PY=$((ELAPSED_PY % 60))

echo ""
echo "=========================================="
echo "All preprocessing completed!"
echo "=========================================="
echo "CDO stage:    ${ELAPSED_MIN_CDO}m ${ELAPSED_SEC_CDO}s (exit code: $EXIT_CODE_CDO)"
echo "Python stage: ${ELAPSED_MIN_PY}m ${ELAPSED_SEC_PY}s (exit code: $EXIT_CODE_PY)"
echo ""

TOTAL_ELAPSED=$((ELAPSED_CDO + ELAPSED_PY))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))
echo "Total time:   ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "=========================================="

# Print output file information
echo ""
echo "Intermediate files (CDO output):"
ls -lh ${ROOT_DIR}/Data/Intermediate/ 2>/dev/null || echo "No intermediate files found"

echo ""
echo "Final output files:"
ls -lh ${ROOT_DIR}/Data/Processed/ 2>/dev/null || echo "No output files found"

echo ""
echo "Job finished on $(date)"
echo "=========================================="

exit $EXIT_CODE_PY

