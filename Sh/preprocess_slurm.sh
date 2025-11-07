#!/bin/bash
#SBATCH --job-name=preprocess_downscale
#SBATCH --account=cranmer-sl3-cpu
#SBATCH --partition=cclake
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=../Log/preprocess_%j.out
#SBATCH --error=../Log/preprocess_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

################################################################################
# SLURM Job Script for Preprocessing Downscaling Data
#
# This script preprocesses NetCDF files for the XGBoost downscaling system.
# It slices spatiotemporal data and converts to compressed numpy format.
#
# Usage:
#   sbatch preprocess_slurm.sh
#
# Customization:
#   - Adjust memory (--mem) based on dataset size
#   - Adjust time limit (--time) for large datasets
#   - Modify cpus-per-task for parallel processing
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

# Load required modules (adjust for your HPC system)
module purge
module load python/3.10
# module load netcdf/4.9.0  # Uncomment if NetCDF module is needed
# module load hdf5/1.12.0   # Uncomment if HDF5 module is needed

echo "Loaded modules:"
module list

# Activate Poetry environment
echo ""
echo "Activating Poetry environment..."
poetry install --no-interaction --quiet

# Set OpenMP threads for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"

# Print Python environment info
echo ""
echo "Python version:"
poetry run python --version

echo ""
echo "Key package versions:"
poetry run python -c "import xarray; print(f'xarray: {xarray.__version__}')"
poetry run python -c "import numpy; print(f'numpy: {numpy.__version__}')"
poetry run python -c "import netCDF4; print(f'netCDF4: {netCDF4.__version__}')"

# Run preprocessing script
echo ""
echo "=========================================="
echo "Starting data preprocessing..."
echo "=========================================="
echo ""

CONFIG_FILE="${ROOT_DIR}/Const/preprocess_config.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    echo "Please create the configuration file before running this script."
    exit 1
fi

# Run preprocessing with time measurement
START_TIME=$(date +%s)

poetry run python Python/preprocess_data.py \
    --config "$CONFIG_FILE"

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "Preprocessing completed!"
echo "Exit code: $EXIT_CODE"
echo "Elapsed time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "=========================================="

# Print output file information
echo ""
echo "Output files in ${ROOT_DIR}/Data/Processed:"
ls -lh ${ROOT_DIR}/Data/Processed/ 2>/dev/null || echo "No files found (check if directory exists)"

echo ""
echo "Job finished on $(date)"
echo "=========================================="

exit $EXIT_CODE

