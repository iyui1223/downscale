#!/bin/bash
################################################################################
# Wrapper script to submit preprocessing job to SLURM
#
# Usage:
#   ./submit_preprocess.sh
#
# This script submits the preprocessing job and monitors its status
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Submitting Preprocessing Job"
echo "=========================================="

# Check if config file exists
CONFIG_FILE="../Const/preprocess_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Configuration file not found: $CONFIG_FILE${NC}"
    echo "Please create the configuration file before submitting."
    exit 1
fi

echo -e "${GREEN}Configuration file found: $CONFIG_FILE${NC}"

# Check if SLURM script exists
SLURM_SCRIPT="preprocess_slurm.sh"
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo -e "${RED}ERROR: SLURM script not found: $SLURM_SCRIPT${NC}"
    exit 1
fi

# Create Log directory if it doesn't exist
mkdir -p ../Log

# Submit job
echo ""
echo "Submitting job to SLURM..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" 2>&1 | grep -oP "Submitted batch job \K\d+")

if [ -n "$JOB_ID" ]; then
    echo -e "${GREEN}Job submitted successfully!${NC}"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor job status with:"
    echo "  squeue -u \$USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View output logs:"
    echo "  tail -f ../Log/preprocess_${JOB_ID}.out"
    echo "  tail -f ../Log/preprocess_${JOB_ID}.err"
    echo ""
    echo "Cancel job with:"
    echo "  scancel $JOB_ID"
else
    echo -e "${RED}ERROR: Job submission failed${NC}"
    exit 1
fi

echo "=========================================="

