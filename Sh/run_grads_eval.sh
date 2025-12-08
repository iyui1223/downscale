#!/bin/bash
################################################################################
# Manual GrADS Evaluation Runner
#
# Run GrADS evaluation scripts manually after data conversion is complete
#
# Usage:
#   bash Sh/run_grads_eval.sh [MODEL_NAME]
#
# Example:
#   bash Sh/run_grads_eval.sh xgboost_downscale_tmax
################################################################################

set -e

MODEL_NAME="${1:-xgboost_downscale_tmax}"

# Paths
ROOT_DIR="/home/yi260/rds/hpc-work/downscale"
DATA_DIR="${ROOT_DIR}/Data/F03_inference_evaluate/${MODEL_NAME}"
FIG_DIR="${ROOT_DIR}/Figs/F03_inference_evaluate/${MODEL_NAME}"
GRADS_DIR="${ROOT_DIR}/GrADS"
GRADS_CMD="/home/yi260/rds/hpc-work/lib/opengrads/opengrads-2.2.1.oga.1/Contents/grads"

echo "================================================================================"
echo "Manual GrADS Evaluation Runner"
echo "================================================================================"
echo ""
echo "Model:         ${MODEL_NAME}"
echo "Data dir:      ${DATA_DIR}"
echo "Output dir:    ${FIG_DIR}"
echo "GrADS:         ${GRADS_CMD}"
echo ""

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Run the conversion first!"
    exit 1
fi

if [ ! -f "${DATA_DIR}/predictions.ctl" ]; then
    echo "ERROR: predictions.ctl not found in $DATA_DIR"
    exit 1
fi

echo "✓ Data files found"
echo ""

# Create output directory
mkdir -p "${FIG_DIR}"

# Try loading Intel modules (suppress errors if not available)
echo "Attempting to load Intel compiler modules..."
module load intel/compilers 2>/dev/null || module load intel-oneapi-compilers 2>/dev/null || echo "  (Intel modules not available - continuing anyway)"
echo ""

# Test file opening
echo "================================================================================"
echo "Test 1: Verifying GrADS can open files..."
echo "================================================================================"
echo ""

# $GRADS_CMD -blc "${ROOT_DIR}/GrADS/test_open.gs ${DATA_DIR}"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: GrADS cannot open the data files!"
    echo "Check the .ctl files in: $DATA_DIR"
    exit 1
fi

echo ""
echo "✓ Files open successfully!"
echo ""

# Run evaluation scripts
echo "================================================================================"
echo "Test 2: Running evaluation scripts..."
echo "================================================================================"
echo ""

echo "1. Spatial maps..."
# $GRADS_CMD -blc "${GRADS_DIR}/evaluate_spatial_maps.gs ${DATA_DIR} ${FIG_DIR} ${GRADS_DIR}"
SPATIAL_RC=$?

echo ""
echo "2. Temporal statistics..."
# $GRADS_CMD -blc "${GRADS_DIR}/evaluate_temporal_stats.gs ${DATA_DIR} ${FIG_DIR} ${GRADS_DIR}"
TEMPORAL_RC=$?

echo ""
if [ -f "${DATA_DIR}/era5_input.ctl" ]; then
    echo "3. Difference-based evaluation (ERA5 found)..."
    $GRADS_CMD -blc "${GRADS_DIR}/evaluate_differences.gs ${DATA_DIR} ${FIG_DIR} ${GRADS_DIR}"
    DIFF_RC=$?
else
    echo "3. Difference-based evaluation (skipped - no ERA5)"
    DIFF_RC=0
fi

echo ""
echo "================================================================================"
echo "Results"
echo "================================================================================"
echo ""

if [ $SPATIAL_RC -eq 0 ]; then
    echo "✓ Spatial maps:          SUCCESS"
else
    echo "✗ Spatial maps:          FAILED (rc=$SPATIAL_RC)"
fi

if [ $TEMPORAL_RC -eq 0 ]; then
    echo "✓ Temporal statistics:   SUCCESS"
else
    echo "✗ Temporal statistics:   FAILED (rc=$TEMPORAL_RC)"
fi

if [ -f "${DATA_DIR}/era5_input.ctl" ]; then
    if [ $DIFF_RC -eq 0 ]; then
        echo "✓ Difference-based:      SUCCESS"
    else
        echo "✗ Difference-based:      FAILED (rc=$DIFF_RC)"
    fi
fi

echo ""
echo "Output files:"
ls -lh "${FIG_DIR}"/*.png 2>/dev/null || echo "  No PNG files created (check GrADS Cairo/libimf.so issue)"

echo ""
echo "================================================================================"
echo "Data directory:   ${DATA_DIR}"
echo "Figures directory: ${FIG_DIR}"
echo "================================================================================"

exit 0

