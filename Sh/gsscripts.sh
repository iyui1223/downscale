#!/bin/bash
################################################################################
# GrADS Evaluation Scripts Runner
#
# This script calls GrADS scripts to perform downscaling evaluation using
# binary files prepared by write_binary.py
#
# Usage:
#   bash Sh/gsscripts.sh --data-dir DIR --output-dir DIR [OPTIONS]
#
# Requirements:
#   - Binary files with .ctl descriptors prepared by write_binary.py
#   - GrADS executable
#
# Author: Climate Downscaling Team
# Date: November 2024
################################################################################

set -e
set -o pipefail

################################################################################
# Parse arguments
################################################################################

DATA_DIR=""
OUTPUT_DIR=""
GRADS_CMD="/home/yi260/rds/hpc-work/lib/opengrads/opengrads-2.2.1.oga.1/Contents/grads"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --grads-cmd)
            GRADS_CMD="$2"
            shift 2
            ;;
        --help)
            grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate
if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --data-dir and --output-dir are required"
    exit 1
fi

################################################################################
# Setup
################################################################################

echo "================================================================================"
echo "GrADS Evaluation Scripts"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Data directory:    $DATA_DIR"
echo "  Output directory:  $OUTPUT_DIR"
echo "  GrADS command:     $GRADS_CMD"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get GrADS scripts directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../GrADS" && pwd)"
GRADS_DIR="${SCRIPT_DIR}"

################################################################################
# Check required files
################################################################################

echo "Checking required files..."

# Check for .ctl files
PRED_CTL="${DATA_DIR}/predictions.ctl"
TRUTH_CTL="${DATA_DIR}/ground_truth.ctl"
ERA5_CTL="${DATA_DIR}/era5_input.ctl"

if [ ! -f "$PRED_CTL" ]; then
    echo "ERROR: predictions.ctl not found in $DATA_DIR"
    exit 1
fi

if [ ! -f "$TRUTH_CTL" ]; then
    echo "ERROR: ground_truth.ctl not found in $DATA_DIR"
    exit 1
fi

echo "  ✓ Found predictions.ctl"
echo "  ✓ Found ground_truth.ctl"

if [ -f "$ERA5_CTL" ]; then
    echo "  ✓ Found era5_input.ctl (will compute difference-based metrics)"
    HAS_ERA5=true
else
    echo "  ⚠ No era5_input.ctl (skipping difference-based metrics)"
    HAS_ERA5=false
fi

echo ""

################################################################################
# Run GrADS evaluation scripts
################################################################################

echo "================================================================================"
echo "Running GrADS Evaluation"
echo "================================================================================"
echo ""

# 1. Spatial maps
echo "1. Creating spatial maps..."
GRADS_SCRIPT="${SCRIPT_DIR}/evaluate_spatial_maps.gs"

if [ -f "$GRADS_SCRIPT" ]; then
    $GRADS_CMD -blc "$GRADS_SCRIPT $DATA_DIR $OUTPUT_DIR $GRADS_DIR" 
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Spatial maps created"
    else
        echo "  ✗ Failed to create spatial maps"
    fi
else
    echo "  ✗ Script not found: $GRADS_SCRIPT"
fi

echo ""

# 2. Temporal statistics
echo "2. Computing temporal statistics..."
GRADS_SCRIPT="${SCRIPT_DIR}/evaluate_temporal_stats.gs"

if [ -f "$GRADS_SCRIPT" ]; then
    $GRADS_CMD -blc "$GRADS_SCRIPT $DATA_DIR $OUTPUT_DIR $GRADS_DIR"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Temporal statistics computed"
    else
        echo "  ✗ Failed to compute temporal statistics"
    fi
else
    echo "  ✗ Script not found: $GRADS_SCRIPT"
fi

echo ""

# 3. Difference-based evaluation (if ERA5 available)
if [ "$HAS_ERA5" = true ]; then
    echo "3. Computing difference-based evaluation..."
    GRADS_SCRIPT="${SCRIPT_DIR}/evaluate_differences.gs"
    
    if [ -f "$GRADS_SCRIPT" ]; then
        $GRADS_CMD -blc "$GRADS_SCRIPT $DATA_DIR $OUTPUT_DIR $GRADS_DIR"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Difference-based evaluation completed"
        else
            echo "  ✗ Failed to compute difference-based evaluation"
        fi
    else
        echo "  ✗ Script not found: $GRADS_SCRIPT"
    fi
    
    echo ""
fi

################################################################################
# Summary
################################################################################

echo "================================================================================"
echo "GrADS Evaluation Complete"
echo "================================================================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "  No PNG files found"
echo ""
echo "================================================================================"

exit 0

