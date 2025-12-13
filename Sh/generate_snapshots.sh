#!/bin/bash
################################################################################
# Generate Snapshots Script
#
# Core logic for generating temperature snapshots for extreme events.
# Loops through events defined in snapshot_events.yaml and generates
# temperature, anomaly, and difference maps using GrADS.
#
# Usage:
#   bash Sh/generate_snapshots.sh [OPTIONS]
#
# Options:
#   --data-dir DIR      Input data directory (required)
#   --output-dir DIR    Output directory for images (required)
#   --events-file FILE  Path to snapshot_events.yaml (default: Const/snapshot_events.yaml)
#   --event NAME        Process only this event (optional)
#   --grads-cmd PATH    Path to GrADS executable
#
# Author: Climate Downscaling Team
# Date: December 2024
################################################################################

set -e
set -o pipefail

################################################################################
# Default Configuration
################################################################################

DATA_DIR=""
OUTPUT_DIR=""
EVENTS_FILE="${ROOT_DIR:-$(pwd)}/Const/snapshot_events.yaml"
SINGLE_EVENT=""
GRADS_CMD="${GRADS_CMD:-/home/yi260/rds/hpc-work/lib/opengrads/opengrads-2.2.1.oga.1/Contents/grads}"
GRADS_DIR="${ROOT_DIR:-$(pwd)}/GrADS"

################################################################################
# Parse Arguments
################################################################################

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
        --events-file)
            EVENTS_FILE="$2"
            shift 2
            ;;
        --event)
            SINGLE_EVENT="$2"
            shift 2
            ;;
        --grads-cmd)
            GRADS_CMD="$2"
            shift 2
            ;;
        --help|-h)
            grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --data-dir and --output-dir are required"
    exit 1
fi

if [ ! -f "$EVENTS_FILE" ]; then
    echo "ERROR: Events file not found: $EVENTS_FILE"
    exit 1
fi

################################################################################
# Setup
################################################################################

echo "================================================================================"
echo "Generate Snapshots"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Data directory:   $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Events file:      $EVENTS_FILE"
echo "  GrADS command:    $GRADS_CMD"
echo "  GrADS scripts:    $GRADS_DIR"
if [ -n "$SINGLE_EVENT" ]; then
    echo "  Single event:     $SINGLE_EVENT"
fi
echo ""

# Try loading Intel modules for GrADS Cairo support
echo "Loading Intel modules..."
module load intel/compilers 2>/dev/null || module load intel-oneapi-compilers 2>/dev/null || echo "  (Intel modules not available)"
echo ""

################################################################################
# Helper Function: Generate dates between start and end
################################################################################

generate_dates() {
    local start_date=$1
    local end_date=$2
    
    # Convert to seconds since epoch, generate sequence, convert back
    local start_sec=$(date -d "$start_date" +%s)
    local end_sec=$(date -d "$end_date" +%s)
    local current_sec=$start_sec
    
    while [ $current_sec -le $end_sec ]; do
        date -d "@$current_sec" +%Y%m%d
        current_sec=$((current_sec + 86400))  # Add one day
    done
}

################################################################################
# Helper Function: Parse YAML and extract events
################################################################################

parse_events() {
    # Simple YAML parser using Python (more reliable than bash)
    python3 << 'PYTHON_SCRIPT'
import yaml
import sys
import os

events_file = os.environ.get('EVENTS_FILE', 'Const/snapshot_events.yaml')
single_event = os.environ.get('SINGLE_EVENT', '')

with open(events_file, 'r') as f:
    config = yaml.safe_load(f)

for event in config.get('events', []):
    name = event.get('name', '')
    if single_event and name != single_event:
        continue
    
    start_date = event.get('start_date', '')
    end_date = event.get('end_date', '')
    description = event.get('description', '')
    event_type = event.get('type', '')
    
    # Output: name|start_date|end_date|description|type
    print(f"{name}|{start_date}|{end_date}|{description}|{event_type}")
PYTHON_SCRIPT
}

################################################################################
# Main Processing Loop
################################################################################

# Export for Python script
export EVENTS_FILE
export SINGLE_EVENT

echo "================================================================================"
echo "Processing Events"
echo "================================================================================"
echo ""

# Track statistics
total_events=0
total_snapshots=0
failed_snapshots=0

# Parse events and loop
while IFS='|' read -r event_name start_date end_date description event_type; do
    if [ -z "$event_name" ]; then
        continue
    fi
    
    total_events=$((total_events + 1))
    
    echo "--------------------------------------------------------------------------------"
    echo "Event: $description"
    echo "  Name:   $event_name"
    echo "  Type:   $event_type"
    echo "  Period: $start_date to $end_date"
    echo "--------------------------------------------------------------------------------"
    echo ""
    
    # Create event output directory
    event_output_dir="${OUTPUT_DIR}/${event_name}"
    mkdir -p "$event_output_dir"
    
    # Generate dates for this event
    dates=$(generate_dates "$start_date" "$end_date")
    
    for date_str in $dates; do
        echo "  Processing date: $date_str"
        
        # Generate temperature/anomaly maps for each dataset
        for dataset in era5 truth pred; do
            echo "    - ${dataset} temperature/anomaly..."
            
            $GRADS_CMD -blc "${GRADS_DIR}/snapshot_temperature.gs $DATA_DIR $event_output_dir $GRADS_DIR $date_str $dataset" 2>&1 | grep -E "(Saving|ERROR)" || true
            
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                total_snapshots=$((total_snapshots + 2))  # temp + anom
            else
                failed_snapshots=$((failed_snapshots + 1))
            fi
        done
        
        # Generate difference maps
        echo "    - difference maps..."
        $GRADS_CMD -blc "${GRADS_DIR}/snapshot_difference.gs $DATA_DIR $event_output_dir $GRADS_DIR $date_str" 2>&1 | grep -E "(Saving|ERROR)" || true
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            total_snapshots=$((total_snapshots + 2))  # pred-truth + era5-truth
        else
            failed_snapshots=$((failed_snapshots + 1))
        fi
        
        echo ""
    done
    
    # Count images for this event
    event_images=$(ls -1 "${event_output_dir}"/*.png 2>/dev/null | wc -l)
    echo "  ✓ Event complete: $event_images images created"
    echo ""
    
done < <(parse_events)

################################################################################
# Generate Viewer Configuration
################################################################################

echo "================================================================================"
echo "Generating Viewer Configuration"
echo "================================================================================"
echo ""

# Create viewer directory
viewer_dir="${OUTPUT_DIR}/viewer"
mkdir -p "$viewer_dir"

# Generate viewer_config.json using Python
python3 << PYTHON_CONFIG
import json
import os
from pathlib import Path

output_dir = "$OUTPUT_DIR"
config = {"events": [], "generated": "$(date -Iseconds)"}

for event_dir in sorted(Path(output_dir).iterdir()):
    if event_dir.is_dir() and event_dir.name != "viewer":
        event_name = event_dir.name
        images = sorted([f.name for f in event_dir.glob("*.png")])
        
        # Group by date
        dates = {}
        for img in images:
            # Parse: {dataset}_{type}_{YYYYMMDD}.png
            parts = img.replace('.png', '').split('_')
            if len(parts) >= 3:
                date_str = parts[-1]
                if date_str not in dates:
                    dates[date_str] = []
                dates[date_str].append(img)
        
        config["events"].append({
            "name": event_name,
            "dates": dates,
            "image_count": len(images)
        })

with open(f"{output_dir}/viewer/viewer_config.json", 'w') as f:
    json.dump(config, f, indent=2)

print(f"Created viewer_config.json with {len(config['events'])} events")
PYTHON_CONFIG

################################################################################
# Summary
################################################################################

echo ""
echo "================================================================================"
echo "Snapshot Generation Complete"
echo "================================================================================"
echo ""
echo "Statistics:"
echo "  Events processed:   $total_events"
echo "  Snapshots created:  $total_snapshots"
echo "  Failed:             $failed_snapshots"
echo ""
echo "Output directories:"
ls -d "${OUTPUT_DIR}"/*/ 2>/dev/null | head -10
echo ""
echo "Viewer config: ${OUTPUT_DIR}/viewer/viewer_config.json"
echo "================================================================================"

exit 0

