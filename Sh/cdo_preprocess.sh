#!/bin/bash
################################################################################
# CDO Pre-processing Script for Downscaling Data
#
# This script processes climate data files with flexible spatial subsetting:
#   - Spatial subsetting is OPTIONAL and can be enabled/disabled
#   - When enabled, subsetting is applied during batch processing for efficiency
#   - This avoids creating huge intermediate files that exceed memory limits
#
# Configuration:
#   1. To enable spatial subsetting: Set apply_spatial_subset="true"
#   2. To process full spatial domain: Set apply_spatial_subset="false"
#   3. Spatial bounds (when enabled) are defined by LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
#
# Requirements:
#   - CDO (Climate Data Operators)
#   - Environment variables from env_setting.sh
################################################################################

# Don't exit immediately on error - we want better error messages
set -o pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment settings FIRST to get all configuration
if [ -f "${SCRIPT_DIR}/../Const/env_setting.sh" ]; then
    source "${SCRIPT_DIR}/../Const/env_setting.sh"
else
    echo "ERROR: env_setting.sh not found!"
    exit 1
fi

# Use parameters from env_setting.sh (with defaults for backward compatibility)
apply_spatial_subset="${APPLY_SPATIAL_SUBSET:-true}"

# Load CDO module
echo "Loading CDO module..."
module load cdo/2.0.5 2>/dev/null || module load cdo 2>/dev/null || echo "Warning: CDO module not found, assuming CDO is in PATH"

echo ""

# Use paths from env_setting.sh
OUTPUT_DIR="${INTERMEDIATE_DIR}"

# Processing parameters
MAX_FILES_PER_BATCH=100  # Adjust based on system limits
USE_BATCH_PROCESSING=auto  # auto, true, or false

echo "=========================================="
echo "Improved CDO Pre-processing Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo ""

################################################################################
# Function: Check if batch processing is needed
################################################################################
check_batch_processing_needed() {
    local file_count=$1
    local max_open_files=$(ulimit -n)
    local max_args=$(getconf ARG_MAX)
    
    echo "System limits:"
    echo "  Max open files: $max_open_files"
    echo "  Max command args: $max_args bytes"
    echo "  Files to process: $file_count"
    echo ""
    
    # If more than 500 files or approaching system limits, use batch processing
    if [ "$USE_BATCH_PROCESSING" = "auto" ]; then
        if [ $file_count -gt 500 ] || [ $file_count -gt $((max_open_files / 2)) ]; then
            echo "→ Using batch processing (file count: $file_count)"
            return 0
        else
            echo "→ Using direct processing (file count manageable)"
            return 1
        fi
    elif [ "$USE_BATCH_PROCESSING" = "true" ]; then
        return 0
    else
        return 1
    fi
}

################################################################################
# Function: Merge files using file list (avoids shell expansion limits)
#
# Parameters:
#   $1 - input_pattern: glob pattern for input files
#   $2 - output_file: path to output file
#   $3 - apply_spatial_subset: "true" or "false" (optional, default: "false")
################################################################################
merge_with_filelist() {
    local input_pattern=$1
    local output_file=$2
    local apply_spatial_subset=${3:-"false"}
    local temp_list="${OUTPUT_DIR}/temp_filelist_$$.txt"
    
    echo "Creating file list..."
    ls -1 $input_pattern > "$temp_list" 2>/dev/null
    
    local file_count=$(wc -l < "$temp_list")
    echo "Found $file_count files"
    
    if [ $file_count -eq 0 ]; then
        echo "ERROR: No files found matching pattern: $input_pattern"
        rm -f "$temp_list"
        return 1
    fi
    
    if [ $file_count -eq 1 ]; then
        echo "Only one file found, copying instead of merging..."
        local single_file=$(cat "$temp_list")
        cp "$single_file" "$output_file"
        rm -f "$temp_list"
        return 0
    fi
    
    # Get full path to cdo to ensure xargs can find it
    local CDO_PATH=$(command -v cdo)
    if [ -z "$CDO_PATH" ]; then
        echo "ERROR: cdo command not found in PATH"
        rm -f "$temp_list"
        return 1
    fi
    echo "Using CDO at: $CDO_PATH"
    
    # Check if batch processing is needed
    if check_batch_processing_needed $file_count; then
        merge_in_batches "$temp_list" "$output_file" "$CDO_PATH" "$apply_spatial_subset"
        local result=$?
        rm -f "$temp_list"
        return $result
    else
        echo "Merging all files in one operation..."
        echo "Command: $CDO_PATH -O mergetime <$file_count files> $output_file"
        
        # Use command substitution to pass files as arguments
        # CDO syntax requires: cdo mergetime input1.nc input2.nc ... output.nc
        if "$CDO_PATH" -O -v mergetime $(cat "$temp_list") "$output_file" 2>&1; then
            rm -f "$temp_list"
            return 0
        else
            echo "ERROR: Direct merge failed, trying batch processing..."
            merge_in_batches "$temp_list" "$output_file" "$CDO_PATH" "$apply_spatial_subset"
            local result=$?
            rm -f "$temp_list"
            return $result
        fi
    fi
}

################################################################################
# Function: Merge files in batches
#
# Parameters:
#   $1 - filelist: path to file containing list of input files
#   $2 - final_output: path to final merged output file
#   $3 - cdo_path: path to cdo executable
#   $4 - apply_spatial_subset: "true" or "false" (optional, default: "false")
################################################################################
merge_in_batches() {
    local filelist=$1
    local final_output=$2
    local cdo_path=$3
    local apply_spatial_subset=${4:-"false"}  # Default to false for flexibility
    local batch_dir="${OUTPUT_DIR}/temp_batches_$$"
    
    mkdir -p "$batch_dir"
    
    local total_files=$(wc -l < "$filelist")
    local num_batches=$(( (total_files + MAX_FILES_PER_BATCH - 1) / MAX_FILES_PER_BATCH ))
    
    echo ""
    echo "Batch processing configuration:"
    echo "  Total files: $total_files"
    echo "  Files per batch: $MAX_FILES_PER_BATCH"
    echo "  Number of batches: $num_batches"
    if [ "$apply_spatial_subset" = "true" ]; then
        echo "  Spatial subset: ENABLED - lon[$LON_MIN,$LON_MAX], lat[$LAT_MIN,$LAT_MAX]"
    else
        echo "  Spatial subset: DISABLED - processing full spatial domain"
    fi
    echo ""
    
    # Split file list into batches
    split -l $MAX_FILES_PER_BATCH "$filelist" "$batch_dir/batch_"
    
    # Create spatial batches directory if needed
    local spatial_dir="${OUTPUT_DIR}/spatial_batches_$$"
    if [ "$apply_spatial_subset" = "true" ]; then
        mkdir -p "$spatial_dir"
    fi
    
    # Process each batch
    local batch_num=0
    for batch_file in "$batch_dir"/batch_*; do
        batch_num=$((batch_num + 1))
        local batch_merged="$batch_dir/merged_batch_${batch_num}.nc"
        
        echo "[$batch_num/$num_batches] Processing batch $batch_num..."
        echo "  Files in batch: $(wc -l < $batch_file)"
        
        # Step 1: Merge files in this batch
        echo "  Merging..."
        if ! "$cdo_path" -O mergetime $(cat "$batch_file") "$batch_merged" 2>&1 | grep -v "Coordinates variable"; then
            echo "  ERROR: Batch merge failed - files preserved in: $batch_dir"
            return 1
        fi
        
        # Step 2: Apply spatial subset if enabled
        if [ "$apply_spatial_subset" = "true" ]; then
            local batch_spatial="$spatial_dir/batch_${batch_num}_spatial.nc"
            
            echo "  Applying spatial subset..."
            if ! "$cdo_path" -O sellonlatbox,$LON_MIN,$LON_MAX,$LAT_MIN,$LAT_MAX \
                "$batch_merged" "$batch_spatial" 2>&1 | grep -v "Coordinates variable"; then
                echo "  ERROR: Spatial subset failed - files preserved in: $batch_dir"
                return 1
            fi
            
            # Show size reduction
            local orig_size=$(ls -lh "$batch_merged" | awk '{print $5}')
            local new_size=$(ls -lh "$batch_spatial" | awk '{print $5}')
            echo "  Size: $orig_size → $new_size"
            
            # Remove unsubset batch to save space
            rm -f "$batch_merged"
        else
            echo "  Batch merged (full spatial domain)"
            echo "  Size: $(ls -lh $batch_merged | awk '{print $5}')"
        fi
        
        echo "  ✓ Batch $batch_num completed"
        echo ""
    done
    
    # Merge all batches (spatially-subset or full domain)
    if [ "$apply_spatial_subset" = "true" ]; then
        echo "Merging all spatially-subset batches..."
        local merge_files=("$spatial_dir"/batch_*_spatial.nc)
        local merge_count=${#merge_files[@]}
        echo "  Found $merge_count spatially-subset batches"
    else
        echo "Merging all batches (full spatial domain)..."
        local merge_files=("$batch_dir"/merged_batch_*.nc)
        local merge_count=${#merge_files[@]}
        echo "  Found $merge_count batches"
    fi
    
    # Use hierarchical merge for large number of files
    if [ $merge_count -gt 50 ]; then
        echo "  Using hierarchical merge..."
        local hier_dir="${OUTPUT_DIR}/hierarchical_merge_$$"
        mkdir -p "$hier_dir"
        
        local chunk_size=30
        local chunk_num=0
        local total_chunks=$(( (merge_count + chunk_size - 1) / chunk_size ))
        
        for ((i=0; i<${#merge_files[@]}; i+=chunk_size)); do
            chunk_num=$((chunk_num + 1))
            local chunk_files=("${merge_files[@]:i:chunk_size}")
            local chunk_output="$hier_dir/chunk_${chunk_num}.nc"
            
            echo "  Merging chunk $chunk_num/$total_chunks..."
            if ! "$cdo_path" -O mergetime "${chunk_files[@]}" "$chunk_output" 2>&1 | grep -v "Coordinates variable"; then
                echo "ERROR: Chunk merge failed - files preserved in: $hier_dir, $batch_dir"
                [ "$apply_spatial_subset" = "true" ] && echo "  Spatial files also preserved in: $spatial_dir"
                return 1
            fi
        done
        
        echo "  Final merge of all chunks..."
        if "$cdo_path" -O mergetime "$hier_dir"/chunk_*.nc "$final_output" 2>&1 | grep -v "Coordinates variable"; then
            echo "✓ Final merge completed"
            rm -rf "$batch_dir" "$hier_dir"
            [ "$apply_spatial_subset" = "true" ] && rm -rf "$spatial_dir"
            return 0
        else
            echo "ERROR: Final merge failed - files preserved in: $hier_dir, $batch_dir"
            [ "$apply_spatial_subset" = "true" ] && echo "  Spatial files also preserved in: $spatial_dir"
            return 1
        fi
    else
        # Direct merge
        if "$cdo_path" -O mergetime "${merge_files[@]}" "$final_output" 2>&1 | grep -v "Coordinates variable"; then
            echo "✓ Final merge completed"
            rm -rf "$batch_dir"
            [ "$apply_spatial_subset" = "true" ] && rm -rf "$spatial_dir"
            return 0
        else
            echo "ERROR: Final merge failed - files preserved in: $batch_dir"
            [ "$apply_spatial_subset" = "true" ] && echo "  Spatial files also preserved in: $spatial_dir"
            return 1
        fi
    fi
}

################################################################################
# Function: Process dataset (training or target)
################################################################################
process_dataset() {
    local dataset_name=$1
    local input_dir=$2
    local output_prefix=$3
    
    echo "=========================================="
    echo "Processing $dataset_name"
    echo "=========================================="
    echo "Input directory: $input_dir"
    echo ""
    
    # Check if directory exists and has files
    if [ ! -d "$input_dir" ]; then
        echo "ERROR: Directory does not exist: $input_dir"
        return 1
    fi
    
    local file_count=$(ls -1 "$input_dir"/*.nc 2>/dev/null | wc -l)
    if [ $file_count -eq 0 ]; then
        echo "ERROR: No NetCDF files found in $input_dir"
        return 1
    fi
    
    echo "Found $file_count NetCDF files"
    echo ""
    
    # Define output files
    # Note: merged_file is now already spatially subset (done during batch processing)
    local merged_file="$OUTPUT_DIR/${output_prefix}_spatial_merged.nc"
    local final_file="$OUTPUT_DIR/${output_prefix}_preprocessed.nc"
    
    # Check if final output already exists
    if [ -f "$final_file" ]; then
        echo "Output file already exists: $final_file"
        echo "To reprocess, delete this file first"
        echo ""
        return 0
    fi
    
    # Step 1: Merge files (with optional spatial subset during batch processing)
    echo "Step 1: Merging all files..."
    
    if [ "$apply_spatial_subset" = "true" ]; then
        echo "  (Spatial subset will be applied during batch processing for efficiency)"
    fi
    
    if ! merge_with_filelist "$input_dir/*.nc" "$merged_file" "$apply_spatial_subset"; then
        echo "ERROR: Failed to merge files"
        return 1
    fi
    
    echo ""
    if [ "$apply_spatial_subset" = "true" ]; then
        echo "✓ Spatially-subset merged file created: $merged_file"
    else
        echo "✓ Merged file created (full spatial domain): $merged_file"
    fi
    echo "  Size: $(ls -lh $merged_file | awk '{print $5}')"
    echo ""
    
    # Show merged file info
    echo "Merged file information:"
    cdo sinfov "$merged_file" 2>&1 | head -30
    echo ""
    
    # Step 2: Temporal subset
    echo "Step 2: Applying temporal subset..."
    echo "  Time range: $TIME_START to $TIME_END"
    
    if ! cdo -O -v seldate,$TIME_START,$TIME_END \
        "$merged_file" "$final_file" 2>&1 | grep -v "Coordinates variable"; then
        echo "ERROR: Temporal subsetting failed"
        echo ""
        echo "Possible issues:"
        echo "  - Time range might not overlap with data"
        echo "  - Calendar format might be incompatible"
        echo ""
        echo "Debug commands:"
        echo "  cdo showdate $merged_file | head -5"
        echo "  cdo showyear $merged_file"
        echo ""
        echo "Merged file preserved at: $merged_file"
        return 1
    fi
    
    echo ""
    echo "✓ Final preprocessed file created: $final_file"
    echo "  Size: $(ls -lh $final_file | awk '{print $5}')"
    echo ""
    
    # Remove spatial file to save space
    rm -f "$spatial_file"
    
    # Show final file info
    echo "Final file information:"
    cdo sinfov "$final_file" 2>&1 | head -30
    echo ""
    
    echo "✓ $dataset_name processing completed successfully!"
    echo ""
    
    return 0
}

################################################################################
# Main Processing
################################################################################

# Process training data (ERA5)
if ! process_dataset "Training Data (ERA5)" "$TRAINING_DATA_DIR" "training_era5_tmax"; then
    echo ""
    echo "=========================================="
    echo "FAILED: Training data processing failed"
    echo "=========================================="
    exit 1
fi

# Process target data (MSWX)
if ! process_dataset "Target Data (MSWX)" "$TARGET_DATA_DIR" "target_mswx_tmax"; then
    echo ""
    echo "=========================================="
    echo "FAILED: Target data processing failed"
    echo "=========================================="
    exit 1
fi

################################################################################
# Summary
################################################################################
echo "=========================================="
echo "CDO Pre-processing Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output files created:"
ls -lh "$OUTPUT_DIR"/*_preprocessed.nc 2>/dev/null
echo ""
echo "Next step: Run Python script to convert to training format"
echo ""

exit 0

