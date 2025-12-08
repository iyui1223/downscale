#!/bin/bash

module purge

# --------------------- EDIT ---------------------
export ROOT_DIR="/home/yi260/rds/hpc-work/downscale"

# for poetry access
source ~/venvs/c1coursework/bin/activate
# Note: Use 'poetry run python3' in scripts, or 'poetry shell' for interactive use

# Data source directories
export TRAINING_DATA_DIR="/home/yi260/rds/hpc-work/Download/ERA5/Tmax/6hourly"
export TARGET_DATA_DIR="/home/yi260/rds/hpc-work/Download/MSWX_V100/Past/Tmax/Daily"

# Spatial/temporal subsetting parameters
export APPLY_SPATIAL_SUBSET="true"  # Set to "false" to process full spatial domain

# Spatial bounds (UK region)
export LAT_MIN=49.0
export LAT_MAX=61.0
export LON_MIN=-11.0
export LON_MAX=2.0

# Temporal bounds
export TIME_START="2000-01-01"
export TIME_END="2020-12-31"

# ------------------------------------------------

# inputs
export PREPROCESS_CONFIG="${ROOT_DIR}/Const/preprocess_config.yaml"

# Output directories
export INTERMEDIATE_DIR="${ROOT_DIR}/Data/Intermediate"
export PROCESSED_DIR="${ROOT_DIR}/Data/Processed"
export FIGS_DIR="${ROOT_DIR}/Figs"

# Create output directories if they don't exist
mkdir -p "${INTERMEDIATE_DIR}"
mkdir -p "${PROCESSED_DIR}"
mkdir -p "${FIGS_DIR}"

ln -rsf ${TRAINING_DATA_DIR} ${ROOT_DIR}/Data/Training
ln -rsf ${TARGET_DATA_DIR} ${ROOT_DIR}/Data/Target

