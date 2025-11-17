#!/bin/bash

module purge

ROOT_DIR="/home/yi260/rds/hpc-work/downscale"

# Data source directories
TRAINING_DATA_DIR="/home/yi260/rds/hpc-work/Download/ERA5/Tmax/6hourly"
TARGET_DATA_DIR="/home/yi260/rds/hpc-work/Download/MSWX_V100/Past/Tmax/Daily"

# Output directories
INTERMEDIATE_DIR="${ROOT_DIR}/Data/Intermediate"
PROCESSED_DIR="${ROOT_DIR}/Data/Processed"

# Create output directories if they don't exist
mkdir -p "${INTERMEDIATE_DIR}"
mkdir -p "${PROCESSED_DIR}"

ln -rsf ${TRAINING_DATA_DIR} ${ROOT_DIR}/Data/Training
ln -rsf ${TARGET_DATA_DIR} ${ROOT_DIR}/Data/Target

# for poetry access
source ~/venvs/c1coursework/bin/activate


