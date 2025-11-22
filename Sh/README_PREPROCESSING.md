# Preprocessing Pipeline for Downscaling Data

## Overview

This preprocessing pipeline uses a **two-stage approach** to safely handle large NetCDF datasets without memory issues:

1. **Stage 1 (CDO)**: Merge and slice NetCDF files using Climate Data Operators (memory-efficient C code)
2. **Stage 2 (Python)**: Convert to compressed format (Zarr/NPZ) for training

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    env_setting.sh                            │
│  - Defines all paths (TRAINING_DATA_DIR, etc.)              │
│  - Exported as environment variables                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│          F01_preprocess_slurm.sh (MAIN)                      │
│  - Sources env_setting.sh                                   │
│  - Loads modules (CDO, etc.)                                │
│  - Orchestrates both stages                                 │
└──────┬────────────────────────────────────┬─────────────────┘
       │                                    │
       ↓                                    ↓
┌──────────────────────┐          ┌────────────────────────┐
│  cdo_preprocess.sh   │          │ preprocess_data_safe.py│
│  (Stage 1)           │          │ (Stage 2)              │
│                      │          │                        │
│  - Sources env vars  │          │  - Reads env vars      │
│  - Merges 959 files  │          │  - Loads single file   │
│  - Spatial slicing   │          │  - Converts format     │
│  - Time slicing      │          │  - Computes stats      │
│  - Output: 1 file    │          │  - Output: Zarr/NPZ    │
└──────────────────────┘          └────────────────────────┘
```

## Files and Responsibilities

### Configuration
- **`Const/env_setting.sh`**: Central configuration for all paths
  - `TRAINING_DATA_DIR`: Location of ERA5 data
  - `TARGET_DATA_DIR`: Location of MSWX data
  - `INTERMEDIATE_DIR`: CDO output location
  - `PROCESSED_DIR`: Final Python output location

### Shell Scripts
- **`Sh/F01_preprocess_slurm.sh`**: Main SLURM job script
  - Loads environment and modules
  - Calls both CDO and Python stages
  - Reports timing and status
  
- **`Sh/cdo_preprocess.sh`**: CDO processing script
  - Merges all NetCDF files into one
  - Applies spatial bounds (UK region)
  - Applies temporal bounds (2000-2020)
  - Memory-efficient (handles 100s of files)

### Python Scripts
- **`Python/preprocess_data_safe.py`**: Safe Python converter
  - Loads CDO-preprocessed files
  - Converts to Zarr/NPZ format
  - Computes statistics for normalization
  - Memory-safe chunked processing

## Usage

### Quick Start

```bash
cd /home/yi260/rds/hpc-work/downscale/Sh
sbatch F01_preprocess_slurm.sh
```

That's it! The script will:
1. Load paths from `env_setting.sh`
2. Run CDO to merge/slice files → `Data/Intermediate/`
3. Run Python to convert format → `Data/Processed/`

### Monitoring

Check job status:
```bash
squeue -u $USER
```

View output logs:
```bash
tail -f ../Log/preprocess.out
tail -f ../Log/preprocess.err
```

### Expected Output

After successful completion:
```
Data/
├── Intermediate/          (CDO output)
│   ├── training_era5_tmax_preprocessed.nc
│   └── target_mswx_tmax_preprocessed.nc
└── Processed/             (Final output)
    ├── training_era5_tmax.zarr/
    ├── training_era5_tmax_stats.yaml
    ├── target_mswx_tmax.zarr/
    └── target_mswx_tmax_stats.yaml
```

## Customization

### Change Spatial/Temporal Bounds

Edit `Sh/cdo_preprocess.sh`:
```bash
# Spatial bounds
LAT_MIN=49.0
LAT_MAX=61.0
LON_MIN=-11.0
LON_MAX=2.0

# Temporal bounds
TIME_START="2000-01-01"
TIME_END="2020-12-31"
```

### Change Data Paths

Edit `Const/env_setting.sh`:
```bash
TRAINING_DATA_DIR="/path/to/your/ERA5/data"
TARGET_DATA_DIR="/path/to/your/MSWX/data"
```

### Change Output Format

Edit `Const/preprocess_config.yaml`:
```yaml
training:
  save_format: "zarr"  # or "npz"
```

**Recommendation**: Use `zarr` for datasets > 10GB

### Adjust Memory/Time

Edit SLURM parameters in `Sh/F01_preprocess_slurm.sh`:
```bash
#SBATCH --mem=64G        # Increase for larger datasets
#SBATCH --time=06:00:00  # Increase for more files
#SBATCH --cpus-per-task=8
```

## Why This Approach?

### Problem: Memory Segmentation Faults
The original script tried to load **959 NetCDF files** simultaneously with `xr.open_mfdataset()`, causing:
- Too many file handles
- Memory exhaustion during metadata loading
- Segmentation faults (exit code 139)

### Solution: Two-Stage Processing

1. **CDO Stage**: Written in C, highly optimized for NetCDF operations
   - Can merge hundreds of files efficiently
   - Minimal memory footprint
   - Fast spatial/temporal slicing

2. **Python Stage**: Works with single merged file
   - No file handle issues
   - Chunked processing
   - Safe conversion to training format

## Troubleshooting

### CDO not found
```bash
module avail cdo
module load cdo/<version>
```

Or edit `Sh/F01_preprocess_slurm.sh` to use the correct module name.

### Insufficient memory
Increase SLURM memory allocation:
```bash
#SBATCH --mem=128G
```

### Time limit exceeded
Increase SLURM time limit:
```bash
#SBATCH --time=12:00:00
```

### Output format too large
Switch from NPZ to Zarr in `preprocess_config.yaml`:
```yaml
save_format: "zarr"
```

## Alternative: Pure Python Batch Processing

If CDO is not available, use the batch processing script:

```bash
poetry run python Python/preprocess_data_batch.py --config Const/preprocess_config.yaml --batch-size 50
```

This processes files in small batches (default: 50 files at a time).

## Performance Notes

- **CDO stage**: ~5-30 minutes (depends on file count and size)
- **Python stage**: ~2-10 minutes (depends on format and compression)
- **Total memory**: Should stay under 32GB with this approach
- **Disk space**: Intermediate files are ~10-50% of original size

## Support

For issues, check:
1. Log files: `Log/preprocess.out` and `Log/preprocess.err`
2. Python log: `preprocess.log` in the root directory
3. Verify paths in `env_setting.sh` are correct
4. Ensure input NetCDF files exist and are readable

