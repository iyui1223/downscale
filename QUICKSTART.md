# Quick Start Guide

This guide will help you get started with the downscaling system in under 10 minutes.

## 1. Installation (5 minutes)

```bash
# Navigate to project directory
cd /home/yi260/rds/hpc-work/downscale

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Verify installation
poetry run python Python/test_setup.py
```

Expected output: "All tests passed" ✓

## 2. Configuration (2 minutes)

Edit `Const/preprocess_config.yaml`:

```yaml
# Update these paths to match your data location
training_data_dir: "/home/yi260/rds/hpc-work/Download/ERA5/Tmax/6hourly"
target_data_dir: "/home/yi260/rds/hpc-work/Download/MSWX_V100/Past/Tmax/Daily"

# Set your region of interest
training:
  lat_bounds: [49.0, 61.0]    # UK region example
  lon_bounds: [-11.0, 2.0]
  time_start: "2000-01-01"
  time_end: "2020-12-31"
```

## 3. Preprocessing (1 minute to submit, ~1-4 hours to run)

```bash
cd Sh/
./submit_preprocess.sh
```

Monitor progress:
```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f ../Log/preprocess_*.out
```

## 4. Verify Output

```bash
# Check processed files
ls -lh Data/Processed/

# View statistics
cat Data/Processed/training_era5_tmax_stats.yaml
```

Expected files:
- `training_era5_tmax.npz`
- `training_era5_tmax_stats.yaml`
- `target_mswx_tmax.npz`
- `target_mswx_tmax_stats.yaml`

## 5. Next Steps

Once preprocessing is complete:

1. **Train model** (coming soon):
   ```bash
   cd Sh/
   ./submit_training.sh
   ```

2. **Generate predictions** (coming soon):
   ```bash
   cd Sh/
   ./submit_prediction.sh
   ```

## Common Issues

### "Module not found" errors
```bash
# Ensure you're in Poetry environment
poetry shell

# Or use poetry run
poetry run python Python/preprocess_data.py
```

### "No files found" errors
- Check data paths in `Const/preprocess_config.yaml`
- Verify files exist: `ls /path/to/your/data/*.nc`

### "Out of memory" errors
- Increase `--mem` in `Sh/preprocess_slurm.sh`
- Reduce time range in config file
- Use smaller spatial bounds for testing

### Job pending too long
```bash
# Check queue
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>
```

## File Locations

| Type | Location |
|------|----------|
| Raw data | `Data/Training/`, `Data/Target/` (symlinks) |
| Processed data | `Data/Processed/*.npz` |
| Models | `Models/*.json` |
| Logs | `Log/*.out`, `Log/*.err` |
| Config | `Const/*.yaml` |

## Getting Help

1. Check the main `README.md` for detailed documentation
2. Read `Sh/README.md` for SLURM configuration
3. Read `Python/README.md` for Python module documentation
4. Check log files in `Log/` directory for error messages

## Quick Commands Reference

```bash
# Activate environment
poetry shell

# Test setup
poetry run python Python/test_setup.py

# Submit preprocessing
cd Sh/ && ./submit_preprocess.sh

# Check job status
squeue -u $USER

# View logs
tail -f Log/preprocess_*.out

# Cancel job
scancel <JOB_ID>

# List processed files
ls -lh Data/Processed/

# Run Python interactively
poetry run ipython
```

## Tips

1. **Test with small subset**: Use 1 month of data first
2. **Check disk space**: `df -h $HOME/rds/hpc-work`
3. **Monitor memory**: `sstat -j <JOB_ID> --format=MaxRSS`
4. **Use interactive session for debugging**:
   ```bash
   sintr -A cranmer-sl3-cpu -p cclake -N 1 -n 1 -t 1:00:00
   cd /home/yi260/rds/hpc-work/downscale
   poetry shell
   python Python/preprocess_data.py --config Const/preprocess_config.yaml
   ```

---


