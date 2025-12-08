# XGBoost Downscaling Training Pipeline

This directory contains scripts for training and applying an XGBoost-based statistical downscaling model to convert low-resolution ERA5 data to high-resolution MSWX-like predictions.

## Overview

The pipeline consists of four main stages:

1. **Training** - Train XGBoost model on ERA5-MSWX pairs
2. **Inference** - Apply trained model to new ERA5 data
3. **Evaluation** - Assess prediction quality against ground truth
4. **Visualization** - Create diagnostic plots and metrics

## Directory Structure

```
Sh/                               # Shell wrapper scripts (this directory)
  ├── F01_preprocess_slurm.sh     # Preprocessing (SLURM)
  ├── F02_train_slurm.sh          # Training (SLURM)
  ├── F03_inference_evaluate_slurm.sh  # Inference & Evaluation (SLURM)
  ├── train_xgboost.sh            # Training wrapper (XGBoost-specific)
  ├── inference_xgboost.sh        # Inference wrapper (XGBoost-specific)
  ├── evaluate_model.sh           # Evaluation wrapper (model-agnostic)
  └── README_TRAINING.md          # This file

Python/                      # Python core scripts
  ├── train_xgboost.py      # XGBoost training logic
  ├── inference_xgboost.py  # Inference logic (XGBoost-specific)
  └── evaluate_model.py     # Evaluation and visualization (model-agnostic)

Data/
  ├── Processed/            # Preprocessed input data
  ├── Downscaled/           # Model outputs (organized by model)
  │   └── ${MODEL_NAME}/    # E.g., xgboost_downscale_tmax/
  └── Intermediate/         # Temporary files

Models/                     # Trained models storage
  └── xgboost_downscale_tmax/  # Model directory
      ├── *.json            # Model file
      ├── *_metadata.yaml   # Model metadata
      └── *_feature_importance.yaml

Work/                       # Temporary working directories
  ├── train_TIMESTAMP/      # Training work dir
  ├── inference_TIMESTAMP/  # Inference work dir
  └── evaluate_TIMESTAMP/   # Evaluation work dir
```

## Quick Start

### Prerequisites

Ensure preprocessing is complete:
```bash
# Check for required input files
ls -lh Data/Processed/training_era5_tmax.npz
ls -lh Data/Processed/target_mswx_tmax.npz
```

### 1. Training

Train the XGBoost model:

```bash
# Interactive training (faster, use subset of data)
bash Sh/train_xgboost.sh --sample-ratio 0.1

# Submit to SLURM cluster
sbatch Sh/F02_train_slurm.sh

# With custom hyperparameters
bash Sh/train_xgboost.sh \
  --sample-ratio 0.2 \
  --max-depth 10 \
  --learning-rate 0.05 \
  --n-estimators 2000
```

**Training Parameters:**
- `--sample-ratio`: Fraction of data to use (0.0-1.0, default: 0.1)
  - 0.1 = ~1.5M samples, trains in ~10-20 minutes
  - 1.0 = all data (~15M samples), trains in several hours
- `--max-depth`: Maximum tree depth (default: 8)
- `--learning-rate`: Learning rate (default: 0.1)
- `--n-estimators`: Number of boosting rounds (default: 1000)

**Output:**
- Model saved to: `Models/xgboost_downscale_tmax/`
- Training log, metrics, and feature importance included

### 2. Inference

Apply the trained model to ERA5 data:

```bash
# Use same data as training (for validation)
bash Sh/inference_xgboost.sh

# Use different ERA5 data
bash Sh/inference_xgboost.sh --era5-data future_era5_tmax.npz

# Customize output name
bash Sh/inference_xgboost.sh \
  --era5-data training_era5_tmax.npz \
  --output-name downscaled_2000_2020
```

**Parameters:**
- `--era5-data`: ERA5 input file (in Data/Processed/)
- `--output-name`: Base name for output file
- `--chunk-size`: Timesteps to process at once (default: 100)

**Output:**
- Predictions saved to: `Data/Downscaled/${MODEL_NAME}/downscaled_tmax.npz`
- Same format as MSWX data (high-resolution grid)

### 3. Evaluation

Evaluate predictions against ground truth:

```bash
# Evaluate predictions (model-agnostic script)
bash Sh/evaluate_model.sh \
  --predictions downscaled_tmax.npz \
  --model-name xgboost_downscale_tmax

# With custom ground truth
bash Sh/evaluate_model.sh \
  --predictions downscaled_tmax.npz \
  --ground-truth validation_mswx_tmax.npz \
  --model-name xgboost_downscale_tmax \
  --output-name validation_eval
```

**Parameters:**
- `--predictions`: Predictions file (required, in Data/Downscaled/${MODEL_NAME}/)
- `--ground-truth`: Ground truth file (default: target_mswx_tmax.npz, in Data/Processed/)
- `--model-name`: Model name for output organization (required)
- `--output-name`: Base name for outputs (default: evaluation)

**Output:**
- Metrics: `Data/Downscaled/${MODEL_NAME}/evaluation_metrics.yaml`
- Summary: `Data/Downscaled/${MODEL_NAME}/evaluation_summary.txt`
- Plots: `Figs/F03_inference_evaluate/${MODEL_NAME}/`
  - scatter_plot.png - Predictions vs ground truth
  - error_distribution.png - Error histogram and Q-Q plot
  - spatial_maps.png - Spatial patterns of bias and RMSE
  - time_series.png - Temporal dynamics at selected locations

## Complete Workflow Example

```bash
# 1. Train model (quick test with 10% of data)
bash Sh/train_xgboost.sh --sample-ratio 0.1

# 2. Run inference on training data
bash Sh/inference_xgboost.sh --output-name test_predictions

# 3. Evaluate results
bash Sh/evaluate_model.sh \
  --predictions test_predictions.npz \
  --model-name xgboost_downscale_tmax

# 4. Review results
cat Data/Downscaled/xgboost_downscale_tmax/evaluation_metrics.yaml
ls -lh Figs/F03_inference_evaluate/xgboost_downscale_tmax/
```

## HPC Usage (SLURM)

### Submitting Jobs

```bash
# Submit training job
sbatch Sh/F02_train_slurm.sh

# With custom parameters
sbatch Sh/F02_train_slurm.sh --sample-ratio 0.5 --max-depth 12

# Check job status
squeue -u $USER

# Monitor output
tail -f Log/train_JOBID.out
```

### Resource Requirements

**Training:**
- CPUs: 16 cores
- Memory: 64 GB
- Time: 1-4 hours (depends on sample ratio)
- Partition: sapphire (CPU)

**Inference:**
- CPUs: 8 cores
- Memory: 32 GB
- Time: 30 min - 2 hours
- Partition: sapphire (CPU)

## Understanding the Output

### Training Output

After training, you'll find in `Models/xgboost_downscale_tmax/`:

1. **Model file** (`*.json`): XGBoost model in JSON format
2. **Metadata** (`*_metadata.yaml`): Training configuration and metrics
3. **Feature importance** (`*_feature_importance.yaml`): Feature contributions
4. **Training log** (`training_log.txt`): Full training details

### Evaluation Metrics

Key metrics in `evaluation_metrics.yaml`:

- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R²**: Coefficient of determination (higher is better, 1.0 = perfect)
- **Bias**: Mean error (should be close to 0)
- **Correlation**: Pearson correlation (higher is better)

Typical good performance:
- RMSE < 1.5 °C
- MAE < 1.0 °C
- R² > 0.90
- |Bias| < 0.5 °C

## Troubleshooting

### Training fails with memory error
- Reduce `--sample-ratio` (e.g., 0.05 or 0.02)
- Increase SLURM memory allocation in `F02_train_slurm.sh`

### Training is too slow
- Reduce `--sample-ratio` for faster training
- Reduce `--n-estimators`
- Ensure using multiple CPUs (`--n-jobs -1`)

### Poor model performance
- Increase `--sample-ratio` to use more training data
- Tune hyperparameters: try deeper trees (`--max-depth 12`)
- Check input data quality and preprocessing

### Inference fails
- Ensure model exists in `Models/` directory
- Check ERA5 data has same format as training data
- Verify target grid reference file exists

## Advanced Usage

### Custom Hyperparameter Tuning

Edit parameters in `Sh/train_xgboost.sh` or pass via command line:

```bash
bash Sh/train_xgboost.sh \
  --sample-ratio 0.3 \
  --max-depth 12 \
  --learning-rate 0.05 \
  --n-estimators 2000 \
  --subsample 0.9 \
  --colsample-bytree 0.9
```

### Processing Different Time Periods

```bash
# Train on 2000-2015, validate on 2016-2020
# (requires separate preprocessing)
bash Sh/train_xgboost.sh --era5-data era5_2000_2015.npz
bash Sh/inference_xgboost.sh --era5-data era5_2016_2020.npz
bash Sh/evaluate_model.sh \
  --predictions downscaled_2016_2020.npz \
  --ground-truth mswx_2016_2020.npz \
  --model-name xgboost_downscale_tmax
```

## Features Used by the Model

The XGBoost model uses 7 features for each prediction:

1. **ERA5 interpolated temperature**: Bilinear interpolation to target location
2. **Latitude (normalized)**: Captures latitudinal patterns
3. **Longitude (normalized)**: Captures longitudinal patterns
4. **Latitude gradient**: North-south temperature gradient
5. **Longitude gradient**: East-west temperature gradient
6. **Day of year (normalized)**: Seasonal cycle
7. **Year (normalized)**: Long-term trends

These features capture both the coarse-scale ERA5 information and the fine-scale spatial patterns needed for downscaling.

## Contact and Support

For issues or questions:
1. Check preprocessing logs: `Log/preprocess.out`
2. Check training logs: `Log/train_JOBID.out`
3. Review work directories for debugging information
4. Check model metadata for training details

## References

- XGBoost: https://xgboost.readthedocs.io/
- Statistical downscaling methods in climate science
- ERA5 reanalysis: ECMWF
- MSWX dataset: Multi-Source Weather

