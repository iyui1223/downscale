# Downscale: XGBoost-Based Temperature Downscaling System

A machine learning framework for downscaling maximum surface temperature using XGBoost, designed to work with ERA5 and other meteorological reanalysis datasets.

## Overview

This system uses XGBoost (eXtreme Gradient Boosting) to perform statistical downscaling of coarse-resolution meteorological data to higher spatial resolution. The primary focus is on maximum surface temperature (Tmax), but the framework can be extended to other variables.

## Project Structure

```
downscale/
├── Const/              # Configuration files and constants
│   ├── env_setting.sh  # User-specific environment settings
│   └── model_params/   # Model hyperparameters and configuration
├── Data/               # Data directory (gitignored)
│   ├── Training/       # Training data (ERA5, etc.)
│   └── Target/         # Target high-resolution data (MSWX, etc.)
├── Python/             # Python modules for training and prediction
├── Sh/                 # Shell scripts and batch job scripts
├── Models/             # Saved trained models
├── Log/                # Log files
└── pyproject.toml      # Python dependencies and project metadata
```

## Requirements

- Python 3.8 or higher
- Linux/Unix environment (tested on RHEL 8)
- NetCDF4 library (system-level)
- Sufficient disk space for meteorological datasets

## Installation

### Option 1: Using Poetry (Recommended)

1. **Install Poetry (if not already installed):**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   Or using pip:
   ```bash
   pip install poetry
   ```

2. **Clone or navigate to the repository:**
   ```bash
   cd /home/yi260/rds/hpc-work/downscale
   ```

3. **Install the package and dependencies:**
   ```bash
   poetry install
   ```

   This will:
   - Create a virtual environment automatically
   - Install all required dependencies including:
     - XGBoost for machine learning
     - netCDF4, xarray for meteorological data handling
     - numpy, pandas, scipy for scientific computing
     - matplotlib, cartopy for visualization

4. **Activate the Poetry environment:**
   ```bash
   poetry shell
   ```

5. **For development (optional):**
   ```bash
   poetry install --with dev
   ```

### Option 2: Using pip

1. **Clone or navigate to the repository:**
   ```bash
   cd /home/yi260/rds/hpc-work/downscale
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the package and dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

4. **For development (optional):**
   ```bash
   pip install -e ".[dev]"
   ```

### Option 3: Using conda/mamba

1. **Create a conda environment:**
   ```bash
   conda create -n downscale python=3.10
   conda activate downscale
   ```

2. **Install dependencies:**
   ```bash
   # Install scientific and geospatial packages via conda (often faster)
   conda install -c conda-forge numpy scipy pandas xarray netCDF4 dask \
       rasterio pyproj cartopy matplotlib seaborn
   
   # Install XGBoost and remaining packages
   pip install xgboost scikit-learn tqdm pyyaml joblib
   ```

3. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```

### Verify Installation

Test that key packages are available:

**With Poetry:**
```bash
poetry run python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
poetry run python -c "import xarray; print('xarray version:', xarray.__version__)"
poetry run python -c "import netCDF4; print('netCDF4 version:', netCDF4.__version__)"
```

**Or after activating the environment:**
```bash
poetry shell
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
python -c "import xarray; print('xarray version:', xarray.__version__)"
python -c "import netCDF4; print('netCDF4 version:', netCDF4.__version__)"
```

## Configuration

### Environment Settings

Edit `Const/env_setting.sh` to set your data paths:

```bash
#!/bin/bash

ROOT_DIR="/home/yi260/rds/hpc-work/downscale"

# Path to training data (e.g., ERA5 6-hourly Tmax)
TRAINING_DATA_DIR="/path/to/your/training/data"

# Path to target high-resolution data (e.g., MSWX Daily Tmax)
TARGET_DATA_DIR="/path/to/your/target/data"
```

### Model Parameters

Model hyperparameters will be stored in `Const/model_params/` (to be created) in YAML or JSON format for easy modification without code changes.

## Usage

### Quick Start

1. **Verify installation:**
   ```bash
   poetry run python Python/test_setup.py
   ```

2. **Configure preprocessing:**
   Edit `Const/preprocess_config.yaml` to set your data paths and parameters

3. **Submit preprocessing job:**
   ```bash
   cd Sh/
   ./submit_preprocess.sh
   ```

4. **Train model (coming soon):**
   ```bash
   cd Sh/
   ./submit_training.sh
   ```

5. **Run predictions (coming soon):**
   ```bash
   cd Sh/
   ./submit_prediction.sh
   ```

### Detailed Workflow

#### Step 1: Activate Environment

Activate the Poetry environment:
```bash
poetry shell
```

Or prefix commands with `poetry run`:
```bash
poetry run python Python/your_script.py
```

#### Step 2: Data Preprocessing

**Purpose:** Convert NetCDF files to efficient compressed format and slice data

**Configure:** Edit `Const/preprocess_config.yaml`:
```yaml
# Set your data paths
training_data_dir: "/path/to/ERA5/data"
target_data_dir: "/path/to/MSWX/data"

# Set spatial bounds (e.g., UK region)
training:
  lat_bounds: [49.0, 61.0]
  lon_bounds: [-11.0, 2.0]
  time_start: "2000-01-01"
  time_end: "2020-12-31"
```

**Run preprocessing:**
```bash
cd Sh/
./submit_preprocess.sh
```

**Monitor progress:**
```bash
# Check job status
squeue -u $USER

# View output
tail -f ../Log/preprocess_<JOB_ID>.out
```

**Output:** Preprocessed data saved to `Data/Processed/`:
- `training_era5_tmax.npz` - Compressed training data
- `training_era5_tmax_stats.yaml` - Normalization statistics
- `target_mswx_tmax.npz` - Compressed target data
- `target_mswx_tmax_stats.yaml` - Target statistics

#### Step 3: Train Model (Coming Soon)

Train XGBoost model on preprocessed data:
```bash
cd Sh/
./submit_training.sh
```

Features will include:
- Automatic feature engineering
- Hyperparameter tuning with cross-validation
- Model checkpoint saving
- Training metrics logging

#### Step 4: Generate Predictions (Coming Soon)

Run downscaling predictions:
```bash
cd Sh/
./submit_prediction.sh
```

#### Step 5: Evaluate Results (Coming Soon)

Evaluate model performance and generate visualizations:
```bash
poetry run python Python/evaluate.py --model Models/best_model.json
```

## Data Format

### Expected NetCDF Structure

- **Variables:** Maximum temperature (e.g., `t2m_max`, `tmax`)
- **Dimensions:** `time`, `latitude`, `longitude`
- **Coordinates:** CF-compliant coordinate variables

Example:
```python
import xarray as xr
data = xr.open_dataset('training_data.nc')
# Expected structure:
# Dimensions:  time: N, latitude: M, longitude: K
# Variables:   tmax(time, latitude, longitude)
```

## HPC/Batch System Integration

For HPC environments (SLURM, PBS, etc.), batch scripts will be provided in `Sh/` directory:

- `Sh/submit_training.slurm` - Submit training job
- `Sh/submit_prediction.slurm` - Submit prediction job

Modify according to your cluster's configuration.

## Troubleshooting

### Common Issues

1. **NetCDF4 installation issues:**
   - Ensure system-level NetCDF libraries are installed
   - On RHEL/CentOS: `sudo yum install netcdf netcdf-devel hdf5 hdf5-devel`
   - Try using conda: `conda install -c conda-forge netcdf4`

2. **Cartopy installation issues:**
   - Requires GEOS and PROJ libraries
   - Easier via conda: `conda install -c conda-forge cartopy`

3. **Memory issues with large datasets:**
   - Use Dask for lazy loading: `xr.open_mfdataset(..., chunks={'time': 10})`
   - Process data in batches

4. **XGBoost performance:**
   - Ensure OpenMP is available for multi-threading
   - Use GPU acceleration if available: `pip install xgboost[gpu]`

## Development

### Code Style

This project uses:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting

Run formatters:
```bash
poetry run black Python/
poetry run isort Python/
poetry run flake8 Python/
```

Or after activating the environment:
```bash
poetry shell
black Python/
isort Python/
flake8 Python/
```

### Testing

Run tests with pytest:
```bash
poetry run pytest tests/
```

Or after activating the environment:
```bash
poetry shell
pytest tests/
```

## Contributing

Contributions are welcome! Please:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please contact [your.email@example.com]

## References

- XGBoost: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
- ERA5: Hersbach et al. (2020). "The ERA5 global reanalysis"
- Statistical Downscaling: Relevant papers on your methodology

## Acknowledgments

- ERA5 data provided by ECMWF
- MSWX data provided by [data provider]
- Computing resources: [HPC facility name]

