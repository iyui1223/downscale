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

### Basic Workflow

1. **Activate the Poetry environment (if not already active):**
   ```bash
   poetry shell
   ```
   
   Or prefix commands with `poetry run`:
   ```bash
   poetry run python Python/your_script.py
   ```

2. **Prepare your data:**
   - Training data: Coarse-resolution reanalysis (e.g., ERA5)
   - Target data: High-resolution observations or analyses (e.g., MSWX)
   - Ensure data is in NetCDF format

3. **Run training:**
   ```bash
   cd Sh/
   ./train_model.sh
   ```

4. **Run prediction/downscaling:**
   ```bash
   cd Sh/
   ./predict.sh
   ```

5. **Evaluate results:**
   Results and metrics will be saved to the `Models/` and `Log/` directories.

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

