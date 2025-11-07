# Python Scripts Directory

This directory contains Python modules for the XGBoost downscaling system.

## Modules

### `preprocess_data.py`
Preprocessing pipeline for meteorological data.

**Features:**
- Loads NetCDF files (ERA5, MSWX, etc.)
- Slices data by spatial/temporal bounds
- Converts to compressed numpy (.npz) or zarr format
- Computes normalization statistics

**Usage:**
```bash
# With Poetry
poetry run python Python/preprocess_data.py --config Const/preprocess_config.yaml

# Or activate environment first
poetry shell
python Python/preprocess_data.py --config Const/preprocess_config.yaml

# Process only training data
python Python/preprocess_data.py --training-only

# Process only target data
python Python/preprocess_data.py --target-only
```

### `data_utils.py`
Utility functions for loading and handling preprocessed data.

**Features:**
- Load preprocessed .npz or .zarr files
- Load normalization statistics
- Normalize/denormalize data
- Prepare arrays for XGBoost training
- Check data alignment

**Example:**
```python
from data_utils import DataLoader, prepare_training_arrays

# Initialize loader
loader = DataLoader('Data/Processed')

# Load training and target data
training_data, training_stats = loader.load_training_data()
target_data, target_stats = loader.load_target_data()

# Prepare for XGBoost
X, y = prepare_training_arrays(
    training_data['tmax'],
    target_data['tmax']
)
```

## Upcoming Scripts

### `train_model.py` (To be implemented)
Train XGBoost model for downscaling.

**Will include:**
- Feature engineering
- Model training with hyperparameter tuning
- Cross-validation
- Model saving

### `predict.py` (To be implemented)
Run prediction/downscaling on new data.

**Will include:**
- Load trained model
- Process input data
- Generate downscaled predictions
- Save results

### `evaluate.py` (To be implemented)
Evaluate model performance.

**Will include:**
- Compute metrics (RMSE, MAE, R², etc.)
- Generate spatial maps
- Create validation plots
- Statistical analysis

## Data Format

### Preprocessed NPZ Files

Structure of `.npz` files:
```python
import numpy as np

# Load file
data = np.load('training_era5_tmax.npz')

# Available arrays:
# - Variable data: data['variable_name']
# - Coordinates: data['coord_time'], data['coord_latitude'], etc.
# - Dimensions: data['dims']
```

### Statistics Files

YAML format with normalization statistics:
```yaml
variable_name:
  mean: 285.5
  std: 12.3
  min: 250.0
  max: 320.0
```

## Development Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Line length: 100 characters

### Testing
Create tests in `tests/` directory:
```python
import pytest
from Python.data_utils import load_preprocessed_npz

def test_load_data():
    data, coords = load_preprocessed_npz('test_data.npz')
    assert 'tmax' in data
    assert data['tmax'].ndim == 3
```

### Logging
Use Python's logging module:
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processing data...")
```

## Dependencies

All dependencies are managed through Poetry (see `pyproject.toml`):

- **Data handling**: xarray, netCDF4, dask
- **ML**: xgboost, scikit-learn
- **Scientific**: numpy, scipy, pandas
- **Visualization**: matplotlib, seaborn, cartopy

Install with:
```bash
poetry install
```

## Tips

1. **Memory Management**: Use dask for large datasets
   ```python
   ds = xr.open_mfdataset(files, chunks={'time': 100})
   ```

2. **Parallel Processing**: Set OMP threads
   ```python
   import os
   os.environ['OMP_NUM_THREADS'] = '8'
   ```

3. **Debugging**: Use IPython for interactive debugging
   ```python
   from IPython import embed; embed()
   ```

4. **Profiling**: Use line_profiler for performance
   ```bash
   poetry add --group dev line_profiler
   poetry run kernprof -l -v script.py
   ```

