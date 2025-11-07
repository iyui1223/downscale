"""
Utility functions for loading and handling preprocessed data.

This module provides helper functions to load preprocessed data files
and prepare them for training/prediction.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
import yaml

logger = logging.getLogger(__name__)


def load_preprocessed_npz(
    filepath: str,
    return_coords: bool = True
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Load preprocessed data from .npz file.
    
    Args:
        filepath: Path to .npz file
        return_coords: Whether to return coordinate arrays
        
    Returns:
        Tuple of (data_dict, coords_dict) where:
        - data_dict: Dictionary of variable arrays
        - coords_dict: Dictionary of coordinate arrays (if return_coords=True)
    """
    logger.info(f"Loading data from {filepath}")
    
    # Load npz file
    data = np.load(filepath)
    
    # Separate data variables and coordinates
    data_dict = {}
    coords_dict = {}
    
    for key in data.files:
        if key.startswith('coord_'):
            if return_coords:
                coord_name = key.replace('coord_', '')
                coords_dict[coord_name] = data[key]
        elif key == 'dims':
            continue
        else:
            data_dict[key] = data[key]
    
    logger.info(f"Loaded {len(data_dict)} variables")
    for var, arr in data_dict.items():
        logger.info(f"  {var}: shape {arr.shape}, dtype {arr.dtype}")
    
    if return_coords:
        return data_dict, coords_dict
    else:
        return data_dict, None


def load_preprocessed_zarr(filepath: str) -> xr.Dataset:
    """
    Load preprocessed data from .zarr format.
    
    Args:
        filepath: Path to .zarr directory
        
    Returns:
        xarray Dataset
    """
    logger.info(f"Loading data from {filepath}")
    ds = xr.open_zarr(filepath)
    logger.info(f"Loaded dataset with shape: {dict(ds.dims)}")
    return ds


def load_statistics(stats_filepath: str) -> Dict:
    """
    Load normalization statistics from YAML file.
    
    Args:
        stats_filepath: Path to statistics YAML file
        
    Returns:
        Dictionary of statistics
    """
    logger.info(f"Loading statistics from {stats_filepath}")
    with open(stats_filepath, 'r') as f:
        stats = yaml.safe_load(f)
    return stats


def normalize_data(
    data: np.ndarray,
    mean: float,
    std: float,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Input array
        mean: Mean value for normalization
        std: Standard deviation for normalization
        method: Normalization method ('zscore' or 'minmax')
        
    Returns:
        Normalized array
    """
    if method == 'zscore':
        return (data - mean) / std
    elif method == 'minmax':
        return (data - mean) / std  # Can be adapted for actual min-max
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_data(
    data: np.ndarray,
    mean: float,
    std: float,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Denormalize data back to original scale.
    
    Args:
        data: Normalized array
        mean: Mean value used in normalization
        std: Standard deviation used in normalization
        method: Normalization method ('zscore' or 'minmax')
        
    Returns:
        Denormalized array
    """
    if method == 'zscore':
        return data * std + mean
    elif method == 'minmax':
        return data * std + mean  # Can be adapted for actual min-max
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class DataLoader:
    """Convenient class for loading and managing preprocessed data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing preprocessed data
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        logger.info(f"Initialized DataLoader with directory: {data_dir}")
    
    def load_training_data(
        self,
        filename: str = "training_era5_tmax.npz",
        load_stats: bool = True
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Load training data.
        
        Args:
            filename: Name of training data file
            load_stats: Whether to load statistics file
            
        Returns:
            Tuple of (data_dict, stats_dict)
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Training data not found: {filepath}")
        
        # Load data
        if filepath.suffix == '.npz':
            data_dict, coords = load_preprocessed_npz(str(filepath))
        elif filepath.suffix == '.zarr':
            data_dict = load_preprocessed_zarr(str(filepath))
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")
        
        # Load statistics
        stats_dict = None
        if load_stats:
            stats_file = filepath.parent / f"{filepath.stem}_stats.yaml"
            if stats_file.exists():
                stats_dict = load_statistics(str(stats_file))
            else:
                logger.warning(f"Statistics file not found: {stats_file}")
        
        return data_dict, stats_dict
    
    def load_target_data(
        self,
        filename: str = "target_mswx_tmax.npz",
        load_stats: bool = True
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Load target data.
        
        Args:
            filename: Name of target data file
            load_stats: Whether to load statistics file
            
        Returns:
            Tuple of (data_dict, stats_dict)
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Target data not found: {filepath}")
        
        # Load data
        if filepath.suffix == '.npz':
            data_dict, coords = load_preprocessed_npz(str(filepath))
        elif filepath.suffix == '.zarr':
            data_dict = load_preprocessed_zarr(str(filepath))
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")
        
        # Load statistics
        stats_dict = None
        if load_stats:
            stats_file = filepath.parent / f"{filepath.stem}_stats.yaml"
            if stats_file.exists():
                stats_dict = load_statistics(str(stats_file))
            else:
                logger.warning(f"Statistics file not found: {stats_file}")
        
        return data_dict, stats_dict


def prepare_training_arrays(
    training_data: np.ndarray,
    target_data: np.ndarray,
    flatten_spatial: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training and target arrays for XGBoost.
    
    Args:
        training_data: Training data array (time, lat, lon) or (time, lat, lon, features)
        target_data: Target data array (time, lat, lon)
        flatten_spatial: Whether to flatten spatial dimensions
        
    Returns:
        Tuple of (X, y) arrays ready for training
    """
    logger.info(f"Preparing training arrays...")
    logger.info(f"Training data shape: {training_data.shape}")
    logger.info(f"Target data shape: {target_data.shape}")
    
    if flatten_spatial:
        # Flatten spatial dimensions
        if training_data.ndim == 3:
            # (time, lat, lon) -> (samples, features=1)
            X = training_data.reshape(-1, 1)
        elif training_data.ndim == 4:
            # (time, lat, lon, features) -> (samples, features)
            time, lat, lon, feat = training_data.shape
            X = training_data.reshape(-1, feat)
        else:
            raise ValueError(f"Unexpected training data dimensions: {training_data.ndim}")
        
        # Flatten target
        y = target_data.flatten()
        
        logger.info(f"Flattened X shape: {X.shape}")
        logger.info(f"Flattened y shape: {y.shape}")
    else:
        X = training_data
        y = target_data
    
    return X, y


def check_data_alignment(
    training_coords: Dict,
    target_coords: Dict,
    tolerance: float = 1e-4
) -> bool:
    """
    Check if training and target data are spatially/temporally aligned.
    
    Args:
        training_coords: Dictionary of training coordinate arrays
        target_coords: Dictionary of target coordinate arrays
        tolerance: Tolerance for coordinate comparison
        
    Returns:
        True if aligned, raises ValueError if not
    """
    logger.info("Checking data alignment...")
    
    for coord_name in ['time', 'latitude', 'longitude', 'lat', 'lon']:
        if coord_name in training_coords and coord_name in target_coords:
            train_coord = training_coords[coord_name]
            target_coord = target_coords[coord_name]
            
            if len(train_coord) != len(target_coord):
                raise ValueError(
                    f"Coordinate {coord_name} length mismatch: "
                    f"training={len(train_coord)}, target={len(target_coord)}"
                )
            
            if not np.allclose(train_coord, target_coord, atol=tolerance):
                raise ValueError(f"Coordinate {coord_name} values do not match")
            
            logger.info(f"  {coord_name}: aligned ({len(train_coord)} points)")
    
    logger.info("Data alignment check passed!")
    return True

