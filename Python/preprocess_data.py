#!/usr/bin/env python3
"""
Preprocessing script for downscaling data.

This script reads NetCDF files (ERA5, MSWX, etc.), slices them according to
specified spatial/temporal bounds, and saves them in compressed numpy format
for efficient training.

Usage:
    python preprocess_data.py --config path/to/config.yaml
    python preprocess_data.py --training --start-date 2000-01-01 --end-date 2010-12-31
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocess.log')
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles preprocessing of meteorological data for downscaling."""
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.config = config
        self.training_dir = Path(config['training_data_dir'])
        self.target_dir = Path(config['target_data_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training data directory: {self.training_dir}")
        logger.info(f"Target data directory: {self.target_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_netcdf_files(
        self,
        file_pattern: str,
        data_dir: Path,
        variable_name: Optional[str] = None
    ) -> xr.Dataset:
        """
        Load NetCDF files matching pattern.
        
        Args:
            file_pattern: Glob pattern for files (e.g., "*.nc")
            data_dir: Directory containing NetCDF files
            variable_name: Specific variable to load (optional)
            
        Returns:
            xarray Dataset with loaded data
        """
        logger.info(f"Loading files from {data_dir} with pattern {file_pattern}")
        
        files = sorted(data_dir.glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching {file_pattern} in {data_dir}")
        
        logger.info(f"Found {len(files)} files")
        
        # Load with dask for lazy loading
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            parallel=True,
            chunks={'time': 100}  # Chunk size for memory efficiency
        )
        
        if variable_name and variable_name in ds:
            ds = ds[[variable_name]]
            logger.info(f"Selected variable: {variable_name}")
        
        logger.info(f"Loaded dataset shape: {dict(ds.dims)}")
        return ds
    
    def slice_spatiotemporal(
        self,
        dataset: xr.Dataset,
        lat_bounds: Optional[Tuple[float, float]] = None,
        lon_bounds: Optional[Tuple[float, float]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None
    ) -> xr.Dataset:
        """
        Slice dataset by spatial and temporal bounds.
        
        Args:
            dataset: Input xarray Dataset
            lat_bounds: (min_lat, max_lat) tuple
            lon_bounds: (min_lon, max_lon) tuple
            time_start: Start date (e.g., "2000-01-01")
            time_end: End date (e.g., "2010-12-31")
            
        Returns:
            Sliced xarray Dataset
        """
        logger.info("Applying spatiotemporal slicing...")
        
        # Identify dimension names (handle variations)
        lat_dim = self._find_dimension(dataset, ['latitude', 'lat', 'y'])
        lon_dim = self._find_dimension(dataset, ['longitude', 'lon', 'x'])
        time_dim = self._find_dimension(dataset, ['time'])
        
        # Spatial slicing
        if lat_bounds is not None:
            logger.info(f"Slicing latitude: {lat_bounds}")
            dataset = dataset.sel({lat_dim: slice(lat_bounds[0], lat_bounds[1])})
        
        if lon_bounds is not None:
            logger.info(f"Slicing longitude: {lon_bounds}")
            dataset = dataset.sel({lon_dim: slice(lon_bounds[0], lon_bounds[1])})
        
        # Temporal slicing
        if time_start is not None or time_end is not None:
            logger.info(f"Slicing time: {time_start} to {time_end}")
            dataset = dataset.sel({time_dim: slice(time_start, time_end)})
        
        logger.info(f"Sliced dataset shape: {dict(dataset.dims)}")
        return dataset
    
    def _find_dimension(self, dataset: xr.Dataset, possible_names: List[str]) -> str:
        """Find dimension name from possible variations."""
        for name in possible_names:
            if name in dataset.dims:
                return name
        raise ValueError(f"Could not find dimension from {possible_names} in dataset")
    
    def compute_statistics(self, dataset: xr.Dataset) -> Dict:
        """
        Compute statistics for normalization.
        
        Args:
            dataset: Input xarray Dataset
            
        Returns:
            Dictionary of statistics (mean, std, min, max)
        """
        logger.info("Computing dataset statistics...")
        stats = {}
        
        for var in dataset.data_vars:
            logger.info(f"Computing statistics for {var}")
            data = dataset[var]
            stats[var] = {
                'mean': float(data.mean().compute()),
                'std': float(data.std().compute()),
                'min': float(data.min().compute()),
                'max': float(data.max().compute())
            }
            logger.info(f"  Mean: {stats[var]['mean']:.4f}, Std: {stats[var]['std']:.4f}")
        
        return stats
    
    def save_preprocessed_data(
        self,
        dataset: xr.Dataset,
        output_name: str,
        save_format: str = 'npz',
        compression: bool = True
    ):
        """
        Save preprocessed data in efficient format.
        
        Args:
            dataset: xarray Dataset to save
            output_name: Output filename (without extension)
            save_format: 'npz' or 'zarr' (default: 'npz')
            compression: Whether to apply compression
        """
        output_path = self.output_dir / output_name
        
        if save_format == 'npz':
            # Save as compressed NumPy format
            logger.info(f"Saving as compressed .npz: {output_path}.npz")
            
            # Convert to numpy arrays and save
            data_dict = {}
            for var in dataset.data_vars:
                logger.info(f"Converting {var} to numpy array...")
                data_dict[var] = dataset[var].values
            
            # Add coordinates
            for coord in dataset.coords:
                if coord not in dataset.dims:
                    continue
                data_dict[f'coord_{coord}'] = dataset[coord].values
            
            # Add metadata
            data_dict['dims'] = np.array([str(d) for d in dataset.dims], dtype=object)
            
            if compression:
                np.savez_compressed(f"{output_path}.npz", **data_dict)
            else:
                np.savez(f"{output_path}.npz", **data_dict)
            
            # Save statistics
            stats = self.compute_statistics(dataset)
            stats_path = f"{output_path}_stats.yaml"
            with open(stats_path, 'w') as f:
                yaml.dump(stats, f, default_flow_style=False)
            logger.info(f"Saved statistics to {stats_path}")
            
        elif save_format == 'zarr':
            # Save as Zarr format (better for very large datasets)
            logger.info(f"Saving as .zarr: {output_path}.zarr")
            
            # Compute and save
            dataset.to_zarr(
                f"{output_path}.zarr",
                mode='w',
                consolidated=True
            )
            
            # Save statistics
            stats = self.compute_statistics(dataset)
            stats_path = f"{output_path}_stats.yaml"
            with open(stats_path, 'w') as f:
                yaml.dump(stats, f, default_flow_style=False)
            logger.info(f"Saved statistics to {stats_path}")
        
        else:
            raise ValueError(f"Unknown save format: {save_format}")
        
        logger.info(f"Successfully saved preprocessed data to {output_path}")
    
    def preprocess_training_data(self):
        """Preprocess training data (e.g., ERA5)."""
        logger.info("=" * 80)
        logger.info("PREPROCESSING TRAINING DATA")
        logger.info("=" * 80)
        
        config = self.config['training']
        
        # Load data
        ds = self.load_netcdf_files(
            config['file_pattern'],
            self.training_dir,
            config.get('variable_name')
        )
        
        # Slice data
        ds_sliced = self.slice_spatiotemporal(
            ds,
            lat_bounds=config.get('lat_bounds'),
            lon_bounds=config.get('lon_bounds'),
            time_start=config.get('time_start'),
            time_end=config.get('time_end')
        )
        
        # Save
        self.save_preprocessed_data(
            ds_sliced,
            config['output_name'],
            save_format=config.get('save_format', 'npz')
        )
        
        logger.info("Training data preprocessing complete!")
    
    def preprocess_target_data(self):
        """Preprocess target data (e.g., MSWX)."""
        logger.info("=" * 80)
        logger.info("PREPROCESSING TARGET DATA")
        logger.info("=" * 80)
        
        config = self.config['target']
        
        # Load data
        ds = self.load_netcdf_files(
            config['file_pattern'],
            self.target_dir,
            config.get('variable_name')
        )
        
        # Slice data
        ds_sliced = self.slice_spatiotemporal(
            ds,
            lat_bounds=config.get('lat_bounds'),
            lon_bounds=config.get('lon_bounds'),
            time_start=config.get('time_start'),
            time_end=config.get('time_end')
        )
        
        # Save
        self.save_preprocessed_data(
            ds_sliced,
            config['output_name'],
            save_format=config.get('save_format', 'npz')
        )
        
        logger.info("Target data preprocessing complete!")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess meteorological data for downscaling'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='Const/preprocess_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--training-only',
        action='store_true',
        help='Process only training data'
    )
    parser.add_argument(
        '--target-only',
        action='store_true',
        help='Process only target data'
    )
    
    return parser.parse_args()


def main():
    """Main preprocessing pipeline."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Process data
        if args.training_only:
            preprocessor.preprocess_training_data()
        elif args.target_only:
            preprocessor.preprocess_target_data()
        else:
            preprocessor.preprocess_training_data()
            preprocessor.preprocess_target_data()
        
        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETE!")
        logger.info(f"End time: {datetime.now()}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

