#!/usr/bin/env python3
"""
Memory-safe preprocessing script for downscaling data.

This script reads pre-merged NetCDF files (processed by CDO)
and converts them to compressed numpy format for efficient training.

Usage:
    python preprocess_data_safe.py --config path/to/config.yaml
"""

import argparse
import gc
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr
import yaml

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


class SafeDataPreprocessor:
    """Memory-safe preprocessor for meteorological data."""
    
    def __init__(self, config: Dict):
        """Initialize preprocessor with configuration."""
        self.config = config
        
        # Use environment variables if set, otherwise use config
        self.intermediate_dir = Path(os.environ.get('INTERMEDIATE_DIR', 
                                      config.get('intermediate_dir', 
                                      '/home/yi260/rds/hpc-work/downscale/Data/Intermediate')))
        self.output_dir = Path(os.environ.get('PROCESSED_DIR', 
                              config.get('output_dir',
                              '/home/yi260/rds/hpc-work/downscale/Data/Processed')))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Intermediate directory: {self.intermediate_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_single_netcdf(
        self,
        file_path: Path,
        variable_name: Optional[str] = None,
        time_chunk: int = 365  # Process one year at a time
    ) -> xr.Dataset:
        """
        Load a single NetCDF file with chunking for memory safety.
        
        Args:
            file_path: Path to NetCDF file
            variable_name: Specific variable to load (optional)
            time_chunk: Chunk size for time dimension
            
        Returns:
            xarray Dataset with loaded data
        """
        logger.info(f"Loading file: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Open with chunking for memory efficiency
        ds = xr.open_dataset(
            file_path,
            chunks={'time': time_chunk},
            engine='netcdf4'
        )
        
        if variable_name and variable_name in ds:
            ds = ds[[variable_name]]
            logger.info(f"Selected variable: {variable_name}")
        
        logger.info(f"Loaded dataset shape: {dict(ds.dims)}")
        logger.info(f"Dataset size in memory: {ds.nbytes / 1e9:.2f} GB")
        
        return ds
    
    def compute_statistics_chunked(self, dataset: xr.Dataset) -> Dict:
        """
        Compute statistics using chunked computation to avoid memory issues.
        
        Args:
            dataset: Input xarray Dataset
            
        Returns:
            Dictionary of statistics (mean, std, min, max)
        """
        logger.info("Computing dataset statistics (chunked)...")
        stats = {}
        
        for var in dataset.data_vars:
            logger.info(f"Computing statistics for {var}")
            data = dataset[var]
            
            # Compute statistics in chunks to avoid memory issues
            stats[var] = {
                'mean': float(data.mean().compute()),
                'std': float(data.std().compute()),
                'min': float(data.min().compute()),
                'max': float(data.max().compute())
            }
            logger.info(f"  Mean: {stats[var]['mean']:.4f}, Std: {stats[var]['std']:.4f}")
            logger.info(f"  Min: {stats[var]['min']:.4f}, Max: {stats[var]['max']:.4f}")
            
            # Force garbage collection
            gc.collect()
        
        return stats
    
    def save_preprocessed_data_chunked(
        self,
        dataset: xr.Dataset,
        output_name: str,
        save_format: str = 'npz',
        time_batch_size: int = 365  # Save one year at a time
    ):
        """
        Save preprocessed data in efficient format with batched processing.
        
        Args:
            dataset: xarray Dataset to save
            output_name: Output filename (without extension)
            save_format: 'npz' or 'zarr'
            time_batch_size: Number of time steps to process at once
        """
        output_path = self.output_dir / output_name
        
        if save_format == 'zarr':
            # Zarr is inherently chunked and memory-safe
            logger.info(f"Saving as .zarr: {output_path}.zarr")
            
            # Add compression
            encoding = {}
            for var in dataset.data_vars:
                encoding[var] = {
                    'compressor': {'id': 'zlib', 'level': 5},
                    'chunks': (time_batch_size, 
                              min(50, dataset.dims.get('latitude', 50)),
                              min(50, dataset.dims.get('longitude', 50)))
                }
            
            dataset.to_zarr(
                f"{output_path}.zarr",
                mode='w',
                encoding=encoding,
                consolidated=True
            )
            
        elif save_format == 'npz':
            # Chunked processing: write data batch-by-batch using memmap
            # Then save as uncompressed npz (compressed requires full RAM load)
            logger.info(f"Saving as .npz (chunked processing): {output_path}.npz")
            
            # Check total size
            total_size_gb = dataset.nbytes / 1e9
            logger.info(f"Total dataset size: {total_size_gb:.2f} GB")
            
            # Store paths to memmap temp files (keep them open for npz save)
            temp_files = []
            data_dict = {}
            
            for var in dataset.data_vars:
                logger.info(f"Processing {var} in chunks...")
                da = dataset[var]
                shape = da.shape
                dtype = np.float32  # Use float32 to save space
                
                # Create a temporary memory-mapped file
                temp_path = f"{output_path}_{var}_temp.dat"
                temp_files.append(temp_path)
                logger.info(f"  Shape: {shape}, dtype: {dtype}")
                logger.info(f"  Creating temp memmap: {temp_path}")
                
                # Create memory-mapped array on disk
                mmap_array = np.memmap(temp_path, dtype=dtype, mode='w+', shape=shape)
                
                # Get time dimension size and chunk size
                n_times = shape[0]
                chunk_size = time_batch_size
                
                # Process in time batches
                for t_start in range(0, n_times, chunk_size):
                    t_end = min(t_start + chunk_size, n_times)
                    logger.info(f"  Processing timesteps {t_start}-{t_end} of {n_times}")
                    
                    # Load only this chunk into memory
                    chunk_data = da.isel(time=slice(t_start, t_end)).values.astype(dtype)
                    
                    # Write to memmap (directly to disk)
                    mmap_array[t_start:t_end] = chunk_data
                    
                    # Free memory
                    del chunk_data
                    gc.collect()
                
                # Flush to disk and keep reference
                mmap_array.flush()
                data_dict[var] = mmap_array  # Keep memmap reference (not loaded to RAM)
                logger.info(f"  Completed {var}")
            
            # Add coordinates (small, load fully)
            for coord in dataset.coords:
                if coord not in dataset.dims:
                    continue
                logger.info(f"Adding coordinate: {coord}")
                data_dict[f'coord_{coord}'] = dataset[coord].values
            
            # Add metadata
            data_dict['dims'] = np.array([str(d) for d in dataset.dims], dtype=object)
            
            # Save as uncompressed npz (works with memmap without loading to RAM)
            # Note: np.savez iterates and saves arrays one by one
            logger.info("Saving to npz (uncompressed for memory efficiency)...")
            np.savez(f"{output_path}.npz", **data_dict)
            
            # Clean up memmap references and temp files
            for var in dataset.data_vars:
                if var in data_dict:
                    del data_dict[var]
            gc.collect()
            
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"  Removed temp file: {temp_path}")
        
        else:
            raise ValueError(f"Unknown save format: {save_format}")
        
        logger.info(f"Successfully saved preprocessed data to {output_path}")
        
        # Report file size
        if save_format == 'npz':
            file_size = os.path.getsize(f"{output_path}.npz") / 1e9
            logger.info(f"Output file size: {file_size:.2f} GB")
        else:
            # For zarr, it's a directory
            import subprocess
            result = subprocess.run(['du', '-sh', f"{output_path}.zarr"], 
                                  capture_output=True, text=True)
            logger.info(f"Output size: {result.stdout.strip()}")
    
    def preprocess_training_data(self):
        """Preprocess training data (from CDO-processed file)."""
        logger.info("=" * 80)
        logger.info("PREPROCESSING TRAINING DATA")
        logger.info("=" * 80)
        
        config = self.config['training']
        
        # Look for CDO-preprocessed file
        preprocessed_file = self.intermediate_dir / "training_era5_tmax_preprocessed.nc"
        
        if not preprocessed_file.exists():
            logger.error(f"CDO-preprocessed file not found: {preprocessed_file}")
            logger.error("Please run cdo_preprocess.sh first!")
            raise FileNotFoundError(f"File not found: {preprocessed_file}")
        
        # Load data
        ds = self.load_single_netcdf(
            preprocessed_file,
            config.get('variable_name'),
            time_chunk=config.get('time_chunk', 365)
        )
        
        # Save
        self.save_preprocessed_data_chunked(
            ds,
            config['output_name'],
            save_format=config.get('save_format', 'zarr')  # Default to zarr for safety
        )
        
        # Close dataset and free memory
        ds.close()
        del ds
        gc.collect()
        
        logger.info("Training data preprocessing complete!")
    
    def preprocess_target_data(self):
        """Preprocess target data (from CDO-processed file)."""
        logger.info("=" * 80)
        logger.info("PREPROCESSING TARGET DATA")
        logger.info("=" * 80)
        
        config = self.config['target']
        
        # Look for CDO-preprocessed file
        preprocessed_file = self.intermediate_dir / "target_mswx_tmax_preprocessed.nc"
        
        if not preprocessed_file.exists():
            logger.error(f"CDO-preprocessed file not found: {preprocessed_file}")
            logger.error("Please run cdo_preprocess.sh first!")
            raise FileNotFoundError(f"File not found: {preprocessed_file}")
        
        # Load data
        ds = self.load_single_netcdf(
            preprocessed_file,
            config.get('variable_name'),
            time_chunk=config.get('time_chunk', 365)
        )
        
        # Save
        self.save_preprocessed_data_chunked(
            ds,
            config['output_name'],
            save_format=config.get('save_format', 'zarr')  # Default to zarr for safety
        )
        
        # Close dataset and free memory
        ds.close()
        del ds
        gc.collect()
        
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
        description='Memory-safe preprocessing for meteorological data'
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
    logger.info("MEMORY-SAFE DATA PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize preprocessor
        preprocessor = SafeDataPreprocessor(config)
        
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

