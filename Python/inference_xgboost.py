#!/usr/bin/env python3
"""
XGBoost Downscaling Inference Script

This script applies a trained XGBoost downscaling model to ERA5 data
to produce high-resolution MSWX-like predictions.

Author: Climate Downscaling Team
Date: November 2024
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import xgboost as xgb
import yaml
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm


def load_model(model_dir: str, model_name: str) -> Tuple[xgb.Booster, dict]:
    """
    Load trained XGBoost model and metadata.
    
    Args:
        model_dir: Directory containing the model
        model_name: Base name of the model
        
    Returns:
        Tuple of (model, metadata)
    """
    print(f"Loading model from: {model_dir}")
    
    # Load model
    model_path = os.path.join(model_dir, f"{model_name}.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"  ✓ Model loaded: {model_path}")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.yaml")
    with open(metadata_path, 'r') as f:
        metadata = yaml.unsafe_load(f)  # Use unsafe_load to handle numpy objects
    print(f"  ✓ Metadata loaded: {metadata_path}")
    
    print(f"\nModel information:")
    print(f"  Created: {metadata.get('created_at', 'unknown')}")
    print(f"  Boosting rounds: {metadata.get('num_boost_round', 'unknown')}")
    print(f"  Training metrics:")
    for k, v in metadata.get('metrics', {}).items():
        try:
            print(f"    {k}: {float(v):.4f}")
        except (ValueError, TypeError):
            print(f"    {k}: {v}")
    
    return model, metadata


def load_era5_data(era5_path: str):
    """
    Load ERA5 data for inference.
    
    Args:
        era5_path: Path to ERA5 .npz file
        
    Returns:
        Tuple of (temperature, coordinates)
    """
    print(f"\nLoading ERA5 data: {era5_path}")
    
    era5 = np.load(era5_path, allow_pickle=True)
    era5_temp = era5['t2m']
    era5_times = era5['coord_valid_time']
    era5_lat = era5['coord_latitude']
    era5_lon = era5['coord_longitude']
    
    print(f"  Shape: {era5_temp.shape} (time, lat, lon)")
    print(f"  Time range: {era5_times[0]} to {era5_times[-1]}")
    print(f"  Spatial extent: lat [{era5_lat.min():.2f}, {era5_lat.max():.2f}], "
          f"lon [{era5_lon.min():.2f}, {era5_lon.max():.2f}]")
    
    return era5_temp, (era5_lat, era5_lon, era5_times)


def load_target_grid(target_grid_file: str):
    """
    Load target grid specification (from reference MSWX data).
    
    Args:
        target_grid_file: Path to reference .npz file with target grid
        
    Returns:
        Tuple of (lat, lon) coordinates for target grid
    """
    print(f"\nLoading target grid specification: {target_grid_file}")
    
    data = np.load(target_grid_file, allow_pickle=True)
    
    # Try different possible coordinate names
    if 'coord_lat' in data:
        lat = data['coord_lat']
        lon = data['coord_lon']
    elif 'coord_latitude' in data:
        lat = data['coord_latitude']
        lon = data['coord_longitude']
    else:
        raise ValueError("Could not find latitude/longitude coordinates in target grid file")
    
    print(f"  Target grid: {len(lat)} x {len(lon)} = {len(lat) * len(lon)} pixels")
    print(f"  Lat range: [{lat.min():.2f}, {lat.max():.2f}], resolution: ~{abs(lat[1]-lat[0]):.3f}°")
    print(f"  Lon range: [{lon.min():.2f}, {lon.max():.2f}], resolution: ~{abs(lon[1]-lon[0]):.3f}°")
    
    return lat, lon


def create_inference_features(era5_temp: np.ndarray,
                              era5_coords: Tuple,
                              target_lat: np.ndarray,
                              target_lon: np.ndarray,
                              time_idx: int) -> np.ndarray:
    """
    Create features for a single timestep for the entire target grid.
    
    Args:
        era5_temp: ERA5 temperature data (time, lat, lon)
        era5_coords: ERA5 (lat, lon, time) coordinates
        target_lat: Target latitude coordinates
        target_lon: Target longitude coordinates
        time_idx: Index of timestep to process
        
    Returns:
        Feature array (n_pixels, n_features)
    """
    era5_lat, era5_lon, era5_times = era5_coords
    
    # Get data for this timestep
    era5_celsius = era5_temp[time_idx] - 273.15
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (era5_lat, era5_lon),
        era5_celsius,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    # Create target grid
    lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    # Interpolate to target grid
    interp_temp = interpolator(points).reshape(len(target_lat), len(target_lon))
    
    # Compute spatial gradients
    lat_grad = np.gradient(interp_temp, axis=0)
    lon_grad = np.gradient(interp_temp, axis=1)
    
    # Normalize spatial coordinates
    lat_norm = (target_lat - target_lat.mean()) / target_lat.std()
    lon_norm = (target_lon - target_lon.mean()) / target_lon.std()
    
    # Create meshgrid for normalized coordinates
    lon_norm_grid, lat_norm_grid = np.meshgrid(lon_norm, lat_norm)
    
    # Extract temporal features
    date = era5_times[time_idx]
    day_of_year = date.astype('datetime64[D]').astype(int) % 365
    year = date.astype('datetime64[Y]').astype(int) + 1970
    
    # Normalize temporal features
    day_of_year_norm = (day_of_year - 182.5) / 182.5
    year_norm = (year - 2010) / 10
    
    # Assemble features
    n_pixels = len(target_lat) * len(target_lon)
    features = np.zeros((n_pixels, 7), dtype=np.float32)
    
    features[:, 0] = interp_temp.ravel()
    features[:, 1] = lat_norm_grid.ravel()
    features[:, 2] = lon_norm_grid.ravel()
    features[:, 3] = lat_grad.ravel()
    features[:, 4] = lon_grad.ravel()
    features[:, 5] = day_of_year_norm
    features[:, 6] = year_norm
    
    return features


def run_inference(model: xgb.Booster,
                 era5_temp: np.ndarray,
                 era5_coords: Tuple,
                 target_lat: np.ndarray,
                 target_lon: np.ndarray,
                 output_path: str,
                 chunk_size: int = 100):
    """
    Run inference on all timesteps and save results.
    
    Args:
        model: Trained XGBoost model
        era5_temp: ERA5 temperature data
        era5_coords: ERA5 coordinates
        target_lat: Target latitude grid
        target_lon: Target longitude grid
        output_path: Path to save output .npz file
        chunk_size: Number of timesteps to process at once (memory management)
    """
    era5_lat, era5_lon, era5_times = era5_coords
    n_times = len(era5_times)
    n_target_lat = len(target_lat)
    n_target_lon = len(target_lon)
    
    print(f"\nRunning inference...")
    print(f"  Timesteps: {n_times}")
    print(f"  Target grid: {n_target_lat} x {n_target_lon}")
    print(f"  Chunk size: {chunk_size} timesteps")
    
    # Pre-allocate output array
    predictions = np.zeros((n_times, n_target_lat, n_target_lon), dtype=np.float32)
    
    # Process in chunks to manage memory
    n_chunks = (n_times + chunk_size - 1) // chunk_size
    
    start_time = time.time()
    
    with tqdm(total=n_times, desc="  Timesteps") as pbar:
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_times)
            
            # Process each timestep in chunk
            for t in range(start_idx, end_idx):
                # Create features
                features = create_inference_features(
                    era5_temp, era5_coords, target_lat, target_lon, t
                )
                
                # Run prediction
                dmatrix = xgb.DMatrix(features)
                pred = model.predict(dmatrix)
                
                # Reshape and store
                predictions[t] = pred.reshape(n_target_lat, n_target_lon)
                
                pbar.update(1)
    
    elapsed = time.time() - start_time
    print(f"  Inference completed in {elapsed:.1f} seconds ({elapsed/n_times:.3f} s/timestep)")
    
    # Save results
    print(f"\nSaving predictions to: {output_path}")
    np.savez_compressed(
        output_path,
        air_temperature=predictions,
        coord_lat=target_lat,
        coord_lon=target_lon,
        coord_time=era5_times
    )
    
    # Calculate output size
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"  File size: {file_size_mb:.1f} MB")
    
    # Print statistics
    print(f"\nPrediction statistics:")
    print(f"  Min: {np.nanmin(predictions):.2f} °C")
    print(f"  Max: {np.nanmax(predictions):.2f} °C")
    print(f"  Mean: {np.nanmean(predictions):.2f} °C")
    print(f"  Std: {np.nanstd(predictions):.2f} °C")


def main():
    parser = argparse.ArgumentParser(description='XGBoost downscaling inference')
    parser.add_argument('--model-dir', required=True, help='Directory containing trained model')
    parser.add_argument('--model-name', required=True, help='Model name (without extension)')
    parser.add_argument('--era5-data', required=True, help='Path to ERA5 .npz file for inference')
    parser.add_argument('--target-grid', required=True, 
                       help='Path to reference .npz file with target grid')
    parser.add_argument('--output', required=True, help='Output path for predictions (.npz)')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Timesteps to process at once (for memory management)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("XGBoost Downscaling Inference")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model directory: {args.model_dir}")
    print(f"  Model name: {args.model_name}")
    print(f"  ERA5 data: {args.era5_data}")
    print(f"  Target grid: {args.target_grid}")
    print(f"  Output: {args.output}")
    print(f"  Chunk size: {args.chunk_size}")
    
    # Load model
    model, metadata = load_model(args.model_dir, args.model_name)
    
    # Load ERA5 data
    era5_temp, era5_coords = load_era5_data(args.era5_data)
    
    # Load target grid
    target_lat, target_lon = load_target_grid(args.target_grid)
    
    # Run inference
    run_inference(
        model, era5_temp, era5_coords, 
        target_lat, target_lon, 
        args.output, args.chunk_size
    )
    
    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

