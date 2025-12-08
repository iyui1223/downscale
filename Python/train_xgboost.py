#!/usr/bin/env python3
"""
XGBoost-based Statistical Downscaling Training Script

This script trains an XGBoost model to downscale ERA5 (low-resolution) to MSWX (high-resolution)
climate data using statistical downscaling techniques with spatial feature engineering.

Date: November 2024
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import xgboost as xgb
import yaml
from scipy.interpolate import RegularGridInterpolator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from tqdm import tqdm


def load_data(era5_path: str, mswx_path: str) -> Tuple[np.ndarray, ...]:
    """
    Load preprocessed ERA5 and MSWX data using memory-mapped arrays.
       
    Args:
        era5_path: Path to ERA5 .npz file
        mswx_path: Path to MSWX .npz file
        
    Returns:
        Tuple of (era5_data, mswx_data, era5_coords, mswx_coords)
    """
    print("Loading data (memory-mapped)...")
    print(f"  ERA5: {era5_path}")
    print(f"  MSWX: {mswx_path}")
    
    # Load ERA5 (low-resolution input) - memory-mapped
    era5 = np.load(era5_path, mmap_mode='r', allow_pickle=True)
    era5_temp = era5['t2m']  # Shape: (time, lat, lon) - not loaded until accessed
    era5_times = np.array(era5['coord_valid_time'])  # Small, load fully
    era5_lat = np.array(era5['coord_latitude'])      # Small, load fully
    era5_lon = np.array(era5['coord_longitude'])     # Small, load fully
    
    # Load MSWX (high-resolution target) - memory-mapped
    mswx = np.load(mswx_path, mmap_mode='r', allow_pickle=True)
    mswx_temp = mswx['air_temperature']  # Shape: (time, lat, lon) - not loaded until accessed
    mswx_times = np.array(mswx['coord_time'])  # Small, load fully
    mswx_lat = np.array(mswx['coord_lat'])     # Small, load fully
    mswx_lon = np.array(mswx['coord_lon'])     # Small, load fully
    
    # Verify temporal alignment
    if not np.array_equal(era5_times, mswx_times):
        raise ValueError("ERA5 and MSWX time coordinates do not match!")
    
    print(f"\nData loaded (memory-mapped, not in RAM yet):")
    print(f"  ERA5 shape: {era5_temp.shape} (time, lat, lon)")
    print(f"  MSWX shape: {mswx_temp.shape} (time, lat, lon)")
    print(f"  Time steps: {len(era5_times)}")
    print(f"  Time range: {era5_times[0]} to {era5_times[-1]}")
    
    return (era5_temp, mswx_temp, 
            (era5_lat, era5_lon, era5_times),
            (mswx_lat, mswx_lon, mswx_times))


def create_features(era5_temp: np.ndarray,
                    era5_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    mswx_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    sample_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training features by interpolating ERA5 to MSWX grid and adding spatial features.
    
    Features include:
    - Bilinear interpolated ERA5 temperature at target location
    - Target latitude and longitude (normalized)
    - Temporal features (day of year, year)
    - Local ERA5 gradients (spatial derivatives)
    
    Args:
        era5_temp: ERA5 temperature data (time, lat, lon)
        era5_coords: ERA5 (lat, lon, time) coordinates
        mswx_coords: MSWX (lat, lon, time) coordinates
        sample_ratio: Fraction of data to use (for faster training on subset)
        
    Returns:
        Tuple of (features, target_indices) for sampled points
    """
    era5_lat, era5_lon, era5_times = era5_coords
    mswx_lat, mswx_lon, mswx_times = mswx_coords
    
    n_times = era5_temp.shape[0]
    n_mswx_lat = len(mswx_lat)
    n_mswx_lon = len(mswx_lon)
    
    print("\nCreating features with spatial interpolation...")
    print(f"  Target grid: {n_mswx_lat} x {n_mswx_lon} = {n_mswx_lat * n_mswx_lon} pixels per timestep")
    print(f"  Sample ratio: {sample_ratio:.2%}")
    
    # Pre-allocate feature array
    n_total_samples = n_times * n_mswx_lat * n_mswx_lon
    n_samples = int(n_total_samples * sample_ratio)
    n_features = 7  # [interp_temp, lat, lon, lat_grad, lon_grad, day_of_year, year]
    
    features = np.zeros((n_samples, n_features), dtype=np.float32)
    target_indices = np.zeros((n_samples, 3), dtype=np.int32)  # [time_idx, lat_idx, lon_idx]
    
    # Create normalized spatial coordinates
    lat_norm = (mswx_lat - mswx_lat.mean()) / mswx_lat.std()
    lon_norm = (mswx_lon - mswx_lon.mean()) / mswx_lon.std()
    
    # Create meshgrid for target locations
    lon_grid, lat_grid = np.meshgrid(mswx_lon, mswx_lat)
    
    # Generate random sample indices if not using all data
    if sample_ratio < 1.0:
        sample_indices = np.random.choice(n_total_samples, n_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    else:
        sample_indices = np.arange(n_total_samples)
    
    print(f"  Processing {n_samples:,} samples...")
    
    # Process each timestep one at a time (memory-efficient)
    # With mmap_mode='r', only the current timestep slice is loaded into RAM
    sample_idx = 0
    with tqdm(total=n_times, desc="  Timesteps") as pbar:
        for t in range(n_times):
            # Create interpolator for this timestep
            # Only era5_temp[t] is loaded from disk (memory-mapped)
            # Note: ERA5 temperature is in Kelvin, we'll convert to Celsius
            era5_celsius = np.array(era5_temp[t]) - 273.15  # Explicit copy to RAM for processing
            interpolator = RegularGridInterpolator(
                (era5_lat, era5_lon), 
                era5_celsius,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            
            # Interpolate ERA5 to MSWX grid
            points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            interp_temp = interpolator(points).reshape(n_mswx_lat, n_mswx_lon)
            
            # Compute spatial gradients (approximate derivatives)
            lat_grad = np.gradient(interp_temp, axis=0)
            lon_grad = np.gradient(interp_temp, axis=1)
            
            # Extract temporal features
            date = era5_times[t]
            day_of_year = date.astype('datetime64[D]').astype(int) % 365
            year = date.astype('datetime64[Y]').astype(int) + 1970
            
            # Normalize temporal features
            day_of_year_norm = (day_of_year - 182.5) / 182.5  # Center around middle of year
            year_norm = (year - 2010) / 10  # Center around 2010
            
            # Fill features for this timestep
            t_start = t * n_mswx_lat * n_mswx_lon
            t_end = (t + 1) * n_mswx_lat * n_mswx_lon
            
            # Get sample indices for this timestep
            t_sample_mask = (sample_indices >= t_start) & (sample_indices < t_end)
            t_sample_indices = sample_indices[t_sample_mask] - t_start
            n_t_samples = len(t_sample_indices)
            
            if n_t_samples > 0:
                # Convert flat indices to 2D indices
                lat_indices = t_sample_indices // n_mswx_lon
                lon_indices = t_sample_indices % n_mswx_lon
                
                # Build features
                features[sample_idx:sample_idx+n_t_samples, 0] = interp_temp.ravel()[t_sample_indices]
                features[sample_idx:sample_idx+n_t_samples, 1] = lat_norm[lat_indices]
                features[sample_idx:sample_idx+n_t_samples, 2] = lon_norm[lon_indices]
                features[sample_idx:sample_idx+n_t_samples, 3] = lat_grad.ravel()[t_sample_indices]
                features[sample_idx:sample_idx+n_t_samples, 4] = lon_grad.ravel()[t_sample_indices]
                features[sample_idx:sample_idx+n_t_samples, 5] = day_of_year_norm
                features[sample_idx:sample_idx+n_t_samples, 6] = year_norm
                
                # Store target indices
                target_indices[sample_idx:sample_idx+n_t_samples, 0] = t
                target_indices[sample_idx:sample_idx+n_t_samples, 1] = lat_indices
                target_indices[sample_idx:sample_idx+n_t_samples, 2] = lon_indices
                
                sample_idx += n_t_samples
            
            pbar.update(1)
    
    print(f"  Features created: {features.shape}")
    return features, target_indices


def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        params: Dict[str, Any]) -> xgb.Booster:
    """
    Train XGBoost model with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: XGBoost parameters
        
    Returns:
        Trained XGBoost booster
    """
    print("\nTraining XGBoost model...")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Training parameters
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # Train model
    print("\n  XGBoost Parameters:")
    for k, v in params.items():
        print(f"    {k}: {v}")
    
    start_time = time.time()
    
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get('num_boost_round', 1000),
        evals=evals,
        early_stopping_rounds=params.get('early_stopping_rounds', 50),
        verbose_eval=params.get('verbose_eval', 50)
    )
    
    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time:.1f} seconds")
    print(f"  Best iteration: {bst.best_iteration}")
    print(f"  Best score: {bst.best_score:.6f}")
    
    return bst


def evaluate_model(model: xgb.Booster, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating model...")
    
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'bias': bias
    }
    
    print("\n  Test Metrics:")
    print(f"    RMSE: {rmse:.4f} °C")
    print(f"    MAE:  {mae:.4f} °C")
    print(f"    R²:   {r2:.4f}")
    print(f"    Bias: {bias:.4f} °C")
    
    return metrics


def save_model_and_metadata(model: xgb.Booster, 
                           metrics: Dict[str, float],
                           feature_names: list,
                           output_dir: str,
                           model_name: str = "xgboost_downscale"):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained XGBoost model
        metrics: Evaluation metrics
        feature_names: List of feature names
        output_dir: Output directory
        model_name: Base name for model files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.json")
    model.save_model(model_path)
    print(f"\n  Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'xgboost',
        'task': 'ERA5_to_MSWX_downscaling',
        'variable': 'temperature_max',
        'feature_names': feature_names,
        'metrics': metrics,
        'num_boost_round': model.best_iteration + 1,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"  Metadata saved: {metadata_path}")
    
    # Save feature importance
    importance = model.get_score(importance_type='gain')
    importance_dict = {k: float(v) for k, v in importance.items()}
    
    importance_path = os.path.join(output_dir, f"{model_name}_feature_importance.yaml")
    with open(importance_path, 'w') as f:
        yaml.dump(importance_dict, f, default_flow_style=False)
    print(f"  Feature importance saved: {importance_path}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost downscaling model')
    parser.add_argument('--era5-data', required=True, help='Path to ERA5 .npz file')
    parser.add_argument('--mswx-data', required=True, help='Path to MSWX .npz file')
    parser.add_argument('--output-dir', required=True, help='Output directory for model')
    parser.add_argument('--model-name', default='xgboost_downscale_tmax', help='Model name')
    parser.add_argument('--sample-ratio', type=float, default=0.1, 
                       help='Fraction of data to use for training (0.0-1.0)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test split ratio')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # XGBoost parameters
    parser.add_argument('--max-depth', type=int, default=8, help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--n-estimators', type=int, default=1000, help='Number of boosting rounds')
    parser.add_argument('--subsample', type=float, default=0.8, help='Subsample ratio')
    parser.add_argument('--colsample-bytree', type=float, default=0.8, help='Column subsample ratio')
    parser.add_argument('--early-stopping', type=int, default=50, help='Early stopping rounds')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel threads')
    
    args = parser.parse_args()
    
    print("="*80)
    print("XGBoost Statistical Downscaling Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  ERA5 data: {args.era5_data}")
    print(f"  MSWX data: {args.mswx_data}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model name: {args.model_name}")
    print(f"  Sample ratio: {args.sample_ratio:.2%}")
    print(f"  Validation split: {args.val_split:.2%}")
    print(f"  Test split: {args.test_split:.2%}")
    print(f"  Random seed: {args.random_seed}")
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load data
    era5_temp, mswx_temp, era5_coords, mswx_coords = load_data(
        args.era5_data, args.mswx_data
    )
    
    # Create features
    features, target_indices = create_features(
        era5_temp, era5_coords, mswx_coords, 
        sample_ratio=args.sample_ratio
    )
    
    # Extract targets (memory-efficient: process timestep by timestep)
    print("\nExtracting target values...")
    targets = np.zeros(len(target_indices), dtype=np.float32)
    
    # Group indices by timestep to minimize disk reads from memory-mapped array
    unique_times = np.unique(target_indices[:, 0])
    for t in tqdm(unique_times, desc="  Extracting targets"):
        t_mask = target_indices[:, 0] == t
        t_lat_idx = target_indices[t_mask, 1]
        t_lon_idx = target_indices[t_mask, 2]
        # Load only this timestep from disk (memory-mapped)
        mswx_slice = np.array(mswx_temp[t])
        targets[t_mask] = mswx_slice[t_lat_idx, t_lon_idx]
    
    print(f"  Targets shape: {targets.shape}")
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(targets))
    features = features[valid_mask]
    targets = targets[valid_mask]
    print(f"  Valid samples after NaN removal: {len(features):,}")
    
    # Split data: train, validation, test
    print("\nSplitting data...")
    test_size = args.test_split
    val_size = args.val_split / (1 - test_size)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=args.random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=args.random_seed
    )
    
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(features)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(features)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(features)*100:.1f}%)")
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'nthread': args.n_jobs if args.n_jobs > 0 else None,
        'num_boost_round': args.n_estimators,
        'early_stopping_rounds': args.early_stopping,
        'verbose_eval': 50,
        'seed': args.random_seed
    }
    
    # Train model
    model = train_xgboost_model(X_train, y_train, X_val, y_val, xgb_params)
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model and metadata
    feature_names = [
        'era5_interpolated', 'latitude_norm', 'longitude_norm',
        'lat_gradient', 'lon_gradient', 'day_of_year_norm', 'year_norm'
    ]
    
    save_model_and_metadata(
        model, metrics, feature_names, 
        args.output_dir, args.model_name
    )
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

