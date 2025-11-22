#!/usr/bin/env python3
"""
XGBoost Downscaling Evaluation Script

This script evaluates downscaling predictions against ground truth MSWX data,
computing metrics and creating visualizations.

Author: Climate Downscaling Team
Date: November 2024
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml


def load_data(predictions_path: str, ground_truth_path: str) -> Tuple:
    """
    Load predictions and ground truth data.
    
    Args:
        predictions_path: Path to predictions .npz file
        ground_truth_path: Path to ground truth .npz file
        
    Returns:
        Tuple of (predictions, ground_truth, coords)
    """
    print("Loading data...")
    print(f"  Predictions: {predictions_path}")
    print(f"  Ground truth: {ground_truth_path}")
    
    # Load predictions
    pred_data = np.load(predictions_path, allow_pickle=True)
    predictions = pred_data['air_temperature']
    pred_times = pred_data['coord_time']
    pred_lat = pred_data['coord_lat']
    pred_lon = pred_data['coord_lon']
    
    # Load ground truth
    truth_data = np.load(ground_truth_path, allow_pickle=True)
    ground_truth = truth_data['air_temperature']
    truth_times = truth_data['coord_time']
    truth_lat = truth_data['coord_lat']
    truth_lon = truth_data['coord_lon']
    
    print(f"\n  Predictions shape: {predictions.shape}")
    print(f"  Ground truth shape: {ground_truth.shape}")
    
    # Verify alignment
    if not np.array_equal(pred_times, truth_times):
        raise ValueError("Time coordinates do not match!")
    if not np.array_equal(pred_lat, truth_lat):
        raise ValueError("Latitude coordinates do not match!")
    if not np.array_equal(pred_lon, truth_lon):
        raise ValueError("Longitude coordinates do not match!")
    
    print("  ✓ Data shapes and coordinates match")
    
    return predictions, ground_truth, (pred_lat, pred_lon, pred_times)


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    print("\nComputing metrics...")
    
    # Flatten arrays and remove NaNs
    pred_flat = predictions.ravel()
    truth_flat = ground_truth.ravel()
    
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(truth_flat))
    pred_valid = pred_flat[valid_mask]
    truth_valid = truth_flat[valid_mask]
    
    print(f"  Valid samples: {len(pred_valid):,} / {len(pred_flat):,} "
          f"({100*len(pred_valid)/len(pred_flat):.1f}%)")
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(truth_valid, pred_valid))
    mae = mean_absolute_error(truth_valid, pred_valid)
    r2 = r2_score(truth_valid, pred_valid)
    bias = np.mean(pred_valid - truth_valid)
    
    # Percentile-based errors
    errors = pred_valid - truth_valid
    mae_50 = np.median(np.abs(errors))
    mae_90 = np.percentile(np.abs(errors), 90)
    mae_95 = np.percentile(np.abs(errors), 95)
    
    # Correlation
    correlation = np.corrcoef(pred_valid, truth_valid)[0, 1]
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'bias': float(bias),
        'correlation': float(correlation),
        'mae_median': float(mae_50),
        'mae_90th': float(mae_90),
        'mae_95th': float(mae_95),
        'n_valid_samples': int(len(pred_valid)),
        'n_total_samples': int(len(pred_flat))
    }
    
    print("\n  Metrics:")
    print(f"    RMSE:            {rmse:.4f} °C")
    print(f"    MAE:             {mae:.4f} °C")
    print(f"    MAE (median):    {mae_50:.4f} °C")
    print(f"    MAE (90th):      {mae_90:.4f} °C")
    print(f"    MAE (95th):      {mae_95:.4f} °C")
    print(f"    Bias:            {bias:.4f} °C")
    print(f"    R²:              {r2:.4f}")
    print(f"    Correlation:     {correlation:.4f}")
    
    return metrics


def compute_spatial_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Compute spatially-aggregated metrics.
    
    Args:
        predictions: Predicted values (time, lat, lon)
        ground_truth: Ground truth values (time, lat, lon)
        
    Returns:
        Dictionary of spatial metrics
    """
    print("\nComputing spatial metrics...")
    
    # Compute time-averaged fields
    pred_mean = np.nanmean(predictions, axis=0)
    truth_mean = np.nanmean(ground_truth, axis=0)
    
    # Compute spatial RMSE
    spatial_errors = pred_mean - truth_mean
    spatial_rmse = np.sqrt(np.nanmean(spatial_errors**2))
    spatial_bias = np.nanmean(spatial_errors)
    
    # Compute temporal metrics at each grid point
    temporal_rmse = np.sqrt(np.nanmean((predictions - ground_truth)**2, axis=0))
    
    metrics = {
        'spatial_rmse': float(spatial_rmse),
        'spatial_bias': float(spatial_bias),
        'temporal_rmse_mean': float(np.nanmean(temporal_rmse)),
        'temporal_rmse_std': float(np.nanstd(temporal_rmse)),
        'temporal_rmse_min': float(np.nanmin(temporal_rmse)),
        'temporal_rmse_max': float(np.nanmax(temporal_rmse))
    }
    
    print(f"    Spatial RMSE:           {spatial_rmse:.4f} °C")
    print(f"    Spatial bias:           {spatial_bias:.4f} °C")
    print(f"    Temporal RMSE (mean):   {metrics['temporal_rmse_mean']:.4f} °C")
    print(f"    Temporal RMSE (range):  [{metrics['temporal_rmse_min']:.4f}, "
          f"{metrics['temporal_rmse_max']:.4f}] °C")
    
    return metrics, temporal_rmse, pred_mean, truth_mean


def create_visualizations(predictions: np.ndarray,
                         ground_truth: np.ndarray,
                         coords: Tuple,
                         temporal_rmse: np.ndarray,
                         pred_mean: np.ndarray,
                         truth_mean: np.ndarray,
                         output_dir: str):
    """
    Create evaluation visualizations.
    
    Args:
        predictions: Predicted values
        ground_truth: Ground truth values
        coords: (lat, lon, time) coordinates
        temporal_rmse: Temporal RMSE at each grid point
        pred_mean: Time-averaged predictions
        truth_mean: Time-averaged ground truth
        output_dir: Output directory for plots
    """
    print("\nCreating visualizations...")
    
    lat, lon, times = coords
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # 1. Scatter plot: predictions vs ground truth
    print("  - Scatter plot...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Sample for plotting (too many points otherwise)
    sample_size = min(50000, predictions.size)
    sample_idx = np.random.choice(predictions.size, sample_size, replace=False)
    pred_sample = predictions.ravel()[sample_idx]
    truth_sample = ground_truth.ravel()[sample_idx]
    
    # Remove NaNs
    valid = ~(np.isnan(pred_sample) | np.isnan(truth_sample))
    pred_sample = pred_sample[valid]
    truth_sample = truth_sample[valid]
    
    ax.hexbin(truth_sample, pred_sample, gridsize=50, cmap='Blues', mincnt=1)
    ax.plot([truth_sample.min(), truth_sample.max()],
            [truth_sample.min(), truth_sample.max()],
            'r--', lw=2, label='1:1 line')
    
    ax.set_xlabel('Ground Truth (°C)', fontsize=12)
    ax.set_ylabel('Predictions (°C)', fontsize=12)
    ax.set_title('Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'))
    plt.close()
    
    # 2. Error distribution
    print("  - Error distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    errors = (predictions - ground_truth).ravel()
    errors = errors[~np.isnan(errors)]
    
    # Histogram
    axes[0].hist(errors, bins=100, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].set_xlabel('Error (°C)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()
    
    # 3. Spatial maps
    print("  - Spatial maps...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    
    # Ground truth mean
    im0 = axes[0, 0].imshow(truth_mean, extent=extent, origin='lower', 
                            cmap='RdYlBu_r', aspect='auto')
    axes[0, 0].set_title('Ground Truth (Time Average)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im0, ax=axes[0, 0], label='Temperature (°C)')
    
    # Predictions mean
    im1 = axes[0, 1].imshow(pred_mean, extent=extent, origin='lower',
                            cmap='RdYlBu_r', aspect='auto')
    axes[0, 1].set_title('Predictions (Time Average)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 1], label='Temperature (°C)')
    
    # Bias (pred - truth)
    bias_map = pred_mean - truth_mean
    vmax_bias = max(abs(bias_map.min()), abs(bias_map.max()))
    im2 = axes[1, 0].imshow(bias_map, extent=extent, origin='lower',
                            cmap='RdBu_r', aspect='auto', vmin=-vmax_bias, vmax=vmax_bias)
    axes[1, 0].set_title('Bias (Prediction - Truth)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1, 0], label='Temperature Bias (°C)')
    
    # Temporal RMSE
    im3 = axes[1, 1].imshow(temporal_rmse, extent=extent, origin='lower',
                            cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('Temporal RMSE', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[1, 1], label='RMSE (°C)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_maps.png'))
    plt.close()
    
    # 4. Time series at selected locations
    print("  - Time series...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Select 4 representative locations
    lat_idx = [len(lat)//4, len(lat)//4, 3*len(lat)//4, 3*len(lat)//4]
    lon_idx = [len(lon)//4, 3*len(lon)//4, len(lon)//4, 3*len(lon)//4]
    
    for i, (ax, li, loi) in enumerate(zip(axes.ravel(), lat_idx, lon_idx)):
        pred_ts = predictions[:, li, loi]
        truth_ts = ground_truth[:, li, loi]
        
        # Plot first 365 days for clarity
        n_plot = min(365, len(times))
        
        ax.plot(range(n_plot), truth_ts[:n_plot], 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(range(n_plot), pred_ts[:n_plot], 'r-', label='Prediction', alpha=0.7)
        ax.set_xlabel('Day', fontsize=10)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(f'Location: Lat={lat[li]:.2f}°, Lon={lon[loi]:.2f}°',
                    fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series.png'))
    plt.close()
    
    print(f"  ✓ Visualizations saved to: {output_dir}")


def save_metrics(metrics: Dict, spatial_metrics: Dict, output_path: str):
    """
    Save metrics to YAML file.
    
    Args:
        metrics: Overall metrics
        spatial_metrics: Spatial metrics
        output_path: Output file path
    """
    all_metrics = {
        'overall': metrics,
        'spatial': spatial_metrics
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(all_metrics, f, default_flow_style=False)
    
    print(f"\n  ✓ Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate XGBoost downscaling predictions')
    parser.add_argument('--predictions', required=True, help='Path to predictions .npz file')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth .npz file')
    parser.add_argument('--output-dir', required=True, help='Output directory for evaluation results')
    parser.add_argument('--output-name', default='evaluation', help='Base name for output files')
    
    args = parser.parse_args()
    
    print("="*80)
    print("XGBoost Downscaling Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Predictions: {args.predictions}")
    print(f"  Ground truth: {args.ground_truth}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output name: {args.output_name}")
    print()
    
    # Load data
    predictions, ground_truth, coords = load_data(args.predictions, args.ground_truth)
    
    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth)
    spatial_metrics, temporal_rmse, pred_mean, truth_mean = compute_spatial_metrics(
        predictions, ground_truth
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'{args.output_name}_metrics.yaml')
    save_metrics(metrics, spatial_metrics, metrics_path)
    
    # Create visualizations
    create_visualizations(
        predictions, ground_truth, coords,
        temporal_rmse, pred_mean, truth_mean,
        args.output_dir
    )
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Plots: {args.output_dir}/*.png")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

