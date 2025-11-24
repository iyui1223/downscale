#!/usr/bin/env python3
"""
Downscaling Model Evaluation Script

This script evaluates downscaling predictions against ground truth MSWX data,
computing metrics and creating visualizations. Works with any downscaling model.

Author: Climate Downscaling Team
Date: November 2024
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import RegularGridInterpolator
import yaml


def load_data(predictions_path: str, ground_truth_path: str, era5_path: Optional[str] = None) -> Tuple:
    """
    Load predictions, ground truth, and optionally ERA5 input data.
    
    Args:
        predictions_path: Path to predictions .npz file
        ground_truth_path: Path to ground truth .npz file
        era5_path: Optional path to ERA5 input .npz file
        
    Returns:
        Tuple of (predictions, ground_truth, era5_interp, coords)
    """
    print("Loading data...")
    print(f"  Predictions: {predictions_path}")
    print(f"  Ground truth: {ground_truth_path}")
    if era5_path:
        print(f"  ERA5 input: {era5_path}")
    
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
    
    # Load and interpolate ERA5 if provided
    era5_interp = None
    if era5_path:
        print("\n  Interpolating ERA5 to high-resolution grid...")
        era5_data = np.load(era5_path, allow_pickle=True)
        era5_temp = era5_data['t2m']  # ERA5 uses 't2m' for temperature
        era5_lat = era5_data['coord_latitude']
        era5_lon = era5_data['coord_longitude']
        era5_times = era5_data['coord_valid_time']
        
        print(f"    ERA5 shape: {era5_temp.shape}")
        print(f"    ERA5 grid: {len(era5_lat)}x{len(era5_lon)}")
        print(f"    Target grid: {len(pred_lat)}x{len(pred_lon)}")
        
        # Verify time alignment
        if not np.array_equal(era5_times, pred_times):
            raise ValueError("ERA5 time coordinates do not match predictions!")
        
        # Interpolate ERA5 to high-resolution grid
        era5_interp = np.zeros_like(predictions)
        for t in range(len(pred_times)):
            if t % 1000 == 0:
                print(f"    Interpolating timestep {t}/{len(pred_times)}...")
            
            # Create interpolator for this timestep
            interpolator = RegularGridInterpolator(
                (era5_lat, era5_lon),
                era5_temp[t, :, :],
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            
            # Create meshgrid for target coordinates
            lon_grid, lat_grid = np.meshgrid(pred_lon, pred_lat)
            points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            
            # Interpolate
            era5_interp[t, :, :] = interpolator(points).reshape(len(pred_lat), len(pred_lon))
        
        print(f"  ✓ ERA5 interpolated to high-resolution grid")
    
    return predictions, ground_truth, era5_interp, (pred_lat, pred_lon, pred_times)


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray, era5_interp: Optional[np.ndarray] = None) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        ground_truth: Ground truth values
        era5_interp: Optional interpolated ERA5 input (for difference-based correlation)
        
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
    
    # Correlation - standard (between predictions and ground truth)
    correlation = np.corrcoef(pred_valid, truth_valid)[0, 1]
    
    # Difference-based correlation (if ERA5 available)
    correlation_diff = None
    if era5_interp is not None:
        print("\n  Computing difference-based correlation...")
        print("    Diff1 = Ground Truth - ERA5 (validation improvement)")
        print("    Diff2 = Predictions - ERA5 (predicted improvement)")
        
        era5_flat = era5_interp.ravel()
        valid_mask_era5 = ~(np.isnan(pred_flat) | np.isnan(truth_flat) | np.isnan(era5_flat))
        
        pred_valid_era5 = pred_flat[valid_mask_era5]
        truth_valid_era5 = truth_flat[valid_mask_era5]
        era5_valid = era5_flat[valid_mask_era5]
        
        # Compute differences
        diff_truth = truth_valid_era5 - era5_valid  # True improvement from ERA5
        diff_pred = pred_valid_era5 - era5_valid   # Predicted improvement from ERA5
        
        # Correlation of the differences
        correlation_diff = np.corrcoef(diff_truth, diff_pred)[0, 1]
        
        print(f"    Mean true improvement:      {np.mean(diff_truth):.4f} °C")
        print(f"    Mean predicted improvement: {np.mean(diff_pred):.4f} °C")
        print(f"    Std true improvement:       {np.std(diff_truth):.4f} °C")
        print(f"    Std predicted improvement:  {np.std(diff_pred):.4f} °C")
    
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
    
    if correlation_diff is not None:
        metrics['correlation_diff'] = float(correlation_diff)
    
    print("\n  Metrics:")
    print(f"    RMSE:                    {rmse:.4f} °C")
    print(f"    MAE:                     {mae:.4f} °C")
    print(f"    MAE (median):            {mae_50:.4f} °C")
    print(f"    MAE (90th):              {mae_90:.4f} °C")
    print(f"    MAE (95th):              {mae_95:.4f} °C")
    print(f"    Bias:                    {bias:.4f} °C")
    print(f"    R²:                      {r2:.4f}")
    print(f"    Correlation:             {correlation:.4f}")
    if correlation_diff is not None:
        print(f"    Correlation (diff-based): {correlation_diff:.4f}")
    
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
                         output_dir: str,
                         era5_interp: Optional[np.ndarray] = None):
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
        era5_interp: Optional interpolated ERA5 input
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
    
    # 5. Spatial distribution maps for errors and temperatures
    print("  - Spatial distribution maps...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    
    # Temperature distributions
    im0 = axes[0, 0].imshow(truth_mean, extent=extent, origin='lower', 
                            cmap='RdYlBu_r', aspect='auto')
    axes[0, 0].set_title('Ground Truth - Mean Temperature', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im0, ax=axes[0, 0], label='Temperature (°C)')
    
    im1 = axes[0, 1].imshow(pred_mean, extent=extent, origin='lower',
                            cmap='RdYlBu_r', aspect='auto')
    axes[0, 1].set_title('Predictions - Mean Temperature', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 1], label='Temperature (°C)')
    
    # Temperature standard deviation (spatial variability)
    truth_std = np.nanstd(ground_truth, axis=0)
    im2 = axes[0, 2].imshow(truth_std, extent=extent, origin='lower',
                            cmap='YlOrRd', aspect='auto')
    axes[0, 2].set_title('Ground Truth - Temporal Std Dev', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Longitude')
    axes[0, 2].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[0, 2], label='Std Dev (°C)')
    
    # Error distributions
    bias_map = pred_mean - truth_mean
    vmax_bias = max(abs(np.nanpercentile(bias_map, 5)), abs(np.nanpercentile(bias_map, 95)))
    im3 = axes[1, 0].imshow(bias_map, extent=extent, origin='lower',
                            cmap='RdBu_r', aspect='auto', vmin=-vmax_bias, vmax=vmax_bias)
    axes[1, 0].set_title('Spatial Bias Map (Pred - Truth)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[1, 0], label='Bias (°C)')
    
    # Absolute error map
    abs_error_map = np.nanmean(np.abs(predictions - ground_truth), axis=0)
    im4 = axes[1, 1].imshow(abs_error_map, extent=extent, origin='lower',
                            cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('Spatial MAE Map', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    plt.colorbar(im4, ax=axes[1, 1], label='MAE (°C)')
    
    # RMSE map
    im5 = axes[1, 2].imshow(temporal_rmse, extent=extent, origin='lower',
                            cmap='YlOrRd', aspect='auto')
    axes[1, 2].set_title('Spatial RMSE Map', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Longitude')
    axes[1, 2].set_ylabel('Latitude')
    plt.colorbar(im5, ax=axes[1, 2], label='RMSE (°C)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_distribution_maps.png'))
    plt.close()
    
    # 6. If ERA5 available, create difference-based visualizations
    if era5_interp is not None:
        print("  - Difference-based spatial maps...")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        era5_mean = np.nanmean(era5_interp, axis=0)
        
        # ERA5 mean
        im0 = axes[0, 0].imshow(era5_mean, extent=extent, origin='lower', 
                                cmap='RdYlBu_r', aspect='auto')
        axes[0, 0].set_title('ERA5 (Interpolated) - Mean', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        plt.colorbar(im0, ax=axes[0, 0], label='Temperature (°C)')
        
        # True improvement (Truth - ERA5)
        true_improvement = truth_mean - era5_mean
        vmax_imp = max(abs(np.nanpercentile(true_improvement, 5)), 
                      abs(np.nanpercentile(true_improvement, 95)))
        im1 = axes[0, 1].imshow(true_improvement, extent=extent, origin='lower',
                                cmap='RdBu_r', aspect='auto', vmin=-vmax_imp, vmax=vmax_imp)
        axes[0, 1].set_title('True Improvement (Truth - ERA5)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0, 1], label='Temperature Diff (°C)')
        
        # Predicted improvement (Pred - ERA5)
        pred_improvement = pred_mean - era5_mean
        im2 = axes[0, 2].imshow(pred_improvement, extent=extent, origin='lower',
                                cmap='RdBu_r', aspect='auto', vmin=-vmax_imp, vmax=vmax_imp)
        axes[0, 2].set_title('Predicted Improvement (Pred - ERA5)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Longitude')
        axes[0, 2].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[0, 2], label='Temperature Diff (°C)')
        
        # Improvement error (Pred improvement - True improvement)
        improvement_error = pred_improvement - true_improvement
        vmax_err = max(abs(np.nanpercentile(improvement_error, 5)),
                      abs(np.nanpercentile(improvement_error, 95)))
        im3 = axes[1, 0].imshow(improvement_error, extent=extent, origin='lower',
                                cmap='RdBu_r', aspect='auto', vmin=-vmax_err, vmax=vmax_err)
        axes[1, 0].set_title('Improvement Error', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        plt.colorbar(im3, ax=axes[1, 0], label='Error (°C)')
        
        # Temporal correlation of improvements at each grid point
        print("    Computing spatial correlation of improvements...")
        correlation_map = np.zeros((len(lat), len(lon)))
        for i in range(len(lat)):
            for j in range(len(lon)):
                true_imp_ts = ground_truth[:, i, j] - era5_interp[:, i, j]
                pred_imp_ts = predictions[:, i, j] - era5_interp[:, i, j]
                
                valid = ~(np.isnan(true_imp_ts) | np.isnan(pred_imp_ts))
                if np.sum(valid) > 10:  # Need at least 10 valid points
                    correlation_map[i, j] = np.corrcoef(true_imp_ts[valid], 
                                                        pred_imp_ts[valid])[0, 1]
                else:
                    correlation_map[i, j] = np.nan
        
        im4 = axes[1, 1].imshow(correlation_map, extent=extent, origin='lower',
                                cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title('Temporal Correlation of Improvements', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        plt.colorbar(im4, ax=axes[1, 1], label='Correlation')
        
        # RMSE of improvements
        improvement_rmse = np.sqrt(np.nanmean((pred_improvement - true_improvement)**2))
        axes[1, 2].text(0.5, 0.5, 
                       f'Improvement RMSE:\n{improvement_rmse:.4f} °C\n\n'
                       f'Mean True Improvement:\n{np.nanmean(true_improvement):.4f} °C\n\n'
                       f'Mean Pred Improvement:\n{np.nanmean(pred_improvement):.4f} °C',
                       ha='center', va='center', fontsize=14,
                       transform=axes[1, 2].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].set_title('Improvement Statistics', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'difference_based_maps.png'))
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
    parser = argparse.ArgumentParser(description='Evaluate downscaling model predictions')
    parser.add_argument('--predictions', required=True, help='Path to predictions .npz file')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth .npz file')
    parser.add_argument('--era5-input', help='Path to ERA5 input .npz file (for difference-based correlation)')
    parser.add_argument('--output-dir', required=True, help='Output directory for evaluation results')
    parser.add_argument('--output-name', default='evaluation', help='Base name for output files')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Downscaling Model Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Predictions: {args.predictions}")
    print(f"  Ground truth: {args.ground_truth}")
    if args.era5_input:
        print(f"  ERA5 input: {args.era5_input}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output name: {args.output_name}")
    print()
    
    # Load data
    predictions, ground_truth, era5_interp, coords = load_data(
        args.predictions, args.ground_truth, args.era5_input
    )
    
    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth, era5_interp)
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
        args.output_dir, era5_interp
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

