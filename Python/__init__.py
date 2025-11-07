"""
XGBoost-based downscaling system for meteorological data.

This package provides tools for preprocessing, training, and prediction
of high-resolution meteorological fields from coarse-resolution inputs.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import key utilities for convenient access
from .data_utils import (
    DataLoader,
    load_preprocessed_npz,
    load_preprocessed_zarr,
    load_statistics,
    normalize_data,
    denormalize_data,
    prepare_training_arrays,
    check_data_alignment,
)

__all__ = [
    'DataLoader',
    'load_preprocessed_npz',
    'load_preprocessed_zarr',
    'load_statistics',
    'normalize_data',
    'denormalize_data',
    'prepare_training_arrays',
    'check_data_alignment',
]

