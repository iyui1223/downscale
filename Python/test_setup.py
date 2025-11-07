#!/usr/bin/env python3
"""
Test script to verify installation and setup.

This script checks that all required packages are installed and
basic functionality works.

Usage:
    poetry run python Python/test_setup.py
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'pandas': 'pandas',
        'xarray': 'xarray',
        'netCDF4': 'netCDF4',
        'xgboost': 'xgboost',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'yaml': 'pyyaml',
    }
    
    failed = []
    
    for module, package in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {package:20s} version {version}")
        except ImportError as e:
            print(f"✗ {package:20s} FAILED: {e}")
            failed.append(package)
    
    print()
    
    if failed:
        print(f"ERROR: {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"  - {pkg}")
        return False
    else:
        print("SUCCESS: All required packages imported successfully!")
        return True


def test_data_utils():
    """Test data utilities module."""
    print("=" * 60)
    print("Testing Data Utilities")
    print("=" * 60)
    
    try:
        from data_utils import (
            normalize_data,
            denormalize_data,
            prepare_training_arrays
        )
        import numpy as np
        
        # Test normalization
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std = 3.0, 1.5
        
        normalized = normalize_data(data, mean, std)
        denormalized = denormalize_data(normalized, mean, std)
        
        assert np.allclose(data, denormalized), "Normalization round-trip failed"
        print("✓ Normalization/denormalization works")
        
        # Test array preparation
        training = np.random.rand(10, 5, 5)  # time, lat, lon
        target = np.random.rand(10, 5, 5)
        
        X, y = prepare_training_arrays(training, target)
        assert X.shape[0] == y.shape[0], "Sample count mismatch"
        print("✓ Array preparation works")
        
        print()
        print("SUCCESS: Data utilities working correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: Data utilities test failed: {e}")
        return False


def test_xgboost():
    """Test XGBoost functionality."""
    print("=" * 60)
    print("Testing XGBoost")
    print("=" * 60)
    
    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        
        # Create dummy data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.rand(100) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train simple model
        model = xgb.XGBRegressor(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        print(f"✓ XGBoost training successful (RMSE: {rmse:.4f})")
        print(f"✓ Model type: {type(model).__name__}")
        print()
        print("SUCCESS: XGBoost working correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: XGBoost test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_netcdf():
    """Test NetCDF handling with xarray."""
    print("=" * 60)
    print("Testing NetCDF/xarray")
    print("=" * 60)
    
    try:
        import xarray as xr
        import numpy as np
        import tempfile
        import os
        
        # Create dummy dataset
        times = np.arange(10)
        lats = np.linspace(-90, 90, 18)
        lons = np.linspace(-180, 180, 36)
        
        data = np.random.rand(len(times), len(lats), len(lons))
        
        ds = xr.Dataset(
            {
                'temperature': (['time', 'lat', 'lon'], data)
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            }
        )
        
        print(f"✓ Created xarray Dataset with shape: {dict(ds.dims)}")
        
        # Test slicing
        ds_sliced = ds.sel(lat=slice(-45, 45), lon=slice(0, 90))
        print(f"✓ Slicing works, new shape: {dict(ds_sliced.dims)}")
        
        # Test saving to NetCDF
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, 'test.nc')
            ds.to_netcdf(tmpfile)
            ds_loaded = xr.open_dataset(tmpfile)
            assert 'temperature' in ds_loaded, "Failed to load saved dataset"
            print("✓ NetCDF read/write works")
            ds_loaded.close()
        
        print()
        print("SUCCESS: NetCDF/xarray working correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: NetCDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "DOWNSCALE SETUP VERIFICATION" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Utilities", test_data_utils),
        ("XGBoost", test_xgboost),
        ("NetCDF/xarray", test_netcdf),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"ERROR: Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 All tests passed ({passed}/{total})!")
        print("\nYour environment is ready for downscaling!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed ({passed}/{total} passed)")
        print("\nPlease check the error messages above and install missing packages.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

