#!/usr/bin/env python3
"""
Stream NPZ to GrADS Binary Format with Climatology

This script converts .npz files to GrADS-readable binary format using a streaming
approach that processes one year at a time to minimize memory usage.

Features:
- Streams data year-by-year (only 2 years in memory max)
- Optionally interpolates ERA5 to high-resolution grid
- Computes day-of-year climatology during streaming
- Writes GrADS-compatible binary files with .ctl descriptors

Memory usage: ~2x single year of data (much better than loading full dataset)

Author: Climate Downscaling Team
Date: November 2024
"""

import argparse
import os
import sys
import struct
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def get_year_indices(times):
    """
    Get indices for each year in the time array.
    
    Returns:
        dict: {year: (start_idx, end_idx)}
    """
    years = {}
    for i, t in enumerate(times):
        # Handle different time formats
        if isinstance(t, np.datetime64):
            year = int(str(t)[:4])
        elif isinstance(t, datetime):
            year = t.year
        elif isinstance(t, (int, np.integer)):
            # Assume days since 1920-01-01 (for historical climate data)
            dt = datetime(1920, 1, 1) + timedelta(days=int(t))
            year = dt.year
        elif isinstance(t, str):
            # String format - extract first 4 characters as year
            year = int(str(t)[:4])
        else:
            # Try converting to string and extracting year
            try:
                year = int(str(t)[:4])
            except:
                print(f"  ⚠ Warning: Could not parse time at index {i}: {t}")
                year = 2000  # Fallback
        
        if year not in years:
            years[year] = [i, i]
        else:
            years[year][1] = i
    
    return years


def write_grads_ctl(filename, nlat, nlon, ntimes, lat, lon, varname='data', 
                    vardesc='Temperature', varunits='degC', tstart='00z01jan2000'):
    """
    Write GrADS .ctl descriptor file.
    
    Args:
        filename: Base filename (without .ctl extension)
        nlat, nlon, ntimes: Grid dimensions
        lat, lon: Coordinate arrays
        varname: Variable name
        vardesc: Variable description
        varunits: Variable units
        tstart: Starting time string
    """
    ctlfile = f"{filename}.ctl"
    binfile = os.path.basename(filename) + ".bin"
    
    # Check if latitude needs to be reversed (GrADS requires positive increment)
    lat_step = float(lat[1] - lat[0])
    if lat_step < 0:
        # Latitude is decreasing (north to south) - use south to north
        lat_start = float(lat[-1])
        lat_step = abs(lat_step)
    else:
        lat_start = float(lat[0])
    
    lon_start, lon_step = float(lon[0]), float(lon[1] - lon[0])
    
    with open(ctlfile, 'w') as f:
        f.write(f"DSET ^{binfile}\n")
        f.write(f"TITLE {vardesc}\n")
        f.write(f"UNDEF -9.99e8\n")
        f.write(f"OPTIONS sequential big_endian\n")
        f.write(f"XDEF {nlon} LINEAR {lon_start} {lon_step}\n")
        f.write(f"YDEF {nlat} LINEAR {lat_start} {lat_step}\n")
        f.write(f"ZDEF 1 LINEAR 1 1\n")
        f.write(f"TDEF {ntimes} LINEAR {tstart} 1dy\n")
        f.write(f"VARS 1\n")
        f.write(f"{varname} 0 99 {vardesc} ({varunits})\n")
        f.write(f"ENDVARS\n")
    
    return ctlfile


def write_climatology_ctl(filename, nlat, nlon, lat, lon, varname='clim',
                          vardesc='Climatology', varunits='degC'):
    """Write .ctl file for day-of-year climatology (366 days)."""
    ctlfile = f"{filename}.ctl"
    binfile = os.path.basename(filename) + ".bin"
    
    # Check if latitude needs to be reversed (GrADS requires positive increment)
    lat_step = float(lat[1] - lat[0])
    if lat_step < 0:
        lat_start = float(lat[-1])
        lat_step = abs(lat_step)
    else:
        lat_start = float(lat[0])
    
    lon_start, lon_step = float(lon[0]), float(lon[1] - lon[0])
    
    with open(ctlfile, 'w') as f:
        f.write(f"DSET ^{binfile}\n")
        f.write(f"TITLE {vardesc}\n")
        f.write(f"UNDEF -9.99e8\n")
        f.write(f"OPTIONS sequential big_endian\n")
        f.write(f"XDEF {nlon} LINEAR {lon_start} {lon_step}\n")
        f.write(f"YDEF {nlat} LINEAR {lat_start} {lat_step}\n")
        f.write(f"ZDEF 1 LINEAR 1 1\n")
        f.write(f"TDEF 366 LINEAR 00z01jan0001 1dy\n")  # 366 days (includes leap day)
        f.write(f"VARS 1\n")
        f.write(f"{varname} 0 99 {vardesc} ({varunits})\n")
        f.write(f"ENDVARS\n")
    
    return ctlfile


def write_binary_record(f, data):
    """Write a single time record in Fortran sequential format."""
    data_out = data.astype('>f4')  # big-endian float32
    record_size = data_out.nbytes
    f.write(struct.pack('>i', record_size))
    f.write(data_out.tobytes())
    f.write(struct.pack('>i', record_size))


def interpolate_era5_year(era5_data, year_start, year_end, era5_lat, era5_lon,
                          target_lat, target_lon):
    """
    Interpolate ERA5 data for one year to target grid.
    
    Args:
        era5_data: ERA5 array for this year (time, lat, lon)
        year_start, year_end: Year indices in era5_data
        era5_lat, era5_lon: ERA5 coordinates
        target_lat, target_lon: Target grid coordinates
        
    Returns:
        Interpolated data (time, lat, lon)
    """
    n_times = year_end - year_start + 1
    interp_data = np.zeros((n_times, len(target_lat), len(target_lon)), dtype=np.float32)
    
    # Create meshgrid once
    lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    for t in range(n_times):
        # Create interpolator for this timestep
        interpolator = RegularGridInterpolator(
            (era5_lat, era5_lon),
            era5_data[year_start + t, :, :],
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        # Interpolate
        interp_data[t, :, :] = interpolator(points).reshape(len(target_lat), len(target_lon))
    
    return interp_data


def process_npz_streaming(npz_path, output_base, varname_in='air_temperature',
                          era5_path=None, interpolate=False, compute_clim=True):
    """
    Process NPZ file in streaming manner (year-by-year).
    
    Args:
        npz_path: Path to input .npz file
        output_base: Base path for output files (without extension)
        varname_in: Variable name in NPZ file
        era5_path: Optional ERA5 file to interpolate from
        interpolate: Whether to interpolate from ERA5
        compute_clim: Whether to compute climatology
        
    Returns:
        dict with file paths and metadata
    """
    print(f"Processing: {npz_path}")
    print(f"Output base: {output_base}")
    
    # Load coordinates (minimal memory)
    data = np.load(npz_path, mmap_mode='r', allow_pickle=True)
    
    # Detect coordinate naming convention (handle both MSWX and ERA5 formats)
    print(f"  Arrays in file: {data.files}")
    
    # Time coordinate
    if 'coord_time' in data.files:
        times = np.array(data['coord_time'])
    elif 'coord_valid_time' in data.files:
        times = np.array(data['coord_valid_time'])
    else:
        raise KeyError(f"Could not find time coordinate. Available: {data.files}")
    
    # Latitude coordinate
    if 'coord_lat' in data.files:
        lat = np.array(data['coord_lat'])
    elif 'coord_latitude' in data.files:
        lat = np.array(data['coord_latitude'])
    else:
        raise KeyError(f"Could not find latitude coordinate. Available: {data.files}")
    
    # Longitude coordinate
    if 'coord_lon' in data.files:
        lon = np.array(data['coord_lon'])
    elif 'coord_longitude' in data.files:
        lon = np.array(data['coord_longitude'])
    else:
        raise KeyError(f"Could not find longitude coordinate. Available: {data.files}")
    
    print(f"  Grid: {len(lat)} x {len(lon)}")
    print(f"  Times: {len(times)} ({times[0]} to {times[-1]})")
    
    # Check if latitude needs to be reversed for GrADS
    lat_step = lat[1] - lat[0] if len(lat) > 1 else 0
    reverse_lat = lat_step < 0
    if reverse_lat:
        print(f"  Note: Latitude is decreasing (north to south), will reverse for GrADS compatibility")
    
    # Get year indices
    year_indices = get_year_indices(times)
    years = sorted(year_indices.keys())
    print(f"  Years: {years[0]} - {years[-1]} ({len(years)} years)")
    
    # Initialize climatology accumulator if needed
    if compute_clim:
        clim_sum = np.zeros((366, len(lat), len(lon)), dtype=np.float64)
        clim_count = np.zeros(366, dtype=np.int32)
    
    # Open binary output file
    bin_file = f"{output_base}.bin"
    ctl_file = f"{output_base}.ctl"
    
    # ERA5 setup if interpolating
    if interpolate and era5_path:
        print(f"\n  Loading ERA5 for interpolation: {era5_path}")
        era5_data_full = np.load(era5_path, mmap_mode='r', allow_pickle=True)
        
        # Detect ERA5 coordinate names
        print(f"  ERA5 arrays: {era5_data_full.files}")
        
        # Variable name
        if 't2m' in era5_data_full.files:
            era5_var = era5_data_full['t2m']
        elif 'air_temperature' in era5_data_full.files:
            era5_var = era5_data_full['air_temperature']
        else:
            raise KeyError(f"Could not find temperature variable in ERA5. Available: {era5_data_full.files}")
        
        # ERA5 coordinates
        if 'coord_latitude' in era5_data_full.files:
            era5_lat = np.array(era5_data_full['coord_latitude'])
        elif 'coord_lat' in era5_data_full.files:
            era5_lat = np.array(era5_data_full['coord_lat'])
        else:
            raise KeyError(f"Could not find latitude in ERA5. Available: {era5_data_full.files}")
        
        if 'coord_longitude' in era5_data_full.files:
            era5_lon = np.array(era5_data_full['coord_longitude'])
        elif 'coord_lon' in era5_data_full.files:
            era5_lon = np.array(era5_data_full['coord_lon'])
        else:
            raise KeyError(f"Could not find longitude in ERA5. Available: {era5_data_full.files}")
        
        if 'coord_valid_time' in era5_data_full.files:
            era5_times = np.array(era5_data_full['coord_valid_time'])
        elif 'coord_time' in era5_data_full.files:
            era5_times = np.array(era5_data_full['coord_time'])
        else:
            raise KeyError(f"Could not find time in ERA5. Available: {era5_data_full.files}")
        
        # Verify time alignment
        if not np.array_equal(era5_times, times):
            raise ValueError("ERA5 and target times do not match!")
        
        print(f"    ERA5 grid: {len(era5_lat)} x {len(era5_lon)}")
    
    # Process year by year
    total_records = 0
    with open(bin_file, 'wb') as f_out:
        for year in years:
            year_start, year_end = year_indices[year]
            n_times = year_end - year_start + 1
            
            print(f"\n  Processing year {year} ({n_times} timesteps)...")
            
            # Load this year's data
            if interpolate and era5_path:
                # Interpolate from ERA5
                print(f"    Interpolating from ERA5...")
                year_data = interpolate_era5_year(
                    era5_var, year_start, year_end,
                    era5_lat, era5_lon, lat, lon
                )
            else:
                # Load directly from high-res data
                year_data = np.array(data[varname_in][year_start:year_end+1, :, :], dtype=np.float32)
            
            # Write to binary file
            for t in range(n_times):
                # Reverse latitude if needed for GrADS compatibility
                data_to_write = year_data[t, ::-1, :] if reverse_lat else year_data[t, :, :]
                write_binary_record(f_out, data_to_write)
                total_records += 1
            
            # Accumulate for climatology
            if compute_clim:
                year_times = times[year_start:year_end+1]
                for t in range(n_times):
                    # Get day of year (1-366)
                    time_obj = year_times[t]
                    
                    # Handle different time formats
                    if isinstance(time_obj, np.datetime64):
                        # Convert numpy datetime64 to Python datetime
                        time_obj = time_obj.astype('datetime64[D]').astype(datetime)
                    elif isinstance(time_obj, (int, np.integer)):
                        # Assume days since 1920-01-01 (for historical climate data)
                        time_obj = datetime(1920, 1, 1) + timedelta(days=int(time_obj))
                    elif isinstance(time_obj, str):
                        # Parse string date
                        time_obj = datetime.fromisoformat(str(time_obj)[:10])
                    else:
                        # Try to convert to datetime
                        try:
                            time_obj = datetime.fromisoformat(str(time_obj)[:10])
                        except:
                            # If all else fails, use sequential day of year based on position
                            # Assume starting from Jan 1
                            doy = (year_start + t) % 366
                            clim_sum[doy, :, :] += year_data[t, :, :]
                            clim_count[doy] += 1
                            continue
                    
                    doy = time_obj.timetuple().tm_yday - 1  # 0-indexed
                    
                    # Add to climatology (with latitude reversal if needed)
                    data_for_clim = year_data[t, ::-1, :] if reverse_lat else year_data[t, :, :]
                    clim_sum[doy, :, :] += data_for_clim
                    clim_count[doy] += 1
            
            # Free memory
            del year_data
    
    print(f"\n  ✓ Wrote {total_records} records to {bin_file}")
    
    # Write .ctl file
    # Determine time start string from first time
    time0 = times[0]
    if isinstance(time0, np.datetime64):
        time0 = time0.astype('datetime64[D]').astype(datetime)
    elif isinstance(time0, (int, np.integer)):
        # Assume days since 1920-01-01 (for historical climate data)
        time0 = datetime(1920, 1, 1) + timedelta(days=int(time0))
    elif isinstance(time0, str):
        time0 = datetime.fromisoformat(str(time0)[:10])
    else:
        # Default fallback
        try:
            time0 = datetime.fromisoformat(str(time0)[:10])
        except:
            time0 = datetime(2000, 1, 1)  # Fallback date
            print(f"  ⚠ Warning: Could not parse time format, using default: {time0}")
    
    tstart = time0.strftime("%Hz%d%b%Y").lower()
    
    write_grads_ctl(output_base, len(lat), len(lon), len(times), lat, lon,
                   varname='data', vardesc='Temperature', varunits='degC', tstart=tstart)
    print(f"  ✓ Wrote {ctl_file}")
    
    # Compute and write climatology
    clim_file = None
    if compute_clim:
        print(f"\n  Computing climatology...")
        clim_mean = np.zeros((366, len(lat), len(lon)), dtype=np.float32)
        for doy in range(366):
            if clim_count[doy] > 0:
                clim_mean[doy, :, :] = (clim_sum[doy, :, :] / clim_count[doy]).astype(np.float32)
            else:
                clim_mean[doy, :, :] = np.nan
        
        # Write climatology binary
        clim_base = f"{output_base}_clim"
        clim_file = f"{clim_base}.bin"
        with open(clim_file, 'wb') as f_clim:
            for doy in range(366):
                write_binary_record(f_clim, clim_mean[doy, :, :])
        
        write_climatology_ctl(clim_base, len(lat), len(lon), lat, lon,
                             varname='clim', vardesc='Climatology', varunits='degC')
        print(f"  ✓ Wrote climatology to {clim_file}")
    
    return {
        'binary': bin_file,
        'ctl': ctl_file,
        'climatology': clim_file,
        'nlat': len(lat),
        'nlon': len(lon),
        'ntimes': len(times),
        'years': years
    }


def main():
    parser = argparse.ArgumentParser(
        description='Stream NPZ to GrADS binary format with climatology'
    )
    parser.add_argument('--input', required=True, help='Input .npz file')
    parser.add_argument('--output', required=True, help='Output base name (without extension)')
    parser.add_argument('--varname', default='air_temperature', help='Variable name in NPZ')
    parser.add_argument('--era5-input', help='ERA5 .npz file for interpolation')
    parser.add_argument('--interpolate', action='store_true', 
                       help='Interpolate from ERA5 to high-res grid')
    parser.add_argument('--no-climatology', action='store_true',
                       help='Skip climatology computation')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NPZ to GrADS Binary Converter (Streaming)")
    print("="*80)
    print()
    
    # Process the file
    result = process_npz_streaming(
        args.input,
        args.output,
        varname_in=args.varname,
        era5_path=args.era5_input,
        interpolate=args.interpolate,
        compute_clim=not args.no_climatology
    )
    
    print("\n" + "="*80)
    print("Conversion Complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Binary: {result['binary']}")
    print(f"  Control: {result['ctl']}")
    if result['climatology']:
        print(f"  Climatology: {result['climatology']}")
    print(f"\nGrid: {result['nlat']} x {result['nlon']}")
    print(f"Times: {result['ntimes']}")
    print(f"Years: {result['years'][0]} - {result['years'][-1]}")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

