*
* GrADS Script: Temperature Snapshot for a Single Date
*
* Creates temperature and anomaly maps for ERA5, ground truth, and predictions
* for a specific date. Output files:
*   - {dataset}_temp_{date}.png: Temperature map
*   - {dataset}_anom_{date}.png: Anomaly from climatology
*
* Usage: grads -blc "snapshot_temperature.gs DATA_DIR OUTPUT_DIR GRADS_DIR DATE_STR DATASET"
*   DATE_STR: Date in format YYYYMMDD (e.g., 20030810)
*   DATASET: era5 | truth | pred
*
* Author: Climate Downscaling Team
* Date: December 2024
*

function main(args)

* Parse arguments
  data_dir = subwrd(args, 1)
  output_dir = subwrd(args, 2)
  grads_dir = subwrd(args, 3)
  date_str = subwrd(args, 4)
  dataset = subwrd(args, 5)
  
  if (data_dir = '' | output_dir = '' | date_str = '' | dataset = '')
    say 'ERROR: Usage: snapshot_temperature.gs DATA_DIR OUTPUT_DIR GRADS_DIR YYYYMMDD DATASET'
    say '       DATASET: era5 | truth | pred'
    return 1
  endif
  
  if (grads_dir = '')
    grads_dir = '.'
  endif
  
* Convert date string to GrADS time format
* YYYYMMDD -> DDmonYYYY (e.g., 20030810 -> 10aug2003)
  year = substr(date_str, 1, 4)
  month_num = substr(date_str, 5, 2)
  day = substr(date_str, 7, 2)
  
* Convert month number to 3-letter abbreviation
  if (month_num = '01'); month = 'jan'; endif
  if (month_num = '02'); month = 'feb'; endif
  if (month_num = '03'); month = 'mar'; endif
  if (month_num = '04'); month = 'apr'; endif
  if (month_num = '05'); month = 'may'; endif
  if (month_num = '06'); month = 'jun'; endif
  if (month_num = '07'); month = 'jul'; endif
  if (month_num = '08'); month = 'aug'; endif
  if (month_num = '09'); month = 'sep'; endif
  if (month_num = '10'); month = 'oct'; endif
  if (month_num = '11'); month = 'nov'; endif
  if (month_num = '12'); month = 'dec'; endif
  
  grads_date = day%month%year
  
  say '================================================================================'
  say 'Temperature Snapshot: 'dataset' - 'date_str
  say '================================================================================'
  say 'Data directory:   'data_dir
  say 'Output directory: 'output_dir
  say 'GrADS date:       'grads_date
  say ''
  
* Determine which files to open based on dataset
  if (dataset = 'era5')
    data_file = data_dir'/era5_input.ctl'
    clim_file = data_dir'/era5_input_clim.ctl'
    title_prefix = 'ERA5'
    file_prefix = 'era5'
  endif
  if (dataset = 'truth')
    data_file = data_dir'/ground_truth.ctl'
    clim_file = data_dir'/ground_truth_clim.ctl'
    title_prefix = 'Ground Truth (MSWX)'
    file_prefix = 'truth'
  endif
  if (dataset = 'pred')
    data_file = data_dir'/predictions.ctl'
    clim_file = data_dir'/predictions_clim.ctl'
    title_prefix = 'Predictions'
    file_prefix = 'pred'
  endif
  
* Open datasets
  say 'Opening data file: 'data_file
  'open 'data_file
  data_fnum = 1
  
  say 'Opening climatology file: 'clim_file
  'open 'clim_file
  clim_fnum = 2
  
* Set graphics
  'set grads off'
  'set grid off'
  'set mpdset hires'
  
* Load custom colors
  colorfile = grads_dir'/colors.gs'
  say 'Loading colors from: 'colorfile
  'run ' colorfile
  
* Set time to requested date
  'set time 'grads_date
  
* Check if time is valid
  'q dims'
  dims = result
  tline = sublin(dims, 5)
  tval = subwrd(tline, 9)
  say 'Time index: 'tval
  
*------------------------------------------------
* Map 1: Temperature
* Global range: -80 to 52.5 C, 2.5 degree intervals, colors 71-125
* Same scale as evaluate_spatial_maps.gs for consistency
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  
* Temperature color scale: -80 to 52.5 C in 2.5 degree steps (global range)
  'set clevs -80 -77.5 -75 -72.5 -70 -67.5 -65 -62.5 -60 -57.5 -55 -52.5 -50 -47.5 -45 -42.5 -40 -37.5 -35 -32.5 -30 -27.5 -25 -22.5 -20 -17.5 -15 -12.5 -10 -7.5 -5 -2.5 0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25 27.5 30 32.5 35 37.5 40 42.5 45 47.5 50 52.5'
  'set ccols 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125'
  'set clab off'
  
  'd data.'data_fnum
  
  title = title_prefix' Temperature 'date_str' (C)'
  'draw title 'title
  'cbarn'
  
  output_file = output_dir'/'file_prefix'_temp_'date_str'.png'
  say 'Saving: 'output_file
  'printim 'output_file' png white x1200 y900'
  
*------------------------------------------------
* Map 2: Anomaly from climatology
* Range: -10 to +10 C
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  
* Compute anomaly (current day minus climatology)
* Climatology file has 12 monthly values, need to select correct month
  'set dfile 'clim_fnum
  'set t 'month_num
  'define clim_val = data.'clim_fnum
  
  'set dfile 'data_fnum
  'set time 'grads_date
  'define anom = data.'data_fnum' - clim_val'
  
* Anomaly color scale: -10 to +10 C, diverging
  'set clevs -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 1 2 3 4 5 6 7 8 9 10'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
  'set clab off'
  
  'd anom'
  
  title = title_prefix' Anomaly 'date_str' (C)'
  'draw title 'title
  'cbarn'
  
  output_file = output_dir'/'file_prefix'_anom_'date_str'.png'
  say 'Saving: 'output_file
  'printim 'output_file' png white x1200 y900'
  
* Clean up
  'close 'clim_fnum
  'close 'data_fnum
  
  say ''
  say 'Snapshot created for 'dataset' on 'date_str
  say '================================================================================'
  
return 0

