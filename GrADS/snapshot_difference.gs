*
* GrADS Script: Difference Snapshot for a Single Date
*
* Creates difference maps comparing datasets to ground truth:
*   - diff_pred_truth_{date}.png: Predictions - Ground Truth
*   - diff_era5_truth_{date}.png: ERA5 - Ground Truth
*
* Usage: grads -blc "snapshot_difference.gs DATA_DIR OUTPUT_DIR GRADS_DIR DATE_STR"
*   DATE_STR: Date in format YYYYMMDD (e.g., 20030810)
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
  
  if (data_dir = '' | output_dir = '' | date_str = '')
    say 'ERROR: Usage: snapshot_difference.gs DATA_DIR OUTPUT_DIR GRADS_DIR YYYYMMDD'
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
  say 'Difference Snapshot: 'date_str
  say '================================================================================'
  say 'Data directory:   'data_dir
  say 'Output directory: 'output_dir
  say 'GrADS date:       'grads_date
  say ''
  
* Open datasets
  say 'Opening predictions...'
  'open 'data_dir'/predictions.ctl'
  pred_fnum = 1
  
  say 'Opening ground truth...'
  'open 'data_dir'/ground_truth.ctl'
  truth_fnum = 2
  
* Check if ERA5 exists
  has_era5 = 0
  era5_fnum = 0
  rc = read(data_dir'/era5_input.ctl')
  if (subwrd(rc, 1) = 0)
    say 'Opening ERA5 input...'
    'open 'data_dir'/era5_input.ctl'
    era5_fnum = 3
    has_era5 = 1
  else
    say 'ERA5 input not found, skipping ERA5 difference map'
  endif
  
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
  
*------------------------------------------------
* Map 1: Predictions - Ground Truth
* Range: -5 to +5 C, diverging colormap
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  
* Compute difference
  'define diff_pred = data.'pred_fnum' - data.'truth_fnum
  
* Difference color scale: -5 to +5 C
  'set clevs -5 -4.5 -4 -3.5 -3 -2.5 -2 -1.5 -1 -0.5 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
  'set clab off'
  
  'd diff_pred'
  
  title = 'Pred - Truth 'date_str' (C)'
  'draw title 'title
  'cbarn'
  
  output_file = output_dir'/diff_pred_truth_'date_str'.png'
  say 'Saving: 'output_file
  'printim 'output_file' png white x1200 y900'
  
*------------------------------------------------
* Map 2: ERA5 - Ground Truth (if ERA5 available)
*------------------------------------------------
  if (has_era5 = 1)
    'clear'
    'set vpage 0 11 0 8.5'
    'set parea 1.0 10.0 1.0 7.5'
    'set gxout shaded'
    
*   Compute difference
    'define diff_era5 = data.'era5_fnum' - data.'truth_fnum
    
*   Same color scale
    'set clevs -5 -4.5 -4 -3.5 -3 -2.5 -2 -1.5 -1 -0.5 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5'
    'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
    'set clab off'
    
    'd diff_era5'
    
    title = 'ERA5 - Truth 'date_str' (C)'
    'draw title 'title
    'cbarn'
    
    output_file = output_dir'/diff_era5_truth_'date_str'.png'
    say 'Saving: 'output_file
    'printim 'output_file' png white x1200 y900'
    
*   Close ERA5 file
    'close 'era5_fnum
  endif
  
* Clean up
  'close 'truth_fnum
  'close 'pred_fnum
  
  say ''
  say 'Difference snapshots created for 'date_str
  say '================================================================================'
  
return 0

