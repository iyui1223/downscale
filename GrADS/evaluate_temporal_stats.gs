*
* GrADS Script: Temporal Statistics Evaluation
*
* Computes and visualizes temporal statistics:
*   - Temporal standard deviation
*   - Mean absolute error map
*   - Correlation at each grid point
*
* Usage: grads -blc "evaluate_temporal_stats.gs DATA_DIR OUTPUT_DIR GRADS_DIR"
*

function main(args)

  data_dir = subwrd(args, 1)
  output_dir = subwrd(args, 2)
  grads_dir = subwrd(args, 3)
  
  if (data_dir = '' | output_dir = '')
    say 'ERROR: Usage: evaluate_temporal_stats.gs DATA_DIR OUTPUT_DIR GRADS_DIR'
    return 1
  endif
  
  if (grads_dir = '')
    grads_dir = '.'
  endif
  
  say '================================================================================'
  say 'GrADS Temporal Statistics Evaluation'
  say '================================================================================'
  say ''
  
* Open datasets
  say 'Opening datasets...'
  'open 'data_dir'/predictions.ctl'
  pred_file = 1
  'open 'data_dir'/ground_truth.ctl'
  truth_file = 2
  
* Query dimensions
  'q file 'pred_file
  info = result
  line = sublin(info, 5)
  ntimes = subwrd(line, 12)
  
  say '  Number of times: 'ntimes
  say ''
  
* Set graphics
  'set grads off'
  'set grid off'
  'set mpdset hires'
  
* Load custom colors
  colorfile = grads_dir '/colors.gs'
  say 'Loading colors from: 'colorfile
  'run ' colorfile
  
* Create 2x2 panel
  'set vpage 0 11 0 8.5'
  'set parea 0.5 5.0 4.5 8.0'
  
*------------------------------------------------
* Panel 1: Temporal Std Dev (Ground Truth)
*------------------------------------------------
  say 'Computing temporal std dev (ground truth)...'
  'define truthstd = sqrt(ave(pow(data.'truth_file' - ave(data.'truth_file', t=1, t='ntimes'), 2), t=1, t='ntimes'))'
  'set gxout shaded'
  'set clevs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9'
  'set ccols 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35'
  'd truthstd'
  'draw title Ground Truth Temporal Std Dev (C)'
  'cbarn'
  
* Panel 2: Mean Absolute Error
  'set vpage 5.5 11 0 8.5'
  'set parea 0.5 5.0 4.5 8.0'
  say 'Computing mean absolute error...'
  'define mae = ave(abs(data.'pred_file' - data.'truth_file'), t=1, t='ntimes')'
  'set gxout shaded'
  'set clevs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9'
  'set ccols 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35'
  'd mae'
  'draw title Mean Absolute Error (C)'
  'cbarn'
  
* Panel 3 & 4: Climatology comparison
  'set vpage 0 11 0 8.5'
  'set parea 0.5 5.0 0.5 4.0'
  
  say 'Computing mean bias map...'
  'define meanbias = ave(data.'pred_file' - data.'truth_file', t=1, t='ntimes')'
  'set gxout shaded'
  'set clevs -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
  'd meanbias'
  'draw title Mean Bias (C)'
  'cbarn'
  
* Save output
  output_file = output_dir'/temporal_stats.png'
  say ''
  say 'Saving to: 'output_file
  'printim 'output_file' png white x1200 y900'
  
* Clean up
  'close 'truth_file
  'close 'pred_file
  
  say ''
  say '================================================================================'
  say 'Temporal statistics created successfully!'
  say '================================================================================'
  
return 0

