*
* GrADS Script: Spatial Maps Evaluation
*
* Creates 4 separate spatial maps comparing predictions vs ground truth:
*   - truth_mean.png: Mean temperature (ground truth)
*   - pred_mean.png: Mean temperature (predictions)
*   - bias.png: Bias (predictions - ground truth) 
*   - rmse.png: Root Mean Square Error
*
* Usage: grads -blc "evaluate_spatial_maps.gs DATA_DIR OUTPUT_DIR GRADS_DIR"
*
* Author: Climate Downscaling Team
* Date: December 2024
*

function main(args)

* Parse arguments
  data_dir = subwrd(args, 1)
  output_dir = subwrd(args, 2)
  grads_dir = subwrd(args, 3)
  
  if (data_dir = '' | output_dir = '')
    say 'ERROR: Usage: evaluate_spatial_maps.gs DATA_DIR OUTPUT_DIR GRADS_DIR'
    return 1
  endif
  
  if (grads_dir = '')
    grads_dir = '.'
  endif
  
  say '================================================================================'
  say 'GrADS Spatial Maps Evaluation'
  say '================================================================================'
  say ''
  say 'Data directory:   'data_dir
  say 'Output directory: 'output_dir
  say 'GrADS directory:  'grads_dir
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
  
* Extract time range
  line = sublin(info, 5)
  ntimes = subwrd(line, 12)
  
  say '  Predictions file:  'pred_file
  say '  Ground truth file: 'truth_file
  say '  Number of times:   'ntimes
  say ''
  
* Set graphics
  'set grads off'
  'set grid off'
  'set mpdset hires'
  
* Load custom colors
  colorfile = grads_dir'/colors.gs'
  say 'Loading colors from: 'colorfile
  'run ' colorfile
  
* Compute all fields first
  say 'Computing mean temperature (ground truth)...'
  'define truthmean = ave(data.'truth_file', t=1, t='ntimes')'
  
  say 'Computing mean temperature (predictions)...'
  'define predmean = ave(data.'pred_file', t=1, t='ntimes')'
  
  say 'Computing bias...'
  'define bias = predmean - truthmean'
  
  say 'Computing RMSE...'
  'define sqerr = pow(data.'pred_file' - data.'truth_file', 2)'
  'define rmse = sqrt(ave(sqerr, t=1, t='ntimes'))'
  
*------------------------------------------------
* Map 1: Mean Temperature (Ground Truth)
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  'set clevs -20 -15 -10 -5 0 5 10 15 20 25 30'
  'set ccols 16 17 18 19 20 21 22 23 24 25 26 27'
  'd truthmean'
  'draw title Ground Truth Mean Temperature (C)'
  'cbarn'
  
  output_file = output_dir'/truth_mean.png'
  say 'Saving: 'output_file
  'printim 'output_file' png white x1200 y900'
  
*------------------------------------------------
* Map 2: Mean Temperature (Predictions)
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  'set clevs -20 -15 -10 -5 0 5 10 15 20 25 30'
  'set ccols 16 17 18 19 20 21 22 23 24 25 26 27'
  'd predmean'
  'draw title Predicted Mean Temperature (C)'
  'cbarn'
  
  output_file = output_dir'/pred_mean.png'
  say 'Saving: 'output_file
  'printim 'output_file' png white x1200 y900'
  
*------------------------------------------------
* Map 3: Bias (smaller range, white centered at 0)
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  'set clevs -2 -1.5 -1 -0.5 -0.2 0.2 0.5 1 1.5 2'
  'set ccols 30 31 32 33 34 35 36 37 38 39 40'
  'd bias'
  'draw title Bias: Predictions - Ground Truth (C)'
  'cbarn'
  
  output_file = output_dir'/bias.png'
  say 'Saving: 'output_file
  'printim 'output_file' png white x1200 y900'
  
*------------------------------------------------
* Map 4: RMSE
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  'set clevs 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61'
  'd rmse'
  'draw title RMSE (C)'
  'cbarn'
  
  output_file = output_dir'/rmse.png'
  say 'Saving: 'output_file
  'printim 'output_file' png white x1200 y900'
  
  say ''
  
* Clean up
  'close 'truth_file
  'close 'pred_file
  
  say ''
  say '================================================================================'
  say 'Spatial maps created successfully!'
  say '  - truth_mean.png'
  say '  - pred_mean.png'
  say '  - bias.png'
  say '  - rmse.png'
  say '================================================================================'
  
return 0

