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
* Fixed range: -80 to 52.5 C, 2.5 degree intervals, colors 71-125
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  'set clevs -80 -77.5 -75 -72.5 -70 -67.5 -65 -62.5 -60 -57.5 -55 -52.5 -50 -47.5 -45 -42.5 -40 -37.5 -35 -32.5 -30 -27.5 -25 -22.5 -20 -17.5 -15 -12.5 -10 -7.5 -5 -2.5 0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25 27.5 30 32.5 35 37.5 40 42.5 45 47.5 50 52.5'
  'set ccols 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125'
  'set clab off'
  'd truthmean'
  'draw title Ground Truth Mean Temperature (C)'
  'cbarn'
  
  output_file = output_dir'/truth_mean.png'
  say 'Saving: 'output_file
  'printim 'output_file' png white x1200 y900'
  
*------------------------------------------------
* Map 2: Mean Temperature (Predictions)
* Fixed range: -80 to 52.5 C, 2.5 degree intervals, colors 71-125
*------------------------------------------------
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  'set clevs -80 -77.5 -75 -72.5 -70 -67.5 -65 -62.5 -60 -57.5 -55 -52.5 -50 -47.5 -45 -42.5 -40 -37.5 -35 -32.5 -30 -27.5 -25 -22.5 -20 -17.5 -15 -12.5 -10 -7.5 -5 -2.5 0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25 27.5 30 32.5 35 37.5 40 42.5 45 47.5 50 52.5'
  'set ccols 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125'
  'set clab off'
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
   'set clevs -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
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
  'set clevs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9'
  'set ccols 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35'
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

