*
* GrADS Script: Difference-Based Evaluation
*
* Evaluates improvement over ERA5 baseline:
*   - ERA5 mean temperature
*   - True improvement (ground truth - ERA5)
*   - Predicted improvement (predictions - ERA5)
*   - Anomaly Correlation Coefficient between (Pred-ERA5) and (Truth-ERA5)
*
* Creates: difference_based_maps.png, anomaly_correlation.png
*
* Usage: grads -blc "evaluate_differences.gs DATA_DIR OUTPUT_DIR GRADS_DIR"
*

function main(args)

  data_dir = subwrd(args, 1)
  output_dir = subwrd(args, 2)
  grads_dir = subwrd(args, 3)
  
  if (data_dir = '' | output_dir = '')
    say 'ERROR: Usage: evaluate_differences.gs DATA_DIR OUTPUT_DIR GRADS_DIR'
    return 1
  endif
  
  if (grads_dir = '')
    grads_dir = '.'
  endif
  
  say '================================================================================'
  say 'GrADS Difference-Based Evaluation'
  say '================================================================================'
  say ''
  
* Open datasets
  say 'Opening datasets...'
  'open 'data_dir'/predictions.ctl'
  pred_file = 1
  'open 'data_dir'/ground_truth.ctl'
  truth_file = 2
  'open 'data_dir'/era5_input.ctl'
  era5_file = 3
  
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
* Panel 1: ERA5 Mean Temperature
*------------------------------------------------
  say 'Computing ERA5 mean temperature...'
  'define era5mean = ave(data.'era5_file', t=1, t='ntimes')'
  'set gxout shaded'
  'set clevs -20 -15 -10 -5 0 5 10 15 20 25 30'
  'set ccols 16 17 18 19 20 21 22 23 24 25 26 27'
  'd era5mean'
  'draw title ERA5 Mean Temperature (C)'
  'cbarn'
  
* Panel 2: True Improvement (Ground Truth - ERA5)
  'set vpage 5.5 11 0 8.5'
  'set parea 0.5 5.0 4.5 8.0'
  say 'Computing true improvement...'
  'define truthmean = ave(data.'truth_file', t=1, t='ntimes')'
  'define trueimp = truthmean - era5mean'
  'set gxout shaded'
  'set clevs -4 -3 -2 -1 -0.5 0 0.5 1 2 3 4'
  'set ccols 30 31 32 33 34 35 36 37 38 39 40 41'
  'd trueimp'
  'draw title True Improvement: Truth - ERA5 (C)'
  'cbarn'
  
* Panel 3: Predicted Improvement (Predictions - ERA5)
  'set vpage 0 11 0 8.5'
  'set parea 0.5 5.0 0.5 4.0'
  say 'Computing predicted improvement...'
  'define predmean = ave(data.'pred_file', t=1, t='ntimes')'
  'define predimp = predmean - era5mean'
  'set gxout shaded'
  'set clevs -4 -3 -2 -1 -0.5 0 0.5 1 2 3 4'
  'set ccols 30 31 32 33 34 35 36 37 38 39 40 41'
  'd predimp'
  'draw title Predicted Improvement: Predictions - ERA5 (C)'
  'cbarn'
  
* Panel 4: Anomaly Correlation Coefficient
  'set vpage 5.5 11 0 8.5'
  'set parea 0.5 5.0 0.5 4.0'
  say 'Computing anomaly correlation coefficient...'
  say '  ACC between (Pred-ERA5) and (Truth-ERA5)'
  'define predanom = data.'pred_file' - data.'era5_file
  'define truthanom = data.'truth_file' - data.'era5_file
  'define acc = tcorr(predanom, truthanom, t=1, t='ntimes')'
  'set gxout shaded'
  'set clevs 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95 0.98'
  'set ccols 70 71 72 73 74 75 76 77 78 79 80'
  'd acc'
  'draw title Anomaly Correlation Coefficient'
  'cbarn'
  
* Save output
  output_file = output_dir'/difference_based_maps.png'
  say ''
  say 'Saving to: 'output_file
  'printim 'output_file' png white x1200 y900'
  
* Also save individual ACC map
  'clear'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
  'set clevs 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95 0.98'
  'set ccols 70 71 72 73 74 75 76 77 78 79 80'
  'd acc'
  'draw title Anomaly Correlation Coefficient: (Pred-ERA5) vs (Truth-ERA5)'
  'cbarn'
  
  output_file = output_dir'/anomaly_correlation.png'
  say 'Saving to: 'output_file
  'printim 'output_file' png white x1200 y900'
  
* Clean up
  'close 'era5_file
  'close 'truth_file
  'close 'pred_file
  
  say ''
  say '================================================================================'
  say 'Difference-based evaluation created successfully!'
  say '================================================================================'
  
return 0

