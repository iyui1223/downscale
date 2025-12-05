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
  
* Open datasets - open predictions first so its grid is the default
  say 'Opening datasets...'
  'open 'data_dir'/predictions.ctl'
  pred_file = 1
  'open 'data_dir'/ground_truth.ctl'
  truth_file = 2
  'open 'data_dir'/era5_input.ctl'
  era5_file = 3
  
* Query dimensions from predictions file (high-res grid)
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

* Regrid ERA5 to match predictions grid using lterp
* First compute means on their native grids, then regrid ERA5 mean
  say 'Computing means and regridding ERA5 to prediction grid...'
  'set dfile 'pred_file
  'define predmean = ave(data.'pred_file', t=1, t='ntimes')'
  'set dfile 'truth_file
  'define truthmean = ave(data.'truth_file', t=1, t='ntimes')'
  'set dfile 'era5_file
  'define era5meanraw = ave(data.'era5_file', t=1, t='ntimes')'
* Regrid ERA5 mean to prediction grid
  'set dfile 'pred_file
  'define era5mean = lterp(era5meanraw, predmean)'
  
* Create 2x2 panel
  'set vpage 0 11 0 8.5'
  'set parea 0.5 5.0 4.5 8.0'
  
*------------------------------------------------
* Panel 1: ERA5 Mean Temperature (regridded)
* Fixed range: -80 to 52.5 C, 2.5 degree intervals, colors 71-125
*------------------------------------------------
  say 'Plotting ERA5 mean temperature...'
  'set gxout shaded'
  'set clevs -80 -77.5 -75 -72.5 -70 -67.5 -65 -62.5 -60 -57.5 -55 -52.5 -50 -47.5 -45 -42.5 -40 -37.5 -35 -32.5 -30 -27.5 -25 -22.5 -20 -17.5 -15 -12.5 -10 -7.5 -5 -2.5 0 2.5 5 7.5 10 12.5 15 17.5 20 22.5 25 27.5 30 32.5 35 37.5 40 42.5 45 47.5 50 52.5'
  'set ccols 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125'
  'set clab off'
  'd era5mean'
  'draw title ERA5 Mean Temperature (C)'
  'cbarn'
  
* Panel 2: True Improvement (Ground Truth - ERA5)
  'set vpage 5.5 11 0 8.5'
  'set parea 0.5 5.0 4.5 8.0'
  say 'Computing true improvement...'
  'define trueimp = truthmean - era5mean'
  'set gxout shaded'
  'set clevs -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
  'd trueimp'
  'draw title True Improvement: Truth - ERA5 (C)'
  'cbarn'
  
* Panel 3: Predicted Improvement (Predictions - ERA5)
  'set vpage 0 11 0 8.5'
  'set parea 0.5 5.0 0.5 4.0'
  say 'Computing predicted improvement...'
  'define predimp = predmean - era5mean'
  'set gxout shaded'
  'set clevs -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
  'd predimp'
  'draw title Predicted Improvement: Predictions - ERA5 (C)'
  'cbarn'
  
* Panel 4: Anomaly Correlation Coefficient
* ACC = corr((pred - ERA5), (truth - ERA5))
* Manual calculation: r = cov(x,y) / (std_x * std_y)
  'set vpage 5.5 11 0 8.5'
  'set parea 0.5 5.0 0.5 4.0'
  say 'Computing anomaly correlation coefficient...'
  say '  ACC = corr((pred-ERA5), (truth-ERA5))'
  'set dfile 'pred_file
  'set t 1'
* Define anomalies: x = pred - era5, y = truth - era5
* Compute means of anomalies
  'define xmean = ave(data.'pred_file' - data.'era5_file', t=1, t='ntimes')'
  'define ymean = ave(data.'truth_file' - data.'era5_file', t=1, t='ntimes')'
* Compute E[xy], E[x^2], E[y^2] for anomalies
  'define xymean = ave((data.'pred_file' - data.'era5_file') * (data.'truth_file' - data.'era5_file'), t=1, t='ntimes')'
  'define x2mean = ave((data.'pred_file' - data.'era5_file') * (data.'pred_file' - data.'era5_file'), t=1, t='ntimes')'
  'define y2mean = ave((data.'truth_file' - data.'era5_file') * (data.'truth_file' - data.'era5_file'), t=1, t='ntimes')'
* Compute covariance and standard deviations
  'define covar = xymean - xmean * ymean'
  'define xstd = sqrt(x2mean - xmean * xmean)'
  'define ystd = sqrt(y2mean - ymean * ymean)'
* Anomaly Correlation Coefficient
  'define acc = covar / (xstd * ystd)'
  'set gxout shaded'
* BWR palette: -1 to 1 in 0.1 steps, colors 50-70 (blue-white-red)
  'set clevs -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
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
  'run ' colorfile
  'set grads off'
  'set grid off'
  'set mpdset hires'
  'set vpage 0 11 0 8.5'
  'set parea 1.0 10.0 1.0 7.5'
  'set gxout shaded'
* BWR palette: -1 to 1 in 0.1 steps, colors 50-70 (blue-white-red)
  'set clevs -1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'
  'set ccols 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70'
  'd acc'
  'draw title Anomaly Correlation: (Pred-ERA5) vs (Truth-ERA5)'
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

