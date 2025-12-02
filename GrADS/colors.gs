*
* Custom Color Definitions for GrADS
* Generated from matplotlib-style colormaps
*
* Usage: run this script before setting colors
*   'run colors.gs'
*   'set rgb 16 r g b'  (colors 16-99 are available)
*

function colors(args)

* Temperature colormap (indices 16-27) - Cold to Hot
* Similar to earth.nullschool.net style
'set rgb 16 0 0 51'
'set rgb 17 0 51 128'
'set rgb 18 0 102 204'
'set rgb 19 0 178 229'
'set rgb 20 0 255 128'
'set rgb 21 128 255 0'
'set rgb 22 204 255 0'
'set rgb 23 255 204 0'
'set rgb 24 255 153 0'
'set rgb 25 255 102 0'
'set rgb 26 204 0 0'
'set rgb 27 136 0 0'

* Bias colormap (indices 30-41) - Diverging purple-white-orange
* White at center (0), cool colors for negative, warm for positive
'set rgb 30 94 60 153'
'set rgb 31 136 108 178'
'set rgb 32 178 171 210'
'set rgb 33 224 224 224'
'set rgb 34 242 242 242'
'set rgb 35 255 255 255'
'set rgb 36 255 240 230'
'set rgb 37 255 224 208'
'set rgb 38 253 184 99'
'set rgb 39 247 148 30'
'set rgb 40 230 97 1'
'set rgb 41 204 76 2'

* Rainbow colormap for RMSE/errors (indices 50-61) - Turbo style
'set rgb 50 48 18 59'
'set rgb 51 75 22 129'
'set rgb 52 96 50 189'
'set rgb 53 92 101 228'
'set rgb 54 58 146 243'
'set rgb 55 44 183 221'
'set rgb 56 64 210 163'
'set rgb 57 117 220 102'
'set rgb 58 180 226 62'
'set rgb 59 236 212 47'
'set rgb 60 252 154 17'
'set rgb 61 240 80 6'

* Anomaly correlation colormap (indices 70-80) - YlGnBu style
'set rgb 70 255 255 217'
'set rgb 71 237 248 177'
'set rgb 72 199 233 180'
'set rgb 73 127 205 187'
'set rgb 74 65 182 196'
'set rgb 75 29 145 192'
'set rgb 76 34 94 168'
'set rgb 77 37 52 148'
'set rgb 78 8 29 88'
'set rgb 79 8 29 88'
'set rgb 80 8 29 88'

return

