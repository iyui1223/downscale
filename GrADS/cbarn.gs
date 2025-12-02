***************************************************************
* Author:    Beate Klinger, (klinger@gkss.de)                 
* Date:      1998                                             
*
* Purpose:   To plot a colorbar on the screen at location x y
*             In addition to the low and high numbers plotting at the bottom of the bar, the color square numbers also plot on top of each colorbar square.
*            This code is used in my plot scripts.
*
* Arguments:
*   sf = scale factor for  sizing
*   orientation: 0 for horizontal orientation, 1 for vertical
*   xmid, ymid: x and y position of the middle of the bar
*
* Example of use:
*   run cbarn 1 0 4.5 5.5
*
* Modifications:
*   - Bob Hart (rhart@fsu.edu): 
*     Modified the script so bars plot horizontally or vertically
*     Modified script to take arguments for positioning the colorbar
*
*   - Adapted for general use
*     
***************************************************************
function cbarn(args)

sf=subwrd(args,1)
orientation=subwrd(args,2)
xmid=subwrd(args,3)
ymid=subwrd(args,4)

if (sf=''); sf=1; endif
if (orientation=''); orientation=0; endif
if (xmid=''); xmid=5.5; endif
if (ymid=''); ymid=0.5; endif

* Get colorbar info
'query shades'
shdinfo=result

if (subwrd(shdinfo,1)='None')
  say 'Cannot plot color bar: No shading information'
  return
endif

* Set up parameters
num=subwrd(shdinfo,5)
xcen=xmid
ycen=ymid

* Orientation specific settings
if (orientation=0)
  xwid=0.25*sf
  ywid=0.15*sf
  xl=xcen-num/2*xwid
  xr=xl+xwid
  xlab=xr-xwid/2
  yb=ycen-ywid/2
  yt=yb+ywid
else
  xwid=0.15*sf
  ywid=0.25*sf
  yb=ycen-num/2*ywid
  yt=yb+ywid
  ylab=yt-ywid/2
  xl=xcen-xwid/2
  xr=xl+xwid
endif

'set string 1 c 3'
'set strsiz 'sf*0.09

* Plot colorbar
_cnum=0
while (_cnum<num)
  rec=sublin(shdinfo,_cnum+2)
  _col=subwrd(rec,1)
  _hi=subwrd(rec,3)
  
  'set line '_col
  'draw recf 'xl' 'yb' 'xr' 'yt
  
  if (orientation=0)
    'draw string 'xlab' 'yt+0.05*sf' '_hi
    xr=xr+xwid
    xl=xl+xwid
    xlab=xr-xwid/2
  else
    'draw string 'xr+0.05*sf' 'ylab' '_hi
    yt=yt+ywid
    yb=yb+ywid
    ylab=yt-ywid/2
  endif
  
  _cnum=_cnum+1
endwhile

return

