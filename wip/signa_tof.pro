nrcols = 275
nrrows = 275
nrplanes = 89

pixsizemm = 2.0
planesepmm = 2.79

timeres = 0.385

proj = nidef_ge_signa(/raytracer, /true_axialsampling, /true_radialsampling,$ 
                      pixelsizemm = pixsizemm, planesepmm = planesepmm,$ 
                      nrcols = nrcols, nrrows = nrrows, nrplanes = nrplanes,$ 
                      nrsegments = 45, segments = [0], timeres = timeres)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;img = fltarr(nrcols, nrrows, nrplanes)
;img[10,10,10] = 1.0
;
;niproj, img, img_fwd, projdescrip = proj
;
;print, img_fwd.dim


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

rad_bin = 170
view_bins = [0, 74, 148]
plane_bins = [0,44,80]
tof_bin = 11

sino = fltarr(357, 224, 89, 27)

ip = 0

FOREACH plane_bin, plane_bins DO BEGIN
    FOREACH view_bin, view_bins DO BEGIN
        sino[rad_bin, view_bin, plane_bin, tof_bin] = 1.4 + 0.1*ip
        sino[rad_bin+11, view_bin, plane_bin, tof_bin] = 1.6 + 0.1*ip
    ENDFOREACH
    ip = ip + 1
ENDFOREACH

;save, sino, FILENAME="sino_back.sav"

print, "backprojecting sino"
sino_back = fltarr(nrcols,nrrows,nrplanes)
niproj, sino_back, sino, projdescrip = proj, /backproject

save, sino_back, FILENAME="tof_sino_back.sav"

print, max(sino_back)
niviewregis, img1 = sino_back

end
