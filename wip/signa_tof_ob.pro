nrcols = 275
nrrows = 275
nrplanes = 89

pixsizemm = 2.0
planesepmm = 2.79

timeres = 0.385

proj = nidef_ge_signa(/raytracer, /true_axialsampling, /true_radialsampling,$ 
                      pixelsizemm = pixsizemm, planesepmm = planesepmm,$ 
                      nrcols = nrcols, nrrows = nrrows, nrplanes = nrplanes,$ 
                      nrsegments = 45, segments = [0,-1,1,-2,2,-3,3], timeres = timeres)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;img = fltarr(nrcols, nrrows, nrplanes)
;img[10,10,10] = 1.0
;
;niproj, img, img_fwd, projdescrip = proj
;
;print, img_fwd.dim
;stop


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

rad_bins = [150, 179, 200]
view_bins = [8, 148]
plane_bins = [0, 44, 81, 89, 173, 174, 258, 259, 420, 498, 536]
tof_bins = [9, 21]

sino = fltarr(357, 224, 575, 27)

FOREACH view_bin, view_bins DO BEGIN
    ip = 0
    FOREACH plane_bin, plane_bins DO BEGIN
        ir = 0
        FOREACH rad_bin, rad_bins DO BEGIN
            it = 0
            FOREACH tof_bin, tof_bins DO BEGIN
                sino[rad_bin, view_bin, plane_bin, tof_bin] = 0.2*ir + 0.01*ip + (it + 1)
                it = it + 1
            ENDFOREACH
            ir = ir + 1
        ENDFOREACH
        ip = ip + 1
    ENDFOREACH
ENDFOREACH

print, "backprojecting sino"
sino_back = fltarr(nrcols,nrrows,nrplanes)
niproj, sino_back, sino, projdescrip = proj, /backproject

save, sino_back, FILENAME="tof_sino_back_ob.sav"

print, max(sino_back)
niviewregis, img1 = sino_back

end
