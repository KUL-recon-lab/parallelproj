nrcols = 500
nrrows = 500
nrplanes = 89

pixsizemm = 0.5
planesepmm = 2.79

proj = nidef_ge_signa(/nontof, /raytracer, /true_axialsampling, /true_radialsampling,$ 
                      pixelsizemm = pixsizemm, planesepmm = planesepmm,$ 
                      nrcols = nrcols, nrrows = nrrows, nrplanes = nrplanes,$ 
                      nrsegments = 45, /urgent)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;img = fltarr(nrcols, nrrows, nrplanes)
;img[10,10,10] = 1.0
;
;niproj, img, img_fwd, projdescrip = proj
;
;print, img_fwd.dim


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

rad_bin = 170
view_bins = [0, 56, 63]
plane_bins = [0, 44, 81, 89, 173, 174, 258]

sino = fltarr(357, 224, 1981)

ip = 0

FOREACH plane_bin, plane_bins DO BEGIN
    FOREACH view_bin, view_bins DO BEGIN
        sino[rad_bin - 1, view_bin+ip, plane_bin] = 1.0 + 0.1*ip
        sino[rad_bin, view_bin+ip, plane_bin] = 1.4 + 0.1*ip
        sino[rad_bin + 1, view_bin+ip, plane_bin] = 2.0 + 0.1*ip
    ENDFOREACH
    ip = ip + 1
ENDFOREACH

;save, sino, FILENAME="sino_back.sav"

print, "backprojecting sino"
niproj, sino_back, sino, projdescrip = proj, /backproject

save, sino_back, FILENAME="sino_back.sav"

niviewregis, img1 = sino_back

end
