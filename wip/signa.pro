nrcols = 500
nrrows = 500
nrplanes = 89

pixsizemm = 0.5
planesepmm = 2.79

proj = nidef_ge_signa(/nontof, /raytracer, /true_axialsampling, /true_radialsampling,$ 
                      pixelsizemm = pixsizemm, planesepmm = planesepmm,$ 
                      nrcols = nrcols, nrrows = nrrows, nrplanes = nrplanes,$ 
                      nrsegments = 45, segments = [0])

rad_bin = 170
view_bins = [0, 56, 63]
plane_bins = [0, 44, 81]

sino = fltarr(357, 224, 89)

FOREACH plane_bin, plane_bins DO BEGIN
    FOREACH view_bin, view_bins DO BEGIN
        sino[rad_bin - 1, view_bin, plane_bin] = 1.0
        sino[rad_bin, view_bin, plane_bin] = 1.4
        sino[rad_bin + 1, view_bin, plane_bin] = 2.0
    ENDFOREACH
ENDFOREACH

print, "backprojecting sino"
niproj, sino_back, sino, projdescrip = proj, /backproject

save, sino_back, FILENAME="sino_back.sav"

niviewregis, img1 = sino_back

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;vmax = max([max(sino1_back), max(sino2_back)])
;
;sl = 44  ; nrplanes / 2 
;img_plane = sino_back[*, *, sl]
;
;;;;;;;;;;;;;;;;;;;;
;img_plane = reverse(reverse(img_plane,1),2)
;;;;;;;;;;;;;;;;;;;;
;
;; mark corner for orientation check
;img_plane[0:4, 0:14] = vmax
;
;pp = image(img_plane, RGB_TABLE=COLORTABLE(1,/REVERSE))

end