nrcols = 500
nrrows = 500
nrplanes = 89

pixsizemm = 0.5
planesepmm = 2.70

proj = nidef_ge_signa(/nontof, /raytracer, /true_axialsampling, /true_radialsampling,$ 
                      pixelsizemm = pixsizemm, planesepmm = planesepmm,$ 
                      nrcols = nrcols, nrrows = nrrows, nrplanes = nrplanes,$ 
                      nrsegments = 45, segments = [0])

;img = fltarr(nrcols, nrrows, nrplanes)
;img[100,100,44] = 1.0
;niproj, img, img_fwd, projdescrip = proj

rad_bin = 170  ; 357//2 - 8
view_bin = 0
plane_bin = 44

sino1 = fltarr(357, 224, 89)
sino2 = fltarr(357, 224, 89)

sino1[rad_bin - 1, view_bin, plane_bin] = 1.0
sino1[rad_bin, view_bin, plane_bin] = 1.4
sino1[rad_bin + 1, view_bin, plane_bin] = 2.0

sino2[rad_bin - 1, view_bin + 56, plane_bin] = 1.0
sino2[rad_bin, view_bin + 56, plane_bin] = 1.4
sino2[rad_bin + 1, view_bin + 56, plane_bin] = 2.0

print, "backprojecting sino1"
niproj, sino1_back, sino1, projdescrip = proj, /backproject

print, "backprojecting sino2"
niproj, sino2_back, sino2, projdescrip = proj, /backproject

sino_back = sino1_back + sino2_back

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
vmax = max([max(sino1_back), max(sino2_back)])

sl = 44  ; nrplanes / 2 
img_plane = sino_back[*, *, sl]

;;;;;;;;;;;;;;;;;;;
img_plane = reverse(reverse(img_plane,1),2)
;;;;;;;;;;;;;;;;;;;

; mark corner for orientation check
img_plane[0:4, 0:14] = vmax

pp = image(img_plane, RGB_TABLE=COLORTABLE(1,/REVERSE))

end