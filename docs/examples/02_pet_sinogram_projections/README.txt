PET sinogram and listmode projector examples
--------------------------------------------

Here you build projectors and run forward and back projections: non-TOF and TOF
sinogram projectors for cylindrical scanners, the equal-block projector for
modular scanners, the listmode projector, and the unlister that maps listmode
events to sinograms.  These examples show the core ``proj(image)`` /
``proj.adjoint(sinogram)`` interface that the reconstruction galleries rely on.
