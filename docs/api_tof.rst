:tocdepth: 2

PET TOF parameters ``parallelproj.tof``
---------------------------------------

Time-of-flight parameters for TOF projectors.  Construct a
:class:`.TOFParameters` object (number of TOF bins, bin width and TOF
resolution) and assign it to a projector's ``tof_parameters`` attribute to turn a
non-TOF projector into a TOF projector; the forward projection then gains a
trailing TOF-bin axis.  See the ``02_pet_sinogram_projections`` gallery examples
for end-to-end usage.

.. automodule:: parallelproj.tof
