:tocdepth: 2

PET LOR / sinogram descriptors ``parallelproj.pet_lors``
--------------------------------------------------------

A LOR / sinogram descriptor maps detector lines of response to sinogram bins and
is the link between a scanner geometry and a projector (see :doc:`quickstart`).
For cylindrical scanners use :class:`.RegularPolygonPETLORDescriptor` together
with a :class:`.Michelogram` (the axial plane layout); for modular block / panel
scanners use :class:`.EqualBlockPETLORDescriptor`.
:class:`.SinogramSpatialAxisOrder` controls the ordering of the sinogram axes.

.. automodule:: parallelproj.pet_lors
