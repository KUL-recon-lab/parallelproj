:tocdepth: 2

PET projectors ``parallelproj.projectors``
------------------------------------------

A PET sinogram projector is the last link in a short construction chain, and
there are **two routes** depending on the scanner:

* **Regular-polygon route** -- for scanners with cylindrical symmetry and a
  single layer of LOR endpoints (the most common case):
  :class:`.RegularPolygonPETScannerGeometry` → :class:`.Michelogram` →
  :class:`.RegularPolygonPETLORDescriptor` → :class:`.RegularPolygonPETProjector`.
* **Equal-block route** -- for general modular geometries built from detector
  blocks (block / panel scanners, or any layout without cylindrical symmetry):
  :class:`.ModularizedPETScannerGeometry` (from :class:`.BlockPETScannerModule`
  objects) → :class:`.EqualBlockPETLORDescriptor` →
  :class:`.EqualBlockPETProjector` (no :class:`.Michelogram` is needed).

If your scanner has cylindrical symmetry and a single endpoint layer, use the
regular-polygon route (shown in the :doc:`quickstart`); otherwise use the
equal-block route.  Either way the projector is a :class:`.LinearOperator`:
``proj(img)`` forward projects and ``proj.adjoint(sino)`` (or ``proj.H(sino)``)
back projects.  Inputs and outputs are **float32** arrays, and the same code runs
on CPU or GPU depending on the array backend / device used to build the geometry.

.. automodule:: parallelproj.projectors
