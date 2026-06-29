Quickstart
==========

This page shows the shortest path from a fresh install to a **forward and back
projection** with ``parallelproj``.  It uses the built-in
:class:`.DemoPETScannerGeometry` so that the scanner setup is a single line; for
a real scanner you would replace it with
:class:`.RegularPolygonPETScannerGeometry` (or build a custom geometry), but the
rest of the workflow is identical.

Make sure ``parallelproj`` is installed first (see :doc:`installation`).

A minimal PET sinogram projection
----------------------------------

The example below uses the **regular-polygon route**, which is the right choice
for scanners with cylindrical symmetry and a single layer of LOR endpoints:

``scanner geometry`` → ``Michelogram`` (axial plane layout) →
``RegularPolygonPETLORDescriptor`` → ``RegularPolygonPETProjector``.

(General block / panel scanners use the **equal-block route** instead --
:class:`.ModularizedPETScannerGeometry` → :class:`.EqualBlockPETLORDescriptor` →
:class:`.EqualBlockPETProjector`; see the :doc:`API reference <api_projectors>`.)

.. literalinclude:: quickstart_minimal.py
    :language: python

That is the whole core API: a projector is a
:class:`.LinearOperator`, so ``proj(img)`` forward projects, ``proj.adjoint(sino)``
(equivalently ``proj.H(sino)``) back projects, and the two are exact adjoints of
each other -- everything else in the library (reconstruction algorithms, priors,
resolution and attenuation models) is built on top of this.

A few things worth knowing
--------------------------

* **Imports come from submodules.** The top-level ``parallelproj`` namespace is
  intentionally minimal, so classes are imported from their submodule
  (``from parallelproj.projectors import RegularPolygonPETProjector``).  The
  :ref:`import map <api_import_map>` in the API reference lists where each
  public name lives.
* **Arrays are float32.** Projectors expect and return single-precision arrays;
  create your images with ``dtype=xp.float32``.
* **Same code on CPU and GPU.** ``parallelproj`` is
  `python array API <https://data-apis.org/array-api/latest/>`_ compatible.  To
  run the snippet above on a CUDA GPU, change only the import
  (``import array_api_compat.torch as xp`` or ``import array_api_compat.cupy as xp``)
  and set ``dev = "cuda"``.  Use :func:`.to_numpy_array` to bring a result back
  to NumPy (e.g. for plotting).
* **Time-of-flight.** For a TOF projector, create a
  :class:`.TOFParameters` object and assign it to ``proj.tof_parameters``; the
  forward projection then gains a trailing TOF-bin axis.  See the
  ``02_pet_sinogram_projections`` gallery examples.

Next steps
----------

* :doc:`Examples gallery <auto_examples/index>` -- scanner geometries, TOF and
  listmode projectors, and full reconstruction algorithms (MLEM/OSEM, PDHG,
  transmission/MLTR, MLAA, ...).
* :doc:`API reference <api_projectors>` -- all projectors, operators and
  objective functions.
* `Source code & issue tracker <https://github.com/KUL-recon-lab/parallelproj>`_
  on GitHub.
