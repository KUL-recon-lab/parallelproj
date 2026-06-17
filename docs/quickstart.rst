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

Every PET sinogram projector is built from the same short chain:

``scanner geometry`` → ``Michelogram`` (axial plane layout) →
``LOR / sinogram descriptor`` → ``projector``.

.. code-block:: python

    import array_api_compat.numpy as xp  # swap for array_api_compat.torch / .cupy
    from parallelproj.pet_scanners import DemoPETScannerGeometry
    from parallelproj.pet_lors import Michelogram, RegularPolygonPETLORDescriptor
    from parallelproj.projectors import RegularPolygonPETProjector
    from parallelproj import to_numpy_array

    dev = "cpu"  # or "cuda"

    # 1) a ready-made demo cylindrical PET scanner (here trimmed to 4 rings)
    scanner = DemoPETScannerGeometry(xp, dev, num_rings=4)

    # 2) Michelogram (axial plane layout) -> LOR / sinogram descriptor
    lor_desc = RegularPolygonPETLORDescriptor(
        scanner,
        Michelogram(scanner.num_rings, max_ring_difference=3, span=1),
    )

    # 3) geometric (non-TOF) sinogram projector for a 40 x 8 x 40 image
    proj = RegularPolygonPETProjector(
        lor_desc, img_shape=(40, 8, 40), voxel_size=(2.0, 2.0, 2.0)
    )

    # a simple test image: a hot box in the centre
    img = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
    img[15:25, :, 15:25] = 1.0

    sino = proj(img)           # forward projection:  image  -> sinogram
    back = proj.adjoint(sino)  # back projection (adjoint);  proj.H(sino) works too

    print("image shape:    ", proj.in_shape)
    print("sinogram shape: ", proj.out_shape)
    print("backproj shape: ", to_numpy_array(back).shape)

That is the whole core API: a projector is a
:class:`.LinearOperator`, so ``proj(img)`` forward projects, ``proj.adjoint(sino)``
(equivalently ``proj.H(sino)``) back projects, and the two are exact adjoints of
each other -- everything else in the library (reconstruction algorithms, priors,
resolution and attenuation models) is built on top of this.

A few things worth knowing
--------------------------

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
