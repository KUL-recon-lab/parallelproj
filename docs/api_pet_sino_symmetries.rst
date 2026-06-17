:tocdepth: 2

PET sinogram symmetries ``parallelproj.sinogram_symmetries``
------------------------------------------------------------

Utilities for exploiting the geometric symmetries of regular-polygon sinograms to
save memory and computation (for example when precomputing sensitivity images).
The typical workflow is to compute the sinogram-plane symmetry classes, build the
per-axis class indices, reduce a sinogram to its unique classes, operate on the
reduced data, and finally expand back to the full sinogram.  These are advanced
helpers; most users do not need them directly.

.. automodule:: parallelproj.sinogram_symmetries
