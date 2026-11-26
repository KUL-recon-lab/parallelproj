Examples
========

The examples go from PET scanner geometries and projectors, through iterative
reconstruction of sinogram and listmode data, to transmission and joint
activity / attenuation reconstruction, and PyTorch layers on one or several
GPUs.  Every example runs unchanged with NumPy, CuPy or PyTorch, on the CPU or
a GPU, and can be downloaded as a Python script or a Jupyter notebook.

**New here?** Read the :doc:`/quickstart`, then start with these three:

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: 1. Scanner geometry
      :link: sphx_glr_auto_examples_01_pet_geometry_01_run_regular_polygon_pet_scanner.py
      :link-type: ref

      Build a cylindrical PET scanner and look at its detector (LOR endpoint)
      layout.

   .. grid-item-card:: 2. Projectors
      :link: sphx_glr_auto_examples_02_pet_sinogram_projections_01_run_pet_non_tof_sinogram_projector.py
      :link-type: ref

      Forward and back project an image with a sinogram projector.

   .. grid-item-card:: 3. Reconstruction
      :link: sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py
      :link-type: ref

      Reconstruct an image with MLEM and OSEM and compare their convergence.
