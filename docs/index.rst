parallelproj documentation
==========================

**parallelproj** provides simple and fast high-level python routines for tomographic reconstruction
that are `python array API <https://data-apis.org/array-api/latest/>`_ 
compatible meaning that they can be used with a variety of python
array libraries (e.g. numpy, cupy, pytorch) and devices (CPU and CUDA GPUs).

**github repository** `<https://github.com/gschramm/parallelproj>`_

.. note:: 
  **Features of parallelproj**

  * dedicated **sinogram** and **listmode** versions of the projectors
  * **Python array API compatible Python interface** (e.g. directly compatible with numpy, cupy, **pytorch**) 
  * available on `conda-forge <https://github.com/conda-forge/parallelproj-feedstock>`_

.. hint::
  *If you are using parallelproj, we highly recommend to read and cite our publication* :cite:`Schramm2023`
  
  * G. Schramm, K. Thielemans: "**PARALLELPROJ - An open-source framework for fast calculation of projections in tomography**", Front. Nucl. Med., Volume 3 - 2023, doi: 10.3389/fnume.2023.1324562, `link to paper <https://www.frontiersin.org/articles/10.3389/fnume.2023.1324562/abstract>`_, `link to arxiv version <https://arxiv.org/abs/2212.12519>`_

Content
-------

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :caption: Getting started

    Installation <installation>
    Examples <auto_examples/index>

.. toctree::
    :maxdepth: 1
    :caption: API

    PET scanner geometries <api_pet_scanners>
    PET LOR / sinogram descriptors <api_pet_lors>
    PET TOF parameters <api_tof>
    PET projectors <api_projectors>
    Linear operators <api_operators>

References
----------

.. rubric:: References
.. bibliography::
