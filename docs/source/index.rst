.. parallelproj documentation master file, created by
   sphinx-quickstart on Fri Dec 29 09:47:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

parallelproj: a Python array API compatible library for fast tomographic projections
====================================================================================

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

.. hint::
  **For bug reports or feature requests, please open a github issue** `here <https://github.com/gschramm/parallelproj/issues>`_.


.. toctree::
    :caption: Getting started
    :maxdepth: 1

    installation
    auto_examples/index

All examples can be run interactively without installation on `mybinder.org`:

.. image:: https://mybinder.org/badge_logo.svg
  :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples

.. toctree::
    :caption: Python API
    :maxdepth: 1

    python_api

.. rubric:: References
.. bibliography::
