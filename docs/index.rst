parallelproj documentation
==========================

**parallelproj** provides simple and fast high-level python routines for tomographic reconstruction
that are `python array API <https://data-apis.org/array-api/latest/>`_ 
compatible meaning that they can be used with a variety of python
array libraries (e.g. numpy, cupy, pytorch) and devices (CPU and CUDA GPUs).

.. note:: 
  **Features of parallelproj**

  * dedicated **sinogram** and **listmode** versions of the projectors
  * **Python array API compatible Python interface** (e.g. directly compatible with numpy, cupy, **pytorch**)
  * available on `conda-forge <https://github.com/conda-forge/parallelproj-feedstock>`_

**Source code** is on `GitHub <https://github.com/KUL-recon-lab/parallelproj>`_
(`report an issue <https://github.com/KUL-recon-lab/parallelproj/issues>`_).
``parallelproj`` is released under the
`Apache-2.0 license <https://github.com/KUL-recon-lab/parallelproj/blob/main/LICENSE>`_.

.. hint::
  *If you are using parallelproj, we highly recommend to read and cite our publication:*
  
  * G. Schramm, K. Thielemans: "**PARALLELPROJ - An open-source framework for fast calculation of projections in tomography**", Front. Nucl. Med., Volume 3 - 2023, doi: 10.3389/fnume.2023.1324562, `link to paper <https://www.frontiersin.org/articles/10.3389/fnume.2023.1324562/abstract>`_, `link to arxiv version <https://arxiv.org/abs/2212.12519>`_

.. admonition:: parallelproj vs other frameworks -- which to use when

  * **parallelproj** is a fast, GPU-native, `python array API <https://data-apis.org/array-api/latest/>`_
    projection library -- a *toolbox*, not a full pipeline. Reach for it
    when you want to **prototype your own reconstruction algorithms** in Python,
    or build **differentiable, deep-learning-integrated** reconstruction
    (PyTorch autograd) with the same code on CPU and GPU.

  * **Easy to install and get started.** parallelproj is available on
    `conda-forge <https://github.com/conda-forge/parallelproj-feedstock>`_
    (single-command install) and ships with many examples, so you can go
    from install to a running prototype reconstruction quickly.

  * `STIR <https://stir.sourceforge.net>`_ and `CASToR <https://castor-project.org>`_
    are mature, full reconstruction *frameworks* with built-in scanner models,
    established algorithms, data I/O and corrections. Reach for them when you
    want a **complete, validated end-to-end reconstruction** out of the box
    across many scanners and modalities.

  * **They are complementary, not competing.** parallelproj deliberately focuses
    on algorithm prototyping; it does not aim to replace
    STIR or CASToR -- and can even serve as a GPU projection backend for
    higher-level frameworks.

  * **Out of scope for parallelproj (by design).** No vendor-specific raw data
    format readers, and currently no built-in randoms or scatter estimation. If
    your workflow needs these, use a full framework -- optionally with
    parallelproj as the projection backend.

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :hidden:
    :caption: Getting started

    Installation <installation>
    Quickstart <quickstart>
    Changelog <changelog>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Examples

    Examples <auto_examples/index>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: API

    Linear operators <api_operators>
    Functions <api_functions>
    PET scanner geometries <api_pet_scanners>
    PET LOR / sinogram descriptors <api_pet_lors>
    PET sinogram symmetries <api_pet_sino_symmetries>
    PET projectors <api_projectors>
    PET TOF parameters <api_tof>
    PET LM Unlisting <api_pet_unlist>
    Data <api_data>
