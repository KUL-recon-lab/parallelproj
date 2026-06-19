parallelproj documentation
==========================

**parallelproj** provides simple and fast high-level python routines for tomographic reconstruction
that are `python array API <https://data-apis.org/array-api/latest/>`_ 
compatible meaning that they can be used with a variety of python
array libraries (e.g. numpy, cupy, pytorch) and devices (CPU and CUDA GPUs).

.. grid:: 1 2 2 3
    :gutter: 3

    .. grid-item-card:: :octicon:`package;1.5em;sd-mr-1` Multi-backend

        The same reconstruction code runs on **NumPy**, **CuPy** and **PyTorch**
        arrays through the
        `Python array API <https://data-apis.org/array-api/latest/>`_.

    .. grid-item-card:: :octicon:`zap;1.5em;sd-mr-1` GPU-native

        Built on fast C/CUDA projectors — the *same* code executes on the
        **CPU** or a **CUDA GPU**, chosen by the array backend and device.

    .. grid-item-card:: :octicon:`flame;1.5em;sd-mr-1` Differentiable / DL-ready

        Projectors integrate with **PyTorch autograd**, so they can be embedded
        directly in deep-learning reconstruction pipelines.

    .. grid-item-card:: :octicon:`telescope;1.5em;sd-mr-1` Sinogram & listmode

        Dedicated **sinogram** and **listmode** PET projectors, with optional
        time-of-flight (TOF) support.

    .. grid-item-card:: :octicon:`download;1.5em;sd-mr-1` Easy to install

        Available on
        `conda-forge <https://github.com/conda-forge/parallelproj-feedstock>`_ —
        a single command pulls in the right CPU or CUDA build.

    .. grid-item-card:: :octicon:`law;1.5em;sd-mr-1` Open source

        Developed in the open and released under the permissive
        **Apache-2.0** license.

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

    API reference <api>
