parallelproj documentation
==========================

**parallelproj** provides simple and fast high-level python routines for tomographic reconstruction
that are `python array API <https://data-apis.org/array-api/latest/>`_ 
compatible meaning that they can be used with a variety of python
array libraries (e.g. numpy, cupy, pytorch) and devices (CPU and CUDA GPUs).

.. grid:: 1 2 2 3
    :gutter: 3

    .. grid-item-card:: :octicon:`package;1.5em;sd-mr-1` Multi-backend

        The same code runs on **NumPy**, **CuPy** and **PyTorch** arrays
        through the `Python array API <https://data-apis.org/array-api/latest/>`_.

    .. grid-item-card:: :octicon:`zap;1.5em;sd-mr-1` GPU-native

        Fast C/CUDA projectors — the *same* code runs on the **CPU** or a
        **CUDA GPU**, chosen by the array backend and device.

    .. grid-item-card:: :octicon:`telescope;1.5em;sd-mr-1` Sinogram & listmode

        Dedicated **sinogram** and **listmode** PET projectors, with optional
        **time-of-flight** (TOF) support.

    .. grid-item-card:: :octicon:`flame;1.5em;sd-mr-1` Differentiable / DL-ready

        Projectors plug into **PyTorch autograd**, ready to embed in
        deep-learning reconstruction pipelines.

    .. grid-item-card:: :octicon:`workflow;1.5em;sd-mr-1` Reconstruction examples

        Worked examples running **OS-MLEM** and other algorithms on both
        sinogram and listmode data.

    .. grid-item-card:: :octicon:`download;1.5em;sd-mr-1` Open & easy to install

        On `conda-forge <https://github.com/conda-forge/parallelproj-feedstock>`_
        (one-command install) and released under the **Apache-2.0** license.

:octicon:`mark-github;1em;sd-mr-1` `Source code on GitHub <https://github.com/KUL-recon-lab/parallelproj>`_ • :octicon:`issue-opened;1em;sd-mr-1` `Report an issue <https://github.com/KUL-recon-lab/parallelproj/issues>`_

.. hint::
  *If you are using parallelproj, we highly recommend to read and cite our publication:*
  
  * G. Schramm, K. Thielemans: "**PARALLELPROJ - An open-source framework for fast calculation of projections in tomography**", Front. Nucl. Med., Volume 3 - 2023, doi: 10.3389/fnume.2023.1324562, `link to paper <https://www.frontiersin.org/articles/10.3389/fnume.2023.1324562/abstract>`_, `link to arxiv version <https://arxiv.org/abs/2212.12519>`_


.. dropdown:: parallelproj vs other frameworks -- which to use when
  :icon: question

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
