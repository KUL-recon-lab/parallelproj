parallelproj documentation
==========================

:octicon:`mark-github;1em;sd-mr-1` `Source code on GitHub <https://github.com/KUL-recon-lab/parallelproj>`_ 
:octicon:`issue-opened;1em;sd-mr-1` `Report an issue <https://github.com/KUL-recon-lab/parallelproj/issues>`_

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

.. hint::
  *If you are using parallelproj, we highly recommend to read and cite our publication:*
  
  * G. Schramm, K. Thielemans: "**PARALLELPROJ - An open-source framework for fast calculation of projections in tomography**", Front. Nucl. Med., Volume 3 - 2023, doi: 10.3389/fnume.2023.1324562, `link to paper <https://www.frontiersin.org/articles/10.3389/fnume.2023.1324562/abstract>`_, `link to arxiv version <https://arxiv.org/abs/2212.12519>`_


.. rubric:: parallelproj vs other frameworks -- which to use when

.. grid:: 1 2 3 3
    :gutter: 3

    .. grid-item-card:: :octicon:`goal;1.5em;sd-mr-1` Aims of parallelproj

        A fast, GPU-native
        `python array API <https://data-apis.org/array-api/latest/>`_
        projection library -- a *toolbox*, not a full pipeline. Use it to
        **prototype reconstruction algorithms** or build **differentiable,
        DL-integrated** recon (PyTorch autograd) on CPU and GPU.

    .. grid-item-card:: :octicon:`git-compare;1.5em;sd-mr-1` Compared to STIR and CASToR

        `STIR <https://stir.sourceforge.net>`_ and
        `CASToR <https://castor-project.org>`_ are mature, full
        reconstruction *frameworks* with built-in scanner models,
        algorithms, data I/O and corrections -- for **complete, validated
        end-to-end reconstruction** across many scanners and modalities.

    .. grid-item-card:: :octicon:`heart;1.5em;sd-mr-1` Complementary, not competing

        parallelproj focuses on algorithm prototyping, not on replacing
        STIR or CASToR -- and can even serve as their **GPU projection
        backend**.

    .. grid-item-card:: :octicon:`download;1.5em;sd-mr-1` Easy to install

        Available on
        `conda-forge <https://github.com/conda-forge/parallelproj-feedstock>`_
        (one-command install) with many examples -- from install to a
        running prototype reconstruction, quickly.

    .. grid-item-card:: :octicon:`link-external;1.5em;sd-mr-1` More frameworks

        Also see the
        `Yale Reconstruction Toolbox <https://yrt-pet.readthedocs.io/en/latest/>`_
        and `PyTomography <https://pytomography.readthedocs.io/en/latest/>`_,
        built on the libparallelproj projectors.

    .. grid-item-card:: :octicon:`no-entry;1.5em;sd-mr-1` Out of scope (by design)

        No vendor-specific raw data readers, and no built-in randoms or
        scatter estimation. Need those? Use a full framework -- optionally
        with parallelproj as the projection backend, or get them from vendor
        toolboxes.

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
