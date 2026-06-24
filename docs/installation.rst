Installation
============

``parallelproj`` is a **pure Python** package for PET image reconstruction.
It relies on two lower-level dependencies:

- `libparallelproj <https://libparallelproj.readthedocs.io>`_ -- a compiled C++/CUDA library providing the core projector implementations
- `parallelproj-core <https://libparallelproj.readthedocs.io>`_ -- a minimal Python interface to ``libparallelproj``

Both ``libparallelproj`` and ``parallelproj-core`` are available on **conda-forge** and are documented at `libparallelproj.readthedocs.io <https://libparallelproj.readthedocs.io>`_.

.. note::
    We strongly recommend installing ``parallelproj`` from **conda-forge**, which automatically pulls in the correct pre-compiled ``libparallelproj`` variant (CPU or CUDA) for your system.

**Requirements**

- **Python ≥ 3.12**.
- A platform for which ``libparallelproj`` is built on conda-forge (Linux, macOS and Windows; CUDA builds are available on the platforms supported by the feedstock). conda-forge selects the right build automatically; you do not need to compile anything yourself.

.. important::
    ``parallelproj`` cannot be installed with ``pip`` alone.  Its compiled
    backend (``libparallelproj`` / ``parallelproj-core``) is distributed **only
    through conda-forge**, not on PyPI.  Installing the pure-Python part with
    ``pip`` will import but fail at the first projection because the backend is
    missing.  Always install from conda-forge as shown below.

.. tip::

   You can get **miniforge** (a minimal conda installer configured for conda-forge) `here <https://github.com/conda-forge/miniforge>`_.
   Alternatively, `pixi <https://pixi.sh>`_ is a modern, cross-platform package manager built on conda-forge that handles environments automatically.
   We recommend installing into a **dedicated virtual environment** regardless of the tool you choose.

Default install (auto CUDA detection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following commands create a new environment and install the package along with all required compiled libraries.

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console

           $ mamba create -n parallelproj -c conda-forge parallelproj

    .. tab-item:: conda

        .. code-block:: console

           $ conda create -n parallelproj -c conda-forge parallelproj

    .. tab-item:: pixi

        Run the following from your project directory.  ``pixi`` ties the
        environment to the directory rather than a global name.

        .. code-block:: console

           $ pixi init -c conda-forge
           $ pixi add parallelproj

After creation, activate the environment:

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console

           $ mamba activate parallelproj

    .. tab-item:: conda

        .. code-block:: console

           $ conda activate parallelproj

    .. tab-item:: pixi

        .. code-block:: console

           $ pixi shell

.. tip::

   To use ``parallelproj`` with **PyTorch** or **CuPy**, add them as extra dependencies:

   .. tab-set::

       .. tab-item:: mamba

           .. code-block:: console

              $ mamba create -n parallelproj -c conda-forge parallelproj pytorch

           .. code-block:: console

              $ mamba create -n parallelproj -c conda-forge parallelproj cupy

       .. tab-item:: pixi

           .. code-block:: console

              $ pixi add pytorch

           .. code-block:: console

              $ pixi add cupy

Force a specific CUDA build
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need a particular CUDA toolkit version of ``libparallelproj``, you can pin it explicitly when creating the environment.
Replace ``cuda129`` below with the CUDA version matching your system (e.g. ``cuda129``, ``cuda13``).

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console

           $ mamba create -n parallelproj-cuda129 -c conda-forge cuda-version=12.9 parallelproj

    .. tab-item:: conda

        .. code-block:: console

           $ conda create -n parallelproj-cuda129 -c conda-forge cuda-version=12.9 parallelproj

    .. tab-item:: pixi

        .. code-block:: console

           $ pixi add 'cuda-version=12.9' parallelproj

Force a CPU-only build
^^^^^^^^^^^^^^^^^^^^^^

To explicitly install the CPU-only variant of ``libparallelproj`` (e.g. on a machine without a GPU):

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console

           $ mamba create -n parallelproj-cpu -c conda-forge parallelproj "libparallelproj=*=cpu*"

    .. tab-item:: conda

        .. code-block:: console

           $ conda create -n parallelproj-cpu -c conda-forge parallelproj "libparallelproj=*=cpu*"

    .. tab-item:: pixi

        .. code-block:: console

           $ pixi add 'libparallelproj=*=cpu*' parallelproj

Verifying the installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

First check that the backend imports and report whether it was compiled with CUDA support:

.. code-block:: python

   import parallelproj
   print(parallelproj.__version__)  # print version of parallelproj python package

   import parallelproj_core
   print(parallelproj_core.__version)  # print version of compiled projector backend core library
   print(parallelproj_core.cuda_enabled)  # 1 = CUDA enabled, 0 = CPU only

Then confirm that the full stack works end to end by building a small projector
and running a forward and back projection (the same minimal example as the
:doc:`quickstart`).
