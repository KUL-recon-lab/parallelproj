Installation
============

``parallelproj`` is a **pure Python** package for PET image reconstruction.
It relies on two lower-level dependencies:

- `libparallelproj <https://libparallelproj.readthedocs.io>`_ — a compiled C++/CUDA library providing the core projector implementations
- `parallelproj-core <https://libparallelproj.readthedocs.io>`_ — a minimal Python interface to ``libparallelproj``

Both ``libparallelproj`` and ``parallelproj-core`` are available on **conda-forge** and are documented at `libparallelproj.readthedocs.io <https://libparallelproj.readthedocs.io>`_.

.. note::
    We strongly recommend installing ``parallelproj`` from **conda-forge**, which automatically pulls in the correct pre-compiled ``libparallelproj`` variant (CPU or CUDA) for your system.

.. tip::

   You can get **miniforge** (a minimal conda installer configured for conda-forge) `here <https://github.com/conda-forge/miniforge>`_.
   We recommend installing into a **dedicated virtual environment**.

Default install (auto CUDA detection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following commands create a new environment named ``parallelproj`` and install the package along with all required compiled libraries.

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console

           $ mamba create -n parallelproj -c conda-forge parallelproj

    .. tab-item:: conda

        .. code-block:: console

           $ conda create -n parallelproj -c conda-forge parallelproj

After creation, activate the environment:

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console

           $ mamba activate parallelproj

    .. tab-item:: conda

        .. code-block:: console

           $ conda activate parallelproj

.. tip::

   To use ``parallelproj`` with **PyTorch** or **CuPy**, add them as extra dependencies directly in the environment creation call, e.g.:

   .. code-block:: console

      $ mamba create -n parallelproj -c conda-forge parallelproj pytorch

   .. code-block:: console

      $ mamba create -n parallelproj -c conda-forge parallelproj cupy

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

           $ conda create -n parallelproj-cuda129 -c conda-forge cuda-version=13.0 parallelproj

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

Verifying the installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

To check whether the installed ``parallelproj-core`` (backend) package was compiled with CUDA support, run the following in Python:

.. code-block:: python

   import parallelproj_core
   print(parallelproj_core.cuda_enabled)  # 1 = CUDA enabled, 0 = CPU only
