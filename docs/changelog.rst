Changelog
=========

2.0.0 (TBD)
-----------

Breaking Changes
~~~~~~~~~~~~~~~~

- **Python ≥ 3.12 required** (dropped support for 3.9, 3.10, 3.11)
- **New required dependency:** ``parallelproj-core >= 2.0.5`` — the compiled C/CUDA
  projection kernels have been extracted into a separate conda-forge package
  (``parallelproj-core``). The old shared-library loading via ctypes is gone.
- **Low-level projection functions moved to** ``parallelproj_core``: ``joseph3d_fwd``,
  ``joseph3d_back``, and all TOF variants (e.g. ``joseph3d_fwd_tof_sino``) are no longer
  in the ``parallelproj`` namespace. Import them from ``parallelproj_core`` instead. Note
  that the TOF function names were also reordered
  (e.g. ``joseph3d_fwd_tof_sino`` → ``joseph3d_tof_sino_fwd``).
- **Top-level namespace reduced**: only ``Array``, ``empty_cuda_cache``,
  ``to_numpy_array``, and ``count_event_multiplicity`` are exported from ``parallelproj``
  directly. Everything else must be imported from the relevant submodule:

  .. code-block:: python

     from parallelproj.operators import LinearOperator
     from parallelproj.projectors import RegularPolygonPETProjector
     from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry
     from parallelproj.pet_lors import RegularPolygonPETLORDescriptor

- **Runtime-detection variables removed** from the ``parallelproj`` namespace:
  ``cuda_present``, ``cupy_enabled``, ``torch_enabled``, ``num_visible_cuda_devices``,
  ``lib_parallelproj_c_fname``, ``lib_parallelproj_cuda_fname``, ``cuda_kernel_file``,
  ``is_cuda_array``. Use ``parallelproj_core.cuda_enabled`` for CUDA detection.
- **``RegularPolygonPETLORDescriptor`` signature changed**: ``max_ring_difference``
  parameter replaced by a ``michelogram`` parameter that accepts a ``Michelogram``
  object. See new ``Michelogram`` class below.
- **``TOFNonTOFElementwiseMultiplicationOperator`` removed**.
- **``MatrixOperator.iscomplex`` and ``ElementwiseMultiplicationOperator.iscomplex``
  changed from method to property**: replace ``op.iscomplex()`` calls with
  ``op.iscomplex``.
- **``GradientFieldProjectionOperator`` numeric change**: the ``eta`` normalisation
  formula was corrected (``sqrt(sum(g²) + η²)`` instead of ``sqrt(sum(g² + η²))``).
  Results will differ from v1.x.
- **License changed** from MIT to Apache-2.0.
- ``scipy >= 1.15`` now required (was ``~=1.0``).
- ``array-api-compat >= 1.7`` now required.
- Import-time banner and ``PARALLELPROJ_SILENT_IMPORT`` environment variable removed;
  ``import parallelproj`` is now silent.

New Features
~~~~~~~~~~~~

- **``parallelproj.functions`` submodule**: new module providing abstract base classes
  (``C1Function``, ``C2Function``, ``FunctionWithProx``, ``FunctionWithConjProx``) and
  concrete loss/regularisation implementations for optimisation:

  - ``NegPoissonLogL``, ``NegPoissonLogLSafe`` — Poisson log-likelihood (sinogram and
    masked variants)
  - ``NegPoissonLogLListmode`` — listmode Poisson log-likelihood with built-in forward
    model
  - ``HalfSquaredL2Deviation`` — weighted least-squares deviation
  - ``SumC1Function`` / ``SumC2Function`` — also created via ``f1 + f2`` operator
    overloading
  - ``C1AffineObjective`` / ``C2AffineObjective`` — compose a loss with an affine
    forward model
  - ``NonNegativeIndicator`` — non-negativity constraint with proximal operator
  - ``MixedL21Norm`` — mixed L2,1 norm for group sparsity / TV-type regularisation

- **``Michelogram`` class** (``parallelproj.pet_lors``): encapsulates the full axial
  plane layout for cylindrical PET scanners under odd-span compression, including
  ring-pair-to-plane tables and visualisation methods.
- **``SinogramAxialCompressionOperator``** (``parallelproj.pet_lors``): ``LinearOperator``
  that axially compresses a span-1 sinogram to a higher odd span (``mode="sum"`` or
  ``mode="average"``).
- **``LinearOperator.H`` property** and **``AdjointLinearOperator``** class: obtain the
  adjoint of any operator via ``A.H``.
- **``EqualBlockPETProjector`` ``num_chunks`` parameter**: split block-pair projections
  into chunks to reduce peak GPU memory usage.
- **``RegularPolygonPETProjector.convert_sinogram_to_listmode``** gained a ``shuffle``
  parameter to randomly permute the returned event list.
- **``VstackOperator``** now raises ``ValueError`` on inconsistent ``in_shape`` across
  stacked operators (previously silent).
- **``parallelproj.sinogram_symmetries`` submodule**: new module for exploiting
  the cylindrical symmetry of regular-polygon PET scanners to speed up geometric
  sensitivity calculations. Provides:

  - ``compute_sinogram_plane_symmetries`` -- partition all axial ring pairs into
    equivalence classes under axial block-shift, midplane reflection, and endpoint-swap
    symmetries (with optional edge-ring correction)
  - ``build_plane_class_indices``, ``build_view_class_indices``,
    ``build_radial_class_indices`` -- per-class index arrays for the three sinogram axes
  - ``reduce_sinogram_by_symmetry_class`` / ``expand_sinogram_by_symmetry_class`` --
    array-API-compatible reduce/expand operations for the typical
    reduce -> compute -> expand sensitivity workflow

- **``parallelproj.unlist`` submodule**: new module for histogramming listmode PET
  data into sinograms for ``RegularPolygonPETScannerGeometry``-based scanners.
  Provides:

  - ``regular_polygon_events_to_sinogram`` -- histogram per-event crystal and ring
    indices into a non-TOF or TOF sinogram array; supports numpy, cupy, and torch
  - ``detection_times_to_tof_bin`` -- convert raw detection-time differences
    (nanoseconds) to projector-convention unsigned TOF bin indices ready for
    histogramming

- **``parallelproj.__version__``** is now exposed at the top level.

1.10.2 (Aug 20, 2025)
----------------------

- add compatibility for latest cupy version (>= 13.5) which require ``from_dlpack`` to
  convert from torch tensors
- fix minor issues to be compatible with ``array-api-strict~=2.0``

1.10.1 (Jan 15, 2025)
----------------------

- add a check whether sum of tof bins along LOR is non-zero before running
  TOF sinogram back projector
- update installation instructions after conda-forge recipe was updated
- clean up RTD docs build

1.10.0 (July 29, 2024)
-----------------------

- add support for numpy>=2.0
- add tests with numpy 2.0 on python 3.9 and 3.12
- remove tox.ini

1.9.1 (June 19, 2024)
----------------------

- BUGFIX: add missing device in BlockPET LOR descriptor (needed for pytorch + cuda backend)

1.9 (June 18, 2024)
--------------------

- add functionality to create scanners, LOR descriptors and projectors for scanners
  consisting of equal "block" modules
- **BUGFIX:** correct behavior of TOF kernel truncation which was wrong in the case that
  the tof bin width was >> tof resolution

1.8 (March 20, 2024)
---------------------

- add function to count event multiplicity
- add more examples (e.g. DePierro and LM SPDHG)
- re-organize folder structure and pyproject.toml
- force array-api-compat<1.5 (bug in 1.5.0)
- use array-api-strict instead of numpy.array_api

1.7.3 (January 26, 2024)
-------------------------

- print banner
- test also on Windows

1.7.2 (January 26, 2024)
-------------------------

- require python>=3.9
- replace ``distuils.spawn`` by ``shutil.which``

1.7.1 (January 19, 2024)
-------------------------

- BUGFIX: correct bug in the "chunking" of TOF sinogram projections in the python
  interface

1.7.0 (January 15, 2024)
-------------------------

- update of documentation
- addition of more examples
- addition of high-level classes for RegularPolygonPETScanner and LOR descriptors

1.6.2 (December 01, 2023)
--------------------------

- BUGFIX: correct use of ``conj()`` of scalar value to be array api compatible
- BUGFIX: divided by ``float()`` to be array api compatible
- add scipy dependency

1.6.1 (October 18, 2023)
-------------------------

- BUGFIX: add sigma as explicit argument in ``GaussianFilterOperator`` and convert
  correctly to numpy/cupy arrays

1.6.0 (October 16, 2023)
-------------------------

- rewrite ``LinearOperator`` base class to support python array api including devices
- add missing type hints
- add finite difference operator
- remove obsolete functions

1.5.0 (July 29, 2023)
----------------------

- add compatibility of python wrapper to python array api (via array-api-compat)
  such that numpy, cupy, pytorch arrays can be directly projected
- no changes to the C/CUDA libs

1.4.0 (June 11, 2023)
----------------------

- add Linear Operators

1.3.7 (April 27, 2023)
-----------------------

- update documentation

1.3.6 (April 25, 2023)
-----------------------

- enable readthedocs

1.3.5 (April 23, 2023)
-----------------------

- add py.typed for mypy type checker

1.3.4 (April 21, 2023)
-----------------------

- rename python binding back to parallelproj

1.3.3 (April 20, 2023)
-----------------------

- import annotations from ``__future__`` to be compatible with older versions

1.3.2 (April 18, 2023)
-----------------------

- rename test folder
- lower absolute tolerance for forward TOF tests (otherwise windows builds might fail)

1.3.1 (April 17, 2023)
-----------------------

- add ``num_visible_devices`` definition when cuda is not present

1.3.0 (April 17, 2023)
-----------------------

- clean up pyproject.toml
- move tests and rename imports in tests
- rename python package to parallelprojpy and adapt setup.cfg
- add first version of pyproject.toml

1.2.16 (April 16, 2023)
------------------------

- improve way to detect whether visible GPUs are present in the python API
- remove AS approximation of ``erff`` in openMP lib (too large inaccuracies)
- add TOF LM tests
- add listmode wrappers
- add TOF sino fwd test

1.2.15 (April 15, 2023)
------------------------

- add TOF sino projector wrappers and first test
- BUGFIX: correct start and stop of loop over planes in cuda TOF sino projector when
  direction=2
- add adjointness test (indirect test for back projection)
- add first python unit test for non-tof fwd projection
- add first python wrappers for non-tof Joseph projectors

1.2.14 (February 15, 2023)
---------------------------

- make target link libraries (m and OpenMP) private

1.2.13 (January 13, 2023)
--------------------------

- fix variable expansion in Config.cmake.in
- update README
- add link to arxiv preprint

1.2.12 (January 08, 2023)
--------------------------

- set CUDA_HOST_COMPILER only when using clang
- skip build of cuda lib if cuda is not present

1.2.11 (January 05, 2023)
--------------------------

- set default ``CMAKE_CUDA_HOST_COMPILER`` to ``CMAKE_CXX_COMPILER``

1.2.10 (December 30, 2022)
---------------------------

- link parallelproj_c against libm (using PUBLIC link interface)
- use better way to test whether we have to link against libm
- add adjoint back projection test
- add more generic non-tof test that tests rays in all 3 directions

1.2.9 (December 09, 2022)
--------------------------

- BUGFIX: correct calculation of ``x_pr2`` when principal direction is 0

1.2.8 (December 02, 2022)
--------------------------

- do not install test binaries
- require CXX compiler only for CUDA

1.2.6 (November 18, 2022)
--------------------------

- clean up CMake logic

1.2.5 (November 11, 2022)
--------------------------

- add conditions to nested if-else when adding cuda subdir

1.2.4 (November 10, 2022)
--------------------------

- add fatal error if cuda lib is to be built but no cuda compiler is found

1.2.3 (November 04, 2022)
--------------------------

- add skip option for cmake

1.2.2 (November 03, 2022)
--------------------------

- read version from package.json
- add conda build
