Changelog
=========

2.x
---

2.0.0 (TBD)
^^^^^^^^^^^

New Features
~~~~~~~~~~~~

Major new capabilities
""""""""""""""""""""""

- **``parallelproj.functions`` submodule**: new module providing abstract base classes
  (``C1Function``, ``C2Function``, ``FunctionWithProx``, ``FunctionWithConjProx``) and
  concrete loss/regularisation implementations for optimisation:

  - ``NegPoissonLogL`` — Poisson log-likelihood with two evaluation modes.  The
    default ("safe epsilon") mode evaluates a shifted-Poisson surrogate with
    ``eps = rel_eps * mean(y)`` added to data and expectation: finite for any
    non-negative expectation, per-bin minimiser unchanged, gradient bias
    proportional to the residual; the dual prox is shifted consistently.
    ``exact=True`` evaluates the unmodified log-likelihood (bins with ``y == 0``,
    e.g. virtual bins, are handled exactly) and requires a positive expectation in
    all bins with counts.  An optional absolute ``eps`` allows sharing one epsilon
    across subset objectives, and ``enable_extra_checks=True`` warns on inputs
    producing ``nan`` / ``inf``
  - ``NegPoissonLogLListmode`` — listmode Poisson log-likelihood with built-in forward
    model; optional ``eps`` kwarg (default ``0.0``) smooths the per-event log /
    division terms (expectation-only shift — the symmetric shifted-Poisson surrogate
    of ``NegPoissonLogL`` would require full-sinogram projections and is not
    listmode-compatible)
  - ``LogCosh`` — edge-preserving log-cosh prior (smooth, with bounded curvature);
    used by the penalised reconstruction examples (MAPTR, MLAA)
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
- **GE-style axial sinogram plane layout** for ``Michelogram``: new
  ``MichelogramLayout`` enum and ``layout=`` argument, plus a
  ``Michelogram.ge(num_rings, max_ring_difference)`` convenience constructor.
  The GE-style layout uses segment 0 = ring differences {−1, 0, +1} (cross
  planes merged into virtual direct planes) and oblique segments = ring-
  difference pairs {±2k, ±(2k+1)}, ordered 0, +1, −1, +2, −2, … (the segment /
  ring-difference plane ordering used by GE-style sinograms; also known as
  "span 2" in STIR).  ``span`` is ignored for this layout and
  :attr:`Michelogram.span` returns ``None``.  Combine it with a matching
  ``RegularPolygonPETLORDescriptor`` for the GE scanner of interest.
- **``SinogramAxialCompressionOperator``** (``parallelproj.pet_lors``): ``LinearOperator``
  that axially compresses a span-1 sinogram to a higher odd span (``mode="sum"`` or
  ``mode="average"``).  A ``target_layout=MichelogramLayout.GE`` option compresses a
  span-1 sinogram to the **GE layout**; its ``mode="average"`` adjoint distributes a
  GE sinogram back onto the span-1 grid while preserving counts (e.g. to convert GE
  data to span-1 before detector mashing).
- **``SinogramMashingOperator``** (``parallelproj.pet_lors``): detector mashing for a
  span-1 regular-polygon sinogram.  Groups ``transaxial_factor`` within-side crystals
  and ``axial_factor`` rings into larger virtual detectors at the averaged endpoint
  position, mapping the fine sinogram to a much smaller mashed one (``mode="sum"`` for
  counts, ``mode="average"`` for multiplicative factors) with a genuine transpose and
  closed-form norm.  The mashed geometry is exposed as ``coarse_scanner`` /
  ``coarse_lor_descriptor`` (a regular-polygon descriptor), so a standard
  ``RegularPolygonPETProjector`` projects directly along the mashed LORs.  By default
  the coarse radial trim is derived automatically from the fine->coarse mapping so
  that no fine LOR is lost to trimming and no empty peripheral coarse radial bins
  remain (only the geometrically unavoidable degenerate self-pairs are dropped); pass
  ``coarse_radial_trim`` to override.  The operator is span-1 only; **GE-layout**
  sinograms are mashed by composition -- convert GE -> span-1 with the
  ``mode="average"`` adjoint of a span-1 <-> GE ``SinogramAxialCompressionOperator``,
  then mash, giving a pure span-1 coarse sinogram (see the
  ``01_pet_geometry/07_run_detector_mashing.py`` example).
- **``TOFBinMashingOperator``** (``parallelproj.pet_lors``): mashes (groups) every
  ``mashing_factor`` neighbouring TOF bins along the trailing TOF axis into fewer,
  wider bins (``mode="sum"`` for counts, ``mode="average"`` for multiplicative
  factors), with a genuine transpose and closed-form norm (``sqrt(G)`` / ``1/sqrt(G)``).
  Geometry-agnostic (takes ``tof_parameters`` and the leading ``non_tof_data_shape``),
  it exposes the matching ``coarse_tof_parameters`` so a projector can target the
  mashed TOF grid, and composes with ``SinogramMashingOperator`` via
  ``CompositeLinearOperator``.  For ``mode="sum"`` the mashed forward projection equals
  a direct coarse-TOF projection (erf additivity over adjacent bins).  See the
  ``01_pet_geometry/08_run_tof_bin_mashing.py`` example.
- **``parallelproj.sinogram_symmetries`` submodule**: new module for exploiting
  the cylindrical symmetry of regular-polygon PET scanners to speed up geometric
  sensitivity calculations. Provides:

  - ``compute_sinogram_plane_symmetries`` ^^ partition all axial ring pairs into
    equivalence classes under axial block-shift, midplane reflection, and endpoint-swap
    symmetries (with optional edge-ring correction)
  - ``build_plane_class_indices``, ``build_view_class_indices``,
    ``build_radial_class_indices`` ^^ per-class index arrays for the three sinogram axes
  - ``reduce_sinogram_by_symmetry_class`` / ``expand_sinogram_by_symmetry_class`` ^^
    array-API-compatible reduce/expand operations for the typical
    reduce -> compute -> expand sensitivity workflow

- **``parallelproj.data`` submodule**: new module for memory-mapped, ordered-subset
  access to sinogram data, enabling out-of-core OSEM on datasets larger than RAM.
  Provides ``SubsetArrayMmap`` (a lazily-loaded per-subset view of an on-disk array)
  and ``to_subset_mmap`` (write a sinogram to disk as subset-ordered memory maps).
  ``count_event_multiplicity`` now also lives here (see the breaking change below —
  it is no longer exported at the top level).
- **``parallelproj.unlist`` submodule**: new module for histogramming listmode PET
  data into sinograms for ``RegularPolygonPETScannerGeometry``-based scanners.
  Provides:

  - ``regular_polygon_events_to_sinogram`` ^^ histogram per-event crystal and ring
    indices into a non-TOF or TOF sinogram array; supports numpy, cupy, and torch
  - ``detection_times_to_tof_bin`` ^^ convert raw detection-time differences
    (nanoseconds) to projector-convention unsigned TOF bin indices ready for
    histogramming

Smaller additions and improvements
""""""""""""""""""""""""""""""""""

- **New sinogram / scanner ordering options**: ``SinogramZigZagOrder`` (a
  ``zig_zag_order`` argument on ``RegularPolygonPETLORDescriptor``) and
  ``RingEndpointOrdering`` (with new ``phis``, ``phi0``, ``ring_endpoint_ordering``
  and ``lor_endpoint_positions`` arguments on ``RegularPolygonPETScannerGeometry``)
  make the crystal / LOR endpoint ordering explicit and configurable.
- **``ParallelViewProjector3D`` now supports any odd span**: the projector
  accepts a :class:`~parallelproj.pet_lors.Michelogram` and uses the
  averaged-LOR z-position per plane (exact for span=1, standard approximation
  for span>1), with no loop over ring-pair multiplicities.
- **``LinearOperator.H`` property** and **``AdjointLinearOperator``** class: obtain the
  adjoint of any operator via ``A.H``.
- **``LinearOperator.adjointness_test`` and ``LinearOperator.norm`` infer ``xp`` /
  ``dev``**: when omitted, the array namespace and device are taken from the operator
  (``self.xp`` / ``self.dev``) so both can be called without arguments; backend-agnostic
  operators (e.g. ``FiniteForwardDifference``, ``CompositeLinearOperator``) still require
  ``xp`` explicitly.
- **``EqualBlockPETProjector`` ``num_chunks`` parameter**: split block-pair projections
  into chunks to reduce peak GPU memory usage.
- **``RegularPolygonPETProjector.convert_sinogram_to_listmode``** gained a ``shuffle``
  parameter to randomly permute the returned event list.
- **``VstackOperator``** now raises ``ValueError`` on inconsistent ``in_shape`` across
  stacked operators (previously silent).
- **``TOFParameters`` validates its arguments**: ``num_tofbins`` must be a positive
  integer and ``tofbin_width`` / ``sigma_tof`` / ``num_sigmas`` strictly positive and
  finite (and ``tofcenter_offset`` finite), raising a clear ``ValueError`` otherwise (a
  common cause is passing timing quantities in seconds/ps instead of the expected
  spatial mm).
- **Clearer fail-fast errors**: ``SumC1Function`` / ``SumC2Function`` raise ``ValueError``
  on an empty function sequence, and the ``start_plane_index`` / ``end_plane_index``
  accessors raise ``ValueError`` (instead of ``AttributeError``) for descriptors with
  more than one ring pair per plane (span>1 or GE), where a single ring pair per plane
  is undefined.
- **``parallelproj.__version__``** is now exposed at the top level.
- **Citation metadata**: ``import parallelproj`` is silent, but the reference to cite
  is available on demand as ``parallelproj.__citation__`` (plain text) and
  ``parallelproj.__bibtex__`` (BibTeX); a ``CITATION.cff`` file is also provided
  (GitHub "Cite this repository").

New examples and documentation
""""""""""""""""""""""""""""""

- **Example gallery substantially reorganised and expanded**, now grouped into PET
  scanner / sinogram geometry, projectors, iterative algorithms, listmode algorithms,
  transmission / joint estimation, and PyTorch integration. Highlights below.
- **New example: Michelograms and axial sinogram compression** — how the
  ``Michelogram`` maps ring pairs to sinogram planes/segments, and using
  ``SinogramAxialCompressionOperator`` to compress a span-1 sinogram to a higher odd
  span (and to/from the GE layout).
- **New example: zig-zag LOR sampling in a sinogram view** — visualises the
  ``SinogramZigZagOrder`` crystal/LOR endpoint pairing within a view.
- **New example: sinogram symmetries** — partitioning ring pairs into symmetry
  classes and the reduce -> compute -> expand workflow for geometric sensitivity
  (``parallelproj.sinogram_symmetries``).
- **New example: detector mashing** — ``SinogramMashingOperator`` groups within-side
  crystals and rings into larger virtual detectors (exact vs fast coarse projector,
  multiplicity, count-preserving up/downsampling), including mashing GE sinograms by
  composition.
- **New example: TOF-bin mashing** — ``TOFBinMashingOperator`` groups neighbouring TOF
  bins, the matching ``coarse_tof_parameters``, and composition with detector mashing.
- **New example: histogramming listmode data into sinograms** — using
  ``parallelproj.unlist`` to bin per-event crystal/ring (and TOF) indices into a
  sinogram.
- **New example: transmission reconstruction (MLTR / SPS / L-BFGS-B)** — exact Poisson
  transmission model with strictly positive scatter background, presenting MLTR
  (Nuyts et al.) and monotone SPS with optimal curvature (Erdoğan & Fessler) as one
  preconditioned gradient ascent differing only in the diagonal preconditioner, and
  L-BFGS-B on the same smooth objective with a non-negativity box constraint.
- **New example: accelerating MLTR with ordered subsets and SVRG** — OS-MLTR and a
  preconditioned SVRG variant compared against full MLTR and a converged L-BFGS-B
  reference, showing the per-epoch speed-up of subset-based transmission reconstruction.
- **New example: penalised transmission reconstruction (MAPTR)** — MLTR / OS-MLTR / SVRG
  on the penalised objective with an edge-preserving log-cosh prior, using the
  transmission "harmonic-mean" preconditioner (inverse of data plus prior curvature).
- **New example: joint activity/attenuation reconstruction (MLAA) for TOF PET** —
  interleaved penalised OS-MLEM (activity) and OS-MLTR (attenuation, with the activity
  forward projection as the transmission blank scan), NAC warm-start, support-constrained
  attenuation update, and a known-water region to fix the TOF scale ambiguity.

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
- **Top-level namespace reduced**: only ``Array``, ``empty_cuda_cache``, and
  ``to_numpy_array`` (plus the ``__version__``, ``__citation__`` and ``__bibtex__``
  dunders) are exported from ``parallelproj`` directly. Everything else
  must be imported from the relevant submodule:

  .. code-block:: python

     from parallelproj.pet_scanners import RegularPolygonPETScannerGeometry
     from parallelproj.pet_lors import RegularPolygonPETLORDescriptor
     from parallelproj.projectors import RegularPolygonPETProjector

- **``count_event_multiplicity`` moved** to the new ``parallelproj.data`` submodule
  and is **no longer exported at the top level**. Replace
  ``parallelproj.count_event_multiplicity`` / ``from parallelproj import
  count_event_multiplicity`` with ``from parallelproj.data import
  count_event_multiplicity``.

- **Runtime-detection variables removed** from the ``parallelproj`` namespace:
  ``cuda_present``, ``cupy_enabled``, ``torch_enabled``, ``num_visible_cuda_devices``,
  ``lib_parallelproj_c_fname``, ``lib_parallelproj_cuda_fname``, ``cuda_kernel_file``,
  ``is_cuda_array``. Use ``parallelproj_core.cuda_enabled`` for CUDA detection.
- **``RegularPolygonPETScannerGeometry`` signature changed**: ``ring_positions``
  and ``symmetry_axis`` are now **required** (and no longer accept ``None``); all
  remaining arguments after them (``num_lor_endpoints_per_side``, ``lor_spacing``,
  ``phis``, ``ring_endpoint_ordering``, ``phi0``, ``lor_endpoint_positions``) are
  **keyword-only**. Omitting a required argument now raises a clear ``TypeError``
  instead of failing later. Calls that already pass these by keyword
  (the documented style) are unaffected; calls that passed them positionally
  must switch to keywords.
- **``RegularPolygonPETLORDescriptor`` signature changed**: ``max_ring_difference``
  parameter replaced by a ``michelogram`` parameter that accepts a ``Michelogram``
  object. See new ``Michelogram`` class below. It now also raises a ``ValueError``
  if ``radial_trim`` is so large that no radial bins remain (``num_rad < 1``).
- **``TOFNonTOFElementwiseMultiplicationOperator`` removed**.
- **``ParallelViewProjector3D`` signature changed**: the ``span`` and
  ``max_ring_diff`` keyword arguments have been replaced by a single
  ``michelogram`` parameter (a :class:`~parallelproj.pet_lors.Michelogram`
  object).  This enables support for any odd span and makes the axial plane
  layout explicit.  Replace::

     ParallelViewProjector3D(..., span=1, max_ring_diff=d)

  with::

     from parallelproj.pet_lors import Michelogram
     ParallelViewProjector3D(..., michelogram=Michelogram(num_rings, d, span=1))

- **``TOFParameters`` defaults removed**: ``num_tofbins``, ``tofbin_width``,
  and ``sigma_tof`` are now required arguments (no defaults).  ``num_sigmas``
  defaults to ``3.0`` and ``tofcenter_offset`` defaults to ``0``.
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

1.x
---

1.10.2 (Aug 20, 2025)
^^^^^^^^^^^^^^^^^^^^^

- add compatibility for latest cupy version (>= 13.5) which require ``from_dlpack`` to
  convert from torch tensors
- fix minor issues to be compatible with ``array-api-strict~=2.0``

1.10.1 (Jan 15, 2025)
^^^^^^^^^^^^^^^^^^^^^

- add a check whether sum of tof bins along LOR is non-zero before running
  TOF sinogram back projector
- update installation instructions after conda-forge recipe was updated
- clean up RTD docs build

1.10.0 (July 29, 2024)
^^^^^^^^^^^^^^^^^^^^^^

- add support for numpy>=2.0
- add tests with numpy 2.0 on python 3.9 and 3.12
- remove tox.ini

1.9.1 (June 19, 2024)
^^^^^^^^^^^^^^^^^^^^^

- BUGFIX: add missing device in BlockPET LOR descriptor (needed for pytorch + cuda backend)

1.9 (June 18, 2024)
^^^^^^^^^^^^^^^^^^^

- add functionality to create scanners, LOR descriptors and projectors for scanners
  consisting of equal "block" modules
- **BUGFIX:** correct behavior of TOF kernel truncation which was wrong in the case that
  the tof bin width was >> tof resolution

1.8 (March 20, 2024)
^^^^^^^^^^^^^^^^^^^^

- add function to count event multiplicity
- add more examples (e.g. DePierro and LM SPDHG)
- re-organize folder structure and pyproject.toml
- force array-api-compat<1.5 (bug in 1.5.0)
- use array-api-strict instead of numpy.array_api

1.7.3 (January 26, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^

- print banner
- test also on Windows

1.7.2 (January 26, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^

- require python>=3.9
- replace ``distuils.spawn`` by ``shutil.which``

1.7.1 (January 19, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^

- BUGFIX: correct bug in the "chunking" of TOF sinogram projections in the python
  interface

1.7.0 (January 15, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^

- update of documentation
- addition of more examples
- addition of high-level classes for RegularPolygonPETScanner and LOR descriptors

1.6.2 (December 01, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^

- BUGFIX: correct use of ``conj()`` of scalar value to be array api compatible
- BUGFIX: divided by ``float()`` to be array api compatible
- add scipy dependency

1.6.1 (October 18, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^

- BUGFIX: add sigma as explicit argument in ``GaussianFilterOperator`` and convert
  correctly to numpy/cupy arrays

1.6.0 (October 16, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^

- rewrite ``LinearOperator`` base class to support python array api including devices
- add missing type hints
- add finite difference operator
- remove obsolete functions

1.5.0 (July 29, 2023)
^^^^^^^^^^^^^^^^^^^^^

- add compatibility of python wrapper to python array api (via array-api-compat)
  such that numpy, cupy, pytorch arrays can be directly projected
- no changes to the C/CUDA libs

1.4.0 (June 11, 2023)
^^^^^^^^^^^^^^^^^^^^^

- add Linear Operators

1.3.7 (April 27, 2023)
^^^^^^^^^^^^^^^^^^^^^^

- update documentation

1.3.6 (April 25, 2023)
^^^^^^^^^^^^^^^^^^^^^^

- enable readthedocs

1.3.5 (April 23, 2023)
^^^^^^^^^^^^^^^^^^^^^^

- add py.typed for mypy type checker

1.3.4 (April 21, 2023)
^^^^^^^^^^^^^^^^^^^^^^

- rename python binding back to parallelproj

1.3.3 (April 20, 2023)
^^^^^^^^^^^^^^^^^^^^^^

- import annotations from ``__future__`` to be compatible with older versions

1.3.2 (April 18, 2023)
^^^^^^^^^^^^^^^^^^^^^^

- rename test folder
- lower absolute tolerance for forward TOF tests (otherwise windows builds might fail)

1.3.1 (April 17, 2023)
^^^^^^^^^^^^^^^^^^^^^^

- add ``num_visible_devices`` definition when cuda is not present

1.3.0 (April 17, 2023)
^^^^^^^^^^^^^^^^^^^^^^

- clean up pyproject.toml
- move tests and rename imports in tests
- rename python package to parallelprojpy and adapt setup.cfg
- add first version of pyproject.toml

1.2.16 (April 16, 2023)
^^^^^^^^^^^^^^^^^^^^^^^

- improve way to detect whether visible GPUs are present in the python API
- remove AS approximation of ``erff`` in openMP lib (too large inaccuracies)
- add TOF LM tests
- add listmode wrappers
- add TOF sino fwd test

1.2.15 (April 15, 2023)
^^^^^^^^^^^^^^^^^^^^^^^

- add TOF sino projector wrappers and first test
- BUGFIX: correct start and stop of loop over planes in cuda TOF sino projector when
  direction=2
- add adjointness test (indirect test for back projection)
- add first python unit test for non-tof fwd projection
- add first python wrappers for non-tof Joseph projectors

1.2.14 (February 15, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- make target link libraries (m and OpenMP) private

1.2.13 (January 13, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^

- fix variable expansion in Config.cmake.in
- update README
- add link to arxiv preprint

1.2.12 (January 08, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^

- set CUDA_HOST_COMPILER only when using clang
- skip build of cuda lib if cuda is not present

1.2.11 (January 05, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^

- set default ``CMAKE_CUDA_HOST_COMPILER`` to ``CMAKE_CXX_COMPILER``

1.2.10 (December 30, 2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- link parallelproj_c against libm (using PUBLIC link interface)
- use better way to test whether we have to link against libm
- add adjoint back projection test
- add more generic non-tof test that tests rays in all 3 directions

1.2.9 (December 09, 2022)
^^^^^^^^^^^^^^^^^^^^^^^^^

- BUGFIX: correct calculation of ``x_pr2`` when principal direction is 0

1.2.8 (December 02, 2022)
^^^^^^^^^^^^^^^^^^^^^^^^^

- do not install test binaries
- require CXX compiler only for CUDA

1.2.6 (November 18, 2022)
^^^^^^^^^^^^^^^^^^^^^^^^^

- clean up CMake logic

1.2.5 (November 11, 2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- add conditions to nested if-else when adding cuda subdir

1.2.4 (November 10, 2022)
^^^^^^^^^^^^^^^^^^^^^^^^^

- add fatal error if cuda lib is to be built but no cuda compiler is found

1.2.3 (November 04, 2022)
^^^^^^^^^^^^^^^^^^^^^^^^^

- add skip option for cmake

1.2.2 (November 03, 2022)
^^^^^^^^^^^^^^^^^^^^^^^^^

- read version from package.json
- add conda build
