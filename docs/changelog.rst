Changelog
=========

2.x
---

2.1.0 (unreleased)
^^^^^^^^^^^^^^^^^^

New features
~~~~~~~~~~~~

- **Tabulated TOF kernels and per-LOR TOF parameters.** (`#166
  <https://github.com/KUL-recon-lab/parallelproj/issues/166>`_)

  - new :class:`.TOFKernel`: a TOF kernel given as a table of its cumulative
    distribution, built with ``TOFKernel.gaussian``, ``TOFKernel.from_pdf`` or
    ``TOFKernel.from_cdf``. Non-Gaussian (measured, asymmetric, with tails)
    kernels are passed as ``TOFParameters(..., kernel=k, kernel_scale=s)``
    instead of ``sigma_tof``;
  - ``sigma_tof`` / ``kernel_scale`` and ``tofcenter_offset`` can be given per
    LOR (sinogram projectors: one value per LOR of the LOR descriptor;
    listmode: per event). Scalars cost nothing extra;
  - the sinogram and listmode projectors **always use the tabulated TOF
    projectors** of parallelproj-core. A Gaussian ``sigma_tof`` becomes a
    2001-point table of the normal CDF; projections differ from the closed-form
    Gaussian by < 1e-6 (relative to the maximum). The listmode forward
    projection uses 256 CUDA threads per block (faster with the table, #165).

Changes
~~~~~~~

- **GPU projections run on the current CUDA stream (asynchronous).** (`#176
  <https://github.com/KUL-recon-lab/parallelproj/issues/176>`_) All projectors
  pass the current stream of the GPU of their arrays to parallelproj-core
  (PyTorch: ``torch.cuda.current_stream(device)``, CuPy:
  ``cupy.cuda.get_current_stream()``). A GPU projection now behaves like any
  other PyTorch / CuPy operation: it is queued on that stream and returns
  without waiting for the GPU, it is correctly ordered with the other work on
  that stream (also inside ``with torch.cuda.stream(s):``, where the previous
  projections on the legacy default stream did not wait for the work queued on
  ``s``), and projections on several GPUs run at the same time even when
  called one after another from one Python thread. Results are unchanged, and
  everything that reads a result (``.cpu()``, ``.item()``, ``print``,
  ``to_numpy_array``, ...) waits for it automatically; only hand-written timing
  code has to synchronize (``torch.cuda.synchronize()``) before reading the
  clock. CPU arrays are not affected.
- **TOFParameters**: ``sigma_tof`` is optional (but required unless ``kernel``
  is given), and the class has the new fields ``kernel`` and ``kernel_scale``
  and properties ``tof_kernel``, ``scale`` and ``is_lor_dependent``. Existing
  code that constructs ``TOFParameters(num_tofbins, tofbin_width, sigma_tof,
  ...)`` works unchanged. The TOF unlisting helpers and ``show_tof_bins``
  support only a scalar ``tofcenter_offset`` (``NotImplementedError``
  otherwise).

- **Requires parallelproj-core >= 3.0.** (`#164
  <https://github.com/KUL-recon-lab/parallelproj/issues/164>`_) Numerical
  results of the TOF projectors change slightly, through the libparallelproj
  3.0 TOF kernel:

  - the Gaussian TOF kernel is truncated at ``num_sigmas`` sigma **inside** a
    TOF bin (only the part of a bin within the window counts) and normalised to
    unit total weight. Bins entirely outside the window get exactly zero weight,
    and the TOF projections summed over the TOF bins equal the non-TOF
    projection. Previously the full Gaussian integrals of the bins overlapping
    the window were used, and listmode weights were normalised differently, so
    single bin weights change by up to ~1e-3 (of a total of 1) near the window
    edges;
  - a listmode event now gets exactly the value of its TOF sinogram bin.

  See the libparallelproj 3.0 changelog for all changes of the backend
  (CUDA streams, tabulated TOF kernels, event sorting, ...).

- **``NegPoissonLogL`` returns the Poisson deviance, summed in float64.** The
  function value is now :math:`\sum_i \bar{y}_i - y_i - y_i \log(\bar{y}_i /
  y_i)` (terms with :math:`y_i = 0`: :math:`\bar{y}_i`), i.e. the textbook
  form :math:`\sum_i \bar{y}_i - y_i \log \bar{y}_i` plus the constant
  :math:`\sum_i y_i \log y_i - y_i`. Gradient, Hessian, proximal operators and
  minimisers are unchanged; **function values differ from 2.0 by that
  constant.** Every term is :math:`\geq 0` and :math:`0` at the fit, and the
  sum is accumulated in float64 (where the backend supports it), so the value
  is now usable for convergence monitoring and line searches with float32 data
  and :math:`10^7`--:math:`10^9` bins. Previously, the float32 sum lost all
  significant digits of the change between late iterations. (`#167
  <https://github.com/KUL-recon-lab/parallelproj/issues/167>`_)
- **``NegPoissonLogLListmode``: consistent constant and float64 sums.** The
  value now subtracts the number of events :math:`N_\text{ev}`, so it equals the
  sinogram deviance of ``NegPoissonLogL`` minus :math:`\sum_i y_i \log y_i`
  (:math:`0` if no sinogram bin contains more than one event), and all sums are
  accumulated in float64. Gradient and Hessian are unchanged; function values
  differ from 2.0 by :math:`-N_\text{ev}`.

Documentation
~~~~~~~~~~~~~

- **Shorter documentation build.** (`#169
  <https://github.com/KUL-recon-lab/parallelproj/issues/169>`_) The six slowest
  iterative examples run fewer iterations on the CPU (as on Read the Docs) and
  print a note that the shown results are less converged. GPU builds keep the
  full numbers; ``PARALLELPROJ_EXAMPLES_FULL_ITERATIONS=1`` runs them on the
  CPU as well.
- **New example: projection layers on multiple GPUs with CUDA streams.**
  (`#177 <https://github.com/KUL-recon-lab/parallelproj/issues/177>`_) One
  projector per GPU, a layer that splits a mini batch across all GPUs and
  gathers the results (with gradients), and how to time GPU code correctly.
- **Example gallery look (part 1).** (`#178
  <https://github.com/KUL-recon-lab/parallelproj/issues/178>`_) Shared figure
  style for all examples (colour-vision-deficiency-safe colour cycle, font
  sizes, high-resolution images), ``show_vol_cuts`` draws a static figure
  (three cuts and a colorbar, no widgets) with non-interactive matplotlib
  backends (documentation, Jupyter inline), shorter gallery section titles, a
  "New here?" start row, and the first example rewritten with the new
  example template (learning goals, numbered sections, summary and next steps).

2.0.2 (Jul 29, 2026)
^^^^^^^^^^^^^^^^^^^^

Bug fixes
~~~~~~~~~

- **``GaussianFilterOperator``: fully fix tensor-valued ``sigma``.** The 2.0.1
  fix converted the input array to NumPy, but the actual trigger is a *tensor*
  ``sigma``: ``scipy.ndimage`` builds the Gaussian kernel in ``sigma``'s array
  namespace, so a torch / CuPy tensor ``sigma`` yields a tensor kernel that
  ``gaussian_filter1d`` reverses with a negative-step slice (``weights[::-1]``),
  which PyTorch does not support — failing even for a plain NumPy input array.
  ``GaussianFilterOperator`` now coerces ``sigma`` (and other array-valued
  kwargs) to native Python values (the input NumPy-view safeguard from 2.0.1 is
  retained). This fully fixes downstream use via e.g. DeepInv.

2.0.1 (Jul 24, 2026)
^^^^^^^^^^^^^^^^^^^^

Bug fixes
~~~~~~~~~

- **``GaussianFilterOperator`` now works with PyTorch (and array-api-strict) CPU
  arrays under modern scipy.** The CPU path previously passed the input array
  straight to ``scipy.ndimage.gaussian_filter``; with array-API-aware scipy
  (``SCIPY_ARRAY_API`` / scipy ≥ 1.16) this made scipy compute natively in the
  input's namespace, where ``gaussian_filter1d`` reverses the kernel with a
  negative-step slice (``weights[::-1]``) that PyTorch does not support. The
  operator now filters on a NumPy view (zero-copy on CPU) and converts back, so
  it is robust regardless of the scipy version or ``SCIPY_ARRAY_API``. This
  fixes downstream use via e.g. DeepInv.

2.0.0 (Jul 17, 2026)
^^^^^^^^^^^^^^^^^^^^

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
- **Selectable segment ordering** for ``Michelogram``: new ``SegmentOrder`` enum and
  ``segment_order=`` argument (also on ``Michelogram.ge`` and, as
  ``target_segment_order=``, on ``SinogramAxialCompressionOperator``).
  ``SegmentOrder.POSITIVE_FIRST`` (default) keeps the existing
  ``0, +1, -1, +2, -2, …`` layout; ``SegmentOrder.NEGATIVE_FIRST`` gives
  ``0, -1, +1, -2, +2, …`` (negative segment before the positive one of each
  ``±k`` pair).  This is a pure permutation of the sinogram planes -- segment
  numbering, plane counts and multiplicities are unchanged -- and works for both
  the ``STANDARD`` and ``GE`` layouts.
- **``SinogramAxialCompressionOperator``** (``parallelproj.pet_lors``): ``LinearOperator``
  that axially compresses a span-1 sinogram to a higher odd span (``mode="sum"`` or
  ``mode="average"``).  A ``target_layout=MichelogramLayout.GE`` option compresses a
  span-1 sinogram to the **GE layout**; its ``mode="average"`` adjoint distributes a
  GE sinogram back onto the span-1 grid while preserving counts (e.g. to convert GE
  data to span-1 before detector mashing).
- **Segment selection**: new ``Michelogram.select_segments([...])`` returns a
  restricted ``Michelogram`` keeping only the requested signed segments (the
  order-preserving subsequence, with plane indices renumbered but segment labels
  unchanged), exposing ``parent_plane_indices`` for gather/scatter against the
  full sinogram.  The companion ``SinogramSegmentSelectionOperator``
  (``parallelproj.pet_lors``) is a ``LinearOperator`` built from a full LOR
  descriptor and a list of ``segments``: it provides a
  ``restricted_lor_descriptor`` (for building the matching projector), gathers
  the selected planes out of a full sinogram (forward) and scatters them back
  into a zero-filled full sinogram (adjoint).  It is a pure plane selection, so
  it supports non-TOF and TOF sinograms and has operator 2-norm 1.
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

  - ``compute_sinogram_plane_symmetries`` — partition all axial ring pairs into
    equivalence classes under axial block-shift, midplane reflection, and endpoint-swap
    symmetries (with optional edge-ring correction)
  - ``build_plane_class_indices``, ``build_view_class_indices``,
    ``build_radial_class_indices`` — per-class index arrays for the three sinogram axes
  - ``reduce_sinogram_by_symmetry_class`` / ``expand_sinogram_by_symmetry_class`` —
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

  - ``regular_polygon_events_to_sinogram`` — histogram per-event crystal and ring
    indices into a non-TOF or TOF sinogram array; supports numpy, cupy, and torch
  - ``detection_times_to_tof_bin`` — convert raw detection-time differences
    (nanoseconds) to projector-convention unsigned TOF bin indices ready for
    histogramming

Smaller additions and improvements
""""""""""""""""""""""""""""""""""

- **New sinogram / scanner ordering options**: ``SinogramZigZagOrder`` (a
  ``zig_zag_order`` argument on ``RegularPolygonPETLORDescriptor``) and
  ``RingEndpointOrdering`` (with new ``phis``, ``phi0``, ``ring_endpoint_ordering``
  and ``lor_endpoint_positions`` arguments on ``RegularPolygonPETScannerGeometry``)
  make the crystal / LOR endpoint ordering explicit and configurable.
- **Selectable LOR start/end (TOF-bin) convention**: new ``LOREndpointOrder`` enum
  and ``lor_endpoint_order`` argument on ``RegularPolygonPETLORDescriptor``.
  ``START_END`` (default) keeps the current behaviour; ``END_START`` swaps
  ``xstart``/``xend`` for every LOR.  Non-TOF projections are unchanged; for TOF
  this reverses the TOF-bin axis, letting users match a given vendor's start/end
  (and hence first-TOF-bin) convention.  See the "PET TOF sinogram projector"
  example.
- **Ready-made demo scanners**: ``get_lor_descriptor_G1`` /
  ``get_lor_descriptor_G2`` (in ``parallelproj.pet_lors``) return fully
  configured ``RegularPolygonPETLORDescriptor`` objects for two built-in
  cylindrical TOF scanner geometries, and the matching ``get_tof_parameters_G1``
  / ``get_tof_parameters_G2`` (in ``parallelproj.tof``) return their
  ``TOFParameters``.  Users can start projecting without assembling the scanner,
  Michelogram, sinogram conventions and TOF model themselves.  All geometry /
  convention / timing parameters are exposed as keyword overrides (e.g.
  ``num_units``, ``radial_trim``, ``max_ring_difference``, ``num_tofbins``,
  ``sigma_tof``).
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
  ``Michelogram`` maps ring pairs to sinogram planes/segments, using
  ``SinogramAxialCompressionOperator`` to compress a span-1 sinogram to a higher odd
  span (and to/from the GE layout), a ``SegmentOrder`` comparison
  (positive-first vs negative-first) for both the STANDARD and GE layouts, and a
  ``SinogramSegmentSelectionOperator`` section that projects/back-projects only a
  chosen subset of segments.
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
- **Example helpers now ship inside the package** as the private
  ``parallelproj._examples_utils`` module (interactive slice viewer
  ``show_vol_cuts``, ``suggest_array_backend_and_device``, analytic phantoms and
  demo priors). Examples and downloaded scripts/notebooks now import them
  straight from ``parallelproj`` — no ``PYTHONPATH`` setup and no separate
  ``example_utils.py`` download are required. The module is private and
  examples-only (not part of the public API) and may change without notice.

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
- **Sinogram transaxial convention changed (breaking)**: view 0's *central* radial
  bin now connects detector 0 and detector ``N/2`` (diametrically opposing),
  matching STIR and the PET vendors.  Previously view 0 was anchored a
  quarter-ring away, so **the bin <-> detector-pair mapping of existing
  regular-polygon sinograms differs** from v1.x / earlier v2.0 pre-releases.  Two
  new descriptor kwargs make the convention fully configurable:
  ``view_direction`` (:class:`~parallelproj.pet_lors.ViewDirection`) and
  ``radial_direction`` (:class:`~parallelproj.pet_lors.RadialDirection`), which
  flip the view / radial index directions.  Together with the scanner's
  ``ring_endpoint_ordering`` (crystal numbering), ``phi0`` (module-0 azimuth;
  ``phi0=0`` puts module 0 at the top -- ``-y`` for ``symmetry_axis=2`` -- see
  the coordinate-convention change below) and ``zig_zag_order`` they reproduce
  any vendor's ``(view, radial) <-> detector-pair`` convention.  See the
  ``01_pet_geometry/02_run_regular_polygon_pet_sino.py`` example.
- **World coordinate / detector-orientation convention changed (breaking)**:
  for ``symmetry_axis=2`` the transaxial detector layout is now defined for the
  standard viewpoint -- standing in front of the scanner, looking along the bore from
  ``-z`` toward ``+z``, with ``x0`` running left->right and ``x1`` top->bottom
  (``+x1`` points down), aligned with the DICOM/LPS axes of a head-first-supine
  patient.  Concretely, relative to earlier v2.0 pre-releases:

  - module/side 0 (``phi0=0``) now sits at the **top** (``-x1``) instead of
    ``+x1``;
  - ``phi0`` is now a **right-hand rotation about the symmetry axis** -- a
    positive ``phi0`` moves module 0 toward ``+x0`` (previously ``-x0``);
  - ``RingEndpointOrdering.CLOCKWISE`` / ``COUNTERCLOCKWISE`` are defined as
    clockwise / counterclockwise **as seen in that ``-z`` view** (the physical
    numbering for each enum value therefore swapped).

  These follow from reflecting the transaxial ``ax0`` coordinate of every
  ``RegularPolygonPETScannerGeometry`` endpoint (for ``symmetry_axis=2`` this is
  the ``x1``/``y`` axis), so **absolute endpoint coordinates and the module-0 /
  crystal-0 anchor of existing scanners differ**.  Segment numbering, plane
  counts and sinogram shapes are unaffected.  The 3-D example plots now render
  this viewpoint via ``ax.view_init(elev=..., azim=..., roll=180,
  vertical_axis="y")``.
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
