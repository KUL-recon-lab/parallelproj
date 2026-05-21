"""Shared utilities for the parallelproj example gallery.

This module is **not** part of the parallelproj package.  It is a
helper for the Sphinx-Gallery examples only and lives exclusively under
``docs/examples/``.  It is excluded from the gallery by the
``ignore_pattern`` in ``docs/conf.py`` via the ``array_utils`` entry.
"""

from __future__ import annotations

from importlib import import_module, util as iutil
from types import ModuleType
from typing import Any

import parallelproj_core as ppc


def suggest_array_backend_and_device(
    backend: str | None = None,
    dev: str | None = None,
) -> tuple[ModuleType, Any]:
    """Select an array API-compatible module and compute device.

    When called without arguments the function probes the environment and
    returns the most capable available backend in the priority order
    ``"torch"`` > ``"cupy"`` > ``"numpy"``, paired with a CUDA device when
    one is available and enabled in ``parallelproj_core``.

    To force a specific backend or device — for example to benchmark on CPU
    or to reproduce a result without a GPU — pass explicit ``backend``
    and/or ``dev`` arguments::

        # auto-detect (default)
        xp, dev = suggest_array_backend_and_device()

        # force NumPy on CPU regardless of available hardware
        xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu")

        # force PyTorch on CPU
        xp, dev = suggest_array_backend_and_device(backend="torch", dev="cpu")

    Parameters
    ----------
    backend : {"torch", "cupy", "numpy"} or None, optional
        Name of the desired array backend.  ``None`` (default) triggers
        automatic selection.  Raises :class:`ValueError` for an
        unrecognised string and :class:`ImportError` if the requested
        backend is not installed.
    dev : str or None, optional
        Compute device string, e.g. ``"cpu"`` or ``"cuda"``.  ``None``
        (default) lets the function choose the best available device for
        the selected backend.  Note that ``"cupy"`` only supports CUDA
        devices; combining ``backend="cupy"`` with ``dev="cpu"`` raises
        :class:`ValueError`.  To select a specific CUDA device for
        cupy pass ``"cuda:N"`` (e.g. ``"cuda:1"``); plain ``"cuda"``
        selects device 0.

    Returns
    -------
    xp : module
        Array API-compatible namespace (``array_api_compat.torch``,
        ``array_api_compat.cupy``, or ``array_api_compat.numpy``).
    dev : str or cupy.cuda.Device
        Compute device compatible with the returned ``xp``.  For the
        ``"cupy"`` backend this is a :class:`cupy.cuda.Device` object;
        for ``"torch"`` and ``"numpy"`` it is a plain string
        (``"cuda"`` or ``"cpu"``).

    Raises
    ------
    ValueError
        If ``backend`` is not one of ``"torch"``, ``"cupy"``,
        ``"numpy"``, or if ``backend="cupy"`` is combined with
        ``dev="cpu"``.
    ImportError
        If the explicitly requested ``backend`` is not installed.
    """

    _valid = ("torch", "cupy", "numpy")
    if backend is not None and backend not in _valid:
        raise ValueError(f"Unknown backend {backend!r}. Choose from {_valid}.")

    # ------------------------------------------------------------------ torch
    if backend == "torch" or (backend is None and iutil.find_spec("torch") is not None):
        if iutil.find_spec("torch") is None:
            raise ImportError("Backend 'torch' was requested but is not installed.")
        xp = import_module("array_api_compat.torch")
        if dev is None:
            dev = "cuda" if xp.cuda.is_available() and ppc.cuda_enabled == 1 else "cpu"

    # ------------------------------------------------------------------ cupy
    elif backend == "cupy" or (
        backend is None
        and iutil.find_spec("cupy") is not None
        and ppc.cupy_enabled == 1
    ):
        if iutil.find_spec("cupy") is None:
            raise ImportError("Backend 'cupy' was requested but is not installed.")
        if dev == "cpu":
            raise ValueError("cupy only supports CUDA devices; dev='cpu' is not valid.")
        xp = import_module("array_api_compat.cupy")
        if dev is None:
            dev = xp.cuda.Device(0)
        elif isinstance(dev, str):
            # accept "cuda" → Device(0) or "cuda:N" → Device(N)
            device_id = int(dev.split(":")[-1]) if ":" in dev else 0
            dev = xp.cuda.Device(device_id)

    # ----------------------------------------------------------------- numpy
    else:
        xp = import_module("array_api_compat.numpy")
        if dev is None:
            dev = "cpu"

    print(f"Using array API: {xp.__name__}, device: {dev}")
    return xp, dev
