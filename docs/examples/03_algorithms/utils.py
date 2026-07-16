"""DEPRECATED shim -- kept only as a tombstone.

The analytic-phantom / prior helpers used by the algorithm examples now live in
:mod:`parallelproj._examples_utils` (a private, examples-only module shipped
inside the ``parallelproj`` package).

This file should be removed:

    git rm docs/examples/03_algorithms/utils.py

It re-exports the public names for backward compatibility until then.
"""

from parallelproj._examples_utils import (  # noqa: F401
    RadonObject,
    RadonObjectSequence,
    RadonDisk,
    RadonSquare,
    neighbor_offsets,
    neighbor_difference_and_sum,
    neighbor_product,
    SmoothFunction,
    SmoothFunctionWithDiagonalHessian,
    RDP,
    show_vol_cuts,
    show_3d_cuts,
)
