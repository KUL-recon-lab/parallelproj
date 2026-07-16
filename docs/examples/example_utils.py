"""DEPRECATED shim -- kept only as a tombstone.

The example helpers now live in :mod:`parallelproj._examples_utils` (a private,
examples-only module shipped inside the ``parallelproj`` package), so the
gallery examples run with nothing but ``parallelproj`` installed -- no
``PYTHONPATH`` and no separate download.

This file is intentionally empty of real code and should be removed:

    git rm docs/examples/example_utils.py

It re-exports the public helper names for backward compatibility until then.
"""

from parallelproj._examples_utils import (  # noqa: F401
    suggest_array_backend_and_device,
    elliptic_cylinder_phantom,
    show_vol_cuts,
    show_3d_cuts,
    poisson_transmission_terms,
)
