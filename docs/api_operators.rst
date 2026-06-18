:tocdepth: 2

Linear operators ``parallelproj.operators``
-------------------------------------------

This module defines the :class:`.LinearOperator` base class used throughout
parallelproj.  The public interface is ``op(x)`` (forward) and
``op.adjoint(y)`` -- or the adjoint operator ``op.H`` -- while subclasses
implement the private ``_apply`` / ``_adjoint``.  Composite operators such as
:class:`.CompositeLinearOperator` let you chain a projector with image-based
models (e.g. a :class:`.GaussianFilterOperator` resolution model or an
:class:`.ElementwiseMultiplicationOperator` for attenuation) while the correct
adjoint is assembled automatically.  Operators act on **float32** array-API
arrays on CPU or GPU; the PET projectors in :doc:`api_projectors` are themselves
linear operators.

.. automodule:: parallelproj.operators
