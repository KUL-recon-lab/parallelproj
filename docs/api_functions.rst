:tocdepth: 2

Functions ``parallelproj.functions``
------------------------------------

This module provides the objective-function building blocks for iterative
reconstruction: data-fidelity terms and priors together with their gradients,
(approximate) Hessians and proximal operators.  The main data-fidelity term is
:class:`.NegPoissonLogL` (the negative Poisson log-likelihood); :class:`.LogCosh`
is an edge-preserving prior; and :class:`.C2AffineObjective` composes a function
with an affine map (e.g. a projector).  The abstract base classes define the
differentiability and proximal interfaces that the algorithms rely on -- as a
user you normally start from the concrete classes above.

.. automodule:: parallelproj.functions
