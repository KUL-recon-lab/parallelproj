Iterative algorithm examples
----------------------------

A tour of reconstruction algorithms for the (regularised) Poisson problem,
all built on the data-fidelity and prior objects in ``parallelproj.functions``.
Start with the MLEM / OSEM / SVRG convergence comparison, then explore the
stochastic-gradient variants, PDHG / SPDHG with edge-preserving priors,
filtered back projection, De Pierro's MAP-EM, the effect of TOF on variance,
and out-of-core (memory-mapped) OSEM.
