Custom parallelproj pytorch layer examples
------------------------------------------

How to wrap a parallelproj operator as a differentiable PyTorch layer, with an
autograd-compatible forward and adjoint, so that projectors can be embedded in
deep-learning reconstruction pipelines (e.g. unrolled / model-based networks).
