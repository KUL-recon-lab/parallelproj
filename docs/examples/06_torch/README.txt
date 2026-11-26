PyTorch layers and multiple GPUs
--------------------------------

How to wrap a parallelproj operator as a differentiable PyTorch layer, with an
autograd-compatible forward and adjoint, so that projectors can be embedded in
deep-learning reconstruction pipelines (e.g. unrolled / model-based networks),
and how to project the samples of a mini batch on several GPUs at the same
time with CUDA streams.
