"""
PyTorch projection layer
========================

:bdg-primary:`pytorch` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

Networks for PET image reconstruction or processing often contain a forward
or back projection.  To train such a network with PyTorch, the projection has
to be part of PyTorch's automatic differentiation (autograd).  In this example
we wrap a parallelproj projector (or any :class:`.LinearOperator`) in two
custom autograd functions, a forward and a back projection layer, that act on
mini batches of single-channel images and can be used in any network.

**You will learn**

- how to subclass :class:`torch.autograd.Function` to turn a linear operator
  into a PyTorch layer,
- why the backward pass of a linear operator is its adjoint (and vice versa),
- how to apply the layers to a mini batch and back propagate a loss through
  them,
- how to check the gradients with :func:`torch.autograd.gradcheck`.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_01_run_pet_non_tof_sinogram_projector.py`
(sinogram projector, forward and back projection).
"""

# %%
# 0. Setup
# --------
#
# This example works with PyTorch tensors directly (the layers are PyTorch
# autograd functions), so instead of ``suggest_array_backend_and_device`` we
# use the array API compatible PyTorch namespace as array backend ``xp`` and
# choose the device ourselves: a CUDA GPU if PyTorch sees one and
# parallelproj-core is a CUDA build, otherwise the CPU.

from __future__ import annotations

import array_api_compat.torch as xp
import matplotlib.pyplot as plt
import parallelproj_core as ppc
import torch

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors

if torch.cuda.is_available() and ppc.cuda_enabled == 1:
    dev = "cuda"
else:
    dev = "cpu"

# random test images and sinograms below
torch.manual_seed(0)

# %%
# 1. The forward projection layer
# --------------------------------
#
# We subclass :class:`torch.autograd.Function` (see the PyTorch tutorial on
# `defining new autograd functions
# <https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html>`_)
# and implement two static methods:
#
# - ``forward`` applies the linear operator :math:`A` to every sample of the
#   mini batch (the first and only channel),
# - ``backward`` receives the gradient of the loss with respect to the output,
#   :math:`\partial L / \partial y`, and returns the gradient with respect to
#   the input.  For :math:`y = A x` this is
#   :math:`\partial L / \partial x = A^T \, \partial L / \partial y`: the
#   **adjoint** of the operator (the back projection for a projector).
#
# Images have the shape ``(batch_size, 1, *operator.in_shape)``, the outputs
# (sinograms) ``(batch_size, *operator.out_shape)``.


class LinearSingleChannelOperator(torch.autograd.Function):
    """linear operator acting on a mini batch of single-channel images"""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.operators.LinearOperator
    ) -> torch.Tensor:
        """forward pass: apply the linear operator

        Parameters
        ----------
        ctx : context object
            used to store information for the backward pass
        x : torch.Tensor
            mini batch of images, shape (batch_size, 1, *operator.in_shape)
        operator : parallelproj.operators.LinearOperator
            linear operator that acts on a single image

        Returns
        -------
        torch.Tensor
            mini batch of outputs, shape (batch_size, *operator.out_shape)
        """

        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size,) + operator.out_shape, dtype=x.dtype, device=x.device
        )

        # apply the operator to the first channel of every sample
        for i in range(batch_size):
            y[i, ...] = operator(x[i, 0, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backward pass: apply the adjoint of the linear operator

        Parameters
        ----------
        ctx : context object
            used to obtain information from the forward pass
        grad_output : torch.Tensor
            gradient w.r.t. the output, shape (batch_size, *operator.out_shape)

        Returns
        -------
        torch.Tensor, None
            gradient w.r.t. the images, shape (batch_size, 1, *operator.in_shape),
            and None for the operator
        """

        # forward takes two input arguments (x, operator), so backward has to
        # return two gradients (the latter is None)
        if grad_output is None:
            return None, None

        operator = ctx.operator

        batch_size = grad_output.shape[0]
        x = torch.zeros(
            (batch_size, 1) + operator.in_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        # apply the adjoint to every sample
        for i in range(batch_size):
            x[i, 0, ...] = operator.adjoint(grad_output[i, ...].detach())

        return x, None


# %%
# 2. The back projection layer
# ----------------------------
#
# The back projection layer is the mirror image of the forward projection
# layer: its ``forward`` applies the adjoint :math:`A^T`, and its ``backward``
# applies the operator :math:`A` itself, since :math:`(A^T)^T = A`.


class AdjointLinearSingleChannelOperator(torch.autograd.Function):
    """adjoint of a linear operator, returning mini batches of single-channel
    images"""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.operators.LinearOperator
    ) -> torch.Tensor:
        """forward pass: apply the adjoint of the linear operator

        Parameters
        ----------
        ctx : context object
            used to store information for the backward pass
        x : torch.Tensor
            mini batch of inputs (sinograms), shape
            (batch_size, *operator.out_shape)
        operator : parallelproj.operators.LinearOperator
            linear operator that acts on a single image

        Returns
        -------
        torch.Tensor
            mini batch of images, shape (batch_size, 1, *operator.in_shape)
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size, 1) + operator.in_shape, dtype=x.dtype, device=x.device
        )

        # apply the adjoint to every sample, the result is the first channel
        for i in range(batch_size):
            y[i, 0, ...] = operator.adjoint(x[i, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backward pass: apply the linear operator

        Parameters
        ----------
        ctx : context object
            used to obtain information from the forward pass
        grad_output : torch.Tensor
            gradient w.r.t. the output images, shape
            (batch_size, 1, *operator.in_shape)

        Returns
        -------
        torch.Tensor, None
            gradient w.r.t. the inputs (sinograms), shape
            (batch_size, *operator.out_shape), and None for the operator
        """

        # forward takes two input arguments (x, operator), so backward has to
        # return two gradients (the latter is None)
        if grad_output is None:
            return None, None

        operator = ctx.operator

        batch_size = grad_output.shape[0]
        x = torch.zeros(
            (batch_size,) + operator.out_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        # apply the operator to the first channel of every sample
        for i in range(batch_size):
            x[i, ...] = operator(grad_output[i, 0, ...].detach())

        return x, None


# %%
# 3. A minimal non-TOF PET projector
# ----------------------------------
#
# Both layers work with any :class:`.LinearOperator`.  Here we use a non-TOF
# sinogram projector of a small scanner with 12 sides and 3 rings, and an image
# of 20 x 20 x 5 voxels of 2 mm.  All arrays of the projector are on ``dev``.

num_rings = 3
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=35.0,
    num_sides=12,
    num_lor_endpoints_per_side=6,
    lor_spacing=3.0,
    ring_positions=xp.linspace(-4, 4, num_rings, device=dev),
    symmetry_axis=2,
)

# the LOR descriptor defines the sinogram
lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=1, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=(20, 20, 5), voxel_size=(2.0, 2.0, 2.0)
)

# %%
# The scanner geometry and the image field of view (FOV):

fig = plt.figure(figsize=(6, 6), layout="constrained")
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")
proj.show_geometry(ax)
# the scanner is only 8 mm long along x2: two ticks avoid overlapping labels
ax.set_zticks([-4.0, 4.0])
ax.set_title("Scanner and image field of view")
fig.show()

# %%
# The three rings (one colour per ring, at ``x2`` = -4, 0 and 4 mm) each have
# 12 sides with 6 LOR endpoints.  The image FOV (the box, 40 x 40 x 10 mm)
# lies in the centre of the bore.

# %%
# 4. A mini batch of images and sinograms
# ---------------------------------------
#
# A mini batch of 2 random images (one channel) and 2 random sinograms.
# ``requires_grad=True`` tells PyTorch to track the operations on them, so
# that gradients with respect to them can be computed.

batch_size = 2

x = torch.rand(
    (batch_size, 1) + proj.in_shape,
    device=dev,
    dtype=torch.float32,
    requires_grad=True,
)

y = torch.rand(
    (batch_size,) + proj.out_shape,
    device=dev,
    dtype=torch.float32,
    requires_grad=True,
)

# %%
# 5. Applying the layers
# ----------------------
#
# The layers are used through the ``apply`` method of the autograd functions.
# They map images to sinograms (:math:`A x`), sinograms to images
# (:math:`A^T y`), and can be chained like any other layer (:math:`A^T A x`).

fwd_op_layer = LinearSingleChannelOperator.apply
adjoint_op_layer = AdjointLinearSingleChannelOperator.apply

f1 = fwd_op_layer(x, proj)
print("forward projection (Ax) .:", f1.shape, type(f1), f1.device)

b1 = adjoint_op_layer(y, proj)
print("back projection (A^T y) .:", b1.shape, type(b1), b1.device)

fb1 = adjoint_op_layer(fwd_op_layer(x, proj), proj)
print("back + forward projection (A^TAx) .:", fb1.shape, type(fb1), fb1.device)

# %%
# The forward projections are sinograms of 51 radial bins, 36 views and 7
# planes (3 direct planes and 4 with ring difference 1); the back projections
# are images with one channel, like the input images.

# %%
# 6. Back propagation through the layers
# --------------------------------------
#
# We define a dummy loss :math:`L = \sum_i \|A^T A x_i\|^2` and let PyTorch
# compute its gradient with respect to the images.  The autograd engine calls
# the ``backward`` methods of both layers.  Analytically, the gradient is
# :math:`2 A^T A A^T A x_i` (since :math:`A^T A` is symmetric), which we use to
# check the result.

dummy_loss = (fb1**2).sum()
# trigger the backpropagation
dummy_loss.backward()

print(f"backpropagated gradient: {tuple(x.grad.shape)}")
with torch.no_grad():
    expected_grad = 2 * adjoint_op_layer(
        fwd_op_layer(adjoint_op_layer(fwd_op_layer(x, proj), proj), proj), proj
    )
rel_err = float((x.grad - expected_grad).abs().max() / expected_grad.abs().max())
print(f"max. rel. difference to 2 A^T A A^T A x: {rel_err:.1e}")

# %%
# The back propagated gradient has the shape of the mini batch of images and
# agrees with the analytic gradient (up to rounding, since parallelproj
# computes in single precision).

# %%
# 7. Checking the gradients numerically
# -------------------------------------
#
# :func:`torch.autograd.gradcheck` compares the gradients computed by the
# ``backward`` methods with finite-difference approximations.  This test is
# slow, which is why we only run it on a GPU.  Since parallelproj's projectors
# compute in single precision, we have to use larger tolerances (``atol`` and
# ``rtol``) than the default ones.

if dev == "cpu":
    print("skipping (slow) gradient checks on cpu")
else:
    print("Running forward projection layer gradient test")
    grad_test_fwd = torch.autograd.gradcheck(
        fwd_op_layer, (x, proj), eps=1e-1, atol=1e-3, rtol=1e-3
    )
    print(f"forward projection layer gradient test passed: {grad_test_fwd}")

    print("Running adjoint projection layer gradient test")
    grad_test_adjoint = torch.autograd.gradcheck(
        adjoint_op_layer, (y, proj), eps=1e-1, atol=1e-3, rtol=1e-3
    )
    print(f"adjoint projection layer gradient test passed: {grad_test_adjoint}")

# %%
# Summary
# -------
#
# - A subclass of :class:`torch.autograd.Function` turns any
#   :class:`.LinearOperator` into a PyTorch layer: ``forward`` applies the
#   operator to every sample of the mini batch, ``backward`` applies its
#   adjoint.
# - The back projection layer is the same with operator and adjoint swapped.
# - The layers are used via ``.apply``, can be chained, and gradients flow
#   through them with ``loss.backward()``.
# - :func:`torch.autograd.gradcheck` verifies the implementation of the
#   backward passes (with tolerances suited for single precision).
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_06_torch_02_run_multi_gpu_projection_layer.py`
#    -- the same projection layers for mini batches split across several GPUs.

# sphinx_gallery_thumbnail_number = 1
