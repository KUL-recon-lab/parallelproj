"""
pytorch parallelproj projection layer
=====================================

In this example, we show how to define a custom pytorch layer that can be used
to define a feed forward neural network that includes a parallelproj forward and back 
backward projections (or any :class:`.LinearOperator`) that can be used with pytorch's
autograd engine.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%

from __future__ import annotations

import array_api_compat.torch as torch
import matplotlib.pyplot as plt
import parallelproj
from array_api_compat import device


# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu
if parallelproj.cuda_present:
    dev = "cuda"
else:
    dev = "cpu"

# %%
# Setup the forward projection layer
# ----------------------------------
#
# We subclass :class:`torch.autograd.Function` to define a custom pytorch layer
# that is compatible with pytorch's autograd engine.
# see also: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html


class LinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
    ) -> torch.Tensor:
        """forward pass of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, num_voxels_x, num_voxels_y, num_voxels_z)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, opertor.out_shape)
        """

        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size,) + operator.out_shape, dtype=x.dtype, device=device(x)
        )

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, ...] = operator(x[i, 0, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, operator.out_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.in_shape)
        """

        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros(
                (batch_size, 1) + operator.in_shape,
                dtype=grad_output.dtype,
                device=device(grad_output),
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, 0, ...] = operator.adjoint(grad_output[i, ...].detach())

            return x, None


# %%
# Setup the back projection layer
# -------------------------------
#
# We subclass :class:`torch.autograd.Function` to define a custom pytorch layer
# that is compatible with pytorch's autograd engine.
# see also: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html


class AdjointLinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing the adjoint of a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
    ) -> torch.Tensor:
        """forward pass of the adjoint of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, operator.out_shape)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, opertor.in_shape)
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size, 1) + operator.in_shape, dtype=x.dtype, device=device(x)
        )

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, 0, ...] = operator.adjoint(x[i, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, 1, operator.in_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.out_shape)
        """

        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros(
                (batch_size,) + operator.out_shape,
                dtype=grad_output.dtype,
                device=device(grad_output),
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, ...] = operator(grad_output[i, 0, ...].detach())

            return x, None


# %%
# Setup a minimal non-TOF PET projector
# -------------------------------------
#
# We setup a minimal non-TOF PET projector of small scanner with
# three rings.

num_rings = 3
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    torch,
    dev,
    radius=35.0,
    num_sides=12,
    num_lor_endpoints_per_side=6,
    lor_spacing=3.0,
    ring_positions=torch.linspace(-4, 4, num_rings, device=dev),
    symmetry_axis=1,
)

# setup the LOR descriptor that defines the sinogram
lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=1,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=(20, 5, 20), voxel_size=(2.0, 2.0, 2.0)
)

# %%
# Define a mini batch of input and output tensors
# -----------------------------------------------

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
# Define the forward and backward projection layers
# -------------------------------------------------

fwd_op_layer = LinearSingleChannelOperator.apply
adjoint_op_layer = AdjointLinearSingleChannelOperator.apply

f1 = fwd_op_layer(x, proj)
print("forward projection (Ax) .:", f1.shape, type(f1), device(f1))

b1 = adjoint_op_layer(y, proj)
print("back projection (A^T y) .:", b1.shape, type(b1), device(b1))

fb1 = adjoint_op_layer(fwd_op_layer(x, proj), proj)
print("back + forward projection (A^TAx) .:", fb1.shape, type(fb1), device(fb1))


# %%
# Define a dummy loss function and trigger the backpropagation
# ------------------------------------------------------------

# define a dummy loss function
dummy_loss = (fb1**2).sum()
# trigger the backpropagation
dummy_loss.backward()

print(f" backpropagted gradient {x.grad}")


# %%
# Check whether the gradients are calculated correctly
# ----------------------------------------------------
#
# We use pytorch's gradcheck function to check whether the implementation
# of the backward pass, needed to calculate the gradients, is correct.
#
# This test can be slow which is why we only execute it on the gpu.
# Note that parallelproj's projectors use single precision precision
# which is why we have to use a larger atol and rtol.

if dev == "cpu":
    print("skipping (slow) gradient checks on cpu")
else:
    print("Running forward projection layer gradient test")
    grad_test_fwd = torch.autograd.gradcheck(
        fwd_op_layer, (x, proj), eps=1e-1, atol=1e-3, rtol=1e-3
    )

    print("Running adjoint projection layer gradient test")
    grad_test_fwd = torch.autograd.gradcheck(
        adjoint_op_layer, (y, proj), eps=1e-1, atol=1e-3, rtol=1e-3
    )

# %%
# Visualize the scanner geometry and image FOV
# --------------------------------------------

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
proj.show_geometry(ax)
fig.tight_layout()
fig.show()
