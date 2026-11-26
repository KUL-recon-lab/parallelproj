"""The projectors run on the current CUDA stream of their arrays (#176).

* every call of a ``parallelproj_core`` projector passes ``stream`` =
  :func:`parallelproj._backend._current_cuda_stream` of the projector's arrays
  (``None`` for CPU arrays, the current stream handle for torch / CuPy GPU arrays);
* GPU only: projections inside ``with torch.cuda.stream(s):`` are ordered with
  the other work on ``s`` and give the same result as on the default stream;
* GPU only, >= 2 GPUs: projections on two GPUs give the same results.
"""

from __future__ import annotations

from types import ModuleType

import array_api_compat
import pytest

import parallelproj.pet_lors as ppl
import parallelproj.pet_scanners as pps
import parallelproj.projectors as ppp
import parallelproj.tof as ppt
from parallelproj._backend import _current_cuda_stream

from .config import pytestmark, torch_available, num_cuda_devices

import parallelproj_core

TOF_PARAMS = ppt.TOFParameters(num_tofbins=11, tofbin_width=3.0, sigma_tof=4.0)


class _SpyCore:
    """Stand-in for ``parallelproj_core`` that records the ``stream`` of every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __getattr__(self, name: str):
        attr = getattr(parallelproj_core, name)
        if not name.startswith("joseph3d"):
            return attr

        def wrapper(*args, **kwargs):
            self.calls.append((name, kwargs.get("stream", "<missing>")))
            return attr(*args, **kwargs)

        return wrapper


def _projectors(xp: ModuleType, dev) -> dict[str, ppp.LinearOperator]:
    """One small projector of every kind (non-TOF and TOF)."""
    projs: dict[str, ppp.LinearOperator] = {}

    projs["parallel2d"] = ppp.ParallelViewProjector2D(
        (4, 4),
        xp.asarray([-1.0, 0.0, 1.0], dtype=xp.float32, device=dev),
        xp.asarray([0.0, 1.0], dtype=xp.float32, device=dev),
        6.0,
        (-1.5, -1.5),
        (1.0, 1.0),
    )
    projs["parallel3d"] = ppp.ParallelViewProjector3D(
        (4, 4, 3),
        xp.asarray([-1.0, 0.0, 1.0], dtype=xp.float32, device=dev),
        xp.asarray([0.0, 1.0], dtype=xp.float32, device=dev),
        6.0,
        (-1.5, -1.5, -1.0),
        (1.0, 1.0, 1.0),
        xp.asarray([-1.0, 0.0, 1.0], dtype=xp.float32, device=dev),
        michelogram=ppl.Michelogram(num_rings=3, max_ring_difference=1, span=1),
    )

    scanner = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=20.0,
        num_sides=8,
        num_lor_endpoints_per_side=4,
        lor_spacing=2.0,
        ring_positions=xp.linspace(-2.0, 2.0, 3, device=dev),
        symmetry_axis=2,
    )
    lor_desc = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(num_rings=3, max_ring_difference=1, span=1),
        radial_trim=3,
    )
    for tof in (False, True):
        p = ppp.RegularPolygonPETProjector(
            lor_desc, img_shape=(8, 8, 3), voxel_size=(2.0, 2.0, 2.0)
        )
        if tof:
            p.tof_parameters = TOF_PARAMS
        projs[f"polygon{'_tof' if tof else ''}"] = p

    xstart = xp.asarray(
        [[-20.0, 0.0, 0.0], [0.0, -20.0, 1.0], [-15.0, -15.0, -1.0]],
        dtype=xp.float32,
        device=dev,
    )
    for tof in (False, True):
        p = ppp.ListmodePETProjector(xstart, -xstart, (8, 8, 3), (2.0, 2.0, 2.0))
        if tof:
            p.tof_parameters = TOF_PARAMS
            p.event_tofbins = xp.asarray([0, 1, -1], dtype=xp.int16, device=dev)
            p.tof = True
        projs[f"listmode{'_tof' if tof else ''}"] = p

    aff1 = xp.eye(4, device=dev, dtype=xp.float32)
    aff1[1, -1] = 10.0
    aff2 = xp.eye(4, device=dev, dtype=xp.float32)
    aff2[1, -1] = -10.0
    block_scanner = pps.ModularizedPETScannerGeometry(
        [
            pps.BlockPETScannerModule(
                xp, dev, (2, 2, 2), (4.0, 3.0, 2.0), affine_transformation_matrix=a
            )
            for a in (aff1, aff2)
        ]
    )
    block_desc = ppl.EqualBlockPETLORDescriptor(block_scanner, xp.asarray([[0, 1]]))
    for tof in (False, True):
        p = ppp.EqualBlockPETProjector(block_desc, (8, 8, 3), (1.0, 1.0, 1.0))
        if tof:
            p.tof_parameters = TOF_PARAMS
        projs[f"equalblock{'_tof' if tof else ''}"] = p

    return projs


def test_current_cuda_stream(xp: ModuleType, dev) -> None:
    x = xp.zeros(3, device=dev)
    stream = _current_cuda_stream(x)

    if array_api_compat.is_torch_array(x) and x.device.type == "cuda":
        import torch

        assert stream == torch.cuda.current_stream(x.device).cuda_stream
        side = torch.cuda.Stream(device=x.device)
        with torch.cuda.stream(side):
            assert _current_cuda_stream(x) == side.cuda_stream
        assert _current_cuda_stream(x) == stream
    elif array_api_compat.is_cupy_array(x):  # pragma: no cover
        import cupy

        assert stream == cupy.cuda.get_current_stream().ptr
        side = cupy.cuda.Stream(non_blocking=True)
        with side:
            assert _current_cuda_stream(x) == side.ptr
    else:
        assert stream is None


def test_projectors_pass_current_stream(xp: ModuleType, dev, monkeypatch) -> None:
    """Every core call of every projector (fwd and adjoint) gets the current stream."""
    spy = _SpyCore()
    monkeypatch.setattr(ppp, "parallelproj_core", spy)

    expected = _current_cuda_stream(xp.zeros(1, device=dev))

    for name, proj in _projectors(xp, dev).items():
        spy.calls.clear()
        x = xp.ones(proj.in_shape, dtype=xp.float32, device=dev)
        y = proj(x)
        proj.adjoint(xp.ones_like(y))
        assert len(spy.calls) == 2, name
        for core_function, stream in spy.calls:
            assert stream == expected, (name, core_function, stream)


def test_projectors_side_stream(xp: ModuleType, dev) -> None:
    """GPU: projections on a side stream are ordered with the work on that stream."""
    if not (torch_available and array_api_compat.is_torch_namespace(xp)):
        pytest.skip("torch only")
    if xp.zeros(1, device=dev).device.type != "cuda":
        pytest.skip("needs a CUDA device")
    import torch

    for name, proj in _projectors(xp, dev).items():
        torch.manual_seed(0)
        x = torch.rand(proj.in_shape, dtype=torch.float32, device=dev)
        y_ref = proj(x)
        x_back_ref = proj.adjoint(y_ref)
        torch.cuda.synchronize()

        side = torch.cuda.Stream(device=x.device)
        side.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(side):
            # produce the input on the side stream right before the projection
            # (a projection on another stream could see the old values)
            x_side = torch.zeros_like(x)
            for _ in range(20):
                x_side = x_side + x / 20
            y = proj(x_side)
            x_back = proj.adjoint(y)
        torch.cuda.current_stream(x.device).wait_stream(side)

        assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5), name
        assert torch.allclose(x_back, x_back_ref, rtol=1e-4, atol=1e-4), name


@pytest.mark.skipif(num_cuda_devices < 2, reason="needs two GPUs")
def test_projectors_two_gpus(xp: ModuleType, dev) -> None:  # pragma: no cover
    """>= 2 GPUs: the same projectors on cuda:0 and cuda:1, queued without waiting."""
    if not (array_api_compat.is_torch_namespace(xp) and dev == "cuda"):
        pytest.skip("torch cuda only")
    import torch

    projs = {d: _projectors(xp, d) for d in ("cuda:0", "cuda:1")}
    for name in projs["cuda:0"]:
        torch.manual_seed(0)
        x = torch.rand(projs["cuda:0"][name].in_shape, dtype=torch.float32)
        # queue both GPUs first, then compare (no synchronization in between)
        res = {
            d: projs[d][name].adjoint(projs[d][name](x.to(d)))
            for d in ("cuda:0", "cuda:1")
        }
        assert torch.allclose(
            res["cuda:0"].cpu(), res["cuda:1"].cpu(), rtol=1e-4, atol=1e-4
        ), name
