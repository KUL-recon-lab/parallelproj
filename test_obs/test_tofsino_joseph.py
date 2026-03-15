import parallelproj
import array_api_compat.numpy as np
import math
from scipy.special import erf

from types import ModuleType

from .config import pytestmark


def isclose(x: float, y: float, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """check if two floating point numbers are close to each other, given absolute and relative error
    inspired by numpy.isclose
    """
    return bool(abs(x - y) <= (atol + rtol * abs(y)))


def test_tof_sino_fwd(
    xp: ModuleType,
    dev: str,
    verbose: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-4,
) -> None:
    """compare a TOF forward projection of a central point source against the theoretical value
    for different combinations of TOF bin width and TOF resolution
    """
    sig_t = 10.0

    n = 101
    vsize = 1.0

    voxsize = xp.asarray([vsize, vsize, vsize], dtype=xp.float32, device=dev)
    tmp = (-0.5 * n + 0.5) * float(voxsize[0])
    img_origin = xp.asarray([tmp, tmp, tmp], dtype=xp.float32, device=dev)

    img = xp.zeros((n, n, n), dtype=xp.float32, device=dev)
    img[n // 2, n // 2, n // 2] = 1.0

    # loop over different number of sigmas used for kernel truncation
    for ns in [2.5, 3.0, 3.5]:
        # loop over all principal directions
        for direction in [0, 1, 2]:
            # loop over different TOF bin widths
            for delta in [
                0.1 * sig_t,
                0.3 * sig_t,
                0.5 * sig_t,
                sig_t,
                3 * sig_t,
                5 * sig_t,
            ]:

                xstart = xp.zeros((1, 3), dtype=xp.float32, device=dev)
                xend = xp.zeros((1, 3), dtype=xp.float32, device=dev)

                xstart[0, direction] = 2 * float(img_origin[0])
                xend[0, direction] = -2 * float(img_origin[0])

                num_tofbins = max(4 * int(n * vsize / delta / 2) + 1, 11)

                p_tof = parallelproj.joseph3d_fwd_tof_sino(
                    xstart,
                    xend,
                    img,
                    img_origin,
                    voxsize,
                    tofbin_width=delta,
                    sigma_tof=xp.asarray([sig_t], dtype=xp.float32, device=dev),
                    tofcenter_offset=xp.asarray([0], dtype=xp.float32, device=dev),
                    nsigmas=ns,
                    ntofbins=num_tofbins,
                )

                trunc_factor = 1.0 / erf(ns / math.sqrt(2))
                trunc_dist = ns * math.sqrt(sig_t**2 + (delta**2) / 12)

                for i in range(num_tofbins // 2):

                    if i * delta <= 0.999 * trunc_dist:
                        # the theoretical value of the TOF projection which is equal to the
                        # difference of two error functions corrected for kernel truncation
                        theory_value = (
                            0.5
                            * trunc_factor
                            * (
                                erf((i * delta + 0.5 * delta) / (math.sqrt(2) * sig_t))
                                - erf(
                                    (i * delta - 0.5 * delta) / (math.sqrt(2) * sig_t)
                                )
                            )
                        )

                        if verbose:
                            print(
                                ns,
                                direction,
                                delta,
                                sig_t,
                                i,
                                theory_value,
                                float(p_tof[0, num_tofbins // 2 + i]) - theory_value,
                            )

                        abs_diff = abs(
                            float(p_tof[0, num_tofbins // 2 + i]) - theory_value
                        )
                        assert abs_diff < atol

                        rel_diff = abs_diff / theory_value
                        assert rel_diff < rtol

                if verbose:
                    print()


def test_adjointness(
    xp: ModuleType, dev: str, nLORs=1000000, seed=1, verbose=True
) -> None:
    """test whether backprojection is the adjoint of forward projection
    indirect test whether back projection is correct (assuming fwd projection is correct)
    """

    np.random.seed(seed)
    n0, n1, n2 = (16, 15, 17)

    img_dim = (n0, n1, n2)
    voxel_size = xp.asarray([0.7, 0.8, 0.6], dtype=xp.float32, device=dev)
    img_origin = (
        -xp.asarray(img_dim, dtype=xp.float32, device=dev) / 2 + 0.5
    ) * voxel_size

    img = xp.asarray(np.random.rand(n0, n1, n2), dtype=xp.float32, device=dev)

    # generate random LORs on a sphere around the image volume
    R = 0.8 * xp.max((xp.asarray(img_dim, dtype=xp.float32, device=dev) * voxel_size))

    phis = xp.asarray(np.random.rand(nLORs) * 2 * np.pi, device=dev, dtype=xp.float32)
    costheta = xp.asarray(np.random.rand(nLORs) * 2 - 1, device=dev, dtype=xp.float32)
    sintheta = xp.sqrt(1 - costheta**2)

    xstart = xp.zeros((nLORs, 3), dtype=xp.float32, device=dev)
    xstart[:, 0] = R * sintheta * xp.cos(phis)
    xstart[:, 1] = R * sintheta * xp.sin(phis)
    xstart[:, 2] = R * costheta

    phis = xp.asarray(np.random.rand(nLORs) * 2 * np.pi, device=dev, dtype=xp.float32)
    costheta = xp.asarray(np.random.rand(nLORs) * 2 - 1, device=dev, dtype=xp.float32)
    sintheta = xp.sqrt(1 - costheta**2)

    xend = xp.zeros((nLORs, 3), dtype=xp.float32, device=dev)
    xend[:, 0] = R * sintheta * xp.cos(phis)
    xend[:, 1] = R * sintheta * xp.sin(phis)
    xend[:, 2] = R * costheta

    # TOF parameters
    tofbin_width = 2.0
    num_tof_bins = 11
    nsigmas = 3.0
    sigma_tof = xp.asarray([5 / 2.35], dtype=xp.float32, device=dev)
    tofcenter_offset = xp.asarray([0], dtype=xp.float32, device=dev)

    # forward project
    img_fwd = parallelproj.joseph3d_fwd_tof_sino(
        xstart,
        xend,
        img,
        img_origin,
        voxel_size,
        tofbin_width,
        sigma_tof,
        tofcenter_offset,
        nsigmas,
        num_tof_bins,
        num_chunks=3,
    )

    # backward project
    sino = xp.asarray(np.random.rand(*img_fwd.shape), dtype=xp.float32, device=dev)
    back_img = parallelproj.joseph3d_back_tof_sino(
        xstart,
        xend,
        img.shape,
        img_origin,
        voxel_size,
        sino,
        tofbin_width,
        sigma_tof,
        tofcenter_offset,
        nsigmas,
        num_tof_bins,
        num_chunks=5,
    )

    ip_a = float(xp.sum((back_img * img)))
    ip_b = float(xp.sum((img_fwd * sino)))

    if verbose:
        print(
            f"module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}"
        )
        print("ip_a = ", ip_a)
        print("ip_b = ", ip_b)
        print("")

    assert isclose(ip_a, ip_b)


def test_tof_sino_fwd_sum(
    xp: ModuleType,
    dev: str,
    verbose: bool = True,
    rtol: float = 4e-3,
    ns=3.0,
) -> None:
    """test whether the sum over all TOF bins of a TOF projection is equal to the non-TOF projection,
    even for the case where the TOF bin is much bigger than the TOF resolution
    """
    # sigma of Gaussian TOF kernel
    sig_t = 10.0
    # number of voxels and voxel size
    n = 101
    vsize = 1.0
    voxsize = xp.asarray([vsize, vsize, vsize], dtype=xp.float32, device=dev)
    tmp = float((-0.5 * n + 0.5) * voxsize[0])
    img_origin = xp.asarray([tmp, tmp, tmp], dtype=xp.float32, device=dev)
    num_off = 30

    # width of TOF bins (much bigger than TOF resolution)
    for delta in [sig_t / 10, sig_t / 2, sig_t, 3 * sig_t, 10 * sig_t]:
        rdiffs = xp.zeros(num_off, dtype=xp.float32, device=dev)

        for i, offset in enumerate(range(num_off)):
            img = xp.zeros((n, n, n), dtype=xp.float32, device=dev)

            img[n // 2 + offset, n // 2, n // 2] = 1.0
            xstart = xp.asarray([[2 * int(img_origin[0]), 0, 0]], device=dev)
            xend = xp.asarray([[-2 * int(img_origin[0]), 0, 0]], device=dev)

            # img[n // 2, n // 2 + offset, n // 2] = 1.0
            # xstart = xp.asarray([[0, 2 * int(img_origin[0]), 0]], device=dev)
            # xend = xp.asarray([[0, -2 * int(img_origin[0]), 0]], device=dev)

            # img[n // 2, n // 2, n // 2 + offset] = 1.0
            # xstart = xp.asarray([[0, 0, 2 * int(img_origin[0])]], device=dev)
            # xend = xp.asarray([[0, 0, -2 * int(img_origin[0])]], device=dev)

            p_nontof = parallelproj.joseph3d_fwd(
                xstart,
                xend,
                img,
                img_origin,
                voxsize,
            )

            p_tof = parallelproj.joseph3d_fwd_tof_sino(
                xstart,
                xend,
                img,
                img_origin,
                voxsize,
                tofbin_width=delta,
                sigma_tof=xp.asarray([sig_t], dtype=xp.float32),
                tofcenter_offset=xp.asarray([0], dtype=xp.float32),
                nsigmas=ns,
                ntofbins=max(4 * int(n * vsize / delta / 2) + 1, 11),
            )

            p_tof_sum = float(xp.sum(p_tof))

            rdiffs[i] = abs(p_tof_sum - float(p_nontof[0])) / float(p_nontof[0])

        max_rdiff = float(xp.max(rdiffs))

        if verbose:
            print(f"sig_t {sig_t} tofbin width {delta}")
            print(
                f"max rel. diff. between summed TOF projection and non-TOF projection = {max_rdiff:.6f}"
            )

        assert max_rdiff < rtol
