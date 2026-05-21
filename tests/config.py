import pytest
import importlib

import array_api_compat.numpy as np

import parallelproj_core as pp

num_cuda_devices = 0

torch_available = (
    importlib.util.find_spec("array_api_compat.torch") is not None
    and importlib.util.find_spec("torch") is not None
)
if torch_available:
    import array_api_compat.torch as torch

    num_cuda_devices = torch.cuda.device_count()

cupy_available = (
    importlib.util.find_spec("array_api_compat.cupy") is not None
    and importlib.util.find_spec("cupy") is not None
)
if cupy_available:
    import array_api_compat.cupy as cp

    try:
        num_cuda_devices = cp.cuda.runtime.getDeviceCount()
    except:
        num_cuda_devices = 0

# %%

xp_dev_list = []

###########
# add numpy
xp_dev_list.append((np, "cpu"))

###########
# add array_api_strict if available
if importlib.util.find_spec("array_api_strict") is not None:
    import array_api_strict as nparr

    xp_dev_list.append((nparr, None))

###########
# add torch cpu
if torch_available:
    xp_dev_list.append((torch, "cpu"))


###########
# add torch gpu and cupy
if pp.cuda_enabled == 1 and num_cuda_devices > 0:
    if torch_available:
        xp_dev_list.append((torch, "cuda"))
    if cupy_available:
        xp_dev_list.append((cp, cp.cuda.Device(0)))

pytestmark = pytest.mark.parametrize("xp,dev", xp_dev_list)
