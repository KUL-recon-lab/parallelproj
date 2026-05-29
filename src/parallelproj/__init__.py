from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parallelproj")
except PackageNotFoundError:
    __version__ = "unknown"


from ._backend import Array, empty_cuda_cache, to_numpy_array, count_event_multiplicity
from .unlist import detection_times_to_tof_bin, regular_polygon_events_to_sinogram

__all__ = [
    "__version__",
    "Array",
    "empty_cuda_cache",
    "to_numpy_array",
    "count_event_multiplicity",
    "regular_polygon_events_to_sinogram",
    "detection_times_to_tof_bin",
]
