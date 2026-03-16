from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parallelproj")
except PackageNotFoundError:
    __version__ = "unknown"


from ._backend import Array, empty_cuda_cache, to_numpy_array

__all__ = [
    "__version__",
    "Array",
    "empty_cuda_cache",
    "to_numpy_array",
]
