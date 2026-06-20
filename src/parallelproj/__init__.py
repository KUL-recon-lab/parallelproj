from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parallelproj")
except PackageNotFoundError:
    __version__ = "unknown"


from ._backend import Array, empty_cuda_cache, to_numpy_array

#: Human-readable citation. ``import parallelproj`` stays silent; if you use
#: parallelproj in your research, please cite this paper (see also ``__bibtex__``).
__citation__ = (
    "Schramm G, Thielemans K. PARALLELPROJ - an open-source framework for fast "
    "calculation of projections in tomography. Frontiers in Nuclear Medicine, "
    "3 (2023). doi:10.3389/fnume.2023.1324562"
)

#: BibTeX entry for parallelproj (see ``__citation__`` for the plain-text form).
__bibtex__ = """@article{Schramm2023,
  author  = {Schramm, Georg and Thielemans, Kris},
  title   = {PARALLELPROJ---an open-source framework for fast calculation of projections in tomography},
  journal = {Frontiers in Nuclear Medicine},
  year    = {2023},
  volume  = {3},
  doi     = {10.3389/fnume.2023.1324562}
}"""

__all__ = [
    "__version__",
    "__citation__",
    "__bibtex__",
    "Array",
    "empty_cuda_cache",
    "to_numpy_array",
]
