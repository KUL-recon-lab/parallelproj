API reference
=============

.. _api_import_map:

Modules
-------

The top-level ``parallelproj`` namespace deliberately exposes only the array
backend helpers (:data:`.Array`, :func:`.to_numpy_array`,
:func:`.empty_cuda_cache`) plus the ``__version__`` / ``__citation__`` /
``__bibtex__`` dunders.  Everything else lives in a submodule and is imported
explicitly.

Each card below links to a submodule's full reference and carries its exact
import line (expand *import statement*):

.. include:: _import_map.rst

.. toctree::
    :hidden:
    :maxdepth: 1

    PET scanner geometries <api_pet_scanners>
    PET LOR / sinogram descriptors <api_pet_lors>
    PET projectors <api_projectors>
    PET TOF parameters <api_tof>
    Linear operators <api_operators>
    Functions <api_functions>
    PET sinogram symmetries <api_pet_sino_symmetries>
    PET LM Unlisting <api_pet_unlist>
    Data <api_data>
