:tocdepth: 2

PET scanner geometries ``parallelproj.pet_scanners``
----------------------------------------------------

The scanner geometry is the first link in the projector construction chain, and
which scanner class you pick selects one of the two projector routes (see
:doc:`api_projectors`):

* **Cylindrical scanners with a single layer of LOR endpoints** (the most common
  case): use :class:`.RegularPolygonPETScannerGeometry`, or
  :class:`.DemoPETScannerGeometry` for a ready-made example.  These feed the
  regular-polygon route (via :class:`.Michelogram` and
  :class:`.RegularPolygonPETLORDescriptor`).
* **General modular scanners** (block / panel geometries, or anything without
  cylindrical symmetry): build a :class:`.ModularizedPETScannerGeometry` from
  :class:`.BlockPETScannerModule` objects, which feeds the equal-block route
  (via :class:`.EqualBlockPETLORDescriptor`).

The array backend ``xp`` and device ``dev`` passed here determine where all
downstream computation runs (CPU or GPU).  See :doc:`quickstart` for a runnable
example of the regular-polygon route.

.. automodule:: parallelproj.pet_scanners
