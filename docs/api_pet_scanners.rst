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

.. _pet-scanner-conventions:

Geometry conventions
~~~~~~~~~~~~~~~~~~~~

Distances are in millimetres.
Use the same units for scanner coordinates, image origin and voxel size.
For :class:`.RegularPolygonPETScannerGeometry`, ``radius`` is the shortest distance from the scanner centre to a flat side of endpoints.
When starting from the inner crystal face, add the chosen depth of interaction, which may be an average, to obtain ``radius``.
Each endpoint has one fixed position.

With the default axis, side 0 is centred on the negative y-axis.
Endpoint indices increase clockwise when looking from negative z towards positive z, with positive x to the right and positive y down.
Rings follow the order of ``ring_positions``.
Named scanners use these conventions and centre their rings on zero.
The depths of interaction can differ between scanner models.

The cylindrical models for mMR, Vision 600, Discovery MI and Discovery 690 use the effective radii and detector counts from STIR.

Ready-made scanner models
~~~~~~~~~~~~~~~~~~~~~~~~~

The following classes provide ready-made geometries for real scanners. See each
class for its references and constructor options.

* :class:`.SiemensMMRPETScannerGeometry`,
  :class:`.SiemensVision600PETScannerGeometry`,
  :class:`.SiemensVisionQuadraPETScannerGeometry` and
  :class:`.SiemensMCTPETScannerGeometry`: flat-block models by default,
  ``model="cylindrical"`` selects a STIR-based cylindrical model.
* :class:`.GEDiscoveryMIPETScannerGeometry`: a cylindrical model with 3, 4, 5
  or 6 axial block rings, selected with ``num_block_rings``.
* :class:`.GEDiscovery690PETScannerGeometry`: a cylindrical model by default,
  ``model="castor_blocks"`` selects a flat-block model.
* :class:`.MedisoAnyScanPETScannerGeometry`: a cylindrical model.
* :class:`.GESignaPETMRScannerGeometry` and
  :class:`.SiemensHRRTPETScannerGeometry`: STIR-inspired cylindrical models.
* :class:`.PhilipsVereosPETScannerGeometry` and
  :class:`.UnitedImagingUExplorerPETScannerGeometry`: flat-block models.

.. automodule:: parallelproj.pet_scanners
