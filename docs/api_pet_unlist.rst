:tocdepth: 2

PET LM Unlisting ``parallelproj.unlist``
----------------------------------------------------

Helpers for converting listmode events into sinograms ("unlisting").  Given
per-event detector (and TOF) indices, these functions produce the corresponding
sinogram-bin mapping, which is useful when histogramming measured listmode data
into sinograms.  For working directly in listmode instead, see the listmode
projector and the ``04_listmode_algorithms`` gallery examples.

.. automodule:: parallelproj.unlist
