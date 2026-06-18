Examples
========

These galleries progress from PET scanner geometry and projectors, through
iterative reconstruction algorithms (sinogram and listmode), to transmission and
joint activity/attenuation (MLAA) reconstruction, and finally PyTorch
integration.  If you are new to parallelproj, read the Quickstart first, then
work through the galleries roughly in numerical order.

Every example selects its array backend and device through the helper
``suggest_array_backend_and_device`` in ``example_utils.py`` (download it into
the same folder when running a script standalone), so the same code runs on
CPU or GPU.
