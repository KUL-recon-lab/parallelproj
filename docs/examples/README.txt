Examples
========

These galleries progress from PET scanner geometry and projectors, through
iterative reconstruction algorithms (sinogram and listmode), to transmission and
joint activity/attenuation (MLAA) reconstruction, and finally PyTorch
integration.  If you are new to parallelproj, read the Quickstart first, then
work through the galleries roughly in numerical order.

Every example selects its array backend and device through the helper
``suggest_array_backend_and_device`` from ``parallelproj._examples_utils`` -- a
private, examples-only module shipped inside parallelproj -- so the same code
runs on CPU or GPU with nothing extra to install or download.
