# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib
from pathlib import Path

from sphinx_gallery.sorting import FileNameSortKey

# make shared example utilities (vis.py) importable from all example subdirectories
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "parallelproj"
# copyright = "2026, parallelproj team"
# author = "parallelproj team"

with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as _f:
    release = tomllib.load(_f)["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples/README.rst"]
bibtex_bibfiles = ["refs.bib"]

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "backreferences_dir": "auto_examples/backreferences",
    "doc_module": ("parallelproj",),
    "filename_pattern": r"[\\/]\d{2,3}_run_.*\.py$",
    "ignore_pattern": r"(^|[\\/])(vis|utils|img|array_utils)\.py$",
    "plot_gallery": True,
    "within_subsection_order": FileNameSortKey,
    "parallel": os.cpu_count(),
    # Inject a setup note at the top of every generated Jupyter notebook so
    # that users who download a notebook know how to obtain the helper modules.
    "first_notebook_cell": (
        "# To run this notebook you need parallelproj and three helper modules.\n"
        "#\n"
        "# 1. Install parallelproj (if not already):\n"
        "#      conda install -c conda-forge parallelproj\n"
        "#\n"
        "# 2. Download the helper modules into the same folder as this notebook:\n"
        "#    array_utils.py: https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/array_utils.py\n"
        "#    img.py:         https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/img.py\n"
        "#    vis.py:         https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/vis.py"
    ),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "_static/parallelproj-logo.png"

# Theme options
html_theme_options = {
    "navigation_with_keys": True,
}

# -- napoleon options --------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- autodoc options ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "private-members": False,
    "special-members": "__call__, __getitem__, __setitem__",
    "show-inheritance": True,
}

autoclass_content = "both"
autodoc_typehints = "both"

suppress_warnings = ["config.cache"]


def _auto_minigallery(app, what, name, obj, options, lines):
    """Append a minigallery directive if sphinx-gallery has backreferences for this object."""
    if what not in ("class", "function", "method", "attribute"):
        return

    gallery_conf = app.config.sphinx_gallery_conf
    backrefs_dir = os.path.join(
        app.srcdir,
        gallery_conf.get("backreferences_dir", "auto_examples/backreferences"),
    )
    stub = os.path.join(backrefs_dir, f"{name}.examples")

    if os.path.isfile(stub) and not any("minigallery" in line for line in lines):
        lines += ["", ".. rubric:: Examples", "", f".. minigallery:: {name}", ""]


def setup(app):
    app.connect("autodoc-process-docstring", _auto_minigallery)
