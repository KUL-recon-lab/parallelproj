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

# Which examples are *executed*?  Non-executed examples are still rendered
# (their source is shown) but not run.  Two env vars control this:
#   * default (incl. Read the Docs): only ``<NN>_run_*.py`` run, which keeps
#     the RTD build under its 15 min limit;
#   * ``BUILD_ALL_EXAMPLES=1``: execute *every* example (e.g. on a GPU box);
#   * ``BUILD_NO_EXAMPLES=1``: execute *none* (fast local prose/structure
#     build); this takes precedence over ``BUILD_ALL_EXAMPLES``.
_build_all_examples = os.environ.get("BUILD_ALL_EXAMPLES", "0") == "1"
_build_no_examples = os.environ.get("BUILD_NO_EXAMPLES", "0") == "1"
_example_filename_pattern = (
    r".*\.py$" if _build_all_examples else r"[\\/]\d{2,3}_run_.*\.py$"
)

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "backreferences_dir": "auto_examples/backreferences",
    "doc_module": ("parallelproj",),
    "filename_pattern": _example_filename_pattern,
    "ignore_pattern": r"(^|[\\/])(utils|example_utils)\.py$",
    # plot_gallery=False disables execution of ALL examples
    "plot_gallery": not _build_no_examples,
    "within_subsection_order": FileNameSortKey,
    "parallel": os.cpu_count(),
    # Inject a setup note at the top of every generated Jupyter notebook so
    # that users who download a notebook know how to obtain the helper modules.
    "first_notebook_cell": (
        "# To run this notebook you need parallelproj and the example_utils helper.\n"
        "#\n"
        "# 1. Install parallelproj (if not already):\n"
        "#      conda install -c conda-forge parallelproj\n"
        "#\n"
        "# 2. Download example_utils.py into the same folder as this notebook:\n"
        "#    https://raw.githubusercontent.com/KUL-recon-lab/"
        "parallelproj/main/docs/examples/example_utils.py"
    ),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "_static/parallelproj-logo.png"
html_favicon = "_static/favicon.ico"

# Theme options.  The brand colours are taken from the parallelproj logo
# (navy / blue family); lighter shades are used in dark mode for contrast.
html_theme_options = {
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#2c5178",
        "color-brand-content": "#3c75aa",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7aa7d0",
        "color-brand-content": "#adc9e0",
    },
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

suppress_warnings = [
    "config.cache",
    # ``shape`` is a valid property on several classes (SubsetArrayMmap,
    # BlockPETScannerModule, Array Protocol, …).  autodoc_typehints="both"
    # causes Sphinx to emit a "more than one target" warning whenever a bare
    # ``shape`` cross-reference is resolved and finds multiple candidates.
    # ``ref.python`` only fires for ambiguous targets; without nitpicky=True
    # missing targets are silently skipped, so this suppression is safe.
    "ref.python",
]


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
