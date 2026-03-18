# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

from sphinx_gallery.sorting import FileNameSortKey

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "parallelproj"
# copyright = "2026, parallelproj team"
# author = "parallelproj team"

try:
    release = _get_version("parallelproj")
except PackageNotFoundError:
    release = "unknown"

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
    "ignore_pattern": r"utils.py",
    "plot_gallery": True,
    "within_subsection_order": FileNameSortKey,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "_static/parallelproj-logo.png"

# Theme options
html_theme_options = {
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/gschramm/parallelproj",
    "source_branch": "master",
    "source_directory": "docs/",
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
