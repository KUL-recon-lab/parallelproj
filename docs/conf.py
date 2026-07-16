# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import tomllib
from pathlib import Path

from sphinx_gallery.sorting import FileNameSortKey

# The shared example helpers now live in ``parallelproj._examples_utils`` (a
# private, examples-only module shipped inside the package), so the gallery
# examples import them straight from ``parallelproj`` -- no sys.path setup, no
# PYTHONPATH and no separate ``example_utils.py`` download are needed.

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "parallelproj"
# copyright = "2026, parallelproj team"
# author = "parallelproj team"

with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as _f:
    release = tomllib.load(_f)["project"]["version"]

# The documentation is English-only (and intended to stay that way).  Set the
# language explicitly so the HTML ``lang`` attribute, full-text-search stemming
# and date formatting are correct, and so local builds match Read the Docs
# without relying on the ``-D language=en`` CLI override.
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinxext.opengraph",
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
    # Inject a setup note at the top of every generated Jupyter notebook.  The
    # example helpers ship inside parallelproj (parallelproj._examples_utils),
    # so installing parallelproj is all that is required to run a notebook.
    "first_notebook_cell": (
        "# To run this notebook, install parallelproj (which bundles the\n"
        "# example helpers used below in ``parallelproj._examples_utils``):\n"
        "#\n"
        "#      conda install -c conda-forge parallelproj"
    ),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "_static/parallelproj-logo.png"
html_favicon = "_static/favicon.ico"

# Canonical base URL for SEO: tells search engines that the ``stable`` version
# is the authoritative one, so ranking is not split across the per-version
# paths (``/latest/``, ``/v2.0.0/``, ...).  Sphinx emits a
# ``<link rel="canonical">`` on every page pointing at the matching page under
# this base URL.  Read the Docs also uses this for its generated sitemap.
html_baseurl = "https://parallelproj.readthedocs.io/en/stable/"

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

# -- Open Graph / SEO metadata (sphinxext-opengraph) -------------------------
# Adds a <meta name="description"> to every page (from its first paragraphs)
# and Open Graph / Twitter-card tags, improving Google snippets and link
# previews (GitHub, Slack, X, ...).
ogp_site_url = "https://parallelproj.readthedocs.io/en/stable/"
ogp_site_name = "parallelproj documentation"
ogp_image = (
    "https://parallelproj.readthedocs.io/en/stable/_static/parallelproj-logo.png"
)
ogp_type = "website"
ogp_enable_meta_description = True
ogp_description_length = 200
# use the static logo as the preview image rather than auto-generated cards
ogp_social_cards = {"enable": False}

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


# Submodules shown in the API "Modules" grid, in display order.
# Each entry is (module, api_doc_page, card_title, one_line_description).
# Only this doc-structure metadata is hand maintained; the symbol *names* in
# each card are introspected from the package at build time (never hand
# maintained), so the import lines can never drift from the code.
_IMPORT_MAP_MODULES = [
    ("parallelproj.pet_scanners", "api_pet_scanners", "PET scanner geometries",
     "Regular-polygon, demo and general modular / block scanner geometries."),
    ("parallelproj.pet_lors", "api_pet_lors", "PET LOR / sinogram descriptors",
     "Michelogram axial layout, sinogram ordering, LOR descriptors and axial compression."),
    ("parallelproj.projectors", "api_projectors", "PET projectors",
     "Sinogram and list-mode forward / back projectors -- the operators you reconstruct with."),
    ("parallelproj.tof", "api_tof", "PET TOF parameters",
     "``TOFParameters``: the time-of-flight kernel (bin width, resolution, centre offset)."),
    ("parallelproj.operators", "api_operators", "Linear operators",
     "Building blocks: resolution models, finite differences, compositions and stacking."),
    ("parallelproj.functions", "api_functions", "Functions",
     "Data-fidelity terms and priors: Poisson log-likelihood, squared-L2, log-cosh, affine objectives."),
    ("parallelproj.sinogram_symmetries", "api_pet_sino_symmetries", "PET sinogram symmetries",
     "Plane / view / radial symmetry classes and sinogram reduction / expansion helpers."),
    ("parallelproj.unlist", "api_pet_unlist", "PET LM unlisting",
     "Turn detected events into sinograms; TOF-bin assignment from arrival times."),
    ("parallelproj.data", "api_data", "Data",
     "Memory-mapped subset helpers for large list-mode datasets."),
]


def _public_api_names(module):
    """Public, concrete classes/functions *defined* in ``module``.

    Excludes private names, re-imports (``__module__`` mismatch) and abstract
    base classes (``inspect.isabstract``) -- the import map should list the
    things a user actually imports and constructs, not the ABCs they subclass.
    """
    import inspect

    return sorted(
        name
        for name, obj in inspect.getmembers(module)
        if not name.startswith("_")
        and (inspect.isclass(obj) or inspect.isfunction(obj))
        and getattr(obj, "__module__", None) == module.__name__
        and not (inspect.isclass(obj) and inspect.isabstract(obj))
    )


def _import_statement(modname, names):
    """Lines of a ``from <modname> import (...)`` statement (no indentation)."""
    if len(names) == 1:
        return [f"from {modname} import {names[0]}"]
    return [f"from {modname} import ("] + [f"    {n}," for n in names] + [")"]


def _indent(lines, n):
    pad = " " * n
    return [pad + ln if ln else "" for ln in lines]


def _code_block(stmt_lines):
    """A python ``code-block`` directive (col 0) wrapping ``stmt_lines``."""
    return [".. code-block:: python", ""] + _indent(stmt_lines, 3)


def _generate_import_map(app, config):
    """Write ``_import_map.rst`` from the live package so the module grid in
    ``api.rst`` can never drift.  Each submodule is a sphinx-design card
    combining a link to its API page (in the footer) with a collapsible
    dropdown holding its exact import line."""
    import importlib

    lines = [".. grid:: 1 2 2 2", "   :gutter: 3", ""]

    for modname, doc, title, desc in _IMPORT_MAP_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:  # pragma: no cover - keep the docs build resilient
            continue
        names = _public_api_names(mod)
        if not names:
            continue
        # card body (col 0); indented by 3 to sit under the grid below
        card = [f".. grid-item-card:: {title}", ""]
        card += _indent([desc], 3)
        card += [""]
        card += _indent([".. dropdown:: import statement", ""], 3)
        card += _indent(_code_block(_import_statement(modname, names)), 6)
        card += ["", "   +++", ""]
        card += _indent([f":doc:`Full reference → <{doc}>`"], 3)
        lines += _indent(card, 3)
        lines.append("")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_import_map.rst")
    with open(out, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def setup(app):
    app.connect("autodoc-process-docstring", _auto_minigallery)
    app.connect("config-inited", _generate_import_map)
