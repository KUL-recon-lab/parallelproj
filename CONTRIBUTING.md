# Contributing to parallelproj

Thanks for your interest in improving parallelproj! Bug reports, feature
requests, documentation fixes and pull requests are all welcome.

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](./LICENSE) license.

## Scope: what belongs in this repository

This repository is the **pure-Python front end**. The compiled C/CUDA Joseph
projection kernels live in the separate
[`libparallelproj`](https://github.com/KUL-recon-lab/libparallelproj) project and
are distributed as the `parallelproj-core` package on conda-forge.

- Pure-Python features (scanners, LOR/sinogram descriptors, operators,
  projectors' Python API, algorithms, examples, docs) go **here**.
- Changes to the C/CUDA projectors must be made in `libparallelproj` — editing
  this clone does **not** rebuild the backend.

## Development setup

We use [pixi](https://pixi.sh), which builds a fully reproducible environment
from the lock file. The local clone is registered as a path dependency, so
`import parallelproj` resolves to your working tree (no manual environment setup
or `PYTHONPATH` needed).

```bash
git clone https://github.com/KUL-recon-lab/parallelproj.git
cd parallelproj
```

Use `pixi shell` to drop into a terminal that has all dependencies (and the local
`parallelproj`) available, picking the environment that matches your hardware:

```bash
pixi shell                 # default env (CPU)
pixi shell -e cuda12       # CUDA 12 stack
pixi shell -e cuda13       # CUDA 13 stack
```

Inside that shell you can run scripts, tests or the examples directly (e.g.
`python <script>`, `ipython`, `pytest`). The environment is built automatically
on first use. Exit with `exit` or `Ctrl+D`.

Common tasks can also be run without an explicit shell via `pixi run`:

```bash
pixi run test        # run the test suite (GPU tests self-skip without a CUDA device)
pixi run test-cov    # tests with a coverage report
pixi run docs        # build the HTML documentation
pixi run docs-fast   # build docs without executing examples (fast prose/structure check)
```

The first `pixi run` also activates the repo-tracked git hooks (see below); you
can activate them explicitly with `pixi run setup-hooks`.

## Coding conventions

- **Array-API compatibility.** parallelproj is backend-agnostic: code must work
  across NumPy, `array_api_strict`, PyTorch and CuPy. Use the array namespace
  (`xp`) and `array_api_compat` helpers rather than NumPy-only operations
  (e.g. `xp.astype(a, ...)` / `xp.mean(a, ...)` instead of `a.astype(...)` /
  `a.mean()`). The test suite parametrizes over `(xp, dev)` to enforce this.
- **`from __future__ import annotations`** at the top of every module.
- **Docstrings** are NumPy-style (rendered by Sphinx + napoleon). Public
  classes/functions should be documented so they appear correctly in the API
  reference.
- **Formatting** follows `black`; keep type hints where practical.

## Tests

- Run with `pixi run test`. GPU-only paths self-skip when no CUDA device is
  present.
- New code should come with tests in `tests/`. Tests are parametrized over array
  backends and devices via `from .config import pytestmark` — follow the
  existing modules for the pattern.
- The project aims to keep test coverage at ~100% (`pixi run test-cov`, and
  Codecov enforces no decrease). Genuinely untestable or examples-only code is
  excluded from the coverage target (e.g. `parallelproj._examples_utils`, via
  `[tool.coverage.run] omit` in `pyproject.toml` and `ignore` in `codecov.yml`).

## Documentation and examples

- Update [`docs/changelog.rst`](./docs/changelog.rst) for any user-facing change.
- The example gallery lives in `docs/examples/` (sphinx-gallery). Files named
  `NN_run_*.py` are **executed** when the docs are built (including on Read the
  Docs); other example files are rendered but not executed. Shared example
  helpers are imported from `parallelproj._examples_utils` (shipped inside the
  package) — no `PYTHONPATH` or separate download is needed.
- Build locally with `pixi run docs`, or `pixi run docs-fast` for a quick
  prose/structure check (it uses `-W --keep-going` to mirror Read the Docs
  strictness).

## Git hooks

The repository tracks its hooks in `.githooks/` (currently a pre-push
version/tag consistency check). They are activated automatically on the first
`pixi run`, or explicitly via `pixi run setup-hooks`.

## Pull request workflow

1. Create a topic branch and keep the PR focused on a single change.
2. Add or update tests, and make sure `pixi run test` passes and coverage holds.
3. Make sure `pixi run docs-fast` builds cleanly.
4. Update `docs/changelog.rst`.
5. Open a pull request describing the change and its motivation. CI runs the
   test suite and uploads coverage.

## Releases (maintainers)

Releases are tag-driven. Bump `[project].version` in `pyproject.toml`, then:

```bash
pixi run release-dry-run   # validate version, print what would happen
pixi run release           # tag HEAD and push (refuses if the tag already exists)
```

## Reporting issues

Please use the GitHub issue tracker for bug reports and feature requests. For
bugs, include a minimal reproduction, the parallelproj and `parallelproj-core`
versions, the array backend/device, and the full error message where relevant.
