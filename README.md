[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/parallelproj.svg)](https://anaconda.org/conda-forge/parallelproj)
[![documentation](https://readthedocs.org/projects/parallelproj/badge/?version=stable)](https://parallelproj.readthedocs.io/en/stable/)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![codecov](https://codecov.io/gh/KUL-recon-lab/parallelproj/graph/badge.svg)](https://codecov.io/gh/KUL-recon-lab/parallelproj)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KUL-recon-lab/parallelproj/main?labpath=docs%2Fexamples)

<p align="center">
<img src="./docs/_static/parallelproj-logo.png" width="350">
</p>

**parallelproj - high-level python routines for tomographic reconstruction**


**parallelproj** provides simple and fast high-level python routines for tomographic reconstruction
that are [python array API](https://data-apis.org/array-api/latest/) 
compatible meaning that they can be used with a variety of python
array libraries (e.g. numpy, cupy, **pytorch**) and devices (CPU and CUDA GPUs).

<br/>

**If you are using parallelproj, we recommend to read and cite our publication** 
  - G. Schramm, K. Thielemans: "**PARALLELPROJ - An open-source framework for fast calculation of projections in tomography**", Front. Nucl. Med., Volume 3 - 2023, doi: 10.3389/fnume.2023.1324562, [link to paper](https://www.frontiersin.org/articles/10.3389/fnume.2023.1324562/abstract), [link to arxiv version](https://arxiv.org/abs/2212.12519)

<br/>

## Installation, Documentation & Examples (for users)

`parallelproj` is distributed through **conda-forge** (it cannot be installed with `pip`
alone — its compiled backend is only on conda-forge, not on PyPI):

```bash
mamba create -n parallelproj -c conda-forge parallelproj
```

Full installation instructions, the API reference and the example gallery are in the
official documentation on Read the Docs: **https://parallelproj.readthedocs.io/en/stable/**

You can also try the examples in your browser — no install — via
[Binder](https://mybinder.org/v2/gh/KUL-recon-lab/parallelproj/main?labpath=docs%2Fexamples),
or start from the minimal [quickstart example](./docs/quickstart_minimal.py).

<br/>

## Developer Setup

For development we use [pixi](https://pixi.sh), which builds a fully reproducible
environment from the lock file. The local clone is registered as a path dependency, so
`import parallelproj` resolves to your working tree (no manual environment setup or
`PYTHONPATH` needed).

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

Inside that shell you can run scripts, tests, or the examples directly (e.g.
`python <script>`, `ipython`, `pytest`). The environment is built automatically on first
use. Exit with `exit` or `Ctrl+D`.

Common tasks can also be run without an explicit shell via `pixi run`:

```bash
pixi run test        # run the test suite (GPU tests self-skip when no CUDA device is present)
pixi run docs        # build the HTML documentation
```

The first `pixi run` also activates the repo-tracked git hooks (a pre-push version/tag
check). Contributions are welcome via pull request.

> **Note:** this repository is the pure-Python front end. The compiled projection kernels
> live in the separate [`libparallelproj`](https://github.com/KUL-recon-lab/libparallelproj)
> project (distributed as `parallelproj-core` on conda-forge): pure-Python features go here,
> but changes to the C/CUDA Joseph projectors must be made there — editing this clone does
> not rebuild the backend.
