[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KUL-recon-lab/parallelproj/main?labpath=docs%2Fexamples)
[![codecov](https://codecov.io/gh/KUL-recon-lab/parallelproj/graph/badge.svg)](https://codecov.io/gh/KUL-recon-lab/parallelproj)

<p align="center">
<img src="./docs/_static/parallelproj-logo.png" width="350">
</p>

**parallelproj - high-level python routines for tomographic reconstruction**


**parallelproj** provides simple and fast high-level python routines for tomographic reconstruction
that are [python array API](https://data-apis.org/array-api/latest/) 
compatible meaning that they can be used with a variety of python
array libraries (e.g. numpy, cupy, **pytorch**) and devices (CPU and CUDA GPUs).

</br>

**If you are using parallelproj, we recommend to read and cite our publication** 
  - G. Schramm, K. Thielemans: "**PARALLELPROJ - An open-source framework for fast calculation of projections in tomography**", Front. Nucl. Med., Volume 3 - 2023, doi: 10.3389/fnume.2023.1324562, [link to paper](https://www.frontiersin.org/articles/10.3389/fnume.2023.1324562/abstract), [link to arxiv version](https://arxiv.org/abs/2212.12519)

</br>

## Installation, Documentation & Examples (for users)

Installation instructions, the API reference and the example gallery are in the official
documentation on Read the Docs: **https://parallelproj.readthedocs.io/en/stable/**

</br>

## Developer Setup (pixi)

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
