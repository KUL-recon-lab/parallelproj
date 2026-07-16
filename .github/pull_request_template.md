<!--
Thanks for contributing to parallelproj!
See CONTRIBUTING.md for setup, conventions and the full workflow.
Keep PRs focused on a single change.
-->

## Summary

<!-- What does this PR change, and why? Link any related issue (e.g. "Fixes #123"). -->

## Checklist

- [ ] Tests added/updated and `pixi run test` passes (parametrized over `(xp, dev)` where relevant)
- [ ] Coverage holds (`pixi run test-cov`)
- [ ] Code is array-API compatible (works across NumPy / array_api_strict / PyTorch / CuPy)
- [ ] Docs build cleanly (`pixi run docs-fast`)
- [ ] `docs/changelog.rst` updated for user-facing changes
- [ ] This change belongs in the pure-Python front end (C/CUDA kernel changes go in `libparallelproj`)
