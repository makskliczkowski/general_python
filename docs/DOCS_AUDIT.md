# Documentation Audit

## Scope

This audit covers Sphinx docs in `docs/`, package-level docstrings, and Read the Docs (RTD) configuration.

## 1) What documentation currently exists

### User-facing docs in `docs/`

- Core navigation and overview pages exist: `index.rst`, `introduction.rst`, `usage.rst`, `api.rst`, `contributing.rst`, `license.rst`.
- Setup and architecture pages exist: `installation.rst`, `getting_started.rst`, `design_principles.rst`.
- Process notes exist: `BUILD_AND_CI.md`, `TESTS_CURRENT.md`, `TESTS_NEW.md`, `BENCHMARKS.md`, `FUTURE_BRANCHES.md`.

### API surface documentation

- `docs/api.rst` uses `automodule` entries for major packages (`algebra`, `common`, `lattices`, `maths`, `ml`, `physics`) and selected submodules.
- Sphinx config enables autodoc + napoleon, so module and function docstrings are directly exposed in generated docs.

### Build configuration

- Local Sphinx build path is configured via `docs/conf.py` and `docs/Makefile`.
- RTD build path is configured in `.readthedocs.yaml` and points to `docs/conf.py`.

## 2) What is missing

- Some module-level docstrings were previously inconsistent in scientific contracts (shape, dtype, determinism) across top-level packages.
- The docs set did not clearly connect `pyproject.toml` optional extras (`docs`, `dev`, `ml`, `jax`) to quick-start workflows in one concise place.
- Backend expectations (NumPy baseline, optional JAX paths) were present but not consistently emphasized as operational guidance.

## 3) What is outdated or potentially confusing

- Earlier prose in audit notes referenced RTD environment details that no longer match current `.readthedocs.yaml` values.
  - Current RTD config uses Ubuntu 24.04 and Python 3.12.
- Existing architecture pages discussed backend behavior in broad terms but did not always spell out determinism caveats (floating-point ordering differences between backends).

## 4) How docs are built

### Local build

From repository root:

```bash
pip install -e ".[docs]"
cd docs
make html
```

Output:

- HTML site in `docs/_build/html/`
- Entry point: `docs/_build/html/index.html`

### Read the Docs build

RTD configuration source: `.readthedocs.yaml`.

- Uses config version 2.
- Uses Ubuntu 24.04 and Python 3.12.
- Uses Sphinx config at `docs/conf.py`.
- Installs package from repository root with `docs` extra and also installs `docs/requirements.txt`.

## 5) Improvements made in this task

- Refined module-level docstrings for:
  - `algebra`
  - `lattices`
  - `maths`
  - `ml`
  - `physics`
  - `common`
- Docstrings now consistently call out:
  - module purpose
  - input/output contracts
  - dtype and shape expectations
  - numerical stability notes
  - determinism and reproducibility caveats
- Refreshed quick-start and design pages (`getting_started.rst`, `design_principles.rst`) to emphasize:
  - installation paths from `pyproject.toml` extras
  - backend expectations (NumPy baseline, optional JAX)
  - practical test commands

## 6) Recommended next steps (non-blocking)

- Add small API examples for each major top-level package in `docs/usage.rst` that include explicit shapes and dtypes.
- Add a short troubleshooting page for backend mismatch and import-time optional dependencies.
- Add a CI docs job that runs `sphinx-build -W` to catch warning regressions early.
