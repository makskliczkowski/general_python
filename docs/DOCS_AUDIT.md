# Documentation Audit

## Scope

This audit covers the current state of documentation in `docs/` and inline docstrings within the codebase (`algebra/`, `lattices/`, `maths/`, `ml/`, `physics/`, `common/`).

## 1) Existing Documentation

### Pages
- **Getting Started** (`docs/getting_started.rst`): Covers installation, backend expectations, and verification.
- **Design Principles** (`docs/design_principles.rst`): Outlines scientific contracts, backend awareness, and coding standards.
- **API Reference** (`docs/api.rst`): Uses `automodule` to pull docstrings from source.
- **Benchmarks** (`docs/BENCHMARKS.md`): Documents performance tests.
- **Tests** (`docs/TESTS_CURRENT.md`): Inventory of tests.

### Build System
- **Local:** Sphinx (`Makefile`, `conf.py`). Run `make html` in `docs/`.
- **ReadTheDocs:** Configured in `.readthedocs.yaml` (uses Ubuntu 24.04, Python 3.12).

## 2) Issues Identified

- **Missing Module Docstrings:** Several key files were missing module-level docstrings, including:
  - `ml/net_impl/utils/net_utils_np.py`
  - `common/timer.py`
  - `physics/eigenlevels.py`
  - `physics/sp/__init__.py`
  - `maths/random.py`
  - `maths/statistics.py`
  - `lattices/hexagonal.py`
  - `algebra/utilities/pfaffian_jax.py`
  - `algebra/utilities/hafnian_jax.py`

- **Syntax Warnings:** Numerous `SyntaxWarning: invalid escape sequence` errors were present due to LaTeX sequences (e.g., `\sigma`, `\alpha`, `\Delta`) in normal string literals instead of raw strings. This affects python 3.12+ and can lead to incorrect rendering or runtime warnings.

## 3) Improvements Made

- **Added Docstrings:** Comprehensive module-level docstrings were added to the files listed above, detailing purpose, input/output contracts, and stability notes.
- **Fixed Syntax Warnings:** A targeted script was used to convert string literals containing invalid escape sequences into raw strings (`r"..."`). This covered:
  - LaTeX in docstrings (e.g., `r"""... \sigma ..."""`).
  - Regex patterns (e.g., `r"\d"`).
  - Scientific constants/symbols in comments or strings.
- **Validation:** Checked using `ast` parsing to ensure docstrings are present and no syntax warnings are emitted.

## 4) Current Status

- **Docstring Coverage:** significantly improved for core scientific modules.
- **Code Hygiene:** Source code is free of invalid escape sequence warnings.
- **Docs Build:** Ready for Sphinx build (locally and RTD).

## 5) Recommended Next Steps

- Add specific API examples in `docs/usage.rst`.
- Expand docstrings for `tests/` directories if needed (currently excluded from audit).
