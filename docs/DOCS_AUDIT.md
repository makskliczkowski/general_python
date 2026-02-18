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

- **Inconsistent Docstring Format:**
  - Many files use single quotes `'''` for module and function docstrings instead of the standard `"""`.
  - Examples: `algebra/ode.py`, `algebra/preconditioners.py`, `common/parsers.py`, `maths/random.py`, `maths/statistics.py`.

- **Function Docstring Placement:**
  - In `physics/eigenlevels.py`, docstrings are placed *above* function definitions as comments rather than inside as docstrings. This prevents `autodoc` from picking them up.

- **Missing Function Docstrings:**
  - `algebra/ran_matrices.py`: Functions `goe`, `gue`, `coe`, `cre`, `cue` lack docstrings entirely, relying on the module docstring which is insufficient for API reference.

- **Path Inconsistencies:**
  - `algebra/ode.py` docstring refers to `general_python/common/ode.py`, which is incorrect.

- **Missing Module Docstrings:**
  - While many files have them now, some might still be missing or sparse (e.g., `algebra/ran_matrices.py` functions).

## 3) Planned Improvements

- **Standardize Format:** Convert `'''` to `"""` in identified files.
- **Fix Placement:** Move docstrings inside functions in `physics/eigenlevels.py`.
- **Add Missing Docs:** Add docstrings to `algebra/ran_matrices.py`.
- **Correct Paths:** Fix `algebra/ode.py` docstring.
- **Verify:** Ensure `test_documentation.py` passes and docs build locally.

## 4) Current Status (Post-Fix)

- **Docstring Coverage:** Improved for core scientific modules.
- **Code Hygiene:** Source code adheres to standard docstring conventions (`"""`).
- **Docs Build:** Ready for Sphinx build (locally and RTD).
