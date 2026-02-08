# Future Development and Structural Improvements

This document outlines structural issues and potential improvements for the `general_python` codebase.

## Structural Pain Points

### Flat Layout and Package Discovery

The current project uses a "flat" layout where the root directory (`.`) is mapped to the `general_python` package.
This is configured explicitly in `pyproject.toml` using `package-dir = {general_python = "."}`.

**Issue**:
- New submodules must be manually added to the `packages` list in `pyproject.toml`.
- Mixing source code with configuration files (`pyproject.toml`, `README.md`, etc.) in the root directory can be confusing for packaging tools and developers.

**Proposal**:
- Move source code to a `src/` directory (src-layout).
- Use `find:` directive in `pyproject.toml` (or `setuptools.find_packages()`) to automatically discover packages.

### Heavy Imports in `__init__.py`

The root `__init__.py` attempts to eagerly import submodules (`algebra`, `common`, `lattices`, `maths`, `physics`, `ml`) inside a `try/except` block for CI compatibility.

**Issue**:
- Eagerly importing all submodules can lead to slow startup times if dependencies are large (e.g., `tensorflow`, `jax`).
- It defeats the purpose of the lazy loading mechanism implemented via `__getattr__`.
- If optional dependencies are missing, the eager import fails silently, potentially masking issues until runtime access.

**Proposal**:
- Remove the eager `try/except` import block in `__init__.py` and rely fully on lazy loading.
- Ensure that CI environments explicitly import modules they need to test.

### Unguarded Test Imports

Some tests (e.g., `tests/test_activations.py`, `algebra/utilities/tests/test_pfaffian.py`) import optional dependencies like `jax` at the module level without checking for their presence.

**Issue**:
- Running `pytest` without optional dependencies installed results in `ImportError` during test collection, failing the entire test suite.
- This forces users to install all optional dependencies even if they only want to test the core functionality.

**Proposal**:
- Use `pytest.importorskip("jax")` or similar guards inside test files or functions.
- Mark tests requiring optional dependencies with custom markers (e.g., `@pytest.mark.jax`) so they can be deselected easily (`pytest -m "not jax"`).

### Circular Dependencies

(Placeholder: If any circular dependencies are identified during future audits, document them here.)

## Maintenance

When addressing these issues, ensure backward compatibility is maintained or clearly communicated.
Moving to a `src` layout is a breaking change for editable installs if not handled carefully, but is generally recommended for modern Python packaging.
