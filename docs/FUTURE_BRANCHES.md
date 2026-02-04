# Future Improvements and Structural Proposals

This document outlines observed structural issues and proposals for future refactoring.

## Packaging and Structure

### Root-level Package
The current structure uses the repository root as the `general_python` package source. This works but can be confusing and may include unintended files in the package if `MANIFEST.in` is not carefully maintained.
**Proposal:** Move source code into a `src/general_python` directory or a `general_python` subdirectory to isolate the package from repository configuration files.

## Import Heaviness

### Heavy Top-level Imports
Several modules import heavy dependencies (Numba, JAX, TensorFlow) at the top level.
- `ml/net_impl/net_general.py` imports `numba`.
- `ml/net_impl/interface_net_flax.py` imports `jax`.
- `ml/keras/__init__.py` conditionally imports `tensorflow` but does extensive imports if present.

**Proposal:** Use lazy imports or move imports inside functions/classes where possible to reduce startup time for users who only use a subset of features (e.g., only `algebra` with NumPy).

### Numba Dependency
`numba` is a core dependency used in `common`, `maths`, and `physics`. It significantly increases installation size and complexity.
**Proposal:** Evaluate if Numba usage can be optional or if it can be replaced by NumPy vectorization for lighter-weight installations.
