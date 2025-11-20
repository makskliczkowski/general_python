# Embedded Kernels

`general_python.common.embedded` provides low-level routines that accelerate the higher-level `binary` utilities and remain safe for both NumPy and JAX execution.

## Modules
| Path | Purpose | Notes |
|------|---------|-------|
| `binary_jax.py` | JAX implementations of bit flips, Gray-code walks, and population counts. | Uses `jax.numpy` primitives so callbacks remain JIT-compilable. |
| `binary_search.py` | Search helpers for locating bit positions and computing prefix masks. | Shared by NumPy and JAX code paths. |
| `bit_extract.py` | Numba-friendly extraction of bit ranges and masks from integers. | Exported functions are imported in `common.binary` as `extract`. |

## Usage Guidance
- Importers should treat these modules as internal building blocks; end users interact through `common.binary`.
- Functions avoid side effects and operate on plain integers or array types supplied by the caller, which keeps them compatible with JIT compilation and vectorization.

## Cross-References
- High-level overview and usage examples: `../README.md`.
- Consumers inside algebra/physics packages: see `../binary.py` docstrings.

## Copyright
Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.
