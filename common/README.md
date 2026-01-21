# Common Runtime Utilities

`general_python.common` collects supporting infrastructure that is reused across algebra, physics, and solver packages.  Utilities range from bit-level transformations to filesystem helpers and HDF5 adapters.  Everything is backend-aware (NumPy/JAX) where appropriate and written to be importable without pulling heavier dependencies.

## Capability Overview
- **Bitstring algebra** — Convert between integers, binary strings, and spin configurations, perform bit masking, and accelerate lookups with Numba/JAX.
- **Data aggregation** — Concatenate histogram outputs, average profiles across runs, and prune near-zero slices from matrices.
- **Filesystem orchestration** — Recursively create/remove directories, generate temporary workspaces, and stream datasets from HDF5 archives.
- **Plotting and display** — Produce publication-ready Matplotlib figures, render kets/bra expansions in LaTeX, and expose notebook-friendly printing helpers.
- **Diagnostics** — Structured logging, wall-clock timers with lap support, and lightweight parser convenience functions.

## Module Map
| Path | Focus | Notes |
|------|-------|-------|
| `binary.py` | Bit manipulation primitives (`check`, `flip`, Gray-code iterators) with NumPy/JAX backends. | Uses `common.embedded` kernels for low-level operations. |
| `datah.py` | Histogram concatenation, interpolation, and zero-slice pruning. | Designed for combining Monte Carlo observables before statistical analysis. |
| `directories.py` | Path wrapper with operator overloading (`/`, `+`) plus directory creation, removal, and size utilities. | Interoperates with `pathlib.Path`. |
| `display.py` | IPython/LaTeX rendering of many-body states and operators. | Supplies `ket`, `bra`, and coefficient formatters that respect phases. |
| `embedded/` | Low-level kernels shared by `binary.py`: see `embedded/README.md`. | Houses JAX-compatible routines. |
| `flog.py` | Facility logger that emits to console/files with rotating handlers. | Wraps Python `logging` but adds color/indent helpers. |
| `hdf5_lib.py`, `hdf5man.py` | HDF5 convenience layers for storing dense/sparse arrays with metadata. | Abstracts PyTables/h5py access patterns. |
| `parsers.py` | Simple token parsers and string utilities for configuration files. | Keeps dependencies minimal (pure Python). |
| `plot.py` | Matplotlib wrappers for line/heatmap plots with consistent styling. | Includes scientific unit formatters and colour maps. |
| `tests.py` | Shared assertions (e.g., approximate equality across backends). | Imported by `binary.py` and algebra tests. |
| `timer.py` | High-resolution stopwatch with laps, context-manager, and decorator support. | Uses `time.perf_counter_ns` for nanosecond resolution. |

## Patterns and Safety
- **Backend negotiation**: functions that accept `backend` route through `algebra.utils.get_backend`, so JAX arrays stay on device.
- **Numba acceleration**: the bit-extraction kernels in `binary.py` are marked with `@numba.njit(inline='always')` to remove call overhead in tight loops.
- **HDF5 writes**: managers add dataset integrity checks (shape/dtype) before overwriting to avoid silent corruption.
- **Timers/logging**: `Timer` can enforce deadlines via `deadline_s` and integrates with provided `logging.Logger` instances by default.

## Cross-References
- Algebra solvers consume the binary toolkit for basis encodings (`../algebra/README.md`).
- HDF5 helpers underpin data ingestion in physics response modules (`../physics/README.md`).
- Embedded kernels are documented separately: `embedded/README.md`.

## Copyright
Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.
