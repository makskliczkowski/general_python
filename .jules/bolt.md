## 2025-11-01 - [Structure Factor Optimization]
**Learning:** In computing dynamical structure factors (and other spectral functions), transitioning to a Numba kernel over all eigenstates is efficient, but passing zero matrix elements (which represent forbidden transitions) wastes significant kernel execution time. Physical operators are typically sparse in the eigenbasis, so many transitions have zero weight.
**Action:** Filter out zero matrix elements (`> 1e-15`) and their corresponding energy differences *before* passing them to Numba kernels (`_structure_factor_kernel`). This early skipping mechanism can yield up to a 2-3x performance boost, especially for multi-initial-state finite-temperature evaluations.

## 2025-11-01 - [Avoid Temporary Array Allocations in Reductions]
**Learning:** In loops over parameters like temperature (`thermal_scan`), doing `np.sum(arr1 * arr2)` where array sizes match the system Hilbert space creates huge temporary arrays per loop iteration. Memory allocation dominates execution time.
**Action:** Use `np.dot(arr1, arr2)` instead of `np.sum(arr1 * arr2)` for 1D arrays to evaluate reductions in C-level BLAS/NumPy functions, bypassing Python/NumPy array allocations and boosting performance significantly with less peak memory.
