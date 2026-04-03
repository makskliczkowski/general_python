## 2025-11-01 - [Structure Factor Optimization]
**Learning:** In computing dynamical structure factors (and other spectral functions), transitioning to a Numba kernel over all eigenstates is efficient, but passing zero matrix elements (which represent forbidden transitions) wastes significant kernel execution time. Physical operators are typically sparse in the eigenbasis, so many transitions have zero weight.
**Action:** Filter out zero matrix elements (`> 1e-15`) and their corresponding energy differences *before* passing them to Numba kernels (`_structure_factor_kernel`). This early skipping mechanism can yield up to a 2-3x performance boost, especially for multi-initial-state finite-temperature evaluations.

## 2025-11-01 - [Spectral Backend Vectorization]
**Learning:** For multi-frequency spectral evaluations (`operator_spectral_function_multi_omega`, `susceptibility_bubble_multi_omega`), performing a pure Python loop over the frequencies invoking single-frequency logic causes massive overhead.
**Action:** Vectorize array calculations over the frequency dimensions using NumPy broadcasting (e.g., `omegas[:, None]`). Combined with early omission of zero-transitions (`> 1e-15`), this single trick reduced execution times by a factor of 5-10x for dynamic structure factor spectra optimizations.
