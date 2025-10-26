# Structured Matrix Functionals

The `general_python.algebra.utilities` package implements Pfaffian and Hafnian evaluators for structured matrices that appear in Gaussian-state simulations and combinatorial enumeration.  NumPy and JAX variants share the same numerical algorithms.

## Definitions
- **Pfaffian**: For an antisymmetric matrix $A \in \mathbb{C}^{2n \times 2n}$,
  $$
  \mathrm{pf}(A) = \frac{1}{2^n n!} \sum_{\sigma \in S_{2n}} \mathrm{sgn}(\sigma) \prod_{j=1}^{n} A_{\sigma(2j-1),\, \sigma(2j)},
  $$
  and satisfies $\mathrm{pf}(A)^2 = \det(A)$.
- **Hafnian**: For a symmetric matrix $B \in \mathbb{C}^{2n \times 2n}$,
  $$
  \mathrm{haf}(B) = \sum_{M \in \mathcal{M}_{2n}} \prod_{(i,j) \in M} B_{i j},
  $$
  where $\mathcal{M}_{2n}$ is the set of perfect matchings.  Hafnians count pairings in bosonic Gaussian states.

## Module Map
| Path | Functionality | Numerical approach |
|------|---------------|--------------------|
| `pfaffian.py` | Pfaffian via Parlett-Reid style elimination with partial pivoting. | Reduces $A$ to block-triangular form $P^\top A P = \bigoplus_k \begin{bmatrix} 0 & d_k \\ -d_k & 0 \end{bmatrix}$ and multiplies $d_k$. |
| `pfaffian_jax.py` | JAX-compatible Pfaffian with custom gradients disabled. | Mirrors the NumPy algorithm using `jax.lax.fori_loop` to maintain determinism. |
| `hafnian.py` | Hafnian using Glynn’s formula with Gray-code traversal. | Computes $\mathrm{haf}(B) = \frac{1}{2^{n-1}} \sum_{\epsilon \in \{\pm1\}^{n-1}} \left(\prod_{k=1}^{n-1} \epsilon_k\right) \prod_{i=1}^{2n} \left(\sum_{j=1}^{2n} \epsilon_{\pi(j)} B_{i j}\right)$ with scaling safeguards. |
| `hafnian_jax.py` | JAX version of the Hafnian evaluator with vectorized Gray-code updates. | Uses `jax.lax.scan` to propagate accumulator states without Python loops. |

## Numerical Safeguards
- Pivoting strategies maintain antisymmetry; when swaps change sign, the running Pfaffian sign is updated explicitly.
- Scaling factors prevent overflow/underflow during Hafnian summations by normalizing intermediate vectors and tracking logarithmic weights.
- Functions accept optional dtype arguments; default complex dtype is promoted from input tensors (e.g., real antisymmetric matrices yield real Pfaffians).

## Usage Notes
- Pfaffians/Hafnians require even dimensions; routines raise `ValueError` for odd-sized inputs.
- Hafnian evaluation cost scales as $O(n^2 2^n)$.  For $n \gtrsim 8$ prefer approximations or Monte Carlo estimators (not yet included).
- JAX implementations can be transformed with `jax.jit` but are marked `inline=False` to keep compilation manageable.

## Cross-References
- Linear solver backends that consume Pfaffians (e.g., fermionic Gaussian overlaps): `../solvers/README.md`.
- Random matrix sampling utilities that generate antisymmetric inputs: `../ran_matrices.py`.
- High-level physics modules using Hafnians (bosonic correlators): see `../../physics/sp/correlation_matrix.py`.

## Copyright
Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.
