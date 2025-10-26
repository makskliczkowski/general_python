# Physics Toolkit

`general_python.physics` hosts finite-temperature and dynamical observables for quantum lattice models.  Each module implements a clearly specified quantity and keeps NumPy/JAX compatibility.

## Core Equations

- **Canonical ensemble**: the partition function is $Z(\beta) = \sum_n e^{-\beta E_n}$ with inverse temperature $\beta = 1/(k_B T)$.  Thermal averages obey $\langle O \rangle_\beta = Z(\beta)^{-1} \sum_n O_{nn} e^{-\beta E_n}$.
- **Density matrices**: given a pure bipartite state $|\psi\rangle$, reduced states are $\rho_A = \mathrm{Tr}_B |\psi\rangle \langle \psi|$.  Von Neumann entropy is $S(\rho_A) = -\mathrm{Tr}(\rho_A \log \rho_A)$ and Rényi-$\alpha$ entropy is $S_\alpha(\rho_A) = (1-\alpha)^{-1} \log \mathrm{Tr}(\rho_A^\alpha)$.
- **Spectral data**: retarded Green's functions $G(\omega) = [(\omega + i \eta)I - H]^{-1}$ produce spectral densities $A(\omega) = -\pi^{-1}\mathrm{Im}\, G(\omega)$.  Densities of states are $D(\omega) = \sum_n \delta(\omega - E_n)$ with controllable broadening.
- **Dynamical response**: the spin structure factor is $S(\mathbf{q},\omega) = \sum_{f,i} \rho_i |\langle f|S_\mathbf{q}|i\rangle|^2 \delta(\omega - (E_f - E_i))$.  The susceptibility implemented in `response/susceptibility.py` evaluates
  $$
  \chi(\mathbf{q}, \omega) = \sum_{f,i} \frac{|\langle f|O_\mathbf{q}|i\rangle|^2 (\rho_i - \rho_f)}{\omega + E_i - E_f + i \eta}.
  $$
- **Correlation matrices**: for quadratic fermions the single-particle correlator obeys $C_{i j} = \langle \psi| c_i^\dagger c_j |\psi\rangle$, with subsystem restrictions $C_A = W_A^\dagger \mathrm{diag}(n) W_A$.

## Module Inventory
| Path | Observable / Operation | Mathematical focus |
|------|------------------------|--------------------|
| `thermal.py` | Partition sums, Boltzmann weights, free energy, heat capacity, susceptibilities. | Provides numerically stable evaluations of $Z(\beta)$, $\langle O \rangle_\beta$, and $\chi = \partial \langle O \rangle_\beta / \partial h$. |
| `density_matrix.py` / `density_matrix_jax.py` | Reduced density matrices, Schmidt decompositions, entanglement spectra. | Implements exact partial traces and SVD-based Schmidt coefficients for bipartite Hilbert spaces. |
| `entropy.py` / `entropy_jax.py` | Von Neumann, Rényi, and participation entropies. | Operates on eigenvalues $\lambda_i$ of $\rho$ via $S = -\sum_i \lambda_i \log \lambda_i$ and $S_\alpha$. |
| `eigenlevels.py` | Level statistics (spacing ratios, unfolding helpers). | Works with ordered spectra $\{E_i\}$ to compute $r_i = \min(\delta_i,\delta_{i+1})/\max(\delta_i,\delta_{i+1})$. |
| `operators.py` | Parsing and selection of simulation operators. | Determines Hilbert-space cutoffs and spectral windows used by response routines. |
| `statistical.py` | Windowed averaging, binning, and jackknife estimators. | Supplies unbiased estimators for Monte Carlo or exact-diagonalization data series. |

## Subpackages

- `spectral/`: constructs densities of states (`dos.py`), retarded/advanced Green's functions (`greens.py`), and spectral functions (`spectral_function.py`) via contour-regularized resolvents with Lorentzian or Gaussian kernels.
- `response/`: evaluates dynamic structure factors and generalized susceptibilities by summing matrix elements in the energy eigenbasis with adaptive broadening.
- `sp/`: builds single-particle correlation matrices for quadratic Hamiltonians, supporting pure Slater determinants and Bogoliubov–de Gennes quasiparticle occupations.

## Cross-References

- Mathematical derivations: `PHYSICS_MATH.md`.
- API guide and naming conventions: `PHYSICS_MODULE.md`.
- End-to-end usage notebooks and scripts: `PHYSICS_EXAMPLES.md`.

## Copyright
Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.
