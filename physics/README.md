# Physics Module

Finite-temperature and dynamical observables for quantum systems with NumPy/JAX compatibility.

## Core Features

**Thermal Physics** (`thermal.py`)
- Partition functions, Boltzmann weights, free energy
- Heat capacity, susceptibilities, temperature scans
- Numerically stable evaluations of $Z(\beta)$ and thermal averages

**Spectral Functions** (`spectral/`)
- Green's functions: `greens_function_quadratic`, `greens_function_manybody`, `greens_function_lanczos`
- Spectral functions from Green's functions or eigenvalues
- Momentum-resolved spectral functions $A(k,\omega)$
- Density of states with Lorentzian/Gaussian broadening

**Response Functions** (`response/`)
- Dynamic structure factors $S(q,\omega)$
- Generalized susceptibilities $\chi(q,\omega)$
- Unified interface for many-body and mean-field systems

**Entanglement** (`density_matrix.py`, `entropy.py`)
- Reduced density matrices, Schmidt decompositions
- Von Neumann, Rényi, and participation entropies
- JAX-accelerated versions available

**Single-Particle** (`sp/`)
- Correlation matrices for quadratic Hamiltonians
- Slater determinant and BdG quasiparticle support

**Utilities**
- `eigenlevels.py`: Level statistics (spacing ratios, unfolding)
- `statistical.py`: Windowed averaging, binning, jackknife estimators
- `operators.py`: Operator parsing and selection

## Quick Examples

```python
# Thermal properties
from QES.general_python.physics import thermal
Z = thermal.partition_function(energies, beta=1.0)
C_V = thermal.heat_capacity(energies, beta=1.0)

# Spectral functions
from QES.general_python.physics.spectral import spectral_function
A = spectral_function.spectral_function(omega=0.5, eigenvalues=E, eigenvectors=U, eta=0.01)

# Green's functions
from QES.general_python.physics.spectral.spectral_backend import greens_function_quadratic
G = greens_function_quadratic(omega, eigenvalues, eigenvectors, eta=0.01)

# Response functions
from QES.general_python.physics.response import unified_response
chi, method = unified_response.compute_response(E, V, operator, omega_grid)
```

## Module Organization

```
physics/
├── backend.py              # Unified interface to spectral & thermal
├── thermal.py              # Temperature-dependent properties
├── spectral/               # Green's functions & spectral functions
│   ├── spectral_backend.py # Core implementations
│   ├── greens_function.py  # Green's function wrappers
│   ├── spectral_function.py# Spectral function wrappers
│   └── dos.py              # Density of states
├── response/               # Dynamic response functions
│   ├── unified_response.py # Auto-selecting response calculator
│   ├── structure_factor.py # Structure factors
│   └── susceptibility.py   # Susceptibilities
├── sp/                     # Single-particle correlations
├── density_matrix.py       # Density matrices
├── entropy.py              # Entanglement entropies
└── statistical.py          # Data analysis utilities
```

## Mathematical Background

**Thermal averages**: $\langle O \rangle_\beta = Z^{-1} \sum_n O_{nn} e^{-\beta E_n}$

**Green's function**: $G(\omega) = [(\omega + i\eta)I - H]^{-1}$ or Lehmann representation

**Spectral function**: $A(\omega) = -\frac{1}{\pi} \mathrm{Im}\, G(\omega)$

**Susceptibility**: $\chi(\omega) = \sum_{m,n} \frac{(\rho_m - \rho_n)}{E_n - E_m - \omega - i\eta} |\langle m|O|n\rangle|^2$

**Structure factor**: $S(q,\omega) = \sum_{m,n} \rho_n |\langle m|S_q|n\rangle|^2 \delta(\omega - (E_m - E_n))$

---

Copyright © 2025 Maksymilian Kliczkowski. All rights reserved.
