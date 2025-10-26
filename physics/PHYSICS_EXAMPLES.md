
# Physics Module Usage Examples

This document provides code examples for the main features of the physics module. For structure and API, see PHYSICS_MODULE.md. For mathematical background, see PHYSICS_MATH.md.

## Thermal Physics Example

This example calculates the partition function $Z(\beta)$, internal energy $U$, and heat capacity $C_V$ for a simple three-level system at $\beta=1.0$.

```python
from general_python.physics import thermal
import numpy as np

energies    = np.array([-1.0, 0.0, 1.0])
Z           = thermal.partition_function(energies, beta=1.0)
U           = thermal.internal_energy(energies, beta=1.0)
C_V         = thermal.heat_capacity(energies, beta=1.0)
print(f"Partition function: {Z}, Internal energy: {U}, Heat capacity: {C_V}")
```


## Spectral Function Example

This example computes the retarded Green's function $G(\omega)$ and the spectral function $A(\omega) = -1/\pi \mathrm{Im}\, G(\omega)$ for a simple diagonal Hamiltonian at $\omega=0.5$.

```python
from general_python.physics.spectral import dos, greens, spectral_function
import numpy as np

# Eigenvalues and eigenvectors for a simple system
E       = np.linspace(-2, 2, 10)
V       = np.eye(10)
omega   = 0.5
G       = greens.greens_function_eigenbasis(omega, E, V, eta=0.01)
A       = spectral_function.spectral_function(G)
print(f"Spectral function at \omega ={omega}: {A}")
```


## Structure Factor Example

This example shows how to calculate the dynamic structure factor $S(q,\omega)$ for a quantum spin system, given the ground state, eigenvalues, eigenvectors, and a spin operator at momentum $q$.

```python
from general_python.physics.response import structure_factor
# ...setup ground state, eigvals, eigvecs, operator, omega_grid...
# S_q_omega = structure_factor.structure_factor_spin(gs, eigvals, eigvecs, spin_q_op, omega_grid, eta=0.05)
```

## Quadratic Thermal Example

This example computes the thermodynamic quantities (chemical potential, internal energy, heat capacity, etc.) for a quadratic non-interacting fermion system as a function of temperature, fixing the particle number.

```python
from QES.Algebra.Properties import quadratic_thermal
import numpy as np

epsilon = np.linspace(-2, 2, 100)       # energies
temps   = np.linspace(0.1, 5.0, 50)     # temperatures (inverse)
results = quadratic_thermal.quadratic_thermal_scan(epsilon, temps, particle_type='fermion', particle_number=50)
print(results)

```

## Copyright

Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.

See [PHYSICS_MODULE.md](./PHYSICS_MODULE.md) for structure/API and [PHYSICS_MATH.md](./PHYSICS_MATH.md) for mathematical background. For combined workflows that stitch algebra + physics together, see [EXAMPLES.md](../../../../EXAMPLES.md).
