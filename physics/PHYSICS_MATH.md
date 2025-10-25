# Physics Module: Mathematical Background

This document describes the main mathematical objects and algorithms implemented in the physics module. For structure and API, see PHYSICS_MODULE.md. For code examples, see PHYSICS_EXAMPLES.md.

## Partition Function

The partition function $Z(\beta)$ is given by:

$$
Z(\beta) = \sum_i e^{-\beta E_i}
$$

where $E_i$ are the energy eigenvalues and $\beta$ is the inverse temperature.

## Thermal Averages

The thermal average of an observable $O$ is:

$$
\langle O \rangle = \frac{\sum_i O_i e^{-\beta E_i}}{Z(\beta)}
$$

where $O_i$ are the observable values in the energy basis.

## Green's Function

The retarded Green's function $G(\omega)$ is:

$$
G(\omega) = [(\omega + i\eta)I - H]^{-1}
$$

where $H$ is the Hamiltonian and $\eta$ is a small broadening parameter.

## Spectral Function

The spectral function $A(k, \omega)$ is:

$$
A(k, \omega) = -\frac{1}{\pi} \mathrm{Im} G(k, \omega)
$$

## Structure Factor

The dynamic structure factor $S(q, \omega)$ is:

$$
S(q, \omega) = \sum_{m,n} |\langle m | S_q | n \rangle|^2 \delta(\omega - (E_m - E_n)) p_n
$$

where $p_n$ is the Boltzmann weight of state $n$.

## Susceptibility

The dynamical susceptibility $\chi(q, \omega)$ is:

$$
\chi(q, \omega) = \sum_{m,n} \frac{|\langle m | O_q | n \rangle|^2 (p_n - p_m)}{\omega + E_n - E_m + i\eta}
$$

See PHYSICS_MODULE.md for structure and API, and PHYSICS_EXAMPLES.md for code examples.

## Copyright

Copyright (c) 2025 Maksymilian Kliczkowski. All rights reserved.
