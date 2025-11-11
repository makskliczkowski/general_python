# Response Function Stack — Comprehensive Physics Guide

**PyQuSolver Complete Framework for Spectral Functions, Response Functions & Transport**

A unified theoretical and computational guide explaining response functions from first principles, their physical necessity, mathematical foundations, and computational implementations for both exact many-body and quadratic systems.

-----

## Table of Contents

1. [Fundamental Theory](#fundamental-theory)
2. [Why These Quantities Matter](#why-these-quantities-matter)
3. [The Two Computational Regimes](#the-two-computational-regimes)
4. [Complete Function Reference](#complete-function-reference)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Practical Examples](#practical-examples)
7. [Implementation Guide](#implementation-guide)
8. [Verification & Sum Rules](#verification--sum-rules)

-----

## Fundamental Theory

### Linear Response & the Kubo Formula

The foundation of all response functions is **linear response theory**. When a quantum system with Hamiltonian $H_0$ experiences a small time-dependent perturbation $V(t)$:

$$H(t) = H_0 + V(t)$$

the expectation value of an observable $A$ changes according to:

$$\langle A(t) \rangle = \langle A \rangle_0 + \int_{-\infty}^{t} dt' \chi_{AB}(t - t') \left\langle \frac{\partial V(t')}{\partial B} \right\rangle$$

This **linear susceptibility** $\chi_{AB}(t)$ relates the response of $A$ to a perturbation coupled to $B$. In frequency space (via Fourier transform), this is:

$$\chi_{AB}(\omega) = \frac{i}{\hbar} \int_0^{\infty} dt e^{i(\omega + i\eta)t} \langle [A(t), B(0)] \rangle$$

**Physical origin:** This is the **Kubo formula**. It measures how the commutator between $A$ (in the Heisenberg picture) and $B$ (at $t=0$) evolves in time. The key insight is that response functions encode the *correlation functions* of the system.

### Lehmann Representation

For a system with Hamiltonian $H$ with eigenvalues $E_n$ and eigenstates $|n\rangle$, the susceptibility can be written in the **Lehmann representation**:

$$\chi_{AB}(\omega) = \sum_{m,n} \frac{(\rho_m - \rho_n)}{E_n - E_m - \omega - i\eta} \langle m|A|n \rangle \langle n|B|m \rangle$$

where:

* $\rho_n = \exp(-\beta(E_n - E_0))/Z$ is the Boltzmann weight.
* $\eta$ is an infinitesimal positive broadening.
* The sum runs over **all many-body eigenstates**.

**Physical interpretation:**

* Each term represents a transition from state $|m\rangle$ to state $|n\rangle$.
* **Height** is proportional to the transition amplitude product $\langle m|A|n \rangle \langle n|B|m \rangle$ and the thermal occupation difference $(\rho_m - \rho_n)$.
* **Position** is at $\omega = E_n - E_m$, the energy required for the transition.
* **Occupation difference** $(\rho_m - \rho_n)$ ensures that transitions only happen from a more populated state to a less populated one (for absorption).

### Spectral Function from Green's Function

The **single-particle Green's function** is defined as:

$$G_{ij}(\omega) = \langle i | (\omega + i\eta - H)^{-1} | j \rangle$$

The **spectral function** $A_{ij}(\omega)$ is extracted from its imaginary part:

$$A_{ij}(\omega) = -\frac{1}{\pi} \text{Im} G_{ij}(\omega + i\eta)$$

**Physical content:** $A(\omega)$ represents the density of excitations you can create or destroy at energy $\omega$ by acting with a creation $a_i^\dagger$ or annihilation $a_i$ operator.

**Why it matters:**

* Directly observable in Angle-Resolved Photoemission Spectroscopy (ARPES).
* Shows the "band structure" of the interacting system.
* **Peak positions** = quasiparticle excitation energies.
* **Peak width** = inverse lifetime of the excitation, $\Gamma$, where width $\propto \eta + \Gamma$.

-----

## Why These Quantities Matter

### 1\. **Thermal Weights** — Foundation of Everything at Finite $T$

**Where it comes from:** Statistical mechanics. At thermal equilibrium, the probability of finding the system in an eigenstate $|n\rangle$ with energy $E_n$ is:

$$\rho_n = \frac{e^{-\beta E_n}}{Z}, \quad Z = \sum_n e^{-\beta E_n}$$

where $\beta = 1/(k_B T)$ and $Z$ is the partition function.

**Why it's necessary:**

* All observables at finite temperature $T$ depend on which states are populated.
* The mean value of an operator $A$ is $\langle A \rangle = \sum_n \rho_n \langle n|A|n \rangle$.
* Response functions (Lehmann representation) explicitly include these thermal factors.
* At $T=0$, only the ground state contributes ($\rho_0 = 1$). At $T>0$, the entire thermal spectrum matters.

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import thermal_weights
# Returns ρ_n normalized so Σ_n ρ_n = 1
rho = thermal_weights(eigenvalues, temperature)
```

**Physical example:** In a metal at room temperature, the thermal energy $k_B T \approx 25 \text{ meV}$ is small but non-zero, affecting transport by exciting electrons near the Fermi surface. This is essential for semiconductors, where $T$ relative to the gap energy dictates all properties.

-----

### 2\. **Single-Particle Spectral Function $A(\omega)$** — Band Structure & DOS

**Where it comes from:** The Green's function formalism.

$$A_{ii}(\omega) = -\frac{1}{\pi} \text{Im} G_{ii}(\omega + i\eta)$$

**Why it's necessary:**

* **Photoemission (ARPES):** Directly measures $A(k, \omega)$, showing the energy-momentum band structure.
* **Density of States (DOS):** The total DOS is $\rho(\omega) = \sum_i A_{ii}(\omega)$, representing the total number of available states per unit energy.
* **Transport:** Conductivity and other transport coefficients depend on $A(\omega)$ near the Fermi level.
* **Quantum Oscillations:** The frequency content of $A(\omega)$ at the Fermi level reveals the Fermi surface geometry.

**Many-body vs. single-particle:**

* **Single-particle:** $A(\omega)$ from a one-body (quadratic) Hamiltonian shows the "bare" bands.
* **Many-body exact:** $A(\omega)$ from the full interacting Hamiltonian shows quasiparticle bands, satellite peaks, and lifetime broadening effects from interactions.

**Physical example:** In graphene, the single-particle $A(k, \omega)$ shows sharp, linear Dirac cones crossing at the $K$ point. With interactions, these peaks broaden (finite lifetime) and can be renormalized.

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import spectral_function_multi_omega
# omega-dependent density of single-particle excitations
A = spectral_function_multi_omega(omegas, eigenvalues, eigenvectors)
```

-----

### 3\. **Operator-Projected Spectral Function $A_O(\omega)$** — Many-Body Response

**Where it comes from:** The many-body Lehmann representation, projected onto a specific operator $O$. This is often defined as the imaginary part of the susceptibility.

$$A_O(\omega) = \sum_{m,n} (\rho_m - \rho_n) |\langle m|O|n \rangle|^2 \delta(\omega - (E_n - E_m))$$

*(In computation, the $\delta$-function is replaced by a Lorentzian of width $\eta$.)*

**Why it's necessary:**

* **Extends beyond single particles:** Can measure the response of *any* operator (e.g., total spin $S_z$, charge density $n_q$, current $j$).
* **Shows many-body excitations:** Peaks appear at collective mode energies (magnons, excitons) and multi-particle continua, not just single-particle energies.
* **Captures correlations:** The matrix elements $\langle m|O|n \rangle$ include all interaction effects.
* **Finite temperature:** Thermal populations are automatically included.

**Many-body exclusive:** This quantity cannot be obtained from single-particle Hamiltonians; it requires the full $2^L \times 2^L$ (for spin-1/2) Hilbert space.

**Physical example:** In a strongly-correlated Hubbard model:

* $A_{c_i}(\omega)$ (single-particle) shows the upper and lower Hubbard bands, separated by the Mott gap.
* $A_{S_z}(\omega)$ (spin) shows the magnon (spin-wave) excitations in the ordered state.

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import operator_spectral_function_multi_omega
# Many-body spectral function for arbitrary operator O
A_O = operator_spectral_function_multi_omega(
    omegas, eigenvalues, eigenvectors, operator, 
    temperature=T
)
```

-----

### 4\. **Linear Response Susceptibility $\chi(\omega)$** — How System Responds

**Where it comes from:** The full Kubo formula (Lehmann representation).

$$\chi_{AB}(\omega) = \sum_{m,n} \frac{(\rho_m - \rho_n) \langle m|A|n \rangle \langle n|B|m \rangle}{\omega - (E_n - E_m) + i\eta}$$

**Why it's necessary:**

* **Fundamental definition** of "response." It tells you the system's reaction to a perturbation.
* **Complex function:**
  * $\text{Re}[\chi(\omega)]$ (Real part) = reactive response (e.g., energy level shifts, screening).
  * $\text{Im}[\chi(\omega)]$ (Imaginary part) = dissipative response (e.g., absorption, heating).
* **Links to experiments:**
  * Magnetic susceptibility $\chi_M$: Response to a magnetic field.
  * Charge susceptibility $\chi_c$: Response to an electric field (compressibility).
  * Shear viscosity $\eta$: Response to shear stress.
  * All inelastic scattering (Neutron, Raman, X-ray) cross-sections are proportional to $\text{Im}[\chi(\omega)]$ via the Fluctuation-Dissipation Theorem.

**Sum rules:** $\chi(\omega)$ must obey fundamental constraints, such as:

$$\int_{-\infty}^{\infty} d\omega \text{ Im}[\chi(\omega)] = \text{Constant} \times \langle [A,B] \rangle$$

This **sum rule** provides an exact, non-perturbative constraint on the total spectral weight.

**Physical example:**

* Paramagnet: $\chi_M(0) \propto 1/T$ (Curie's Law).
* Superconductor: $\chi_M(0) = -1/(4\pi)$ (Meissner effect, perfect diamagnetism).
* Fermi liquid: $\chi_c(0) \propto \rho(E_F)$ (Pauli susceptibility, density of states at $E_F$).

**In our code:**

```python
# Many-body exact (Lehmann)
from QES.general_python.physics.response.susceptibility import susceptibility_lehmann
chi = susceptibility_lehmann(E, V, operator_A, operator_B, omega, T)

# Note: Our A_O(ω) is related by A_O(ω) = -Im[χ_OO]/π
```

-----

### 5\. **Bubble Susceptibility $\chi^0(\omega)$** — Single-Particle Approximation

**Where it comes from:** The Lindhard function for non-interacting (or mean-field) fermions.

$$\chi_0(\omega) = \sum_{m,n} \frac{(f_m - f_n) |V_{mn}|^2}{\omega - (E_n - E_m) + i\eta}$$

where:

* $E_m, E_n$ are **single-particle** energies (NOT many-body).
* $f_m, f_n$ are single-particle occupations (Fermi-Dirac distribution).
* $V_{mn}$ is the vertex (e.g., identity for density-density, velocity for current-current).

**Why it's necessary:**

* **Computational efficiency:** Scales as $O(L^2)$ or $O(L^3)$, not $O((2^L)^3)$.
* **Exact for non-interacting systems:** This is the correct answer when interactions are zero.
* **RPA starting point:** The Random Phase Approximation (RPA) is built upon the bubble: $\chi_{RPA} = \chi_0 / (1 - V \chi_0)$.
* **Scales to large systems:** Can be computed for $L \sim 1000+$ sites.

**Single-particle origin:** This calculation does *not* include correlations beyond the mean-field level. It misses bound states, Mott gaps, and other genuine many-body phenomena.

**Physical example:**

* Tight-binding chain: $\chi_0(\omega)$ shows Van Hove singularities at the band edges.
* 2D free electron gas: $\chi_0(\omega)$ shows the Kohn anomaly, an instability towards a Charge Density Wave (CDW).

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import susceptibility_bubble_multi_omega
# Fast bubble from single-particle energies and occupations
chi0 = susceptibility_bubble_multi_omega(
    omegas, E_single_particle, vertex=None, occupation=f
)
```

-----

### 6\. **Kubo Conductivity $\sigma(\omega)$** — Charge Transport

**Where it comes from:** The linear response of the current operator $j$ to an electric field.

$$\sigma(\omega) = \frac{1}{i\omega} \chi_{jj}(\omega)$$

where $j$ is the current operator (e.g., $j = \sum_i v_i c_i^\dagger c_i$).

**Why it's necessary:**

* **Experimental observable:** Directly measured via optical reflectance/transmittance.
* **Transport physics:**
  * $\text{Re}[\sigma(\omega)]$ = dissipation (Joule heating, absorption).
  * $\text{Im}[\sigma(\omega)]$ = reactive part (screening).
* **Drude weight:** The $\omega \rightarrow 0$ limit. A delta-function peak $\sigma(\omega) = D \delta(\omega)$ indicates a perfect conductor (metal at $T=0$). $D$ is the Drude weight.
* **Interband transitions:** Peaks at $\omega > 0$ correspond to exciting electrons across band gaps.

**Many-body vs. single-particle:**

* **Many-body:** The full $\sigma(\omega)$ includes all correlations, showing correct damping and interaction-driven gap features (e.g., Mott gap).
* **Bubble:** $\sigma^0(\omega)$ (from the Kubo-Greenwood formula) shows the bare Drude peak and interband transitions.

**Physical example:**

* Metal: $\text{Re}[\sigma(\omega)]$ has a Drude peak centered at $\omega=0$.
* Semiconductor: $\text{Re}[\sigma(\omega)]$ is zero for $\omega < E_{\text{gap}}$, then rises sharply (optical gap).
* Superconductor: $\sigma(\omega)$ has a $\delta(\omega)$ function (zero resistance) and a gap $2\Delta$ for $\omega > 0$.

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import conductivity_kubo_bubble
# Computes \sigma(ω) using the bubble approximation
sigma = conductivity_kubo_bubble(omega, E_sp, velocity_matrix, occupation=f)
```

-----

### 7\. **Kramers-Kronig Relations** — Causality & Consistency

**Where it comes from:** The principle of **causality** (a response cannot precede its cause). Mathematically, this means $\chi(t)$ must be zero for $t < 0$.

$$\text{Re}[\chi(\omega)] = \frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} d\omega' \frac{\text{Im}[\chi(\omega')]}{\omega' - \omega}$$
$$\text{Im}[\chi(\omega)] = -\frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} d\omega' \frac{\text{Re}[\chi(\omega')]}{\omega' - \omega}$$
(where $\mathcal{P}$ denotes the principal value of the integral)

**Why it's necessary:**

* **Fundamental consistency check:** If your calculated $\text{Re}[\chi]$ and $\text{Im}[\chi]$ do not satisfy this, your calculation is physically incorrect (violates causality).
* **Determines full response from half:** If you measure the absorption ($\text{Im}[\chi]$) at all frequencies, you can calculate the reactive part ($\text{Re}[\chi]$) for free.
* **Sum rules emerge:** Integrating these relations gives constraints on the total spectral weight.

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import kramers_kronig_transform
# If this result matches your directly-calculated Re[χ], 
# then causality is verified.
Re_chi_from_Im = kramers_kronig_transform(Im_chi, omega_grid)
```

-----

## The Two Computational Regimes

### Regime I: Exact Many-Body (Full Diagonalization)

**Applicable to:** Small systems, typically $L \le 12$ sites (Hilbert space dimension $N = 2^L \approx 4096$).

**Method:**

1. Build the full many-body Hamiltonian $H$ as an $N \times N$ matrix.
2. Numerically diagonalize $H$ to get all $N$ eigenvalues $E_n$ and eigenvectors $|n\rangle$. Cost: $O(N^3) = O((2^L)^3)$.
3. Compute all matrix elements $\langle m|O|n \rangle$ for the operator $O$.
4. Use the Lehmann formula (sum over all $N^2$ pairs) to compute $\chi(\omega)$ or $A_O(\omega)$.

**Advantages:**

* ✅ **Exact:** No approximations. It is the numerically exact solution.
* ✅ **Any operator:** Works for spin, charge, current, or any exotic operator.
* ✅ **All correlations:** Includes everything: bound states, collective modes, magnons, Mott physics, etc.
* ✅ **Finite $T$:** Natively handles thermal effects.
* ✅ **Transparent physics:** Each peak in the spectrum corresponds to a real transition $|m\rangle \rightarrow |n\rangle$.

**Limitations:**

* (error) **Exponential scaling:** Impossible for $L > 15$ (storage) or $L > 18$ (computation time).
* (error) **Memory intensive:** Must store the $N \times N$ eigenvectors.
* (error) **Limited to small clusters:** Finite-size effects can be significant.

**Example:** 4-site Hubbard model
$$H = -t \sum_{\langle i,j \rangle, \sigma} c_{i,\sigma}^\dagger c_{j,\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}$$

* Diagonalize the $2^4 = 16$ states (or $70$ states in the $S_z=0$, $N=4$ sector).
* Get all 16 eigenvalues.
* Compute $A_O(\omega)$ exactly.

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import operator_spectral_function_multi_omega

# H_full is the 2^L times  2^L matrix
E, V = np.linalg.eigh(H_full)
A_O = operator_spectral_function_multi_omega(
    omegas, E, V, operator,
    temperature=T, eta=0.05
)
# Exact many-body result for any operator O
```

-----

### Regime II: Quadratic Approximation (Single-Particle)

**Applicable to:** Large systems ($L$ up to $\sim 1000+$); **only when interactions are weak, absent, or treated at a mean-field level.**

**Method:**

1. Build the single-particle Hamiltonian $H_{sp}$ as an $L \times L$ matrix.
2. Diagonalize $H_{sp}$ to get $L$ single-particle energies $E_\alpha$ and occupations $f_\alpha$. Cost: $O(L^3)$.
3. Use the bubble formula ($\chi_0$) with these single-particle quantities.

**Advantages:**

* ✅ **Fast:** Polynomial $O(L^3)$ scaling.
* ✅ **Large systems:** Can easily handle $L \sim 100-1000$ sites.
* ✅ **Exact for quadratic:** This is the *exact* solution for non-interacting fermions/bosons.
* ✅ **Standard approximation:** The foundation for RPA and other diagrammatic techniques.
* ✅ **Scalable:** Can be used for realistic lattice sizes and geometries.

**Limitations:**

* (error) **No genuine many-body excitations:** Misses bound states, Mott gaps, etc.
* (error) **Correlation effects missing:** Cannot describe phenomena driven by strong interactions.
* (error) **Only single-particle excitations** (particle-hole pairs) are visible.

**Example:** Tight-binding chain
$$H = -t \sum_{\langle i,j \rangle} c_i^\dagger c_j$$

* Diagonalize the $L \times L$ matrix.
* Get $E_k = -2t \cos(k)$ for $k = 2\pi n / L$.
* Compute $\chi_0(\omega)$ (the Lindhard function).

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import susceptibility_bubble_multi_omega

# H_single_particle is the L times  L matrix
E_sp = np.linalg.eigvals(H_single_particle)
f = 1/(1 + np.exp(beta * E_sp))  # Fermi-Dirac occupations
chi0 = susceptibility_bubble_multi_omega(
    omegas, E_sp, occupation=f
)
# Fast bubble approximation for large systems
```

-----

## Complete Function Reference

### Location: `QES/general_python/algebra/spectral_backend.py`

#### **1. Thermal Weights** (Foundation)

```python
thermal_weights(eigenvalues, temperature=0.0, backend="default")
    -> ρ_n = exp(-β(E_n - E_0))/Z
    Shape: (N,), dtype: float
```

**Theory:** Statistical mechanics Boltzmann distribution $\rho_n \propto \exp(-\beta E_n)$.
**When to use:**

* Any calculation at $T > 0$.
* Foundation for all thermal Lehmann sums.
    **Mathematical guarantee:** $\sum_n \rho_n = 1$ (normalized probability).
    **Behavior:**
* $T=0$: $\rho_0 = 1$, all others = 0.
* $T \rightarrow \infty$: $\rho_n \rightarrow 1/N$ (all states equally populated).

-----

#### **2. Many-Body Lehmann Spectral Function** (Exact)

```python
operator_spectral_function_lehmann(
    omega, eigenvalues, eigenvectors, operator,
    eta=0.01, temperature=0.0, backend="default"
)
    -> A_O(ω) = Σ_{m,n} (ρ_m - ρ_n) |⟨m|O|n⟩|² delta_η(ω - (E_n - E_m))
    Returns: float (single frequency)
```

*(where $\delta_\eta(x) = \frac{1}{\pi} \frac{\eta}{x^2 + \eta^2}$ is a Lorentzian)*

**Theory:** Lehmann representation of the operator spectral function (related to $\text{Im}[\chi]$).
**Physical content:**

* Probability amplitude to excite the system from state $m$ to $n$ using operator $O$.
* Matrix element $|\langle m|O|n \rangle|^2$ weights the transition strength.
* Thermal factor $(\rho_m - \rho_n)$ enforces thermal populations.
    **Exact for:** Full many-body Hamiltonian, any operator, all interactions.
    **Cost:** $O(N^2)$ where $N = 2^L$.
    **When to use:**
* $L \le 12$ sites.
* You need the exact spectral function for a specific operator.
* Studying strong correlations, bound states, magnons, etc.
    **Experimental connection:** Measures what a spectroscopy experiment sees:
* $O = c_i^\dagger$: ARPES (photoemission)
* $O = S_i^z$: Inelastic Neutron Scattering (INS)
* $O = n_i$: X-ray Scattering

-----

#### **3. Many-Body Lehmann Multi-omega** (Vectorized)

```python
operator_spectral_function_multi_omega(
    omegas, eigenvalues, eigenvectors, operator,
    eta=0.01, temperature=0.0, backend="default"
)
    -> A_O(ω) for all ω in grid
    Returns: (n_omega,) array, dtype: float
```

**Same physics as above,** but vectorized over the frequency grid for efficiency.

-----

#### **4. Bubble Susceptibility** (Fast)

```python
susceptibility_bubble(
    omega, eigenvalues, vertex=None, occupation=None,
    eta=0.01, backend="default"
)
    -> χ⁰(ω) = Σ_{m,n} (f_m - f_n) |V_{mn}|² / (ω + iη - (E_n - E_m))
    Returns: complex (single frequency)
```

**Theory:** Lindhard function for non-interacting (or mean-field) fermions.
$$\chi_0(\omega) = -\frac{1}{N} \sum_{k,q} \frac{f(E_k) - f(E_{k+q})}{E_{k+q} - E_k - \omega - i\eta}$$
**Physical content:**

* Sum over all possible single-particle transitions (particle-hole pairs).
* Fermi-Dirac factors $(f_m - f_n)$ determine if transitions are Pauli-allowed.
    **Exact for:** Non-interacting systems.
    **Cost:** $O(L^2)$ per frequency (where $L$ = \# of single-particle states).
    **When to use:**
* $L > 20$ (too large for full Exact Diagonalization).
* System is non-interacting or weakly interacting (mean-field).
* As the basis for an RPA calculation.

-----

#### **5. Bubble Multi-omega** (Vectorized)

```python
susceptibility_bubble_multi_omega(
    omegas, eigenvalues, vertex=None, occupation=None,
    eta=0.01, backend="default"
)
    -> χ⁰(ω) for all ω 
    Returns: (n_omega,) array, dtype: complex
```

**Same physics as bubble,** vectorized over the frequency grid.

-----

#### **6. Kubo Conductivity** (Transport)

```python
conductivity_kubo_bubble(
    omega, eigenvalues, velocity_matrix,
    occupation=None, eta=0.01, backend="default"
)
    -> \sigma(ω) = (1/(2ω)) χ⁰_{vv}(ω)
    Returns: complex (single frequency)
```

**Theory:** Kubo-Greenwood formula (bubble approximation for conductivity).
$$\sigma(\omega) \propto \frac{1}{\omega} \sum_{m,n} (f_m - f_n) \frac{|v_{mn}|^2}{\omega - (E_n - E_m) + i\eta}$$
**Physical content:**

* $\text{Re}[\sigma(\omega)]$: dissipation (Drude peak and interband absorption).
* $\text{Im}[\sigma(\omega)]$: reactive part (screening).
* Drude weight ($D \propto \lim_{\omega \to 0} \omega \text{Im}[\sigma(\omega)]$) measures free carrier density.
    **Exact for:** Non-interacting electrons.
    **When to use:**
* Computing optical conductivity $\sigma(\omega)$.
* Studying transport in metals/semiconductors.
* Analyzing Drude weight and interband transitions.

-----

#### **7. Kramers-Kronig Transform** (Verification)

```python
kramers_kronig_transform(Im_chi, omega_grid, backend="default")
    -> Re[χ(ω)] from Im[χ(ω)] via Hilbert transform
    Returns: (n_omega,) array, dtype: float
```

**Theory:** Causality via dispersion relations.
$$\text{Re}[\chi(\omega)] = \frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} d\omega' \frac{\text{Im}[\chi(\omega')]}{\omega' - \omega}$$
**Physical content:** Relates absorption ($\text{Im}[\chi]$) to reaction ($\text{Re}[\chi]$).
**When to use:**

* To verify the consistency of a calculated $\chi(\omega)$.
* To reconstruct $\text{Re}[\chi]$ from a measured $\text{Im}[\chi]$ (or vice-versa).
* To check the causality of an approximation.

-----

### Location: `QES/general_python/physics/response/susceptibility.py`

#### **8. Many-Body Susceptibility (Correlation)** (Exact)

```python
susceptibility_lehmann(
    hamiltonian_eigvals, hamiltonian_eigvecs, operator_q,
    omega, eta=0.01, temperature=0.0
)
    -> χ(ω) = Σ_{m,n} (ρ_m - ρ_n) ⟨m|A|n⟩⟨n|B|m⟩ / (ω - (E_n-E_m) + iη)
    Returns: complex
```

**Theory:** Full, exact Lehmann representation for the susceptibility $\chi_{AB}(\omega)$. `operator_q` can be a single operator (for $\chi_{AA}$) or a tuple `(A, B)`.

-----

#### **9. Static Susceptibility** ($\chi(\omega=0)$)

```python
static_susceptibility(
    hamiltonian_eigvals, hamiltonian_eigvecs, operator_q,
    temperature=0.0
)
    -> χ(0) 
    Returns: float
```

**Theory:** The $\omega=0$ limit of the Lehmann formula. In the classical limit, this is related to the fluctuation-dissipation theorem: $\chi(0) \approx \beta \langle (\Delta A)^2 \rangle$.
**Physical example:**

* Magnetic susceptibility: $\chi_M(0)$ (Curie constant / $T$).
* Charge compressibility: $\kappa = \partial n / \partial \mu \propto \chi_c(0)$.

-----

#### **10. Structure Factor** (Scattering Cross-Section)

```python
susceptibility_to_structure_factor(chi, omega_grid, temperature=0.0)
    -> S(q,ω) = -(1/π) Im[χ] / [1 - exp(-βω)]
    Returns: real array
```

**Theory:** The Fluctuation-Dissipation Theorem.
$$S(q,\omega) = -\frac{1}{\pi} \frac{\text{Im}[\chi(q,\omega)]}{1 - e^{-\beta\omega}}$$
**Physical content:**

* $S(q, \omega)$ is the dynamical structure factor, what is *actually* measured in scattering.
* $\text{Im}[\chi]$ is the *response* (absorption/emission difference).
* $S(q, \omega)$ is the *correlation* (measures total fluctuations at $\omega$).
* At $T=0$ (for $\omega > 0$): $S(q, \omega) = -(1/\pi) \text{Im}[\chi(q, \omega)]$.

**Experimental connection:** Inelastic Neutron Scattering (INS), Raman, and Resonant Inelastic X-ray Scattering (RIXS) measure $S(q, \omega)$.

-----

## Mathematical Foundations

### Eigenvalue Problem & Green's Function

For a Hamiltonian $H$ with eigenvalues $E_n$ and eigenvectors $|n\rangle$:
$$H |n\rangle = E_n |n\rangle$$
The **retarded Green's function** is the resolvent of the Hamiltonian:
$$G^R(\omega) = (\omega + i\eta - H)^{-1} = \sum_n \frac{|n\rangle\langle n|}{\omega + i\eta - E_n}$$
In the eigenbasis (with $U$ being the matrix of eigenvectors):
$$G^R_{ij}(\omega) = \langle i | G^R(\omega) | j \rangle = \sum_n \frac{U_{in} U^*_{jn}}{\omega + i\eta - E_n}$$

### Spectral Function from Green's Function

The **spectral function** (single-particle density of states):
$$A(\omega) = -\frac{1}{\pi} \text{Im} G^R(\omega + i0^+)$$
In the eigenbasis, this becomes a sum of delta functions:
$$A(\omega) = \sum_n \delta(\omega - E_n)$$
With finite broadening $\eta$, this is a sum of Lorentzians:
$$A(\omega) = \sum_n \frac{1}{\pi} \frac{\eta}{(\omega - E_n)^2 + \eta^2}$$

### Thermal Ensemble

At thermal equilibrium (temperature $T = 1/(k_B \beta)$):

**Canonical ensemble (Many-Body):**
$$\rho_n = \frac{e^{-\beta E_n}}{Z}, \quad Z = \text{Tr}(e^{-\beta H})$$

**Fermi-Dirac occupation (Single-Particle):**
$$f(\epsilon) = \frac{1}{1 + e^{\beta(\epsilon - \mu)}}$$

**Bose-Einstein occupation (Single-Particle):**
$$f(\epsilon) = \frac{1}{e^{\beta(\epsilon - \mu)} - 1}$$

### Sum Rules

**f-sum rule / Energy-Weighted Sum Rule (Many-Body):**
$$\int_{-\infty}^{\infty} d\omega \, \omega \, A_O(\omega) = \frac{1}{2\pi} \langle [O, [H, O^\dagger]] \rangle$$
*(Note: $A_O(\omega)$ is the operator spectral function, not the single-particle one)*

**Thomas-Reiche-Kuhn sum rule (specific case for conductivity):**
$$\int_0^{\infty} d\omega \, \frac{\omega \, \sigma(\omega)}{\pi} = \frac{Ne^2}{2m}$$
*(Note: This is one of several optical sum rules, and its form depends on the definition of $\sigma$)*

### Wick's Theorem & Mean-Field

**For quadratic Hamiltonians only,** correlation functions factorize:
$$\langle c_i^\dagger c_j c_k^\dagger c_l \rangle = \langle c_i^\dagger c_j \rangle \langle c_k^\dagger c_l \rangle - \langle c_i^\dagger c_l \rangle \langle c_k^\dagger c_j \rangle$$
This theorem is the mathematical foundation for:

* **RPA:** Summing bubble diagrams.
* **Hartree-Fock:** Mean-field approximations.
* **BCS/BdG:** Bogoliubov transformations for superconductivity.
* It justifies why the quadratic/bubble approximation works for non-interacting systems.

-----

## Practical Examples

### Example 1: Many-Body Spin Response in a 4-Site Cluster

```python
import numpy as np
import matplotlib.pyplot as plt
from QES.general_python.algebra.spectral_backend import (
    operator_spectral_function_multi_omega
)
# Assume build_hubbard_4site and build_total_spin_z exist
# from QES.systems.hubbard import build_hubbard_4site
# from QES.operators.spin import build_total_spin_z

# Build many-body Hamiltonian (16times 16 for 4 sites)
# H = -t Σ_{<ij>,\sigma} c†_{i,\sigma} c_{j,\sigma} + U Σ_i n_{i↑} n_{i↓}
H = build_hubbard_4site(t=1.0, U=4.0)

# Diagonalize
E, V = np.linalg.eigh(H)

# Define spin operator (total S_z)
S_z = build_total_spin_z(4)

# Compute spectral function at different temperatures
omegas = np.linspace(-8, 8, 300)
T_list = [0.0, 0.5, 1.0]

plt.figure()
for T in T_list:
    A_sz = operator_spectral_function_multi_omega(
        omegas, E, V, S_z, temperature=T, eta=0.05
    )
    
    # Physics: 
    # - T=0: Only shows excitations from the ground state.
    # - T>0: Higher-energy states get populated, 
    #   allowing new transitions (peaks) and broadening.
    
    plt.plot(omegas, A_sz, label=f"T={T}")

plt.xlabel("$\omega$ (Excitation Energy)")
plt.ylabel("$A_{S_z}(\omega)$")
plt.legend()
plt.show()
```

**Physics observed:**

* Peaks show energies of spin excitations (magnons, spin-flips).
* Heights encode matrix elements $\langle \text{ground} | S_z | \text{excited} \rangle$.
* Finite $T$ activates more transition channels.

-----

### Example 2: Conductivity in a Tight-Binding Chain

```python
import numpy as np
import matplotlib.pyplot as plt
from QES.general_python.algebra.spectral_backend import conductivity_kubo_bubble

# 1D tight-binding: H = -t Σ_i c†_i c_{i+1}
L = 100
t_hop = 1.0
k = np.linspace(-np.pi, np.pi, L)
E_k = -2 * t_hop * np.cos(k)

# Velocity: v = ∂H/∂k = 2t sin(k)
v_k = 2 * t_hop * np.sin(k)
v_matrix = np.diag(v_k)

# T=0 occupation (half-filling, Fermi energy = 0)
f = (E_k < 0).astype(float)

# Compute conductivity
omegas = np.linspace(0.01, 4.5, 200)
sigma = np.array([
    conductivity_kubo_bubble(w, E_k, v_matrix, occupation=f, eta=0.1)
    for w in omegas
])

# Extract dissipation (Real part) and reaction (Imaginary)
Re_sigma = np.real(sigma)
Im_sigma = np.imag(sigma)

plt.figure(figsize=(10, 5))
plt.plot(omegas, Re_sigma, label="Re[$\sigma$] (Absorption)")
# Plot Drude weight proxy
plt.plot(omegas, -Im_sigma * omegas, label="$-\omega$ Im[$\sigma$] (Drude)")
plt.xlabel("$\omega$")
plt.ylabel("$\sigma(\omega)$")
plt.legend()
plt.show()
```

**Physics observed:**

* **Low $\omega$:** Drude peak ($\text{Re}[\sigma]$) from free carriers.
* **$\omega \approx 4t$:** $\text{Re}[\sigma]$ drops off. This is the bandwidth ($4t$).
* **$\omega > 4t$:** No absorption, as no single-particle transitions exist.

-----

## Implementation Guide

### Step 1: Prepare Your System

```python
import numpy as np

# Option A: Many-body (small system, L ≤ 12)
# H_full = build_hubbard_hamiltonian(L=4, U=2.0, t=1.0)
# E, V = np.linalg.eigh(H_full)
# print(f"Hilbert space dimension: {len(E)}") # 2^4 = 16

# Option B: Single-particle (larger system)
# H_sp = build_tight_binding_hamiltonian(L=50)
# E_sp = np.linalg.eigvals(H_sp)
# V_sp = ... (eigenvectors if needed for vertex)
# print(f"Number of orbitals: {len(E_sp)}") # 50
```

### Step 2: Choose Your Operator

```python
# For Many-Body
# O_spin_z = build_spin_z_operator(L)
# O_charge_i = build_charge_operator(i, L)

# For Single-Particle (Vertex matrix)
# V_density = np.eye(L) # Density
# V_velocity = build_velocity_matrix(L) # Current
```

### Step 3: Set Parameters

```python
# Frequency grid
omegas = np.linspace(-5, 5, 300)  # Energy range and resolution

# Physical parameters
T = 0.5      # Temperature (in energy units, k_B=1)
beta = 1.0/T if T > 0 else np.inf
eta = 0.05   # Broadening (simulates scattering, lifetime)

# For quadratic: occupations
# f = 1 / (1 + np.exp(beta * (E_sp - mu))) # Fermi-Dirac
```

### Step 4: Compute

```python
from QES.general_python.algebra.spectral_backend import (
    operator_spectral_function_multi_omega,
    susceptibility_bubble_multi_omega
)

# Many-body
# if L <= 12:
#     A_O = operator_spectral_function_multi_omega(
#         omegas, E, V, O_spin_z,
#         temperature=T, eta=eta
#     )
    
# Quadratic
# else:
#     chi_0 = susceptibility_bubble_multi_omega(
#         omegas, E_sp, vertex=V_density,
#         occupation=f, eta=eta
#     )
#     A_O = -np.imag(chi_0) / np.pi
```

### Step 5: Verify & Analyze

```python
# Check sum rules
# from QES.general_python.algebra.spectral_backend import integrated_spectral_weight
# total_weight = integrated_spectral_weight(A_O, omegas)
# print(f"Integrated spectral weight: {total_weight:.4f}")

# Find peaks
# from scipy.signal import find_peaks
# peak_indices = find_peaks(A_O, height=0.05)[0]
# peak_omegas = omegas[peak_indices]
# print(f"Excitation energies: {peak_omegas}")
```

-----

## Verification & Sum Rules

### f-Sum Rule (Energy-Weighted Sum)

For an operator spectral function $A_O(\omega)$, the first moment is fixed by a double commutator:
$$\int_{-\infty}^{\infty} d\omega \, \omega \, A_O(\omega) = \frac{1}{2\pi} \langle [O, [H, O^\dagger]] \rangle$$
*(Note: $\langle ... \rangle$ denotes the thermal expectation value)*

**Check in code:**

```python
from scipy import integrate

# Compute expectation value of commutator
# commutator = O @ H - H @ O  # [O, H]
# double_commutator = O @ commutator - commutator @ O  # [O, [H, O]]

# Need to compute thermal average
# rho = thermal_weights(E, temperature=T)
# expected_trace = np.sum(rho * np.diag(V.T.conj() @ double_commutator @ V))
# expected = np.real(expected_trace) / (2 * np.pi)

# Compute sum numerically
# computed_sum = np.trapz(omegas * A_O, omegas)

# print(f"Expected sum: {expected:.6f}")
# print(f"Computed sum: {computed_sum:.6f}")
# print(f"Error: {abs(expected - computed_sum)/abs(expected)*100:.2f}%")
```

*Rule of thumb: If error \> 1%, check broadening $\eta$ or frequency range/resolution.*

### Causality Check (Kramers-Kronig)

Verify that the real and imaginary parts of $\chi(\omega)$ are consistent.

```python
from QES.general_python.algebra.spectral_backend import kramers_kronig_transform

# Assume you computed χ(ω) = Re[χ] + i Im[χ]
# chi_computed = susceptibility_lehmann(...)
# Re_computed = np.real(chi_computed)
# Im_computed = np.imag(chi_computed)

# Reconstruct Real part from Imaginary
# Re_kk = kramers_kronig_transform(Im_computed, omegas)

# Compare
# error = np.trapz(np.abs(Re_computed - Re_kk), omegas)
# print(f"Kramers-Kronig (Causality) error: {error:.6f}")
```

-----

## Summary Table: When to Use What

| Question You're Asking | What You Need | Function to Use | Example |
| :--- | :--- | :--- | :--- |
| What are the **exact many-body excitations**? | Many-body spectrum | `operator_spectral_function_multi_omega` | Hubbard model, $L < 12$ |
| What's the **non-interacting band structure**? | Single-particle DOS | `spectral_function_multi_omega` | Tight-binding, ARPES sim. |
| How does the system **respond to a field**? | Linear susceptibility | `susceptibility_lehmann` | Magnetic/charge response |
| What's the **optical conductivity**? | Kubo formula | `conductivity_kubo_bubble` | Optics, Drude weight |
| How to scale to **100+ sites** (approx.)? | Quadratic bubble | `susceptibility_bubble_multi_omega` | Large systems, mean-field |
| Is my $\chi(\omega)$ calculation **consistent**? | Check causality | `kramers_kronig_transform` | Verify $\text{Im} \leftrightarrow \text{Re}$ |
| What states are **populated at $T > 0$**? | Thermal weights | `thermal_weights` | Any finite-T property |
| What does a **scattering experiment** see? | Structure Factor | `susceptibility_to_structure_factor` | INS, RIXS simulation |

-----

## References & Further Reading

* **Foundational Theory:**
  * Mahan, G.D. "Many-Particle Physics"
  * Fetter, A.L. & Walecka, J.D. "Quantum Theory of Many-Particle Systems"
  * Abrikosov, Gorkov, Dzyaloshinski "Methods of Quantum Field Theory in Statistical Physics"
* **Linear Response:**
  * Kubo, R. "Statistical-Mechanical Theory of Irreversible Processes" (1957)
* **Exact Diagonalization:**
  * Dagotto, E. "Correlated electrons in high-temperature superconductors" (1994, Rev. Mod. Phys.)
* **Quadratic & Mean-Field:**
  * Ashcroft, N.W. & Mermin, N.D. "Solid State Physics"

-----

## Contact & Support

**Framework:** PyQuSolver Response Function Stack
**Maintainer:** Maksymilian Kliczkowski
**Email:** [maksymilian.kliczkowski@pwr.edu.pl](mailto:maksymilian.kliczkowski@pwr.edu.pl)
**Version:** 1.0 (November 2025)

**Modules:**

* `QES/general_python/algebra/spectral_backend.py`
* `QES/general_python/physics/response/susceptibility.py`
* `QES/general_python/physics/response/unified_response.py`
