# Response Function Stack — Comprehensive Physics Guide

**PyQuSolver Complete Framework for Spectral Functions, Response Functions & Transport**

A unified theoretical and computational guide explaining response functions from first principles, their physical necessity, mathematical foundations, and computational implementations for both exact many-body and quadratic systems.

---

## Table of Contents

1. [Fundamental Theory](#fundamental-theory)
2. [Why These Quantities Matter](#why-these-quantities-matter)
3. [The Two Computational Regimes](#the-two-computational-regimes)
4. [Complete Function Reference](#complete-function-reference)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Practical Examples](#practical-examples)
7. [Implementation Guide](#implementation-guide)
8. [Verification & Sum Rules](#verification--sum-rules)

---

## Fundamental Theory

### Linear Response & the Kubo Formula

The foundation of all response functions is **linear response theory**. When a quantum system experiences a small time-dependent perturbation:

$$H(t) = H_0 + V(t)$$

the observable A changes according to:

$$\langle A(t) \rangle = \langle A \rangle_0 + \int_{-\infty}^{t} dt' \chi_{AB}(t - t') \langle \partial V(t')/\partial B \rangle$$

This **linear susceptibility** χ_{AB}(t) relates the response of A to a perturbation in B. In frequency space (Fourier transform):

$$\chi_{AB}(\omega) = \frac{i}{\hbar} \int_0^{\infty} dt e^{i(\omega + i0^+)t} \langle [A(t), B] \rangle$$

**Physical origin:** This is the **Kubo formula**. It measures how the commutator between A and B evolves. The key insight: response functions encode *correlation functions* between observables.

### Lehmann Representation

For a system with Hamiltonian H with eigenvalues E_n and eigenstates |n⟩, the susceptibility can be written in the **Lehmann representation**:

$$\chi_{AB}(\omega) = \sum_{m,n} \frac{(\rho_m - \rho_n)}{E_n - E_m - \omega - i\eta} \langle m|A|n \rangle \langle n|B|m \rangle$$

where:

- ρ_n = Boltzmann weight = exp(-β(E_n - E_0))/Z
- η = infinitesimal broadening
- Sum runs over **all many-body eigenstates**

**Physical interpretation:**

- Each term represents a transition |m⟩ → |n⟩
- Height ∝ |⟨m|A|n⟩|² × |⟨n|B|m⟩|² — matrix elements determine strength
- Position = E_n - E_m — energy required for transition
- Difference in occupations (ρ_m - ρ_n) determines if transition is allowed

### Spectral Function from Green's Function

The **single-particle Green's function** is defined as:

$$G_{ij}(\omega) = \langle i | (\omega + i\eta - H)^{-1} | j \rangle$$

The **spectral function** is extracted via:

$$A_{ij}(\omega) = -\frac{1}{\pi} \text{Im} G_{ij}(\omega)$$

**Physical content:** A(ω) represents the density of excitations you can create/destroy at energy ω by acting with operator O = a_i† or a_i.

**Why it matters:**

- Directly observable in photoemission spectroscopy (ARPES)
- Shows band structure with lifetime broadening (η)
- Peak positions = excitation energies
- Peak width = lifetime = 1/(2Γ) where Γ is scattering rate

---

## Why These Quantities Matter

### 1. **Thermal Weights** — Foundation of Everything at Finite T

**Where it comes from:** Statistical mechanics. At thermal equilibrium:

$$\rho_n = \frac{e^{-\beta E_n}}{Z}, \quad Z = \sum_n e^{-\beta E_n}$$

**Why it's necessary:**

- All observables at finite T depend on which states are populated
- Mean value: ⟨A⟩ = Σ_n ρ_n ⟨n|A|n⟩
- Response functions automatically include thermal factors
- At T=0: only ground state contributes; at T>0: whole spectrum matters

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import thermal_weights
rho = thermal_weights(eigenvalues, temperature)
# Returns ρ_n normalized so Σ_n ρ_n = 1
```

**Physical example:** In a metal at room temperature, the Fermi-Dirac distribution kT ≈ 25 meV is tiny but nonzero, affecting transport via rare thermal excitations. This is essential for semiconductors/insulators where T ≲ gap.

---

### 2. **Single-Particle Spectral Function A(ω)** — Band Structure & DOS

**Where it comes from:** Green's function formalism in Hartree-Fock or exact diagonalization.

$$A_{ii}(\omega) = -\frac{1}{\pi} \text{Im} G_{ii}(\omega + i\eta)$$

**Why it's necessary:**

- **Photoemission (ARPES):** Directly measures A(ω) — shows band structure
- **Density of states:** ρ(ω) = Σ_i A_{ii}(ω) — total number of states per energy
- **Transport:** Conductivity depends on A(ω) near Fermi surface
- **Quantum oscillations:** Frequency content reveals Fermi surface geometry

**Many-body vs. single-particle:**

- **Single-particle:** A(ω) from one-body Green's function — shows "bare" bands
- **Many-body exact:** A(ω) from full diagonalization — shows bands with all interactions

**Physical example:** In graphene:

- Single-particle: Linear crossing at K point
- With interactions: Landau damping broadens peaks, interaction effects visible

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import spectral_function_multi_omega
A = spectral_function_multi_omega(omegas, eigenvalues, eigenvectors)
# ω-dependent density of single-particle excitations
```

---

### 3. **Operator-Projected Spectral Function A_O(ω)** — Many-Body Response

**Where it comes from:** Many-body Lehmann representation with specific operator O.

$$A_O(\omega) = \sum_{m,n} (\rho_m - \rho_n) |\langle m|O|n \rangle|^2 \delta(\omega - (E_n - E_m))$$

**Why it's necessary:**

- **Extends beyond single particles:** Can measure response of ANY operator (spin, charge, current)
- **Shows many-body excitations:** Peaks at multi-particle energies, not just single excitations
- **Captures correlations:** Matrix elements include all interactions
- **Finite temperature:** Thermal population automatically included

**Many-body exclusive** — Cannot be obtained from single-particle Hamiltonians; requires full diagonalization of 2^L Hilbert space.

**Physical origins:**

- Comes directly from Kubo formula for general operators
- Each peak represents true many-body transition |m⟩ ↔ |n⟩
- Example: In Hubbard model, upper/lower Hubbard bands are visible; in quadratic case, they don't exist

**Physical example:** In a strongly-correlated magnet:

- Spin spectral A_Sz(ω) shows magnon excitations
- Heights reveal matrix elements ⟨ground|S_z|excited state⟩
- Temperature shifts weight to higher energies

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import operator_spectral_function_multi_omega
A_O = operator_spectral_function_multi_omega(
    omegas, eigenvalues, eigenvectors, operator, 
    temperature=T
)
# Many-body spectral function for arbitrary operator O
```

---

### 4. **Linear Response Susceptibility χ(ω)** — How System Responds

**Where it comes from:** Kubo formula applied to correlation functions.

$$\chi_{AB}(\omega) = \sum_{m,n} \frac{(\rho_m - \rho_n) \langle m|A|n \rangle \langle n|B|m \rangle}{\omega - (E_n - E_m) + i\eta}$$

**Why it's necessary:**

- **Fundamental definition** of "response" — tells you system's reaction to perturbation
- **Complex function:** Real part = reactive (shift in energy levels), imaginary = dissipative
- **Observables from experiments:**
  - Magnetic susceptibility χ_M: Response to magnetic field
  - Charge susceptibility χ_c: Response to electric field
  - Shear viscosity η: Response to shear stress
- **Links to experiments:** All inelastic scattering cross-sections (INS, Raman, X-ray) are proportional to Im[χ]

**Sum rules:** Important constraints from first principles:

$$\int_{-\infty}^{\infty} d\omega \text{ Im}[\chi(\omega)] = \text{specific constant depending on} [A,B]$$

This is a **sum rule** — provides exact relation between different frequency ranges.

**Physical example:**

- In paramagnet: χ_M = Σ_i ⟨S_i^2⟩ / (3k_B T) — Curie law
- In superconductor: χ_M = -1/(4π) (Meissner effect, perfect diamagnetism)
- In Fermi liquid: χ_c(0) ∝ density of states at Fermi level

**In our code:**

```python
# Many-body exact (Lehmann)
from QES.general_python.physics.response.susceptibility import susceptibility_lehmann
chi = susceptibility_lehmann(E, V, operator_A, operator_B, omega, T)

# Or use our operator-projected version
# which gives A_O(ω) = -Im[χ]/π
```

---

### 5. **Bubble Susceptibility χ⁰(ω)** — Single-Particle Approximation

**Where it comes from:** Lindhard function for non-interacting fermions.

$$\chi_0(\omega) = \sum_{m,n} \frac{(f_m - f_n) |V_{mn}|^2}{\omega - (E_n - E_m) + i\eta}$$

where:

- E_m = single-particle energies (NOT many-body!)
- f_m = single-particle occupations (Fermi-Dirac)
- V_{mn} = vertex (identity for density-density, velocity for current, etc.)

**Why it's necessary:**

- **Computational efficiency:** O(L³) not O((2^L)³)
- **Exact for non-interacting systems:** Gives exact answer when interactions vanish
- **RPA starting point:** Hartree-Fock + bubble = RPA (standard approximation)
- **Scales to large systems:** L ~ 100 sites easily

**Single-particle origin** — Does NOT include correlations beyond mean-field.

**Difference from many-body:**

- Poles at single-particle excitations (not collective)
- No Hubbard gap, no bound states, no exotic correlations
- Much faster

**Physical example:**

- Tight-binding chain: χ⁰ shows Van Hove singularities at band edges
- 2D free electron gas: χ⁰ shows Kohn anomaly (structure factor instability)
- In real systems: χ = χ⁰ + χ⁰ Γ χ⁰ + ... (RPA ladder, vertex corrections)

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import susceptibility_bubble_multi_omega
chi0 = susceptibility_bubble_multi_omega(
    omegas, E_single_particle, vertex=None, occupation=f
)
# Fast bubble from single-particle energies and occupations
```

---

### 6. **Kubo Conductivity σ(ω)** — Charge Transport

**Where it comes from:** Response to electric field applied to charges.

$$\sigma(\omega) = \frac{1}{i\omega} \chi_{jj}(\omega)$$

where j = current operator = Σ_i v_i (velocity).

**Why it's necessary:**

- **Experimental observable:** Direct measurement via optical reflectance/transmittance
- **Transport physics:**
  - Real part Re[σ] = dissipation (Joule heating)
  - Imaginary part Im[σ] = reactive (screening)
- **Drude weight:** Low-frequency limit shows free-carrier density
- **Interband transitions:** Higher frequencies show band-to-band excitations

**Many-body vs. single-particle:**

- **Many-body:** Full σ(ω) including all correlations — shows correct damping, threshold effects
- **Bubble:** σ⁰(ω) from Kubo-Greenwood formula — shows Drude peak, interband structure

**Experimental connection:** Reflectance R(ω) = |Z-1|²/|Z+1|² where Z = √(ε/σ) — can extract σ from reflectance.

**Physical example:**

- Metal: σ → 1/ω as ω → 0 (Drude limit), then interband absorption at ω > Eg
- Semiconductor: Gap at low ω (optical gap), then strong absorption
- Superconductor: δ-function at ω=0 (zero resistance), gap at 2Δ

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import conductivity_kubo_bubble
sigma = conductivity_kubo_bubble(omega, E_sp, velocity_matrix, occupation=f)
# σ(ω) = (1/2ω) χ_vv(ω)
```

---

### 7. **Kramers-Kronig Relations** — Causality & Consistency

**Where it comes from:** Causality (response must be retarded, H(t) < H(t')).

$$\text{Re}[\chi(\omega)] = \frac{1}{\pi} P \int_{-\infty}^{\infty} \frac{d\omega'}{\omega' - \omega} \text{Im}[\chi(\omega')]$$

$$\text{Im}[\chi(\omega)] = -\frac{1}{\pi} P \int_{-\infty}^{\infty} \frac{d\omega'}{\omega' - \omega} \text{Re}[\chi(\omega')]$$

**Why it's necessary:**

- **Fundamental consistency check:** If Im[χ] violates K-K, calculation has error
- **Determines full response from half:** Knowing Im[χ] alone determines Re[χ]
- **Physical insight:** Low and high-frequency behavior connected by causality
- **Sum rules emerge:** Integration gives constraints

**Always satisfied for:** Physical systems (causal response)

**Violated by:** Approximate calculations, truncated Hilbert spaces, poor numerical integration

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import kramers_kronig_transform
Re_chi = kramers_kronig_transform(Im_chi, omega_grid)
# If this matches what you calculated directly, ✓ causality verified
```

---

## The Two Computational Regimes

### Regime I: Exact Many-Body (Full Diagonalization)

**Applicable to:** L ≤ 12 sites (2^12 ≈ 4000D Hilbert space)

**Method:**

1. Diagonalize full H: O((2^L)³) ~ 10⁶ operations for L=12
2. Compute all matrix elements ⟨m|O|n⟩
3. Use Lehmann formula with exact eigenvalues/eigenvectors

**Advantages:**

- ✅ **Exact:** No approximations
- ✅ **Any operator:** Works for spin, charge, current, exotic
- ✅ **All correlations:** Includes everything: bound states, holon-doublon excitations, magnons, etc.
- ✅ **Finite T:** Native thermal effects
- ✅ **Transparent physics:** Each peak = real transition

**Limitations:**

- ❌ Exponential scaling: L > 13 becomes very slow
- ❌ Memory intensive: Store 2^L × 2^L matrix
- ❌ Limited to small clusters

**Physics captured:**

- Mott insulators with Hubbard gaps
- Bound states (excitons, trions)
- Magnons in magnets
- Charge-transfer excitations
- All genuine many-body phenomena

**Example:** 4-site Hubbard model

```
H = -t Σ_<ij> c†_i c_j + U Σ_i n_i↑ n_i↓
     ↓ hopping           ↓ on-site repulsion

Diagonalize 16×16 matrix (including spin)
Get all 16 eigenvalues
Compute any A_O(ω)
```

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import operator_spectral_function_multi_omega

E, V = np.linalg.eigh(H_full)  # H_full is 2^L × 2^L
A_O = operator_spectral_function_multi_omega(
    omegas, E, V, operator,
    temperature=T, eta=0.05
)
# Exact many-body result for any operator O
```

---

### Regime II: Quadratic Approximation (Single-Particle)

**Applicable to:** L up to ~100 sites; **when interactions are weak or mean-field**

**Method:**

1. Diagonalize single-particle H: O(L³) ~ 10⁶ operations for L=100
2. Get single-particle energies E_α and occupations f_α
3. Use bubble formula with single-particle quantities

**Advantages:**

- ✅ **Fast:** Polynomial O(L³) scaling
- ✅ **Large systems:** L ~ 50-100 easily
- ✅ **Exact for quadratic:** Perfect for non-interacting fermions/bosons
- ✅ **Standard approximation:** RPA built on this
- ✅ **Scalable:** Can handle realistic lattice sizes

**Limitations:**

- ❌ No genuine many-body excitations (bound states, Mott gap)
- ❌ Correlation effects missing
- ❌ Only single-particle excitations visible

**Physics captured:**

- Band structure (with lifetime broadening)
- Drude weight and conductivity
- Van Hove singularities
- Density of states
- Carrier dynamics in metals

**Example:** Tight-binding chain

```
H = -t Σ_<i,j> c†_i c_j  (no interactions)

Diagonalize L×L matrix
E_k = -2t cos(k), k = 2πn/L
χ⁰(ω) = Lindhard function
```

**In our code:**

```python
from QES.general_python.algebra.spectral_backend import susceptibility_bubble_multi_omega

E_sp = np.linalg.eigvals(H_single_particle)  # L eigenvalues
f = 1/(1 + np.exp(beta * E_sp))  # Fermi-Dirac occupations
chi0 = susceptibility_bubble_multi_omega(
    omegas, E_sp, occupation=f
)
# Fast bubble approximation for large systems
```

---

## Complete Function Reference

### Location: `QES/general_python/algebra/spectral_backend.py`

#### **1. Thermal Weights** (Foundation)

```python
thermal_weights(eigenvalues, temperature=0.0, backend="default")
    → ρ_n = exp(-β(E_n - E_0))/Z
    Shape: (N,), dtype: float
```

**Theory:** Statistical mechanics. Boltzmann distribution at thermal equilibrium.

**When to use:**

- ANY calculation at T > 0
- Foundation for all thermal calculations
- Must precede any response calculation

**Mathematical guarantee:** Σ_n ρ_n = 1 (normalized)

**Behavior:**

- T=0: ρ_0 = 1, all others = 0
- T>0: Exponential decay with energy
- T → ∞: ρ_n → 1/N (equal population)

---

#### **2. Many-Body Lehmann Spectral Function** (Exact)

```python
operator_spectral_function_lehmann(
    omega, eigenvalues, eigenvectors, operator,
    eta=0.01, temperature=0.0, backend="default"
)
    → A_O(ω) = Σ_{m,n} (ρ_m - ρ_n) |⟨m|O|n⟩|² Γ(ω - ΔE_mn)
    Returns: float (single frequency)
```

**Theory:** Lehmann representation from Kubo formula.

$$A_O(\omega) = -\frac{1}{\pi} \text{Im} \left\langle 0 | O (\omega + i\eta - H)^{-1} O^\dagger | 0 \rangle \right\rangle$$

**Physical content:**

- Probability amplitude to excite system from state m to n
- Matrix element |⟨m|O|n⟩|² weights each transition
- Thermal factor (ρ_m - ρ_n) determines if transition is accessible
- Peak position = energy cost E_n - E_m
- Peak height = matrix element strength

**Exact for:** Full many-body Hamiltonian, any operator, all interactions

**Cost:** O(N²) where N = 2^L (number of many-body states)

**When to use:**

- L ≤ 12 sites (manageable computation)
- Need exact answer
- Want to see all correlations
- Studying strong correlations, bound states, exotic excitations

**Experimental connection:** Measures what a spectroscopy experiment sees when probing with operator O:

- O = c_i† : ARPES (photoemission)
- O = S_i^z : Neutron scattering
- O = n_i : X-ray scattering
- O = j_i : Optical conductivity

---

#### **3. Many-Body Lehmann Multi-ω** (Vectorized)

```python
operator_spectral_function_multi_omega(
    omegas, eigenvalues, eigenvectors, operator,
    eta=0.01, temperature=0.0, backend="default"
)
    → A_O(ω) for all ω in grid
    Returns: (n_omega,) array, dtype: float
```

**Same physics as above, but vectorized over frequency grid.**

**When to use:**

- Building spectral function plot
- Computing sum rules over full range
- Time-intensive calculations best batched

---

#### **4. Bubble Susceptibility** (Fast)

```python
susceptibility_bubble(
    omega, eigenvalues, vertex=None, occupation=None,
    eta=0.01, backend="default"
)
    → χ⁰(ω) = Σ_{m,n} (f_m - f_n) |V_{mn}|² / (ω + iη - ΔE_mn)
    Returns: complex (single frequency)
```

**Theory:** Lindhard function for non-interacting fermions.

$$\chi_0(\omega) = -\frac{1}{N} \sum_{k,q} \frac{f(E_k) - f(E_{k+q})}{E_{k+q} - E_k - \omega - i\eta}$$

**Physical content:**

- Sum over all single-particle transition pairs
- Fermi-Dirac factors (f_m - f_n) determine accessibility
- |V_{mn}|² = vertex (identity for density, velocity for current)
- Only single-particle transitions visible (not collective)

**Exact for:** Non-interacting systems

**Cost:** O(L²) in frequency (O(L³) total for all frequencies)

**When to use:**

- L > 20 (too large for full ED)
- System is non-interacting or weakly interacting
- Want speed
- Building RPA (bubble is foundation)

**Experimental connection:**

- Zero-vertex bubble: density-density response (charge density waves)
- Velocity-vertex bubble: current response → conductivity
- Spin-vertex bubble: magnetic response

---

#### **5. Bubble Multi-ω** (Vectorized)

```python
susceptibility_bubble_multi_omega(
    omegas, eigenvalues, vertex=None, occupation=None,
    eta=0.01, backend="default"
)
    → χ⁰(ω) for all ω
    Returns: (n_omega,) array, dtype: complex
```

**Same physics as bubble, vectorized.**

---

#### **6. Kubo Conductivity** (Transport)

```python
conductivity_kubo_bubble(
    omega, eigenvalues, velocity_matrix,
    occupation=None, eta=0.01, backend="default"
)
    → σ(ω) = (1/(2ω)) χ⁰_{vv}(ω)
    Returns: complex (single frequency)
```

**Theory:** Kubo-Greenwood formula.

$$\sigma(\omega) = \frac{e^2}{m\omega} \int \frac{d^d k}{(2\pi)^d} \sum_{m,n} (f_m - f_n) \frac{|v_{mn}(k)|^2}{\omega - (E_n - E_m) + i\eta}$$

**Physical content:**

- Real part Re[σ]: dissipation (Drude and interband)
- Imaginary part Im[σ]: reactive part (screening)
- Drude weight at ω→0: carrier density × mobility
- Interband transitions at ω > gap

**Exact for:** Non-interacting electrons with velocity matrix

**Cost:** O(1) single frequency

**When to use:**

- Computing optical conductivity
- Studying transport properties
- Connecting to reflectance measurements
- Analyzing Drude weight and band structure effects

**Experimental connection:**

- Optical reflectance: R(ω) = |1-Z(ω)|²/|1+Z(ω)|², Z = √(ε/σ)
- Conductivity measured via FTIR spectroscopy
- Shows metal-insulator transitions

---

#### **7. Kramers-Kronig Transform** (Verification)

```python
kramers_kronig_transform(Im_chi, omega_grid, backend="default")
    → Re[χ(ω)] from Im[χ(ω)] via Hilbert transform
    Returns: (n_omega,) array, dtype: float
```

**Theory:** Causality via dispersion relations.

$$\text{Re}[\chi(\omega)] = \frac{1}{\pi} P \int_{-\infty}^{\infty} d\omega' \frac{\text{Im}[\chi(\omega')]}{\omega' - \omega}$$

**Physical content:**

- Relates low and high-frequency behavior
- If violated: calculation error, causality broken
- Essential consistency check

**When to use:**

- Verify consistency of χ(ω)
- Reconstruct Re[χ] from Im[χ] only
- Check causality of approximations
- Compute reflectance from conductivity

---

### Location: `QES/general_python/physics/response/susceptibility.py`

#### **8. Many-Body Susceptibility (Correlation)** (Exact)

```python
susceptibility_lehmann(
    hamiltonian_eigvals, hamiltonian_eigvecs, operator_q,
    omega, eta=0.01, temperature=0.0
)
    → χ(ω) = Σ_{m,n} (ρ_m - ρ_n) ⟨m|A|n⟩⟨n|B|m⟩ / (ω - ΔE_mn + iη)
    Returns: complex
```

**Theory:** Full Lehmann representation including both forward/backward parts.

**Difference from operator_spectral_function:**

- operator_spectral: A_O(ω) = |⟨m|O|n⟩|² (single side)
- susceptibility: χ(ω) = ⟨m|O|n⟩⟨n|O†|m⟩ (both sides)

**Standard form** used in literature for response functions.

---

#### **9. Static Susceptibility** (χ(ω=0))

```python
static_susceptibility(
    hamiltonian_eigvals, hamiltonian_eigvecs, operator_q,
    temperature=0.0
)
    → χ(0) = fluctuation-dissipation relation
    Returns: float
```

**Theory:** Zero-frequency limit connects to fluctuations.

$$\chi(0) = \beta \langle (\Delta A)^2 \rangle = \int_{-\infty}^{\infty} \frac{d\omega}{2\pi} \chi(\omega)$$

**Physical example:**

- Magnetic susceptibility: χ_M(0) = Curie constant / T
- Charge compressibility: κ = ∂n/∂μ inverse of χ_c(0)

---

#### **10. Structure Factor** (Scattering Cross-Section)

```python
susceptibility_to_structure_factor(chi, omega_grid, temperature=0.0)
    → S(q,ω) = -(1/π) Im[χ]/[1 - exp(-βω)]
    Returns: real array
```

**Theory:** Fluctuation-dissipation theorem.

$$S(q,\omega) = \frac{1}{\pi} \frac{\text{Im}[\chi(q,\omega)]}{1 - e^{-\beta\omega}}$$

**Physical content:**

- Directly proportional to inelastic scattering cross-section
- What neutrons (INS), photons (Raman), X-rays (RIXS) measure
- T=0 limit: S ∝ Im[χ] (only excitations allowed)
- T>0: includes excitations and de-excitations

**Experimental connection:** Inelastic neutron scattering (INS), Raman scattering, X-ray scattering

---

## Mathematical Foundations

### Eigenvalue Problem & Green's Function

For Hamiltonian H with eigenvalues E_n and eigenvectors |n⟩:

$$H |n\rangle = E_n |n\rangle$$

The **retarded Green's function** is:

$$G^R(\omega) = (\omega + i\eta - H)^{-1} = \sum_n \frac{|n\rangle\langle n|}{\omega + i\eta - E_n}$$

In eigenbasis:

$$G^R_{ij}(\omega) = \sum_n \frac{U_{in} U^*_{jn}}{\omega + i\eta - E_n}$$

### Spectral Function from Green's Function

The **spectral function** (one-particle density of states):

$$A(\omega) = -\frac{1}{\pi} \text{Im} G^R(\omega + i0^+)$$

$$= \sum_n \delta(\omega - E_n)$$

With finite broadening η:

$$A(\omega) = \sum_n \frac{\eta/\pi}{(\omega - E_n)^2 + \eta^2} \equiv \text{Lorentzian}$$

### Thermal Ensemble

At thermal equilibrium (temperature T, chemical potential μ):

**Canonical ensemble:**
$$\rho_n = \frac{e^{-\beta(E_n - E_0)}}{Z}, \quad Z = \text{Tr}(e^{-\beta H})$$

**Fermi-Dirac occupation** (fermions):
$$f(\epsilon) = \frac{1}{1 + e^{\beta(\epsilon - \mu)}}$$

**Bose-Einstein occupation** (bosons):
$$f(\epsilon) = \frac{1}{e^{\beta(\epsilon - \mu)} - 1}$$

### Sum Rules

**f-sum rule** (oscillator strength):
$$\int_{-\infty}^{\infty} d\omega \, \omega \, A(\omega) = \frac{\pi}{2} \langle [O, [H, O^\dagger]] \rangle$$

**Thomas-Reiche-Kuhn sum rule** (specific case):
$$\int_0^{\infty} d\omega \, \omega \, \sigma(\omega) / \pi = Ne^2 / (2m) \, \equiv \text{plasma frequency}$$

### Wick's Theorem & Mean-Field

**For quadratic Hamiltonians**, correlation functions factorize:

$$\langle c_i^\dagger c_j c_k^\dagger c_l \rangle = \langle c_i^\dagger c_j \rangle \langle c_k^\dagger c_l \rangle - \langle c_i^\dagger c_l \rangle \langle c_k^\dagger c_j \rangle + \text{contractions}$$

This allows:

- **RPA:** Bubble diagrams → full response
- **BdG:** Bogoliubov transform → superconductivity
- Justifies quadratic approximation when interactions are weak

---

## Practical Examples

### Example 1: Many-Body Spin Response in a 4-Site Cluster

```python
import numpy as np
from QES.general_python.algebra.spectral_backend import (
    operator_spectral_function_multi_omega,
    thermal_weights
)

# Build many-body Hamiltonian (16×16 for 4 sites with spin)
# H = -t Σ_<ij> c†_i,σ c_j,σ + U Σ_i n_i↑ n_i↓
H = build_hubbard_4site(t=1.0, U=4.0)

# Diagonalize
E, V = np.linalg.eigh(H)

# Define spin operator (total S_z)
S_z = build_total_spin_z(4)

# Compute spectral function at different temperatures
omegas = np.linspace(-8, 8, 300)
T_list = [0.0, 0.5, 1.0]

for T in T_list:
    A_sz = operator_spectral_function_multi_omega(
        omegas, E, V, S_z, temperature=T, eta=0.05
    )
    
    # Physics: 
    # - T=0: only ground state + first few excited states
    # - T>0: more states accessible, spectrum broadens
    # - Each peak = transition between many-body states
    
    plt.plot(omegas, A_sz, label=f"T={T}")
plt.xlabel("ω (excitation energy)")
plt.ylabel("A_Sz(ω)")
plt.legend()
plt.show()
```

**Physics observed:**

- Peaks show where system can emit/absorb spin excitations
- Heights encode matrix elements ⟨ground|S_z|excited⟩
- Finite T activates more channels

---

### Example 2: Conductivity in a Tight-Binding Chain

```python
import numpy as np
from QES.general_python.algebra.spectral_backend import conductivity_kubo_bubble

# 1D tight-binding: H = -t Σ_i c†_i c_{i+1}
L = 100
k = np.linspace(-np.pi, np.pi, L)
E_k = -2 * np.cos(k)

# Velocity: v = ∂H/∂k = 2 sin(k)
v_k = 2 * np.sin(k)
v_matrix = np.diag(v_k)

# T=0 occupation (filled up to Fermi)
f = (E_k < 0).astype(float)

# Compute conductivity
omegas = np.linspace(0.01, 4, 200)
sigma = np.array([
    conductivity_kubo_bubble(w, E_k, v_matrix, occupation=f, eta=0.1)
    for w in omegas
])

# Extract dissipation
Re_sigma = np.real(sigma)
Im_sigma = np.imag(sigma)

plt.figure(figsize=(10, 5))
plt.plot(omegas, -Im_sigma, label="-Im[σ] (Drude weight)")
plt.plot(omegas, Re_sigma, label="Re[σ] (reactive)")
plt.xlabel("ω")
plt.ylabel("σ(ω)")
plt.legend()
plt.show()

# Physics observed:
# - Low ω: Drude peak (free carriers)
# - ω ≈ 2: Threshold for interband excitations
# - High ω: Weak absorption (all electrons scattered)
```

---

### Example 3: Temperature Effects on Magnetic Response

```python
import numpy as np
from QES.general_python.algebra.spectral_backend import (
    operator_spectral_function_multi_omega,
    thermal_weights
)

# Build spin model (small cluster)
H = build_heisenberg_chain(L=4, J=1.0)
E, V = np.linalg.eigh(H)

# Magnetization operator
M = build_magnetization_operator(4)

omegas = np.linspace(-2, 2, 100)

# Plot temperature dependence
for beta in [10, 5, 2, 1, 0.5]:  # β = 1/T
    T = 1.0 / beta
    
    A = operator_spectral_function_multi_omega(
        omegas, E, V, M, temperature=T, eta=0.02
    )
    
    rho = thermal_weights(E, temperature=T)
    Z = np.sum(rho)
    
    plt.plot(omegas, A, label=f"T={T:.2f} (Z={Z:.3f})")

plt.xlabel("ω (excitation energy)")
plt.ylabel("A_M(ω)")
plt.legend()
plt.show()

# Physics observed:
# - T=0: Only lowest-energy excitations
# - T increases: Higher-energy states become populated
# - At very high T: All states equally populated (flat response)
# - Intensity shifts to higher energies
```

---

## Implementation Guide

### Step 1: Prepare Your System

```python
import numpy as np

# Option A: Many-body (small system, L ≤ 12)
H_full = build_hubbard_hamiltonian(L=4, U=2.0, t=1.0)
E, V = np.linalg.eigh(H_full)
print(f"Hilbert space dimension: {len(E)}")  # 2^4 = 16

# Option B: Single-particle (larger system)
H_sp = build_tight_binding_hamiltonian(L=50)
E_sp = np.linalg.eigvals(H_sp)
print(f"Number of orbitals: {len(E_sp)}")  # 50
```

### Step 2: Choose Your Operator

```python
# Example operators
O_charge = np.eye(dim)  # particle number
O_spin_z = build_spin_z_operator()  # magnetization
O_current = build_current_operator()  # charge current
O_custom = my_custom_observable  # your operator
```

### Step 3: Set Parameters

```python
# Frequency grid
omegas = np.linspace(-5, 5, 300)  # range and resolution

# Physical parameters
T = 0.5  # Temperature (in energy units, k_B T)
eta = 0.05  # Broadening (damping, scattering rate)

# For quadratic: occupations
f = 1 / (1 + np.exp(beta * E_sp))  # Fermi-Dirac
```

### Step 4: Compute

```python
from QES.general_python.algebra.spectral_backend import (
    operator_spectral_function_multi_omega,
    susceptibility_bubble_multi_omega
)

# Many-body
if L <= 12:
    A = operator_spectral_function_multi_omega(
        omegas, E, V, O_spin_z,
        temperature=T, eta=eta
    )
    
# Quadratic
else:
    chi = susceptibility_bubble_multi_omega(
        omegas, E_sp, vertex=None,
        occupation=f, eta=eta
    )
    A = -np.imag(chi) / np.pi
```

### Step 5: Verify & Analyze

```python
# Check sum rules
from QES.general_python.algebra.spectral_backend import integrated_spectral_weight

total_weight = integrated_spectral_weight(A, omegas)
print(f"Integrated spectral weight: {total_weight:.4f}")
# Should be close to (degrees of freedom) or 1 depending on normalization

# Find peaks
peak_indices = signal.find_peaks(A, height=0.05)
peak_omegas = omegas[peak_indices[0]]
print(f"Excitation energies: {peak_omegas}")

# Calculate moments
E_avg = np.trapz(omegas * A, omegas) / total_weight
print(f"Mean excitation energy: {E_avg:.4f}")
```

---

## Verification & Sum Rules

### f-Sum Rule (Energy-Weighted Sum)

For any spectral function A(ω), the first moment must equal a specific value:

$$\int_{-\infty}^{\infty} d\omega \, \omega \, A(\omega) = \frac{1}{2\pi} \langle [O, [H, O^\dagger]] \rangle$$

**Check in code:**

```python
from scipy import integrate

# Compute expectation value of commutator
commutator = O @ H - H @ O  # [O, H]
double_commutator = O @ commutator - commutator @ O  # [O, [H, O]]
expected = np.real(np.trace(double_commutator)) / (2 * np.pi)

# Compute sum numerically
computed_sum = np.trapz(omegas * A, omegas)

print(f"Expected: {expected:.6f}")
print(f"Computed: {computed_sum:.6f}")
print(f"Error: {abs(expected - computed_sum)/abs(expected)*100:.2f}%")

# Rule of thumb: if error > 5%, check broadening η or frequency resolution
```

### Thermal Consistency

Verify that thermal weights behave correctly:

```python
from QES.general_python.algebra.spectral_backend import thermal_weights

E_grid = np.linspace(-5, 5, 100)
temps = [0.1, 0.5, 1.0, 2.0, 10.0]

for T in temps:
    rho = thermal_weights(E_grid, temperature=T)
    Z = np.sum(rho)
    E_avg = np.sum(rho * E_grid)
    
    assert np.isclose(Z, 1.0), "Weights not normalized!"
    print(f"T={T}: Z={Z:.4f}, ⟨E⟩={E_avg:.4f}")

# Expected: Z always = 1, E_avg increases monotonically with T
```

### Causality Check (Kramers-Kronig)

If you have only Im[χ], reconstruct Re[χ] and check consistency:

```python
from QES.general_python.algebra.spectral_backend import kramers_kronig_transform

# Assume you computed χ(ω) = Re[χ] + i Im[χ]
chi_computed = np.array([...])  # Full complex array
Re_computed = np.real(chi_computed)
Im_computed = np.imag(chi_computed)

# Reconstruct Real part from Imaginary
Re_kk = kramers_kronig_transform(Im_computed, omegas)

# Compare
error = np.trapz(np.abs(Re_computed - Re_kk), omegas)
print(f"Causality error: {error:.6f}")

if error < 0.01:
    print("✓ Causality satisfied (error < 1%)")
else:
    print("⚠ Causality violated (check calculation)")
```

---

## Summary Table: When to Use What

| Question | Answer | Function | Example |
|----------|--------|----------|---------|
| How many excitations at energy ω? | Many-body spectrum | `operator_spectral_function_multi_omega` | Hubbard model, T<12 sites |
| What's the band structure? | Single-particle | `spectral_function_multi_omega` | Tight-binding, ARPES |
| How does system respond to field? | Linear susceptibility | `susceptibility_lehmann` | Magnetic/charge response |
| What's the optical conductivity? | Kubo formula | `conductivity_kubo_bubble` | Optics, reflectance |
| Can I scale to 100 sites? | Need quadratic | `susceptibility_bubble_multi_omega` | Large systems, mean-field |
| Is my calculation consistent? | Check causality | `kramers_kronig_transform` | Verify Im ↔ Re |
| What's accessible at temp T? | Thermal weights | `thermal_weights` | Finite T properties |
| How many states per energy? | Density of states | `integrated_spectral_weight` | Sum rules, thermo |

---

## Quick Start Code

```python
"""
Quick example: Compute spin response in a 4-site Hubbard model
"""
import numpy as np
from QES.general_python.algebra.spectral_backend import (
    operator_spectral_function_multi_omega,
    thermal_weights
)

# Build system
L = 4
U = 2.0
t = 1.0
H = build_hubbard_hamiltonian(L, U, t)

# Solve
E, V = np.linalg.eigh(H)
print(f"✓ Diagonalized {len(E)}D Hilbert space")

# Define operator
S_z = np.zeros((2**L, 2**L))
for i in range(L):
    S_z += build_spin_z_single(i, L)

# Compute response
omegas = np.linspace(-6, 6, 300)
A_sz = operator_spectral_function_multi_omega(
    omegas, E, V, S_z,
    temperature=0.5, eta=0.03
)

print(f"✓ Computed spectral function")
print(f"  Range: [{A_sz.min():.4f}, {A_sz.max():.4f}]")

# Find excitations
peaks = np.where(np.gradient(A_sz) < -0.01)[0]
print(f"✓ Found {len(peaks)} excitations")

# Plot
import matplotlib.pyplot as plt
plt.plot(omegas, A_sz)
plt.xlabel("ω (excitation energy)")
plt.ylabel("A_Sz(ω)")
plt.show()
```

---

## References & Further Reading

**Foundational Theory:**

- Mahan, G.D. "Many-Particle Physics" (3rd ed., 1990)
- Fetter, A.L. & Walecka, J.D. "Quantum Theory of Many-Particle Systems" (1971)
- Abrikosov, Gorkov, Dzyaloshinski "Methods of Quantum Field Theory in Statistical Physics" (1965)

**Linear Response:**

- Kubo, R. "Statistical-Mechanical Theory of Irreversible Processes" (1957)
- Evans, D.J. & Morris, G.P. "Statistical Mechanics of Nonequilibrium Liquids" (1984)

**Exact Diagonalization:**

- Prelovšek, P. & Bonča, J. "Ground state of strongly interacting fermions in 2D" (2013)
- Dagotto, E. "Correlated electrons in high-temperature superconductors" (1994)

**Quadratic & Mean-Field:**

- Ashcroft, N.W. & Mermin, N.D. "Solid State Physics" (1976)
- Schrieffer, J.R. "Theory of Superconductivity" (1964)

**Experimental Methods:**

- Devereaux, T.P. & Hackl, R. "Electrodynamics of Correlated Electron Materials" (2007)
- Abbamonte et al. "X-ray scattering studies of electronic excitations in the cuprates" (2005)

---

## Contact & Support

**Framework:** PyQuSolver Response Function Stack  
**Maintainer:** Maksymilian Kliczkowski  
**Email:** <maksymilian.kliczkowski@pwr.edu.pl>  
**Version:** 1.0 (November 2025)  
**Status:** Production-Ready

**Modules:**

- `QES/general_python/algebra/spectral_backend.py` — Core implementations
- `QES/general_python/physics/response/susceptibility.py` — Many-body Lehmann
- `QES/general_python/physics/response/unified_response.py` — Unified interface

---

**Last Updated:** November 2025  
**Total Documentation:** 1 comprehensive guide  
**Code Examples:** Integrated throughout  
**Physics Level:** Graduate-level quantum many-body physics
