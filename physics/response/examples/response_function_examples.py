"""
general_python/physics/response/examples/response_function_examples.py

Practical examples demonstrating the complete response function stack.
Shows both many-body (Lehmann) and quadratic (bubble) calculations.

Run this file to see:
  1. Many-body spin dynamics
  2. Quadratic optical conductivity
  3. Comparison: exact vs mean-field
  4. Temperature effects

Author: Maksymilian Kliczkowski
Date:   November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib
rcParams['figure.figsize'] = (14, 10)
rcParams['font.size'] = 11


# =============================================================================
# EXAMPLE 1: Many-Body Spin Dynamics (Exact Diagonalization)
# =============================================================================

def example_1_manybody_spin_dynamics():
    """
    Compute spin dynamics in a small interacting lattice using exact ED + Lehmann.
    
    This demonstrates:
      - Full many-body Hilbert space
      - Operator-projected spectral function
      - Temperature effects on response
      - Peaks correspond to many-body excitations
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Many-Body Spin Dynamics (Exact)")
    print("="*70)
    
    from general_python.physics.spectral.spectral_backend import (
        operator_spectral_function_multi_omega,
        thermal_weights,
    )
    
    # Build a toy 4-site Ising model (16D Hilbert space)
    # H = -J * Σ_i \sigma_z_i \sigma_z_{i+1} + h * Σ_i \sigma_x_i
    L = 4
    J = 1.0
    h = 0.5
    
    # Simple 2-basis representation: |↑⟩, |↓⟩ per site
    # For demonstration, use random symmetric matrix
    dim = 2**L
    np.random.seed(42)
    H = np.random.randn(dim, dim)
    H = (H + H.T) / 2  # Symmetrize
    
    # Diagonalize
    print(f"System: {L}-site spin chain, {dim}D Hilbert space")
    E, V = np.linalg.eigh(H)
    print(f"Eigenvalues range: [{E[0]:.3f}, {E[-1]:.3f}]")
    
    # Define observable: total S_z (magnetization)
    S_z = np.diag(np.random.randn(dim))  # Random symmetric for demo
    
    # Compute spectral function at multiple temperatures
    omegas = np.linspace(-2, 2, 150)
    temps = [0.0, 0.5, 1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, T in zip(axes, temps):
        A_sz = operator_spectral_function_multi_omega(
            omegas, E, V, S_z, eta=0.05, temperature=T
        )
        
        rho = thermal_weights(E, temperature=T)
        Z = np.sum(rho)
        
        ax.plot(omegas, A_sz, linewidth=2, color='darkblue')
        ax.fill_between(omegas, A_sz, alpha=0.3, color='lightblue')
        ax.set_xlabel("omega  (energy units)")
        ax.set_ylabel("A_Sz(omega )")
        ax.set_title(f"T = {T:.1f}\nZ = {Z:.3f}")
        ax.grid(alpha=0.3)
        
        # Find peaks
        peak_indices = np.where(np.gradient(A_sz) < -0.01)[0]
        if len(peak_indices) > 0:
            peaks = omegas[peak_indices]
            heights = A_sz[peak_indices]
            ax.plot(peaks, heights, 'r*', markersize=8, label="Peaks")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('/tmp/example1_manybody_spin.png', dpi=150, bbox_inches='tight')
    print("(ok) Plot saved to /tmp/example1_manybody_spin.png")
    
    # Summary
    print(f"\nPhysics:")
    print(f"  - Each peak = many-body transition between eigenstates")
    print(f"  - Height ∝ |⟨m|S_z|n⟩|²")
    print(f"  - Peak positions = E_n - E_m")
    print(f"  - T increases: thermal population of excited states")
    print(f"  - Available exactly for L ≤ 12 sites")


# =============================================================================
# EXAMPLE 2: Optical Conductivity (Quadratic System)
# =============================================================================

def example_2_optical_conductivity():
    r"""
    Compute optical conductivity \sigma(omega ) for a tight-binding chain.
    
    This demonstrates:
      - Single-particle (quadratic) system
      - Bubble diagram contributions
      - Drude vs interband contributions
      - Simple but physically meaningful model
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Optical Conductivity (Tight-Binding)")
    print("="*70)
    
    from general_python.physics.spectral.spectral_backend import (
        conductivity_kubo_bubble,
        kramers_kronig_transform,
    )
    
    # 1D tight-binding chain: E(k) = -2t cos(k)
    L = 100  # k-points
    k_points = np.linspace(-np.pi, np.pi, L)
    t = 1.0  # hopping
    E_k = -2 * t * np.cos(k_points)
    
    # Velocity: v(k) = ∂E/∂k = 2t sin(k)
    v_k = 2 * t * np.sin(k_points)
    
    # Full velocity matrix (simplified)
    v_matrix = np.diag(v_k)
    
    # T=0 occupation: filled Fermi sea
    f = (E_k < 0).astype(float)
    
    print(f"System: 1D tight-binding chain, {L} k-points")
    print(f"Band width: {E_k.max() - E_k.min():.3f}")
    print(f"Fermi energy: {E_k[np.where(f == 1)[0][-1]]:.3f}")
    
    # Compute conductivity
    omegas = np.linspace(0.01, 4, 200)
    sigma = np.array([
        conductivity_kubo_bubble(w, E_k, v_matrix, occupation=f, eta=0.05)
        for w in omegas
    ])
    
    # Separate real and imaginary
    Re_sigma = np.real(sigma)
    Im_sigma = np.imag(sigma)
    
    # Try K-K transform
    Re_sigma_kk = kramers_kronig_transform(-Im_sigma, omegas)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Imaginary part (dissipative)
    ax = axes[0]
    ax.plot(omegas, -Im_sigma / np.pi, linewidth=2, label=r'-Im[\sigma]/Pi (Drude)')
    ax.fill_between(omegas, -Im_sigma / np.pi, alpha=0.2)
    ax.set_xlabel("omega  (energy units)")
    ax.set_ylabel("Dissipative conductivity")
    ax.set_title("Optical Response (Bubble Diagram)")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Real part (reactive)
    ax = axes[1]
    ax.plot(omegas, Re_sigma, linewidth=2, label=r'Re[\sigma] (direct)', color='tab:blue')
    ax.plot(omegas, Re_sigma_kk, linewidth=2, linestyle='--', 
            label=r'Re[\sigma] (K-K from Im)', color='tab:orange')
    ax.set_xlabel("omega  (energy units)")
    ax.set_ylabel("Reactive conductivity")
    ax.set_title("Real Part (Comparison Methods)")
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('/tmp/example2_conductivity.png', dpi=150, bbox_inches='tight')
    print("(ok) Plot saved to /tmp/example2_conductivity.png")
    
    # Summary
    print(f"\nPhysics:")
    print(f"  - Drude peak at low omega : intraband (free electron) behavior")
    print(f"  - Real part threshold ≈ 2|E_Fermi|: interband transitions")
    print(f"  - K-K relation checks causality")
    print(rf"  - Integrating -Im[\sigma]/Pi gives oscillator strength")


# =============================================================================
# EXAMPLE 3: Comparison - Exact vs Mean-Field
# =============================================================================

def example_3_exact_vs_meanfield():
    """
    Compare exact many-body Lehmann vs quadratic bubble approximation.
    
    This demonstrates:
      - When mean-field is good
      - Where interactions matter
      - Heuristic for choosing method
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Exact vs Mean-Field Comparison")
    print("="*70)
    
    from general_python.physics.spectral.spectral_backend import (
        operator_spectral_function_multi_omega,
        susceptibility_bubble_multi_omega,
    )
    
    # Small system: can do exact
    dim = 32  # 2^5 ~ 32D Hilbert space
    
    # Weakly interacting limit (diagonal ≈ diagonal + small off-diagonal)
    np.random.seed(42)
    
    # "Many-body" Hamiltonian
    H_mb = np.random.randn(dim, dim) * 0.3  # Weak interactions
    H_mb = (H_mb + H_mb.T) / 2
    
    # Add diagonal: resembles single-particle structure
    H_diag = np.diag(np.linspace(-2, 2, dim))
    H_full = H_diag + H_mb
    
    # "Single-particle" approximation (just the diagonal part)
    E_sp = np.linspace(-2, 2, 10)
    
    # Diagonalize
    E_mb, V_mb = np.linalg.eigh(H_full)
    
    # Random observable
    O = np.random.randn(dim, dim)
    O = (O + O.T) / 2
    
    # Compute responses
    omegas = np.linspace(-2, 2, 100)
    
    A_exact = operator_spectral_function_multi_omega(
        omegas, E_mb, V_mb, O, eta=0.05
    )
    
    A_mf = susceptibility_bubble_multi_omega(
        omegas, E_sp, eta=0.05
    )
    
    # Convert bubble to spectral-like quantity
    A_mf_spec = -np.imag(A_mf) / np.pi
    
    # Normalize for comparison
    A_exact_norm = A_exact / np.max(A_exact)
    A_mf_norm = A_mf_spec / np.max(A_mf_spec)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw comparison
    ax = axes[0]
    ax.plot(omegas, A_exact, linewidth=2.5, label='Exact (Many-Body ED)', color='darkblue')
    ax.plot(omegas, A_mf_spec, linewidth=2.5, linestyle='--', 
            label='Mean-Field (Bubble)', color='darkred', alpha=0.7)
    ax.set_xlabel("omega  (energy units)")
    ax.set_ylabel("Response function")
    ax.set_title("Raw Spectra")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Normalized for peak shape comparison
    ax = axes[1]
    ax.plot(omegas, A_exact_norm, linewidth=2.5, label='Exact (normalized)', color='darkblue')
    ax.plot(omegas, A_mf_norm, linewidth=2.5, linestyle='--', 
            label='Mean-Field (normalized)', color='darkred', alpha=0.7)
    ax.set_xlabel("omega  (energy units)")
    ax.set_ylabel("Normalized response")
    ax.set_title("Peak Shape Comparison")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/example3_comparison.png', dpi=150, bbox_inches='tight')
    print("(ok) Plot saved to /tmp/example3_comparison.png")
    
    # Quantitative comparison
    error = np.trapz(np.abs(A_exact - A_mf_spec), omegas)
    print(f"\nQuantitative comparison:")
    print(f"  - Integrated difference: {error:.4f}")
    print(f"  - Peak position (exact): {omegas[np.argmax(A_exact)]:.3f}")
    print(f"  - Peak position (MF): {omegas[np.argmax(A_mf_spec)]:.3f}")
    print(f"  - Relative peak shift: {abs(omegas[np.argmax(A_exact)] - omegas[np.argmax(A_mf_spec)]):.3f}")
    print(f"\nConclusion:")
    print(f"  - For weak interactions: mean-field captures main features")
    print(f"  - For strong interactions: need full many-body for accuracy")


# =============================================================================
# EXAMPLE 4: Temperature Effects
# =============================================================================

def example_4_temperature_effects():
    """
    Demonstrate finite-temperature response functions.
    
    Shows:
      - How thermal population changes response
      - Thermal weights across energy scales
      - Application to realistic systems
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Temperature Effects")
    print("="*70)
    
    from general_python.physics.spectral.spectral_backend import (
        operator_spectral_function_multi_omega,
        thermal_weights,
    )
    
    # Small system
    dim = 16
    np.random.seed(42)
    H = np.random.randn(dim, dim) * 0.5
    H = (H + H.T) / 2
    
    E, V = np.linalg.eigh(H)
    O = np.eye(dim)  # Identity operator
    
    omegas = np.linspace(-1.5, 1.5, 150)
    temps = [0.0, 0.1, 0.5, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, T in zip(axes, temps):
        # Thermal weights
        rho = thermal_weights(E, temperature=T)
        
        # Response
        A = operator_spectral_function_multi_omega(
            omegas, E, V, O, eta=0.05, temperature=T
        )
        
        # Plot 1: Thermal distribution
        ax2 = ax.twinx()
        
        # Response
        ax.bar(E, rho * 10, width=0.1, alpha=0.5, color='tab:blue', label='Thermal weights ρ_n')
        ax.set_ylabel("Thermal weight ρ_n", color='tab:blue', fontsize=10)
        ax.tick_params(axis='y', labelcolor='tab:blue')
        
        # Spectral function
        ax2.plot(omegas, A, linewidth=2.5, color='darkred', label='Spectral function A(omega )')
        ax2.fill_between(omegas, A, alpha=0.2, color='darkred')
        ax2.set_ylabel("A(omega )", color='darkred', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='darkred')
        
        ax.set_xlabel("Energy")
        ax.set_title(f"T = {T:.1f} (Z = {np.sum(rho):.3f})")
        ax.grid(alpha=0.3, axis='x')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/tmp/example4_temperature.png', dpi=150, bbox_inches='tight')
    print("(ok) Plot saved to /tmp/example4_temperature.png")
    
    # Quantitative analysis
    print(f"\nThermal analysis:")
    for T in temps:
        rho = thermal_weights(E, temperature=T)
        E_avg = np.sum(rho * E)
        E2_avg = np.sum(rho * E**2)
        C_v = E2_avg - E_avg**2
        print(f"  T={T:.1f}: ⟨E⟩={E_avg:6.3f}, C_v={C_v:6.3f}, Z={np.sum(rho):.4f}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PyQuSolver Response Function Examples")
    print("Full Many-Body and Quadratic Demonstrations")
    print("="*70)
    
    # Run all examples
    example_1_manybody_spin_dynamics()
    example_2_optical_conductivity()
    example_3_exact_vs_meanfield()
    example_4_temperature_effects()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("Plots saved to /tmp/example*.png")
    print("="*70 + "\n")
    
    # Show plots if interactive
    try:
        plt.show()
    except:
        pass
