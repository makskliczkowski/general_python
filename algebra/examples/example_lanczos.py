#!/usr/bin/env python3
"""
Lanczos Eigenvalue Solver - Example

Demonstrates the Lanczos algorithm for finding extremal eigenvalues and
eigenvectors of large sparse symmetric matrices.

The Lanczos method is particularly effective for:
    - Quantum mechanics: Finding ground states and low-lying excited states
    - Graph Laplacians: Spectral clustering and graph analysis
    - PDEs: Eigenvalues of differential operators
    - Vibration analysis: Natural frequencies and mode shapes

Mathematical Background:
    For symmetric matrix A, Lanczos builds orthonormal basis V for Krylov subspace
    K_m(A, v) = span{v, Av, A^2v, ..., A^(m-1)v} such that:
        A V = V T + residual
    where T is tridiagonal. Eigenvalues of T approximate eigenvalues of A.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from QES.general_python.algebra.eigen import LanczosEigensolver, LanczosEigensolverScipy


def create_test_hamiltonian(n, kind='harmonic'):
    """
    Create test Hamiltonians for quantum mechanics examples.
    
    Args:
        n: Dimension (number of basis states)
        kind: 'harmonic' (quantum harmonic oscillator), 
              'spin_chain' (Heisenberg spin chain),
              'random' (random symmetric matrix)
    
    Returns:
        Symmetric matrix A
    """
    if kind == 'harmonic':
        # Quantum harmonic oscillator in position basis
        # H = -d^2/dx^2 + x^2 (in suitable units)
        # Discretized on grid
        h = 1.0 / (n + 1)
        x = np.linspace(-5, 5, n)
        
        # Kinetic energy (second derivative)
        T = np.zeros((n, n))
        T[range(n), range(n)] = -2.0
        if n > 1:
            T[range(n-1), range(1, n)] = 1.0
            T[range(1, n), range(n-1)] = 1.0
        T = T / h**2
        
        # Potential energy
        V = np.diag(x**2)
        
        H = T + V
        return H
    
    elif kind == 'spin_chain':
        # Heisenberg spin chain: H = \Sigma _i (\sigma ^x_i \sigma ^x_{i+1} + \sigma ^y_i \sigma ^y_{i+1} + \sigma ^z_i \sigma ^z_{i+1})
        # For small n, use full matrix representation
        
        if n > 12:
            raise ValueError("Spin chain only supported for n <= 12 (2^n states)")
        
        n_states = 2**n
        H = np.zeros((n_states, n_states))
        
        # Build Hamiltonian
        for state in range(n_states):
            for site in range(n - 1):
                # \sigma ^z_i \sigma ^z_{i+1}
                bit_i = (state >> site) & 1
                bit_ip1 = (state >> (site + 1)) & 1
                sz_i = 0.5 if bit_i == 0 else -0.5
                sz_ip1 = 0.5 if bit_ip1 == 0 else -0.5
                H[state, state] += sz_i * sz_ip1
                
                # \sigma ^x_i \sigma ^x_{i+1} + \sigma ^y_i \sigma ^y_{i+1}
                # These flip both spins
                new_state = state ^ (1 << site) ^ (1 << (site + 1))
                H[new_state, state] += 0.5
        
        return H
    
    elif kind == 'random':
        # Random symmetric matrix
        A = np.random.randn(n, n)
        A = 0.5 * (A + A.T)
        return A
    
    else:
        raise ValueError(f"Unknown kind: {kind}")


def example_quantum_harmonic_oscillator():
    """Find ground state of quantum harmonic oscillator."""
    print("=" * 70)
    print("Example 1: Quantum Harmonic Oscillator Ground State")
    print("=" * 70)
    
    n = 200  # Grid points
    H = create_test_hamiltonian(n, kind='harmonic')
    
    print(f"Hamiltonian size: {n}\times{n}")
    print(f"Finding 5 lowest eigenvalues (energy levels)...")
    
    # Use Lanczos to find ground state and first excited states
    solver = LanczosEigensolver(k=5, which='smallest', tol=1e-10)
    result = solver.solve(A=H)
    
    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"\nEnergy levels:")
    for i, E in enumerate(result.eigenvalues):
        print(f"  n={i}: E = {E:.6f}")
    
    print(f"\nResidual norms:")
    for i, res in enumerate(result.residual_norms):
        print(f"  n={i}: ||H|\psi > - E|\psi >|| = {res:.2e}")
    
    # Theoretical values for harmonic oscillator: E_n = 2n + 1 (in suitable units)
    print(f"\nNote: For quantum harmonic oscillator, E_n = \omega (n + 1/2)")
    print(f"      Numerical values depend on discretization")
    
    return result


def example_scipy_comparison():
    """Compare native Lanczos with SciPy implementation."""
    print("\n" + "=" * 70)
    print("Example 2: Native vs SciPy Lanczos")
    print("=" * 70)
    
    n = 500
    A = create_test_hamiltonian(n, kind='random')
    k = 6
    
    print(f"Matrix size: {n}\times{n}")
    print(f"Finding {k} smallest eigenvalues...")
    
    # Native implementation
    print(f"\nNative Lanczos:")
    solver_native = LanczosEigensolver(k=k, which='smallest', tol=1e-10)
    result_native = solver_native.solve(A=A)
    print(f"  Converged: {result_native.converged}")
    print(f"  Iterations: {result_native.iterations}")
    print(f"  Eigenvalues: {result_native.eigenvalues}")
    
    # SciPy implementation
    print(f"\nSciPy Lanczos (eigsh):")
    solver_scipy = LanczosEigensolverScipy(k=k, which='SA', tol=1e-10)
    result_scipy = solver_scipy.solve(A=A)
    print(f"  Converged: {result_scipy.converged}")
    print(f"  Eigenvalues: {result_scipy.eigenvalues}")
    
    # Compare results
    error = np.linalg.norm(result_native.eigenvalues - result_scipy.eigenvalues)
    print(f"\nDifference: {error:.2e}")
    
    return result_native, result_scipy


def example_matrix_free():
    """Matrix-free Lanczos using matvec function."""
    print("\n" + "=" * 70)
    print("Example 3: Matrix-Free Lanczos")
    print("=" * 70)
    
    # Large tridiagonal matrix (1D Laplacian)
    n = 1000
    
    def laplacian_matvec(x):
        """Apply 1D Laplacian: (Ax)_i = 2x_i - x_{i-1} - x_{i+1}"""
        y = np.zeros_like(x)
        y[0] = 2*x[0] - x[1]
        y[1:-1] = 2*x[1:-1] - x[:-2] - x[2:]
        y[-1] = 2*x[-1] - x[-2]
        return y
    
    print(f"Problem size: {n}")
    print(f"1D Laplacian (tridiagonal, never stored explicitly)")
    print(f"Finding 8 smallest eigenvalues...")
    
    solver = LanczosEigensolver(k=8, which='smallest', tol=1e-10)
    result = solver.solve(matvec=laplacian_matvec, n=n)
    
    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"\nSmallest eigenvalues:")
    for i, lam in enumerate(result.eigenvalues):
        print(f"  \lambda_{i+1} = {lam:.6f}")
    
    # Analytical eigenvalues: \lambda_k = 2(1 - cos(k\pi/(n+1))) for k=1,2,...,n
    analytical = np.array([2*(1 - np.cos((k+1)*np.pi/(n+1))) for k in range(8)])
    print(f"\nAnalytical eigenvalues:")
    for i, lam in enumerate(analytical):
        print(f"  \lambda_{i+1} = {lam:.6f}")
    
    error = np.linalg.norm(result.eigenvalues - analytical)
    print(f"\nError vs analytical: {error:.2e}")
    
    return result


def example_spin_chain_ground_state():
    """Find ground state of Heisenberg spin chain."""
    print("\n" + "=" * 70)
    print("Example 4: Heisenberg Spin Chain Ground State")
    print("=" * 70)
    
    n_spins = 8  # Chain length
    H = create_test_hamiltonian(n_spins, kind='spin_chain')
    n_states = 2**n_spins
    
    print(f"Spin chain length: {n_spins}")
    print(f"Hilbert space dimension: {n_states}")
    print(f"Finding ground state and first excited states...")
    
    solver = LanczosEigensolverScipy(k=5, which='SA', tol=1e-12)
    result = solver.solve(A=H)
    
    print(f"\nLowest energy levels:")
    for i, E in enumerate(result.eigenvalues):
        print(f"  E_{i} = {E:.8f}")
    
    # Ground state properties
    print(f"\nGround state (E_0 = {result.eigenvalues[0]:.8f}):")
    psi_0 = result.eigenvectors[:, 0]
    print(f"  Norm: {np.linalg.norm(psi_0):.10f}")
    print(f"  Largest amplitude: {np.max(np.abs(psi_0)):.6f}")
    
    # Gap to first excited state
    gap = result.eigenvalues[1] - result.eigenvalues[0]
    print(f"\nEnergy gap to first excited state: Δ = {gap:.8f}")
    
    return result


def example_convergence_behavior():
    """Study convergence behavior with different parameters."""
    print("\n" + "=" * 70)
    print("Example 5: Convergence Behavior")
    print("=" * 70)
    
    n = 300
    A = create_test_hamiltonian(n, kind='random')
    k = 5
    
    print(f"Matrix size: {n}\times{n}")
    print(f"Testing different max_iter values...")
    
    max_iters = [10, 20, 40, 80]
    
    print(f"\n{'max_iter':<10} {'Converged':<12} {'Iterations':<12} {'Min Residual':<15}")
    print("-" * 60)
    
    for max_iter in max_iters:
        solver = LanczosEigensolver(k=k, which='smallest', max_iter=max_iter, tol=1e-10)
        result = solver.solve(A=A)
        
        min_res = np.min(result.residual_norms) if result.residual_norms is not None else float('nan')
        print(f"{max_iter:<10} {str(result.converged):<12} {result.iterations:<12} {min_res:<15.2e}")
    
    print(f"\nObservation: More iterations -> better convergence")
    print(f"             Typically need max_iter ~ 2k to 3k for good convergence")


def example_largest_eigenvalues():
    """Find largest eigenvalues instead of smallest."""
    print("\n" + "=" * 70)
    print("Example 6: Largest Eigenvalues")
    print("=" * 70)
    
    n = 400
    A = create_test_hamiltonian(n, kind='random')
    k = 6
    
    print(f"Matrix size: {n}\times{n}")
    
    # Find smallest
    solver_small = LanczosEigensolver(k=k, which='smallest')
    result_small = solver_small.solve(A=A)
    
    # Find largest
    solver_large = LanczosEigensolver(k=k, which='largest')
    result_large = solver_large.solve(A=A)
    
    # Find both ends
    solver_both = LanczosEigensolver(k=k, which='both')
    result_both = solver_both.solve(A=A)
    
    print(f"\nSmallest {k} eigenvalues:")
    print(f"  {result_small.eigenvalues}")
    
    print(f"\nLargest {k} eigenvalues:")
    print(f"  {result_large.eigenvalues}")
    
    print(f"\nBoth ends (k={k}):")
    print(f"  {result_both.eigenvalues}")
    
    # Verify with full diagonalization (for small matrix)
    if n <= 500:
        evals_full = np.linalg.eigvalsh(A)
        print(f"\nFull spectrum range: [{evals_full[0]:.4f}, {evals_full[-1]:.4f}]")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LANCZOS EIGENVALUE SOLVER - EXAMPLES")
    print("=" * 70)
    print("\nLanczos algorithm for symmetric/Hermitian eigenvalue problems")
    print("Finding extremal eigenvalues of large sparse matrices")
    print("=" * 70)
    
    # Run all examples
    example_quantum_harmonic_oscillator()
    example_scipy_comparison()
    example_matrix_free()
    example_spin_chain_ground_state()
    example_convergence_behavior()
    example_largest_eigenvalues()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Lanczos is ideal for finding a few extremal eigenvalues")
    print("2. Memory-efficient: O(kn) instead of O(n^2)")
    print("3. Works with matrix-free operators (matvec functions)")
    print("4. Reorthogonalization improves numerical stability")
    print("5. SciPy wrapper (eigsh) recommended for production use")
    print("6. Convergence depends on spectral gap and iteration count")
    print("\nApplications: Quantum mechanics, graph theory, PDEs, vibration analysis")
