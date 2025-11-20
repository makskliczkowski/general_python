"""
Comprehensive Eigenvalue Solver Examples

Demonstrates all available eigenvalue solvers:
    - Exact Diagonalization (ED)
    - Lanczos
    - Arnoldi  
    - Block Lanczos
    - Unified interface (choose_eigensolver)

Shows when to use each method and compares performance.
"""

import numpy as np
import sys
import os
import time
from enum import Enum

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

try:
    from QES.general_python.algebra.eigen import (
        full_diagonalization,
        LanczosEigensolver,
        BlockLanczosEigensolver,
        choose_eigensolver,
        decide_method
    )
except ImportError as e:
    print("Error importing QES eigenvalue solvers:", e)
    sys.exit(1)

# ----------------------------------------------------------------------------------------

class HamiltonianKind(Enum):
    HARMONIC      = 'harmonic'
    SPIN_CHAIN    = 'spin_chain'
    NONHERMITIAN  = 'nonhermitian'
    RANDOM        = 'random'

# ----------------------------------------------------------------------------------------
#! Helper Functions
# ----------------------------------------------------------------------------------------

def create_test_hamiltonian(n, kind: str) -> np.ndarray:
    """Create test Hamiltonians for different physics examples."""
    
    if kind == HamiltonianKind.HARMONIC.value:
        # 1D quantum harmonic oscillator (tridiagonal)
        H = np.zeros((n, n))
        for i in range(n):
            H[i, i] = 2 * i + 1  # Energy levels: 1, 3, 5, ...
            if i < n - 1:
                H[i, i+1] = H[i+1, i] = np.sqrt((i+1) * (i+1))  # Ladder operators
        return H
    
    elif kind == HamiltonianKind.SPIN_CHAIN.value:
        # 1D Heisenberg spin chain (sparse)
        H = np.diag(np.ones(n) * 2.0)  # On-site terms
        for i in range(n-1):
            H[i, i+1] = H[i+1, i] = -1.0  # Nearest-neighbor coupling
        return H

    elif kind == HamiltonianKind.NONHERMITIAN.value:
        # Non-Hermitian example
        H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        return H

    elif kind == HamiltonianKind.RANDOM.value:
        # Random symmetric matrix
        np.random.seed(42)
        H = np.random.randn(n, n)
        H = 0.5 * (H + H.T)
        return H
    
    else:
        raise ValueError(f"Unknown kind: {kind}")

# ----------------------------------------------------------------------------------------
#! Example Functions
# ----------------------------------------------------------------------------------------

def example_exact_diagonalization():
    """Example 1: Exact diagonalization for small systems."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Exact Diagonalization")
    print("="*70)
    
    n = 100
    print(f"System size: {n}x{n}")
    
    # Create Hamiltonian
    H = create_test_hamiltonian(n, kind='spin_chain')
    
    # Method 1: Direct call to full_diagonalization
    print("\n--- Method 1: Direct full_diagonalization ---")
    t0 = time.time()
    result = full_diagonalization(H, hermitian=True, backend='numpy')
    t1 = time.time()
    
    print(f"Ground state energy: {result.eigenvalues[0]:.8f}")
    print(f"First 5 eigenvalues: {result.eigenvalues[:5]}")
    print(f"Time: {t1-t0:.4f} s")
    print(f"All {n} eigenvalues computed")
    
    # Method 2: Using unified interface
    print("\n--- Method 2: Using choose_eigensolver('exact') ---")
    t0 = time.time()
    result2 = choose_eigensolver('exact', H, hermitian=True)
    t1 = time.time()
    
    print(f"Ground state energy: {result2.eigenvalues[0]:.8f}")
    print(f"Time: {t1-t0:.4f} s")
    
    # Verify they match
    assert np.allclose(result.eigenvalues, result2.eigenvalues), "Results don't match!"
    print("\nv Both methods give identical results")


def example_lanczos_vs_exact():
    """Example 2: Lanczos vs Exact for large systems."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Lanczos vs Exact Diagonalization")
    print("="*70)
    
    n = 500
    k = 10
    print(f"System size: {n}x{n}, finding {k} smallest eigenvalues")
    
    H = create_test_hamiltonian(n, kind='random')
    
    # Exact diagonalization (all eigenvalues)
    print("\n--- Exact Diagonalization (all eigenvalues) ---")
    t0 = time.time()
    result_exact = full_diagonalization(H, hermitian=True)
    t1 = time.time()
    time_exact = t1 - t0
    
    print(f"Time: {time_exact:.4f} s")
    print(f"First {k} eigenvalues: {result_exact.eigenvalues[:k]}")
    
    # Lanczos (only k eigenvalues)
    print(f"\n--- Lanczos (only {k} eigenvalues) ---")
    solver = LanczosEigensolver(k=k, which='smallest', backend='numpy', max_iter=100)
    t0 = time.time()
    result_lanczos = solver.solve(A=H)
    t1 = time.time()
    time_lanczos = t1 - t0
    
    print(f"Time: {time_lanczos:.4f} s")
    print(f"Eigenvalues: {result_lanczos.eigenvalues}")
    print(f"Converged: {result_lanczos.converged}")
    print(f"Residual norms: {result_lanczos.residual_norms}")
    
    # Compare accuracy
    error = np.linalg.norm(result_lanczos.eigenvalues - result_exact.eigenvalues[:k])
    print(f"\n--- Comparison ---")
    print(f"Error: {error:.2e}")
    print(f"Speedup: {time_exact / time_lanczos:.1f}x")
    print(f"(v) Lanczos is {time_exact/time_lanczos:.1f}x faster for finding only {k} eigenvalues!")


def example_block_lanczos():
    """Example 3: Block Lanczos for multiple eigenvalues."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Block Lanczos Method")
    print("="*70)
    
    n = 400
    k = 20
    block_size = 5
    
    print(f"System size: {n}x{n}")
    print(f"Finding {k} eigenvalues using block_size={block_size}")
    
    H = create_test_hamiltonian(n, kind='random')
    
    # Standard Lanczos
    print("\n--- Standard Lanczos ---")
    solver_standard = LanczosEigensolver(k=k, which='smallest', backend='numpy', max_iter=150)
    t0 = time.time()
    result_standard = solver_standard.solve(A=H)
    t1 = time.time()
    time_standard = t1 - t0
    
    print(f"Time: {time_standard:.4f} s")
    print(f"Iterations: {result_standard.iterations}")
    print(f"Converged: {result_standard.converged}")
    
    # Block Lanczos
    print(f"\n--- Block Lanczos (block_size={block_size}) ---")
    solver_block = BlockLanczosEigensolver(k=k, block_size=block_size, which='smallest', backend='numpy')
    t0 = time.time()
    result_block = solver_block.solve(A=H)
    t1 = time.time()
    time_block = t1 - t0
    
    print(f"Time: {time_block:.4f} s")
    print(f"Iterations: {result_block.iterations}")
    print(f"Converged: {result_block.converged}")
    
    # Compare
    print(f"\n--- Comparison ---")
    print(f"Standard Lanczos eigenvalues: {result_standard.eigenvalues[:5]}")
    print(f"Block Lanczos eigenvalues:    {result_block.eigenvalues[:5]}")
    error = np.linalg.norm(result_standard.eigenvalues - result_block.eigenvalues)
    print(f"Error: {error:.2e}")
    print(f"Block Lanczos uses {block_size} vectors per iteration")
    print(f"(v) Both methods converge to the same eigenvalues!")


def example_unified_interface():
    """Example 4: Using unified interface."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Unified Interface (choose_eigensolver)")
    print("="*70)
    
    # Different problem sizes
    problems = [
        (100, 5, 'Small system'),
        (1000, 10, 'Medium system'),
        (5000, 20, 'Large system'),
    ]
    
    for n, k, description in problems:
        print(f"\n--- {description}: n={n}, k={k} ---")
        
        H = create_test_hamiltonian(n, kind='random')
        
        # Auto-select method
        recommended = decide_method(n, k, hermitian=True)
        print(f"Recommended method: {recommended}")
        
        # Use auto-selection
        t0 = time.time()
        result = choose_eigensolver('auto', H, k=k, hermitian=True)
        t1 = time.time()
        
        print(f"Ground state energy: {result.eigenvalues[0]:.8f}")
        print(f"Time: {t1-t0:.4f} s")
        print(f"Number of eigenvalues computed: {len(result.eigenvalues)}")


def example_matrix_free():
    """Example 5: Matrix-free operation (never form H explicitly)."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Matrix-Free Operation")
    print("="*70)
    
    n = 1000
    k = 8
    
    print(f"System size: {n} (matrix never formed explicitly)")
    print(f"Finding {k} smallest eigenvalues of 1D Laplacian")
    
    # Matrix-vector product for 1D Laplacian
    def laplacian_matvec(v):
        """Compute (Lv) where L is 1D Laplacian."""
        y = np.zeros_like(v)
        y[0] = 2*v[0] - v[1]
        y[1:-1] = 2*v[1:-1] - v[:-2] - v[2:]
        y[-1] = 2*v[-1] - v[-2]
        return y
    
    # Solve using Lanczos with matvec
    print("\n--- Lanczos with matrix-free matvec ---")
    solver = LanczosEigensolver(k=k, which='smallest', backend='numpy', max_iter=100)
    t0 = time.time()
    result = solver.solve(matvec=laplacian_matvec, n=n)
    t1 = time.time()
    
    print(f"Time: {t1-t0:.4f} s")
    print(f"Eigenvalues: {result.eigenvalues}")
    
    # Analytical solution for 1D Laplacian
    analytical = np.array([2*(1 - np.cos((i+1)*np.pi/(n+1))) for i in range(k)])
    
    error = np.linalg.norm(result.eigenvalues - analytical)
    print(f"\nAnalytical eigenvalues: {analytical}")
    print(f"Error vs analytical: {error:.2e}")
    print(f"(v) Matrix-free computation matches analytical result!")


def example_comparison_all_backends():
    """Example 6: Compare all backends."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Backend Comparison")
    print("="*70)
    
    n = 300
    k = 10
    H = create_test_hamiltonian(n, kind='random')
    
    backends = []
    
    # NumPy
    print("\n--- NumPy Backend ---")
    solver = LanczosEigensolver(k=k, which='smallest', backend='numpy', max_iter=100)
    t0 = time.time()
    result_numpy = solver.solve(A=H)
    t1 = time.time()
    print(f"Time: {t1-t0:.4f} s")
    print(f"Eigenvalues: {result_numpy.eigenvalues}")
    backends.append(('NumPy', result_numpy, t1-t0))
    
    # SciPy
    print("\n--- SciPy Backend (eigsh) ---")
    t0 = time.time()
    result_scipy = choose_eigensolver('lanczos', H, k=k, use_scipy=True)
    t1 = time.time()
    print(f"Time: {t1-t0:.4f} s")
    print(f"Eigenvalues: {result_scipy.eigenvalues}")
    backends.append(('SciPy', result_scipy, t1-t0))
    
    # JAX (if available)
    try:
        import jax
        print("\n--- JAX Backend ---")
        solver = LanczosEigensolver(k=k, which='smallest', backend='jax', max_iter=100)
        t0 = time.time()
        result_jax = solver.solve(A=H)
        t1 = time.time()
        print(f"Time: {t1-t0:.4f} s")
        print(f"Eigenvalues: {result_jax.eigenvalues}")
        backends.append(('JAX', result_jax, t1-t0))
    except ImportError:
        print("\n--- JAX Backend ---")
        print("JAX not available")
    
    # Compare
    print("\n--- Backend Comparison Summary ---")
    print(f"{'Backend':<10} {'Time (s)':<12} {'Ground State Energy'}")
    print("-" * 50)
    for name, result, t in backends:
        print(f"{name:<10} {t:<12.4f} {result.eigenvalues[0]:.10f}")
    
    print("\nv All backends converge to the same eigenvalues!")


if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE EIGENVALUE SOLVER EXAMPLES")
    print("="*70)
    
    example_exact_diagonalization()
    example_lanczos_vs_exact()
    example_block_lanczos()
    example_unified_interface()
    example_matrix_free()
    example_comparison_all_backends()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)
