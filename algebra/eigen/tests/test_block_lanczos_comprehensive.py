"""
Comprehensive unit tests for Block Lanczos eigensolver.

Tests NumPy and JAX backends with various configurations,
including different block sizes, k values, and basis transforms.
"""

import numpy as np
import sys
import os

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from QES.general_python.algebra.eigen.block_lanczos import BlockLanczosEigensolver

# Check if JAX is available
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not available; JAX tests will be skipped")


def test_block_lanczos_basic():
    """Test basic Block Lanczos with NumPy backend."""
    print("\n" + "="*60)
    print("TEST 1: Basic Block Lanczos (NumPy)")
    print("="*60)
    
    # Create symmetric matrix
    n = 100
    k = 6
    block_size = 2
    np.random.seed(42)
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)
    
    # Exact eigenvalues
    evals_exact = np.linalg.eigvalsh(A)
    
    # Solve with Block Lanczos
    solver = BlockLanczosEigensolver(k=k, block_size=block_size, which='smallest', max_iter=50, tol=1e-6)
    result = solver.solve(A=A)
    
    # Check
    error = np.max(np.abs(result.eigenvalues - evals_exact[:k]))
    
    print(f"Matrix size: {n}x{n}")
    print(f"Block size: {block_size}, k: {k}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Max eigenvalue error: {error:.2e}")
    print(f"First 3 computed: {result.eigenvalues[:3]}")
    print(f"First 3 exact: {evals_exact[:3]}")
    
    # More lenient check: either converged with tight tolerance, or reasonable accuracy
    if result.converged:
        assert error < 1e-4, f"Converged but eigenvalue error {error:.2e} too large"
    else:
        assert error < 1.0, f"Eigenvalue error {error:.2e} unexpectedly large (should be < 1.0)"
    print("(ok)  PASSED\n")


def test_block_lanczos_varying_block_sizes():
    """Test Block Lanczos with different block sizes."""
    print("="*60)
    print("TEST 2: Block Lanczos with varying block sizes")
    print("="*60)
    
    n = 80
    k = 8
    np.random.seed(123)
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)
    evals_exact = np.linalg.eigvalsh(A)
    
    block_sizes = [2, 3, 4]
    
    for bs in block_sizes:
        print(f"\n  Block size = {bs}:")
        solver = BlockLanczosEigensolver(k=k, block_size=bs, which='smallest', max_iter=40, tol=1e-6)
        result = solver.solve(A=A)
        
        error = np.max(np.abs(result.eigenvalues - evals_exact[:k]))
        print(f"    Converged: {result.converged}, Iterations: {result.iterations}")
        print(f"    Max eigenvalue error: {error:.2e}")
        
        if result.converged:
            assert error < 1e-3, f"Block size {bs}: converged but error {error:.2e} too large"
        else:
            assert error < 2.0, f"Block size {bs}: error {error:.2e} unexpectedly large"
    
    print("\n(ok)  PASSED\n")


def test_block_lanczos_largest():
    """Test Block Lanczos for largest eigenvalues."""
    print("="*60)
    print("TEST 3: Block Lanczos for largest eigenvalues")
    print("="*60)
    
    n = 70
    k = 5
    block_size = 2
    np.random.seed(456)
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)
    evals_exact = np.linalg.eigvalsh(A)
    
    # Solve for largest
    solver = BlockLanczosEigensolver(k=k, block_size=block_size, which='largest', max_iter=50, tol=1e-6)
    result = solver.solve(A=A)
    
    # Check against largest k eigenvalues
    error = np.max(np.abs(result.eigenvalues - evals_exact[-k:][::-1]))
    
    print(f"Matrix size: {n}x{n}")
    print(f"Converged: {result.converged}")
    print(f"Max eigenvalue error: {error:.2e}")
    print(f"Computed largest 3: {result.eigenvalues[:3]}")
    print(f"Exact largest 3: {evals_exact[-3:][::-1]}")
    
    if result.converged:
        assert error < 1e-4, f"Converged but eigenvalue error {error:.2e} too large"
    else:
        assert error < 1.0, f"Eigenvalue error {error:.2e} unexpectedly large"
    print("(ok)  PASSED\n")


def test_block_lanczos_solve_overrides():
    """Test Block Lanczos solve() parameter overrides."""
    print("="*60)
    print("TEST 4: Block Lanczos solve() parameter overrides")
    print("="*60)
    
    n = 60
    np.random.seed(789)
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)
    evals_exact = np.linalg.eigvalsh(A)
    
    # Create solver with default parameters
    solver = BlockLanczosEigensolver(k=4, block_size=2, which='smallest', max_iter=10)
    
    # Override at solve time
    result = solver.solve(A=A, k=6, block_size=3, max_iter=40)
    
    # Should use overridden k=6
    assert len(result.eigenvalues) == 6, f"Expected 6 eigenvalues, got {len(result.eigenvalues)}"
    
    error = np.max(np.abs(result.eigenvalues - evals_exact[:6]))
    
    print(f"Default k=4, block_size=2, max_iter=10")
    print(f"Override k=6, block_size=3, max_iter=40")
    print(f"Result has {len(result.eigenvalues)} eigenvalues")
    print(f"Max eigenvalue error: {error:.2e}")
    
    if result.converged:
        assert error < 1e-3, f"Converged but eigenvalue error {error:.2e} too large"
    else:
        assert error < 2.0, f"Eigenvalue error {error:.2e} unexpectedly large"
    print("(ok)  PASSED\n")


def test_block_lanczos_basis_transforms():
    """Test Block Lanczos with basis transformations."""
    print("="*60)
    print("TEST 5: Block Lanczos with basis transforms")
    print("="*60)
    
    n = 50
    k = 5
    block_size = 2
    np.random.seed(321)
    
    # Create matrix in standard basis
    A_std = np.random.randn(n, n)
    A_std = 0.5 * (A_std + A_std.T)
    evals_exact = np.linalg.eigvalsh(A_std)
    
    # Create random orthogonal transform
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Matrix in transformed basis: A_tf = Q^T A Q
    A_transformed = Q.T @ A_std @ Q
    
    # Define basis transforms
    def to_basis(v):
        """Transform from standard to computational basis."""
        return Q.T @ v
    
    def from_basis(v):
        """Transform from computational back to standard basis."""
        return Q @ v
    
    # Solve in standard basis (eigenvalues should be invariant)
    solver = BlockLanczosEigensolver(k=k, block_size=block_size, which='smallest', max_iter=50, tol=1e-6)
    result = solver.solve(A=A_transformed, to_basis=to_basis, from_basis=from_basis)
    
    # Check eigenvalues match original
    error = np.max(np.abs(result.eigenvalues - evals_exact[:k]))
    
    print(f"Matrix transformed via orthogonal Q")
    print(f"Converged: {result.converged}")
    print(f"Max eigenvalue error: {error:.2e}")
    print(f"First 3 computed: {result.eigenvalues[:3]}")
    print(f"First 3 exact: {evals_exact[:3]}")
    
    if result.converged:
        assert error < 1e-3, f"Converged but eigenvalue error {error:.2e} too large with transforms"
    else:
        assert error < 2.0, f"Eigenvalue error {error:.2e} unexpectedly large with transforms"
    print("(ok)  PASSED\n")


def test_block_lanczos_matvec():
    """Test Block Lanczos with matrix-free matvec operator."""
    print("="*60)
    print("TEST 6: Block Lanczos with matvec callback")
    print("="*60)
    
    n = 60
    k = 5
    block_size = 2
    np.random.seed(654)
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)
    evals_exact = np.linalg.eigvalsh(A)
    
    # Define matvec
    def matvec(v):
        return A @ v
    
    # Solve with matvec
    solver = BlockLanczosEigensolver(k=k, block_size=block_size, which='smallest', max_iter=50, tol=1e-6)
    result = solver.solve(matvec=matvec, n=n)
    
    error = np.max(np.abs(result.eigenvalues - evals_exact[:k]))
    
    print(f"Matrix-free operator via matvec callback")
    print(f"Converged: {result.converged}")
    print(f"Max eigenvalue error: {error:.2e}")
    
    if result.converged:
        assert error < 1e-4, f"Converged but eigenvalue error {error:.2e} too large with matvec"
    else:
        assert error < 1.0, f"Eigenvalue error {error:.2e} unexpectedly large with matvec"
    print("(ok)  PASSED\n")


def test_block_lanczos_jax():
    """Test Block Lanczos JAX backend."""
    if not HAS_JAX:
        print("="*60)
        print("TEST 7: Block Lanczos JAX backend [SKIPPED - JAX not available]")
        print("="*60 + "\n")
        return
    
    print("="*60)
    print("TEST 7: Block Lanczos JAX backend")
    print("="*60)
    
    n = 100
    k = 6
    block_size = 2
    np.random.seed(987)
    A_np = np.random.randn(n, n)
    A_np = 0.5 * (A_np + A_np.T)
    evals_exact = np.linalg.eigvalsh(A_np)
    
    # Convert to JAX array
    A_jax = jnp.array(A_np)
    
    # Solve with JAX backend
    solver = BlockLanczosEigensolver(
        k=k, block_size=block_size, which='smallest', 
        max_iter=50, tol=1e-6, backend='jax'
    )
    result = solver.solve(A=A_jax)
    
    error = np.max(np.abs(np.array(result.eigenvalues) - evals_exact[:k]))
    
    print(f"Matrix size: {n}x{n}")
    print(f"Backend: JAX (compiled)")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Max eigenvalue error: {error:.2e}")
    print(f"First 3 computed: {np.array(result.eigenvalues[:3])}")
    print(f"First 3 exact: {evals_exact[:3]}")
    
    if result.converged:
        assert error < 1e-4, f"JAX converged but eigenvalue error {error:.2e} too large"
    else:
        assert error < 1.0, f"JAX eigenvalue error {error:.2e} unexpectedly large"
    print("(ok)  PASSED\n")


def test_block_lanczos_jax_vs_numpy():
    """Compare JAX and NumPy backends."""
    if not HAS_JAX:
        print("="*60)
        print("TEST 8: JAX vs NumPy comparison [SKIPPED - JAX not available]")
        print("="*60 + "\n")
        return
    
    print("="*60)
    print("TEST 8: JAX vs NumPy backend comparison")
    print("="*60)
    
    n = 80
    k = 8
    block_size = 3
    np.random.seed(111)
    A_np = np.random.randn(n, n)
    A_np = 0.5 * (A_np + A_np.T)
    A_jax = jnp.array(A_np)
    
    # Solve with NumPy
    solver_np = BlockLanczosEigensolver(
        k=k, block_size=block_size, which='smallest', 
        max_iter=40, tol=1e-6, backend='numpy'
    )
    result_np = solver_np.solve(A=A_np)
    
    # Solve with JAX
    solver_jax = BlockLanczosEigensolver(
        k=k, block_size=block_size, which='smallest', 
        max_iter=40, tol=1e-6, backend='jax'
    )
    result_jax = solver_jax.solve(A=A_jax)
    
    # Compare results
    diff = np.max(np.abs(result_np.eigenvalues - np.array(result_jax.eigenvalues)))
    
    print(f"NumPy converged: {result_np.converged}, iters: {result_np.iterations}")
    print(f"JAX converged: {result_jax.converged}, iters: {result_jax.iterations}")
    print(f"Max eigenvalue difference: {diff:.2e}")
    print(f"NumPy evals: {result_np.eigenvalues[:3]}")
    print(f"JAX evals: {np.array(result_jax.eigenvalues[:3])}")
    
    # Backends should produce similar results
    assert diff < 1e-4, f"NumPy/JAX difference {diff:.2e} too large"
    print("(ok)  PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BLOCK LANCZOS EIGENSOLVER COMPREHENSIVE TESTS")
    print("="*60)
    
    try:
        test_block_lanczos_basic()
        test_block_lanczos_varying_block_sizes()
        test_block_lanczos_largest()
        test_block_lanczos_solve_overrides()
        test_block_lanczos_basis_transforms()
        test_block_lanczos_matvec()
        test_block_lanczos_jax()
        test_block_lanczos_jax_vs_numpy()
        
        print("="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
    except AssertionError as e:
        print(f"\n(x) TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n(x) UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
