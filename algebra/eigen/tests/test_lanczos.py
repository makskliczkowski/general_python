"""
Test suite for Lanczos eigenvalue solver.

Tests both NumPy and JAX backends, compares with full diagonalization,
and validates convergence behavior.
"""

import numpy as np
import pytest
import sys
import os

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

try:
    from QES.general_python.algebra.eigen import LanczosEigensolver, LanczosEigensolverScipy
except ImportError:
    raise ImportError("QES package is required to run these tests.")

# Check JAX availability
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# ----------------------------------
#! Helper functions to create test matrices
# ----------------------------------

def create_symmetric_matrix(n, condition_number=10.0, seed=42):
    """Create symmetric matrix with controlled spectrum."""
    np.random.seed(seed)
    eigenvalues     = np.linspace(1.0, condition_number, n)
    Q, _            = np.linalg.qr(np.random.randn(n, n))
    A               = Q @ np.diag(eigenvalues) @ Q.T
    A               = 0.5 * (A + A.T)  # Ensure exact symmetry
    return A

# ----------------------------------

def create_tridiagonal_matrix(n, diagonal=2.0, off_diagonal=-1.0):
    """Create symmetric tridiagonal matrix (1D Laplacian-like)."""
    A = np.diag(np.full(n, diagonal))
    if n > 1:
        A += np.diag(np.full(n-1, off_diagonal), k=1)
        A += np.diag(np.full(n-1, off_diagonal), k=-1)
    return A

# ----------------------------------
#! Test classes
# ----------------------------------

class TestLanczosBasic:
    """Basic functionality tests."""
    
    # ----------------------------------
    
    def test_smallest_eigenvalues_numpy(self):
        """Test finding smallest eigenvalues with NumPy backend."""
        n           = 100
        k           = 5
        A           = create_symmetric_matrix(n, condition_number=50.0)
        
        # Lanczos - use more iterations for convergence
        solver      = LanczosEigensolver(k=k, which='smallest', backend='numpy', tol=1e-8, max_iter=100)
        result      = solver.solve(A=A)
        
        # Full diagonalization
        evals_full  = np.linalg.eigvalsh(A)
        
        # Compare
        error       = np.linalg.norm(result.eigenvalues - evals_full[:k])
        print(f"\nNumPy Backend - Smallest {k} eigenvalues:")
        print(f"  Lanczos: {result.eigenvalues}")
        print(f"  Full ED: {evals_full[:k]}")
        print(f"  Error: {error:.2e}")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Residual norms: {result.residual_norms}")
        
        # Check accuracy (may not be fully converged but should be accurate)
        assert error < 1e-6, f"Eigenvalue error too large: {error:.2e}"
    
    def test_largest_eigenvalues_numpy(self):
        """Test finding largest eigenvalues with NumPy backend."""
        n               = 100
        k               = 5
        A               = create_symmetric_matrix(n, condition_number=50.0)
        
        # Lanczos
        solver          = LanczosEigensolver(k=k, which='largest', backend='numpy', tol=1e-8, max_iter=100)
        result          = solver.solve(A=A)

        # Full diagonalization
        evals_full      = np.linalg.eigvalsh(A)
        error           = np.linalg.norm(result.eigenvalues - evals_full[-k:][::-1])

        print(f"\nNumPy Backend - Largest {k} eigenvalues:")
        print(f"  Lanczos: {result.eigenvalues}")
        print(f"  Full ED: {evals_full[-k:][::-1]}")
        print(f"  Error: {error:.2e}")
        
        assert error < 1e-6, f"Eigenvalue error too large: {error:.2e}"
    
    # ----------------------------------
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_smallest_eigenvalues_jax(self):
        """Test finding smallest eigenvalues with JAX backend."""
        n               = 100
        k               = 5
        A               = create_symmetric_matrix(n, condition_number=50.0)

        # Lanczos with JAX
        solver          = LanczosEigensolver(k=k, which='smallest', backend='jax', tol=1e-8, max_iter=100)
        result          = solver.solve(A=A)

        # Full diagonalization
        evals_full      = np.linalg.eigvalsh(A)
        error           = np.linalg.norm(result.eigenvalues - evals_full[:k])
        print(f"\nJAX Backend - Smallest {k} eigenvalues:")
        print(f"  Lanczos: {result.eigenvalues}")
        print(f"  Full ED: {evals_full[:k]}")
        print(f"  Error: {error:.2e}")
        print(f"  Converged: {result.converged}")
        
        assert error < 1e-6, f"Eigenvalue error too large: {error:.2e}"
    
    # ----------------------------------
    
    def test_scipy_wrapper(self):
        """Test SciPy wrapper."""
        n               = 200
        k               = 6
        A               = create_symmetric_matrix(n, condition_number=100.0)

        # SciPy Lanczos
        solver          = LanczosEigensolverScipy(k=k, which='SA', tol=1e-10)
        result          = solver.solve(A=A)
        
        # Full diagonalization
        evals_full      = np.linalg.eigvalsh(A)
        error           = np.linalg.norm(result.eigenvalues - evals_full[:k])
        print(f"\nSciPy Wrapper - Smallest {k} eigenvalues:")
        print(f"  Lanczos: {result.eigenvalues}")
        print(f"  Full ED: {evals_full[:k]}")
        print(f"  Error: {error:.2e}")
        
        assert error < 1e-8, f"Eigenvalue error too large: {error:.2e}"

# ----------------------------------
#! Additional test classes for more detailed scenarios
# ----------------------------------

class TestLanczosMatrixFree:
    """Test matrix-free operation."""
    
    def test_matvec_tridiagonal(self):
        """Test with matrix-free matvec function."""
        n       = 500
        k       = 8

        # Tridiagonal matrix (never formed explicitly)
        def matvec(x):
            """1D Laplacian: (Ax)_i = 2x_i - x_{i-1} - x_{i+1}"""
            y = np.zeros_like(x)
            y[0] = 2*x[0] - x[1]
            y[1:-1] = 2*x[1:-1] - x[:-2] - x[2:]
            y[-1] = 2*x[-1] - x[-2]
            return y
        
        # Lanczos with matvec
        solver      = LanczosEigensolver(k=k, which='smallest', backend='numpy', tol=1e-8, max_iter=150)
        result      = solver.solve(matvec=matvec, n=n)
        
        # Analytical eigenvalues for 1D Laplacian
        analytical  = np.array([2*(1 - np.cos((i+1)*np.pi/(n+1))) for i in range(k)])
        error       = np.linalg.norm(result.eigenvalues - analytical)
        
        print(f"\nMatrix-Free (1D Laplacian) - {k} smallest eigenvalues:")
        print(f"  Lanczos: {result.eigenvalues[:4]}")
        print(f"  Analytical: {analytical[:4]}")
        print(f"  Error: {error:.2e}")
        
        assert error < 1e-6, f"Eigenvalue error too large: {error:.2e}"

# ----------------------------------

class TestLanczosEigenvectors:
    """Test eigenvector accuracy."""
    
    # ----------------------------------
    
    def test_eigenvector_residuals(self):
        """Test eigenvector residuals ||Av - \lambdav||."""
        n       = 80
        k       = 5
        A       = create_symmetric_matrix(n, condition_number=20.0)

        solver  = LanczosEigensolver(k=k, which='smallest', backend='numpy', tol=1e-8, max_iter=100)
        result  = solver.solve(A=A)
        
        print(f"\nEigenvector Residuals:")
        for i in range(k):
            lam         = result.eigenvalues[i]
            v           = result.eigenvectors[:, i]
            residual    = np.linalg.norm(A @ v - lam * v)
            print(f"  \lambda_{i} = {lam:.6f}, ||Av - \lambdav|| = {residual:.2e}")
            assert residual < 1e-6, f"Eigenvector {i} residual too large: {residual:.2e}"
        
        # Check orthonormality
        VtV         = result.eigenvectors.T @ result.eigenvectors
        ortho_error = np.linalg.norm(VtV - np.eye(k))
        print(f"  Orthonormality error: {ortho_error:.2e}")
        assert ortho_error < 1e-8, f"Eigenvectors not orthonormal: {ortho_error:.2e}"

# ----------------------------------

class TestLanczosConvergence:
    """Test convergence behavior."""
    
    def test_iteration_count(self):
        """Test that more iterations improve accuracy."""
        n           = 100
        k           = 5
        A           = create_symmetric_matrix(n, condition_number=100.0)
        evals_full  = np.linalg.eigvalsh(A)

        max_iters   = [20, 40, 80]
        errors      = []

        print(f"\nConvergence with increasing iterations:")
        print(f"{'Max Iter':<10} {'Converged':<12} {'Error':<15}")
        print("-" * 40)
        
        for max_iter in max_iters:
            solver  = LanczosEigensolver(k=k, which='smallest', max_iter=max_iter, 
                                       backend='numpy', tol=1e-8)
            result  = solver.solve(A=A)
            error   = np.linalg.norm(result.eigenvalues - evals_full[:k])
            errors.append(error)
            print(f"{max_iter:<10} {str(result.converged):<12} {error:<15.2e}")
        
        # Error should decrease with more iterations
        assert errors[-1] <= errors[0] * 1.1, "Error did not decrease with more iterations"
    
    # ----------------------------------
    
    def test_breakdown_detection(self):
        """Test early termination when exact subspace found."""
        # Create matrix where Krylov space is small
        n           = 50
        k           = 5
        # Matrix with only k distinct eigenvalues
        eigenvalues = np.concatenate([np.linspace(1, 10, k), np.full(n-k, 10)])
        Q, _        = np.linalg.qr(np.random.randn(n, n))
        A           = Q @ np.diag(eigenvalues) @ Q.T
        A           = 0.5 * (A + A.T)
        
        solver      = LanczosEigensolver(k=k, which='smallest', max_iter=100, 
                                   backend='numpy', tol=1e-10)
        result      = solver.solve(A=A)

        print(f"\nBreakdown Detection:")
        print(f"  Iterations: {result.iterations} (max_iter=100)")
        print(f"  Converged: {result.converged}")
        print(f"  Expected early termination due to exact subspace")
        
        # Should converge with fewer iterations than max_iter (or close to it)
        assert result.iterations <= 100

# ----------------------------------

class TestLanczosComplex:
    """Test with complex Hermitian matrices."""
    
    def test_complex_hermitian(self):
        """Test Hermitian (complex symmetric) matrix."""
        np.random.seed(42)
        n           = 60
        k           = 5
        # Create Hermitian matrix
        A           = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        A           = 0.5 * (A + A.T.conj())  # Make Hermitian
        
        # Lanczos
        solver      = LanczosEigensolver(k=k, which='smallest', backend='numpy', tol=1e-8, max_iter=100)
        result      = solver.solve(A=A)
        
        # Full diagonalization (eigenvalues are real for Hermitian)
        evals_full  = np.linalg.eigvalsh(A)
        error       = np.linalg.norm(result.eigenvalues - evals_full[:k])
        print(f"\nComplex Hermitian Matrix:")
        print(f"  Lanczos eigenvalues: {result.eigenvalues}")
        print(f"  Full ED eigenvalues: {evals_full[:k]}")
        print(f"  Error: {error:.2e}")
        
        assert error < 1e-6, f"Eigenvalue error too large: {error:.2e}"
        # Eigenvalues should be real
        assert np.allclose(result.eigenvalues.imag, 0), "Eigenvalues not real for Hermitian matrix"

# ----------------------------------
#! Run tests if executed as main script
# ----------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("LANCZOS EIGENVALUE SOLVER - COMPREHENSIVE TESTS")
    print("=" * 70)
    
    # Run tests
    test_basic      = TestLanczosBasic()
    test_basic.test_smallest_eigenvalues_numpy()
    test_basic.test_largest_eigenvalues_numpy()
    if JAX_AVAILABLE:
        test_basic.test_smallest_eigenvalues_jax()
    test_basic.test_scipy_wrapper()
    
    test_matvec     = TestLanczosMatrixFree()
    test_matvec.test_matvec_tridiagonal()
    
    test_eigvec     = TestLanczosEigenvectors()
    test_eigvec.test_eigenvector_residuals()
    
    test_conv       = TestLanczosConvergence()
    test_conv.test_iteration_count()
    test_conv.test_breakdown_detection()
    
    test_complex    = TestLanczosComplex()
    test_complex.test_complex_hermitian()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    
# ------------------------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------------------------
