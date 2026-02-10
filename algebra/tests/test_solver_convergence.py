"""
Comprehensive convergence tests for linear solvers and eigensolvers.

Validates correctness, convergence, and stability across different matrix types
(dense, sparse) and data types (float32, float64, complex128).
"""

import pytest
import numpy as np
from general_python.algebra.eigen import LanczosEigensolver
from general_python.algebra.solvers.minres_qlp import MinresQLPSolver

# --- Helper Functions ---

def create_2d_laplacian(L):
    """
    Creates a 2D Laplacian matrix for an LxL grid with Dirichlet boundary conditions.
    N = L*L.
    """
    N = L * L
    # Diagonals
    diag = 4.0 * np.ones(N)
    off_diag_1 = -1.0 * np.ones(N - 1)
    off_diag_L = -1.0 * np.ones(N - L)

    # Fix boundary effects for off_diag_1 (remove connections between rows)
    for i in range(1, L):
        off_diag_1[i*L - 1] = 0.0

    A = np.diag(diag) + np.diag(off_diag_1, k=1) + np.diag(off_diag_1, k=-1) + \
        np.diag(off_diag_L, k=L) + np.diag(off_diag_L, k=-L)

    return A

def create_random_hermitian(n, seed=None):
    """Creates a random Hermitian matrix."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return 0.5 * (A + A.T.conj())

def create_random_spd(n, seed=None):
    """Creates a random Symmetric Positive Definite matrix."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    return A @ A.T + 0.1 * np.eye(n)

# --- Lanczos Tests ---

class TestLanczosConvergence:

    def test_lanczos_dense_hermitian(self):
        """Test Lanczos convergence on a dense complex Hermitian matrix."""
        n = 50
        k = 5
        A = create_random_hermitian(n, seed=42)

        # Exact eigenvalues
        evals_exact = np.linalg.eigvalsh(A)

        # Solve
        solver = LanczosEigensolver(k=k, which='smallest', maxiter=100, tol=1e-8)
        result = solver.solve(A=A)

        assert result.converged

        # Check eigenvalues
        # Note: Lanczos finds extremal eigenvalues well. 'smallest' corresponds to most negative.
        # evals_exact are sorted small to large.
        error = np.max(np.abs(result.eigenvalues - evals_exact[:k]))
        assert error < 1e-6

        # Check residuals explicitly
        for i in range(k):
            lam = result.eigenvalues[i]
            vec = result.eigenvectors[:, i]
            res = np.linalg.norm(A @ vec - lam * vec)
            assert res < 1e-6

    def test_lanczos_sparse_2d_laplacian(self):
        """
        Test Lanczos on a sparse 2D Laplacian.
        Smallest eigenvalue known analytically: 4 - 4*cos(pi/(L+1)) approx for large L?
        Actually: 4 - 2*cos(pi/(L+1)) - 2*cos(pi/(L+1))
        """
        L = 10
        A = create_2d_laplacian(L)
        k = 1

        expected_min_eval = 4 - 2*np.cos(np.pi/(L+1)) - 2*np.cos(np.pi/(L+1))

        solver = LanczosEigensolver(k=k, which='smallest', maxiter=200, tol=1e-8)
        result = solver.solve(A=A)

        assert result.converged
        assert np.isclose(result.eigenvalues[0], expected_min_eval, atol=1e-6)

    @pytest.mark.parametrize("dtype", [np.float32, np.complex128])
    def test_lanczos_dtypes(self, dtype):
        """Test Lanczos robustness with different dtypes."""
        n = 30
        k = 3
        np.random.seed(123)

        if dtype == np.float32:
            A = np.random.randn(n, n).astype(np.float32)
            A = 0.5 * (A + A.T)
            tol = 1e-4
        else:
            A = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex128)
            A = 0.5 * (A + A.T.conj())
            tol = 1e-8

        solver = LanczosEigensolver(k=k, which='largest', maxiter=100, tol=tol)
        result = solver.solve(A=A)

        # Check residuals
        for i in range(k):
            lam = result.eigenvalues[i]
            vec = result.eigenvectors[:, i]
            res = np.linalg.norm(A @ vec - lam * vec)
            # Use appropriate tolerance for dtype
            assert res < (1e-3 if dtype == np.float32 else 1e-6)

    def test_lanczos_deterministic(self):
        """Test that Lanczos is deterministic given a fixed seed/start vector."""
        n = 20
        A = create_random_spd(n, seed=999)

        # Case 1: Fixed random seed for numpy (if solver uses it internally for v0)
        # Note: Solver usually generates v0 via np.random.randn if not provided.

        np.random.seed(42)
        solver1 = LanczosEigensolver(k=2, maxiter=20)
        res1 = solver1.solve(A=A)

        np.random.seed(42)
        solver2 = LanczosEigensolver(k=2, maxiter=20)
        res2 = solver2.solve(A=A)

        np.testing.assert_allclose(res1.eigenvalues, res2.eigenvalues)
        np.testing.assert_allclose(res1.eigenvectors, res2.eigenvectors)


# --- Minres Tests ---

class TestMinresConvergence:

    def test_minres_dense_spd(self):
        """Test Minres on dense SPD matrix."""
        n = 50
        A = create_random_spd(n, seed=101)
        x_true = np.random.randn(n)
        b = A @ x_true

        solver = MinresQLPSolver(maxiter=100, eps=1e-8, a=A)
        result = solver.solve_instance(b=b)

        assert result.converged
        assert np.linalg.norm(result.x - x_true) / np.linalg.norm(x_true) < 1e-6
        assert np.linalg.norm(A @ result.x - b) / np.linalg.norm(b) < 1e-6

    def test_minres_sparse_2d_laplacian(self):
        """Test Minres on sparse 2D Laplacian (SPD)."""
        L = 10
        A = create_2d_laplacian(L)
        n = L*L
        np.random.seed(202)
        x_true = np.random.randn(n)
        b = A @ x_true

        solver = MinresQLPSolver(maxiter=200, eps=1e-8, a=A)
        result = solver.solve_instance(b=b)

        assert result.converged
        assert np.linalg.norm(result.x - x_true) / np.linalg.norm(x_true) < 1e-6
