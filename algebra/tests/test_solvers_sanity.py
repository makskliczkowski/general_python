
import pytest
import numpy as np
from general_python.algebra.solvers.minres_qlp import MinresQLPSolver
from general_python.algebra.eigen import LanczosEigensolver

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

class TestSolversSanity:

    def test_lanczos_convergence_2d_laplacian(self):
        """
        Test Lanczos convergence on a 2D Laplacian matrix (L=10, N=100).
        The eigenvalues of 2D Laplacian are known:
        lambda_{i,j} = 4 - 2*cos(pi*i/(L+1)) - 2*cos(pi*j/(L+1))
        """
        L = 10
        N = L * L
        A = create_2d_laplacian(L)

        # Expected smallest eigenvalue (i=1, j=1)
        expected_min_eval = 4 - 2*np.cos(np.pi/(L+1)) - 2*np.cos(np.pi/(L+1))

        solver = LanczosEigensolver(k=5, which='smallest', maxiter=100, tol=1e-8, backend='numpy')
        result = solver.solve(A=A)

        assert result.converged
        assert np.isclose(result.eigenvalues[0], expected_min_eval, atol=1e-6)

        # Check residuals
        for i in range(len(result.eigenvalues)):
            lam = result.eigenvalues[i]
            vec = result.eigenvectors[:, i]
            res = np.linalg.norm(A @ vec - lam * vec)
            assert res < 1e-6

    def test_minres_convergence_2d_laplacian(self):
        """
        Test MINRES-QLP convergence on a 2D Laplacian linear system Ax = b.
        A is SPD.
        """
        L = 10
        N = L * L
        A = create_2d_laplacian(L)

        np.random.seed(42)
        x_true = np.random.randn(N)
        b = A @ x_true

        solver = MinresQLPSolver(maxiter=200, eps=1e-8, a=A)
        result = solver.solve_instance(b=b)

        assert result.converged

        # Check relative error
        rel_error = np.linalg.norm(result.x - x_true) / np.linalg.norm(x_true)
        assert rel_error < 1e-6

        # Check residual
        residual = np.linalg.norm(A @ result.x - b) / np.linalg.norm(b)
        assert residual < 1e-6

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
    def test_lanczos_dtypes(self, dtype):
        """Test Lanczos with different dtypes."""
        N = 30
        np.random.seed(123)

        # Use well-separated eigenvalues to ensure easy convergence
        # Random matrices can be hard to converge to 1e-8 in N iterations
        evals = np.linspace(1, 10, N)

        if np.issubdtype(dtype, np.complexfloating):
            # Apply unitary transform
            U, _ = np.linalg.qr(np.random.randn(N, N) + 1j * np.random.randn(N, N))
            A = U @ np.diag(evals) @ U.T.conj()
        else:
            Q, _ = np.linalg.qr(np.random.randn(N, N))
            A = Q @ np.diag(evals) @ Q.T

        A = A.astype(dtype)

        # Note: Native Lanczos implementation has hardcoded 1e-8 tolerance check for 'converged' flag.
        # This will fail for float32. So we check residuals manually.
        solver = LanczosEigensolver(k=3, which='smallest', maxiter=N, tol=1e-5, backend='numpy')
        result = solver.solve(A=A)

        # For double precision, it should converge to 1e-8
        # However, due to numerical issues or hardcoded tolerance in library,
        # we rely on manual residual check instead of result.converged flag.

        # Eigenvalues of Hermitian matrix are real
        assert np.all(np.isreal(result.eigenvalues))

        # Check residuals
        # Relax tolerance for float32
        tol = 1e-4 if dtype == np.float32 else 1e-6

        for i in range(len(result.eigenvalues)):
            lam = result.eigenvalues[i]
            vec = result.eigenvectors[:, i]
            res = np.linalg.norm(A @ vec - lam * vec)
            assert res < tol

    def test_deterministic_behavior(self):
        """Test that solvers are deterministic when seeded."""
        N = 30
        np.random.seed(999)
        A = np.random.randn(N, N)
        A = 0.5 * (A + A.T)

        # Run Lanczos twice
        solver1 = LanczosEigensolver(k=3, which='largest', maxiter=30)
        # Assuming v0 random generation inside uses np.random, which we seeded?
        # Wait, LanczosEigensolver doesn't take seed in __init__.
        # It generates v0 inside solve if not provided.
        # We need to control randomness.

        # If we pass v0, it's deterministic.
        # If we don't, it uses np.random.

        # Let's seed before each call
        np.random.seed(42)
        res1 = solver1.solve(A=A)

        np.random.seed(42)
        res2 = solver1.solve(A=A)

        np.testing.assert_allclose(res1.eigenvalues, res2.eigenvalues)
        np.testing.assert_allclose(res1.eigenvectors, res2.eigenvectors)
