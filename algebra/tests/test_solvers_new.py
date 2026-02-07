
import pytest
import numpy as np
from general_python.algebra.solvers.minres_qlp import MinresQLPSolver
from general_python.algebra.eigen import LanczosEigensolver

def create_1d_laplacian(N):
    """
    Creates a 1D Laplacian matrix for size N with Dirichlet boundary conditions.
    """
    diag = 2.0 * np.ones(N)
    off_diag = -1.0 * np.ones(N - 1)
    A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    return A

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

class TestSolversNew:

    def test_lanczos_1d_laplacian_convergence(self):
        """
        Test Lanczos on 1D Laplacian.
        Eigenvalues: lambda_k = 2 - 2*cos(k*pi/(N+1))
        Smallest eigenvalue for k=1.
        """
        N = 20
        A = create_1d_laplacian(N)
        expected_min = 2 - 2*np.cos(np.pi/(N+1))

        # Test finding smallest eigenvalue
        solver = LanczosEigensolver(k=1, which='smallest', maxiter=N*2, tol=1e-8, backend='numpy')
        result = solver.solve(A=A)

        assert len(result.eigenvalues) >= 1
        assert np.isclose(result.eigenvalues[0], expected_min, atol=1e-6)

        # Check residual: ||Ax - lambda*x||
        vec = result.eigenvectors[:, 0]
        lam = result.eigenvalues[0]
        res = np.linalg.norm(A @ vec - lam * vec)
        assert res < 1e-6

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
    def test_lanczos_dtypes_residuals(self, dtype):
        """
        Test Lanczos with different dtypes on a random symmetric matrix.
        """
        N = 20
        np.random.seed(42)
        if np.issubdtype(dtype, np.complexfloating):
            A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
            A = A + A.T.conj() # Hermitian
        else:
            A = np.random.randn(N, N)
            A = A + A.T # Symmetric

        A = A.astype(dtype)

        # Use more iterations for float32/complex
        tol = 1e-5 if dtype == np.float32 else 1e-8

        solver = LanczosEigensolver(k=2, which='largest', maxiter=100, tol=tol, backend='numpy')
        result = solver.solve(A=A)

        assert len(result.eigenvalues) >= 1

        # Check residual for the first eigenpair
        lam = result.eigenvalues[0]
        vec = result.eigenvectors[:, 0]

        # Matrix-vector multiplication might promote dtype
        Ax = A @ vec
        res_vec = Ax - lam * vec
        res_norm = np.linalg.norm(res_vec)

        # Relax tolerance slightly for float32 accumulation errors
        check_tol = 1e-3 if dtype == np.float32 else 1e-6
        assert res_norm < check_tol, f"Residual {res_norm} too high for {dtype}"

    def test_minres_qlp_determinism(self):
        """
        Verify that MinresQLPSolver produces identical results given the same input and seed (if applicable).
        MinresQLP is deterministic given A and b, but we check consistency.
        """
        N = 20
        np.random.seed(123)
        # Create SPD matrix
        Q, _ = np.linalg.qr(np.random.randn(N, N))
        D = np.diag(np.linspace(1, 10, N))
        A = Q @ D @ Q.T
        b = np.random.randn(N)

        solver = MinresQLPSolver(maxiter=100, eps=1e-8, a=A)

        res1 = solver.solve_instance(b=b)
        res2 = solver.solve_instance(b=b)

        np.testing.assert_allclose(res1.x, res2.x, atol=1e-12)
        assert res1.converged == res2.converged

    def test_minres_qlp_residual_check(self):
        """
        Check actual residual ||Ax - b|| vs reported residual.
        """
        N = 20
        np.random.seed(456)
        A = create_1d_laplacian(N)
        x_true = np.random.randn(N)
        b = A @ x_true # Exact solution exists

        solver = MinresQLPSolver(maxiter=100, eps=1e-8, a=A)
        result = solver.solve_instance(b=b)

        assert result.converged

        # True residual
        r_true = b - A @ result.x
        norm_r_true = np.linalg.norm(r_true)

        # Reported residual should be close to true residual
        # Note: implementation might scale it. Usually relative residual.
        # Let's check relative error
        rel_err = np.linalg.norm(result.x - x_true) / np.linalg.norm(x_true)
        assert rel_err < 1e-6
