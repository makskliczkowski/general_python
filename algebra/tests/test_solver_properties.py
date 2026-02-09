import pytest
import numpy as np
from general_python.algebra.eigen import LanczosEigensolver
from general_python.algebra.solvers.minres_qlp import MinresQLPSolver

def create_2d_laplacian(L, dtype=np.float64):
    """Creates a 2D Laplacian matrix for an LxL grid."""
    N = L * L
    diag = 4.0 * np.ones(N, dtype=dtype)
    off_diag_1 = -1.0 * np.ones(N - 1, dtype=dtype)
    off_diag_L = -1.0 * np.ones(N - L, dtype=dtype)

    for i in range(1, L):
        off_diag_1[i*L - 1] = 0.0

    A = np.diag(diag) + np.diag(off_diag_1, k=1) + np.diag(off_diag_1, k=-1) + \
        np.diag(off_diag_L, k=L) + np.diag(off_diag_L, k=-L)
    return A

class TestSolverProperties:

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_lanczos_dtypes_shapes(self, dtype):
        """Test Lanczos with different dtypes and shapes."""
        L = 6 # N=36
        N = L*L
        np.random.seed(42)

        # Create Hermitian matrix
        A_real = create_2d_laplacian(L, dtype=np.float64 if np.issubdtype(dtype, np.complexfloating) else dtype)

        if np.issubdtype(dtype, np.complexfloating):
            # Make it complex Hermitian
            # Add small imaginary perturbation that preserves Hermiticity
            noise = np.random.randn(N, N) + 1j * np.random.randn(N, N)
            H_noise = 1e-2 * (noise + noise.T.conj())
            A = (A_real + H_noise).astype(dtype)
        else:
            A = A_real.astype(dtype)

        # Relax tolerance for float32/complex64
        tol = 1e-4 if dtype in [np.float32, np.complex64] else 1e-7

        solver = LanczosEigensolver(k=3, which='smallest', maxiter=N*2, tol=tol, backend='numpy')
        result = solver.solve(A=A)

        # Check residuals
        for i in range(len(result.eigenvalues)):
            lam = result.eigenvalues[i]
            vec = result.eigenvectors[:, i]
            res = np.linalg.norm(A @ vec - lam * vec)
            # Higher tolerance for residuals in lower precision
            check_tol = tol * 20 if dtype == np.float32 else tol * 10
            assert res < check_tol, f"Residual {res} too high for dtype {dtype}"

    def test_minres_determinism(self):
        """Test deterministic behavior of MinresQLPSolver with fixed seed."""
        N = 20
        np.random.seed(123)
        A = np.random.randn(N, N)
        A = 0.5 * (A + A.T) # Symmetric
        b = np.random.randn(N)

        # First run
        solver1 = MinresQLPSolver(maxiter=50, eps=1e-8, a=A)
        # MINRES is deterministic if A and b are fixed, but let's ensure no internal randomness leaks
        res1 = solver1.solve_instance(b=b)

        # Second run
        solver2 = MinresQLPSolver(maxiter=50, eps=1e-8, a=A)
        res2 = solver2.solve_instance(b=b)

        np.testing.assert_allclose(res1.x, res2.x)
        assert res1.iterations == res2.iterations

    def test_lanczos_determinism(self):
        """Test deterministic behavior of LanczosEigensolver with fixed seed."""
        N = 20
        np.random.seed(123)
        A = np.random.randn(N, N)
        A = 0.5 * (A + A.T)

        # Lanczos uses random start vector if not provided.
        # We must provide v0 or seed the RNG before solve.

        # Method 1: Seed before solve
        solver = LanczosEigensolver(k=3, which='largest', maxiter=30)

        np.random.seed(999)
        res1 = solver.solve(A=A)

        np.random.seed(999)
        res2 = solver.solve(A=A)

        np.testing.assert_allclose(res1.eigenvalues, res2.eigenvalues)

        # Method 2: Provide v0
        v0 = np.random.randn(N)
        v0 /= np.linalg.norm(v0)

        res3 = solver.solve(A=A, v0=v0)
        res4 = solver.solve(A=A, v0=v0)

        np.testing.assert_allclose(res3.eigenvalues, res4.eigenvalues)
        np.testing.assert_allclose(res3.eigenvectors, res4.eigenvectors)

    def test_lanczos_convergence_laplacian(self):
        """Test convergence on 2D Laplacian (sanity check)."""
        L = 8
        A = create_2d_laplacian(L)
        expected_min = 4 - 2*np.cos(np.pi/(L+1)) - 2*np.cos(np.pi/(L+1))

        solver = LanczosEigensolver(k=1, which='smallest', maxiter=100, tol=1e-8)
        result = solver.solve(A=A)

        assert np.isclose(result.eigenvalues[0], expected_min, atol=1e-6)
