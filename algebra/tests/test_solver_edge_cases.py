import pytest
import numpy as np
from general_python.algebra.eigen import LanczosEigensolver
from general_python.algebra.solvers.minres_qlp import MinresQLPSolver

class TestSolverEdgeCases:

    def test_lanczos_singular_matrix(self):
        """Test Lanczos on a singular matrix (0 eigenvalues)."""
        N = 20
        # Matrix with 0 eigenvalue
        # Create diagonal matrix with one 0
        diag = np.linspace(1, 10, N-1)
        diag = np.concatenate([[0], diag])
        np.random.seed(42)
        Q, _ = np.linalg.qr(np.random.randn(N, N))
        A = Q @ np.diag(diag) @ Q.T
        A = 0.5 * (A + A.T)

        solver = LanczosEigensolver(k=3, which='smallest', maxiter=50, tol=1e-8)
        result = solver.solve(A=A)

        # Should find 0
        min_eval = np.min(result.eigenvalues)
        assert np.isclose(min_eval, 0.0, atol=1e-6)

    def test_lanczos_degenerate_eigenvalues(self):
        """Test Lanczos with degenerate eigenvalues."""
        N = 20
        # Eigenvalues: 1, 1, 1, 2, 3...
        diag = np.concatenate([[1, 1, 1], np.linspace(2, 10, N-3)])
        np.random.seed(123)
        Q, _ = np.linalg.qr(np.random.randn(N, N))
        A = Q @ np.diag(diag) @ Q.T
        A = 0.5 * (A + A.T)

        solver = LanczosEigensolver(k=5, which='smallest', maxiter=100, tol=1e-8)
        result = solver.solve(A=A)

        # Should find values close to real eigenvalues
        evals = np.sort(result.eigenvalues)
        assert np.isclose(evals[0], 1.0, atol=1e-5)

        # Verify all found eigenvalues are valid eigenvalues of A
        # (Lanczos might find 'ghost' eigenvalues if not careful, or duplicates)
        for val in evals:
             # Check distance to nearest true eigenvalue
             dist = np.min(np.abs(val - diag))
             assert dist < 1e-4, f"Found eigenvalue {val} not in spectrum"

    def test_minres_ill_conditioned(self):
        """Test MINRES on ill-conditioned matrix."""
        N = 30
        # Condition number 1e6
        diag = np.logspace(0, 6, N)
        np.random.seed(456)
        Q, _ = np.linalg.qr(np.random.randn(N, N))
        A = Q @ np.diag(diag) @ Q.T
        A = 0.5 * (A + A.T)

        x_true = np.random.randn(N)
        b = A @ x_true

        solver = MinresQLPSolver(maxiter=N*2, eps=1e-6, a=A)
        result = solver.solve_instance(b=b)

        # MINRES minimizes residual norm.
        res_norm = np.linalg.norm(A @ result.x - b)
        b_norm = np.linalg.norm(b)
        rel_res = res_norm / b_norm

        assert rel_res < 1e-5
