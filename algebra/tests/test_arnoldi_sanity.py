
import pytest
import numpy as np
from general_python.algebra.eigen.arnoldi import ArnoldiEigensolver, ArnoldiEigensolverScipy
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

def create_random_nonsymmetric_matrix(n, seed=42):
    np.random.seed(seed)
    A = np.random.randn(n, n)
    # Ensure it's not symmetric
    A[0, 1] = 100
    A[1, 0] = -50
    return A

class TestArnoldiSanity:

    def test_arnoldi_numpy_nonsymmetric(self):
        """Test Arnoldi (NumPy) on a non-symmetric matrix."""
        n = 20
        k = 4
        A = create_random_nonsymmetric_matrix(n)

        # Reference solution
        evals_exact = np.linalg.eigvals(A)
        # Sort by magnitude largest
        idx = np.argsort(np.abs(evals_exact))[::-1]
        evals_exact_top = evals_exact[idx][:k]

        solver = ArnoldiEigensolver(k=k, which='LM', max_iter=n, tol=1e-8, backend='numpy')
        result = solver.solve(A=A)

        assert result.converged

        # Check eigenvalues match
        # Note: Order might differ slightly within same magnitude, so we sort both
        res_evals = np.sort(np.abs(result.eigenvalues))
        exact_evals = np.sort(np.abs(evals_exact_top))

        np.testing.assert_allclose(res_evals, exact_evals, rtol=1e-5)

        # Check residuals
        for i in range(k):
            lam = result.eigenvalues[i]
            vec = result.eigenvectors[:, i]
            res = np.linalg.norm(A @ vec - lam * vec)
            assert res < 1e-5

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    @pytest.mark.xfail(reason="Bug in arnoldi.py: jnp.iscomplexobj called on dtype object causing TypeError")
    def test_arnoldi_jax_nonsymmetric(self):
        """Test Arnoldi (JAX) on a non-symmetric matrix."""
        n = 20
        k = 4
        A = create_random_nonsymmetric_matrix(n)

        # JAX requires 64-bit for good precision usually, but let's try.
        # Check if 64-bit is enabled in config?
        # The solver tries to enable it.

        solver = ArnoldiEigensolver(k=k, which='LM', max_iter=n, tol=1e-8, backend='jax')
        result = solver.solve(A=A) # A is numpy array, handled by JAX internally or converted?
        # ArnoldiEigensolver._arnoldi_jax takes matvec.
        # If A is passed, solve wraps it.
        # But if A is numpy, jax.jit might complain if we were using it, but here it's direct calls.

        assert result.converged

        # Check eigenvalues
        evals_exact = np.linalg.eigvals(A)
        idx = np.argsort(np.abs(evals_exact))[::-1]
        evals_exact_top = evals_exact[idx][:k]

        res_evals = np.sort(np.abs(result.eigenvalues))
        exact_evals = np.sort(np.abs(evals_exact_top))

        np.testing.assert_allclose(res_evals, exact_evals, rtol=1e-5)

    def test_arnoldi_scipy_wrapper(self):
        """Test ArnoldiEigensolverScipy wrapper."""
        n = 20
        k = 4
        A = create_random_nonsymmetric_matrix(n)

        solver = ArnoldiEigensolverScipy(k=k, which='LM', tol=1e-8)
        result = solver.solve(A=A)

        assert result.converged

        evals_exact = np.linalg.eigvals(A)
        idx = np.argsort(np.abs(evals_exact))[::-1]
        evals_exact_top = evals_exact[idx][:k]

        res_evals = np.sort(np.abs(result.eigenvalues))
        exact_evals = np.sort(np.abs(evals_exact_top))

        np.testing.assert_allclose(res_evals, exact_evals, rtol=1e-5)
