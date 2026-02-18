import numpy as np
import pytest
from general_python.algebra.solvers.cg import CgSolver
from general_python.algebra.utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

def test_sigma_in_matvec_func_numpy():
    """
    Test that sigma is correctly applied when using a custom matvec function with NumPy backend.
    """
    N = 10
    # Create a simple diagonal matrix A = diag(1, 2, ..., N)
    diag = np.arange(1, N + 1, dtype=np.float64)

    def matvec(x):
        return diag * x

    # We want to solve (A + sigma * I) x = b
    # Let sigma = 2.0
    sigma = 2.0

    # Let x_true be all ones
    x_true = np.ones(N)

    # b = (A + sigma * I) x_true
    #   = (diag + sigma) * x_true
    b = (diag + sigma) * x_true

    solver = CgSolver(backend='numpy', matvec_func=matvec, eps=1e-8, maxiter=100)

    # Solve with sigma
    x0 = np.zeros_like(b)
    result = solver.solve_instance(b, x0=x0, sigma=sigma)

    assert result.converged

    # If sigma is ignored, it solves Ax = b => x = A^{-1}b = A^{-1}(A+sigma)x_true = (I + sigma A^{-1}) x_true
    # which is NOT x_true (unless sigma=0).

    np.testing.assert_allclose(result.x, x_true, rtol=1e-5,
                               err_msg="Solver failed to apply sigma with matvec_func")

    # Test with a different sigma to ensure no caching issue preventing update
    sigma2 = 5.0
    b2 = (diag + sigma2) * x_true
    result2 = solver.solve_instance(b2, x0=x0, sigma=sigma2)

    assert result2.converged
    np.testing.assert_allclose(result2.x, x_true, rtol=1e-5,
                               err_msg="Solver failed to apply new sigma (recompilation/caching issue?)")

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_sigma_in_matvec_func_jax():
    """
    Test that sigma is correctly applied when using a custom matvec function with JAX backend.
    """
    N = 10
    diag = jnp.arange(1, N + 1, dtype=jnp.float32)

    def matvec(x):
        return diag * x

    sigma = 2.0
    x_true = jnp.ones(N)
    b = (diag + sigma) * x_true

    solver = CgSolver(backend='jax', matvec_func=matvec, eps=1e-6, maxiter=100)

    # Solve with sigma
    # Note: compilation happens here
    result = solver.solve_instance(b, x0=None, sigma=sigma)

    assert result.converged
    np.testing.assert_allclose(result.x, x_true, rtol=1e-4,
                               err_msg="JAX Solver failed to apply sigma with matvec_func")

    # Test with a different sigma
    sigma2 = 5.0
    b2 = (diag + sigma2) * x_true
    result2 = solver.solve_instance(b2, x0=None, sigma=sigma2)

    assert result2.converged
    np.testing.assert_allclose(result2.x, x_true, rtol=1e-4,
                               err_msg="JAX Solver failed to apply new sigma")
