
import pytest
import numpy as np
import time

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from algebra.utilities.pfaffian_jax import Pfaffian
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_scherman_morrison_skew_jax_correctness():
    """
    Verifies that the JAX implementation of Sherman-Morrison skew update
    calculates correct results.
    """
    key = random.PRNGKey(42)
    N = 20

    k1, k2, k3 = random.split(key, 3)

    # 1. Create random skew-symmetric matrix A
    A = random.normal(k1, (N, N))
    A = A - A.T

    # 2. Invert it using standard numpy/jax (Ainv)
    # Note: Inverse of skew-symmetric is skew-symmetric
    Ainv = jnp.linalg.inv(A)

    # 3. Define update: A' = A + u*v^T - v*u^T (rank 2 update)
    # But Sherman-Morrison formula in the code assumes we are updating ONE row and column.
    # The function signature is `_scherman_morrison_skew_jax(Ainv, updIdx, updRow)`.
    # This implies we updated row `updIdx` of A to `updRow` (and symmetric col).

    updIdx = N // 2
    updRow = random.normal(k3, (N,))
    # Ensure updRow[updIdx] = 0 for skew-symmetry
    updRow = updRow.at[updIdx].set(0.0)

    # Construct expected new A
    # A_new = A.copy()
    # A_new[updIdx, :] = updRow
    # A_new[:, updIdx] = -updRow # Skew symmetric
    # But wait, the function takes Ainv and updates it to (A_new)^-1.

    # Let's verify this property.
    A_new = A.at[updIdx, :].set(updRow)
    A_new = A_new.at[:, updIdx].set(-updRow)
    # Skew symmetry implies diagonal is 0, which we set above.

    expected_Ainv_new = jnp.linalg.inv(A_new)

    # Run our function
    # Note: The function assumes Ainv is the inverse of A BEFORE update.
    # It updates Ainv to match the new A where row/col updIdx was replaced by updRow.

    computed_Ainv_new = Pfaffian._scherman_morrison_skew_jax(Ainv, updIdx, updRow)

    # Compare
    # Tolerance might need to be loose due to condition number
    diff = jnp.linalg.norm(expected_Ainv_new - computed_Ainv_new) / jnp.linalg.norm(expected_Ainv_new)

    print(f"Relative Difference: {diff}")
    assert diff < 1e-4

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_scherman_morrison_skew_jax_performance():
    """
    Simple performance regression test.
    Ensures the vectorized implementation is reasonably fast.
    """
    key = random.PRNGKey(123)
    N = 100

    k1, k2, k3 = random.split(key, 3)
    Ainv = random.normal(k2, (N, N)) # Doesn't need to be real inverse for perf test
    updRow = random.normal(k3, (N,))
    updIdx = 10

    # Warmup
    _ = Pfaffian._scherman_morrison_skew_jax(Ainv, updIdx, updRow).block_until_ready()

    start = time.time()
    n_iters = 100
    for _ in range(n_iters):
        _ = Pfaffian._scherman_morrison_skew_jax(Ainv, updIdx, updRow).block_until_ready()
    end = time.time()

    avg_time = (end - start) / n_iters
    print(f"Average time (N={N}): {avg_time:.6f} s")

    # Threshold is arbitrary but should catch massive regressions (e.g. falling back to loops ~1ms)
    # On CPU, vector ops are fast.
    assert avg_time < 0.005 # 5ms is generous for N=100

if __name__ == "__main__":
    test_scherman_morrison_skew_jax_correctness()
    test_scherman_morrison_skew_jax_performance()
