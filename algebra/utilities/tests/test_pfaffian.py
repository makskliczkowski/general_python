
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from algebra.utilities.pfaffian_jax import Pfaffian

class TestPfaffianJax:
    def test_scherman_morrison_skew_jax_correctness(self):
        """
        Verifies that the vectorized Sherman-Morrison update implementation
        is correct by comparing it with a known update or full inverse recalculation.
        """
        key = jax.random.PRNGKey(42)
        N = 20

        # Create random skew-symmetric matrix A
        A = jax.random.normal(key, (N, N))
        A = A - A.T

        # Initial Inverse
        Ainv = jnp.linalg.inv(A)

        # Define an update: A' = A + updMatrix
        # Sherman-Morrison is for rank-1 updates.
        # But here we are doing a row/col replacement?
        # The function signature is `_scherman_morrison_skew_jax(Ainv, updIdx, updRow)`
        # It seems it updates the row `updIdx` and column `updIdx` to `updRow`.

        updIdx = 5
        updRow = jax.random.normal(key, (N,))
        # Enforce skew-symmetry on the update row (element at updIdx must be 0)
        updRow = updRow.at[updIdx].set(0.0)

        # Create the new A matrix manually
        A_new = A.at[updIdx, :].set(updRow)
        A_new = A_new.at[:, updIdx].set(-updRow) # Skew-symmetry

        # Calculate new inverse directly
        Ainv_expected = jnp.linalg.inv(A_new)

        # Calculate update using the function
        # Note: The function expects `updRow` to be the NEW row content?
        # Let's check the docstring or logic.
        # `dots = jnp.dot(Ainv, updRow)`
        # In standard Sherman-Morrison: (A + uv^T)^-1 = A^-1 - ...
        # Here we are replacing a row/col. This is a rank-2 update.
        # The logic in `pfaffian.py` and `pfaffian_jax.py` seems to handle this.

        Ainv_updated = Pfaffian._scherman_morrison_skew_jax(Ainv, updIdx, updRow)

        # Check agreement
        assert jnp.allclose(Ainv_updated, Ainv_expected, atol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(Ainv_updated - Ainv_expected))}"

    def test_skew_symmetry_check(self):
        A = jnp.array([[0., 1.], [-1., 0.]])
        assert jnp.allclose(A, -A.T)
