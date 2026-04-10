'''
Test the UDT decomposition from general_python.algebra.utilities.udt, including pivoted UDT and the Loh inversion method.

--------------
file        : general_python/algebra/utilities/tests/test_udt.py
author      : Maksymilian Kliczkowski
included    : 
            - test_pivoted_udt_decompose_reconstructs_matrix    - pivoted UDT should reconstruct the original matrix exactly up to numerical precision.
            - test_pivoted_udt_fact_mult_matches_direct_product - repeated left multiplication in pivoted UDT space should match the direct dense product
            - test_pivoted_udt_inv_1p_stabilizes_stressed_stack - pivoted UDT plus Loh inversion should keep the inverse residual bounded on a stressed product.
'''


from __future__ import annotations

import numpy as np

try:
    from general_python.algebra.utilities import udt
except ImportError:
    raise ImportError("Could not import udt module. Ensure that the general_python package is correctly installed and accessible.")

def _build_stack(num_slices: int, dim: int, span: float, seed: int = 42):
    ''' Build a stack of num_slices random matrices of size dim x dim, with singular values spanning from exp(span) to exp(-span).
    
    Returns the stack of matrices and their product.
    '''
    rng     = np.random.RandomState(seed)
    mats    = []
    product = np.eye(dim)
    for _ in range(num_slices):
        u_mat, _    = np.linalg.qr(rng.randn(dim, dim))
        v_mat, _    = np.linalg.qr(rng.randn(dim, dim))
        diag        = np.diag(np.exp(np.linspace(span, -span, dim)))
        mat         = u_mat @ diag @ v_mat
        mats.append(mat) # Store the matrix of UDT factorization tests
        product     = mat @ product
    return np.asarray(mats), product

# ==================================================================================

def test_pivoted_udt_decompose_reconstructs_matrix():
    """Pivoted UDT should reconstruct the original matrix exactly up to numerical precision."""
    rng             = np.random.RandomState(7)
    matrix          = rng.randn(6, 6)

    state           = udt.udt_decompose_pivoted(matrix)
    reconstructed   = state.U @ np.diag(state.D) @ state.T
    np.testing.assert_allclose(reconstructed, matrix, rtol=1e-10, atol=1e-12)

def test_pivoted_udt_fact_mult_matches_direct_product():
    """Repeated left multiplication in pivoted UDT space should match the direct dense product."""
    rng             = np.random.RandomState(11)
    left            = rng.randn(5, 5)
    right           = rng.randn(5, 5)

    state           = udt.udt_decompose_pivoted(right)
    updated         = udt.udt_fact_mult_pivoted(left, state)
    reconstructed   = updated.U @ np.diag(updated.D) @ updated.T

    np.testing.assert_allclose(reconstructed, left @ right, rtol=1e-10, atol=1e-12)

def test_pivoted_udt_inv_1p_stabilizes_stressed_stack():
    """
    Pivoted UDT plus Loh inversion should keep the inverse residual bounded on a stressed product.
    """
    Bs, product = _build_stack(num_slices=6, dim=10, span=6.0)
    state       = udt.UDTState(
                    U   = np.eye(product.shape[0], dtype=product.dtype),
                    D   = np.ones((product.shape[0],), dtype=product.dtype),
                    T   = np.eye(product.shape[0], dtype=product.dtype),
                )
    
    # Apply the stack of matrices in pivoted UDT space, starting from the identity state.
    for mat in Bs.reshape(3, 2, 10, 10):
        group_product = np.eye(product.shape[0], dtype=product.dtype)
        for idx in range(mat.shape[0]):
            group_product = mat[idx] @ group_product
        state = udt.udt_fact_mult_pivoted(group_product, state)

    green       = udt.udt_inv_1p(state, backend="numpy")
    residual    = np.linalg.norm((np.eye(product.shape[0]) + product) @ green - np.eye(product.shape[0]))

    assert np.isfinite(residual)
    assert residual < 1e-3

# ==================================================================================
#! EOF
# ==================================================================================
