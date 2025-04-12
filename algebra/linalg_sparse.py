from abc import ABC, abstractmethod
import numpy as np
import scipy as sp

from .utils import JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend as __backend

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax.experimental.sparse import BCOO
    from jax import jit
    
####################################################################################################

if JAX_AVAILABLE:
    
    def _sparse_kron_jax(A, B):
        """
        Computes the Kronecker product of two sparse matrices.
        Parameters
        ----------
        A : array-like, shape (N, M)
            The first matrix.
        B : array-like, shape (P, Q)
            The second matrix.
        
        Returns
        -------
        kron_product : array-like, shape (N * P, M * Q)
            The Kronecker product of the two matrices.
        """
        if not isinstance(A, BCOO):
            A = BCOO.fromdense(A, index_dtype=jnp.int64)
        if not isinstance(B, BCOO):
            B = BCOO.fromdense(B, index_dtype=jnp.int64)

        m, n = A.shape
        p, q = B.shape
        
        # Get nonzero indices and data.
        A_idx   = A.indices  # shape (nnz_A, 2)
        A_data  = A.data    # shape (nnz_A,)
        B_idx   = B.indices  # shape (nnz_B, 2)
        B_data  = B.data    # shape (nnz_B,)
        
        # Compute new indices:
        # For each nonzero in A (i,j) and for each nonzero in B (k,l),
        # the new index is (i * p + k, j * q + l)
        nnz_A       = A_idx.shape[0]
        nnz_B       = B_idx.shape[0]
        
        # Repeat A indices and tile B indices to form all combinations.
        new_i       = jnp.repeat(A_idx[:, 0], nnz_B) * p + jnp.tile(B_idx[:, 0], nnz_A)
        new_j       = jnp.repeat(A_idx[:, 1], nnz_B) * q + jnp.tile(B_idx[:, 1], nnz_A)
        new_indices = jnp.stack([new_i, new_j], axis=1)  # shape (nnz_A*nnz_B, 2)
        
        # Compute new data as the outer product of A_data and B_data.
        new_data    = (A_data[:, None] * B_data[None, :]).reshape(-1)
        
        new_shape   = (m * p, n * q)
        return BCOO((new_data, new_indices), shape=new_shape)

def _sparse_kron_np(A, B):
    """
    Computes the Kronecker product of two sparse matrices.
    Parameters
    ----------
    A : array-like, shape (N, M)
        The first matrix.
    B : array-like, shape (P, Q)
        The second matrix.
    
    Returns
    -------
    kron_product : array-like, shape (N * P, M * Q)
        The Kronecker product of the two matrices.
    """
    if not sp.sparse.issparse(A):
        A = sp.sparse.csr_matrix(A)
    if not sp.sparse.issparse(B):
        B = sp.sparse.csr_matrix(B)
    return sp.sparse.kron(A, B)

def kron(A, B, backend="default"):
    """
    Computes the Kronecker product of two sparse matrices.
    Parameters
    ----------
    A : array-like, shape (N, M)
        The first matrix.
    B : array-like, shape (P, Q)
        The second matrix.
    backend : module, optional
        The numerical backend to use (default is jnp if JAX is available, otherwise np).
    
    Returns
    -------
    kron_product : array-like, shape (N * P, M * Q)
        The Kronecker product of the two matrices.
    """
    backend = __backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return _sparse_kron_np(A, B)
    return _sparse_kron_jax(A, B)

####################################################################################################

def _identity_np(n, dtype=np.float64):
    '''Returns the identity matrix of size n.'''
    return sp.sparse.eye(n, dtype=dtype)

if JAX_AVAILABLE:
    
    def _identity_jax(n, dtype=jnp.float64):
        '''Returns the identity matrix of size n.'''
        indices = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1, dtype=jnp.int32)
        data    = jnp.ones(n, dtype=dtype)
        return BCOO((data, indices), shape=(n, n))

def identity(n, dtype=None, backend="default"):
    '''Returns the identity matrix of size n.'''
    backend = __backend(backend)
    dtype   = dtype if dtype is not None else backend.float64
    
    if backend == np or not JAX_AVAILABLE:
        return _identity_np(n, dtype=dtype)
    return _identity_jax(n, dtype=dtype)

####################################################################################################