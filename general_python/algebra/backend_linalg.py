"""
general_python/algebra/backend_linalg.py

Unified linear algebra backend providing matrix operations, decompositions,
and transformations with support for multiple backends (NumPy, JAX).

This module consolidates linalg.py and linalg_sparse.py functionality into
a single backend-aware interface.

Features:
  - Matrix transformations and basis changes
  - Outer and Kronecker products (dense and sparse)
  - Inner products and overlaps
  - Matrix properties (trace, Hilbert-Schmidt norm)
  - Matrix creation (identity, etc.)
  - Eigendecomposition (dense, sparse, Lanczos)
  - State manipulation (Givens rotations)
  - Backend support: NumPy, SciPy, JAX

Type Safety:
  - Proper dtype handling and promotion
  - Complex values preserved through computations
  - Backend-specific type conversions handled transparently

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from typing import Optional, Union, Tuple, Literal, Any, Callable
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg
from numpy.typing import NDArray

# Backend imports
try:
    from .utils import JAX_AVAILABLE, get_backend, JIT
except ImportError:
    JAX_AVAILABLE = False
    get_backend = lambda x="default": np
    JIT = lambda f: f

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax.experimental.sparse import BCOO
    from functools import partial
else:
    jnp = None
    jsp = None
    BCOO = None
    partial = lambda f, **kwargs: f

# Type alias
Array = Union[np.ndarray, Any]  # Any allows JAX arrays

# =============================================================================
# Basis Transformations
# =============================================================================

def change_basis(
        unitary_matrix  : Array,
        state_vector    : Array,
        backend         : str = "default") -> Array:
    r"""
    Transform state vector to new basis.
    
    V' = U V
    
    Parameters
    ----------
    unitary_matrix : array-like, shape (N, N)
        Unitary transformation matrix.
    state_vector : array-like, shape (N,) or (N, 1)
        State vector to transform.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Transformed state vector.
    """
    be              = get_backend(backend)
    U               = be.asarray(unitary_matrix)
    vec             = be.asarray(state_vector)
    
    # Common dtype
    common_dtype    = np.result_type(U.dtype, vec.dtype) if hasattr(U, 'dtype') else np.complex128
    U               = be.asarray(U, dtype=common_dtype)
    vec             = be.asarray(vec, dtype=common_dtype)
    
    return U @ vec

def change_basis_matrix(
        unitary_matrix  : Array,
        matrix          : Array,
        direction       : Literal['forward', 'backward'] = 'forward',
        backend         : str = "default") -> Array:
    r"""
    Change basis of matrix using unitary transformation.
    
    Forward:  A' = U A Udagger 
    Backward: A' = Udagger  A U
    
    Parameters
    ----------
    unitary_matrix : array-like, shape (N, N)
        Unitary transformation matrix U.
    matrix : array-like, shape (N, N)
        Matrix to transform A.
    direction : {'forward', 'backward'}, optional
        Transformation direction (default: 'forward').
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Transformed matrix.
    """
    be              = get_backend(backend)
    U               = be.asarray(unitary_matrix)
    A               = be.asarray(matrix)
    
    # Common dtype
    common_dtype    = np.result_type(U.dtype, A.dtype) if hasattr(U, 'dtype') else np.complex128
    U               = be.asarray(U, dtype=common_dtype)
    A               = be.asarray(A, dtype=common_dtype)

    U_H             = be.conj(U).T

    if direction == 'forward':
        return U @ A @ U_H
    else:  # backward
        return U_H @ A @ U

# =============================================================================
# Outer and Kronecker Products
# =============================================================================

def outer(A: Array, B: Array, backend: str = "default") -> Array:
    r"""
    Compute outer product of two vectors.
    
    C = A otimes B (element-wise: C[i,j] = A[i] * B[j])
    
    Parameters
    ----------
    A : array-like, shape (N,)
        First vector.
    B : array-like, shape (M,)
        Second vector.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Outer product, shape (N, M).
    """
    be = get_backend(backend)
    return be.outer(A, B)

def kron(A: Array, B: Array, backend: str = "default") -> Array:
    r"""
    Compute Kronecker product of two matrices (dense).
    
    C = A otimes B
    
    Parameters
    ----------
    A : array-like, shape (N, M)
        First matrix.
    B : array-like, shape (P, Q)
        Second matrix.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Kronecker product, shape (N*P, M*Q).
    """
    be = get_backend(backend)
    return be.kron(A, B)

def kron_sparse(A: Array, B: Array, backend: str = "default") -> Array:
    r"""
    Compute Kronecker product of sparse matrices.
    
    Preserves sparsity structure efficiently.
    
    Parameters
    ----------
    A : array-like or sparse, shape (N, M)
        First matrix.
    B : array-like or sparse, shape (P, Q)
        Second matrix.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Kronecker product, sparse if JAX BCOO available, else dense.
    """
    be = get_backend(backend)
    
    if backend == "jax" and JAX_AVAILABLE:
        return _kron_sparse_jax(A, B)
    else:
        return _kron_sparse_numpy(A, B)

def _kron_sparse_numpy(A: Array, B: Array) -> scipy.sparse.csr_matrix:
    """Kronecker product using SciPy sparse matrices."""
    if not sp.sparse.issparse(A):
        A = sp.sparse.csr_matrix(A)
    if not sp.sparse.issparse(B):
        B = sp.sparse.csr_matrix(B)
    return sp.sparse.kron(A, B)

def _kron_sparse_jax(A: Array, B: Array) -> "BCOO":
    """Kronecker product using JAX BCOO sparse matrices."""
    if not isinstance(A, BCOO):
        A = BCOO.fromdense(A, index_dtype=jnp.int64)
    if not isinstance(B, BCOO):
        B = BCOO.fromdense(B, index_dtype=jnp.int64)
    
    m, n = A.shape
    p, q = B.shape
    
    A_idx, A_data = A.indices, A.data
    B_idx, B_data = B.indices, B.data
    
    nnz_A = A_idx.shape[0]
    nnz_B = B_idx.shape[0]
    
    # All combinations of nonzeros
    new_i       = jnp.repeat(A_idx[:, 0], nnz_B) * p + jnp.tile(B_idx[:, 0], nnz_A)
    new_j       = jnp.repeat(A_idx[:, 1], nnz_B) * q + jnp.tile(B_idx[:, 1], nnz_A)
    new_indices = jnp.stack([new_i, new_j], axis=1)
    
    new_data    = (A_data[:, None] * B_data[None, :]).reshape(-1)
    new_shape   = (m * p, n * q)
    
    return BCOO((new_data, new_indices), shape=new_shape)

# =============================================================================
# Inner Products and Overlaps
# =============================================================================

def inner(vec1: Array, vec2: Array, backend: str = "default") -> Array:
    r"""
    Compute inner product of two vectors.
    
    <v1|v2> = v1dagger  \cdot  v2
    
    Parameters
    ----------
    vec1 : array-like, shape (N,)
        First vector.
    vec2 : array-like, shape (N,)
        Second vector.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    scalar
        Inner product (complex if inputs complex, real otherwise).
    """
    be = get_backend(backend)
    return be.dot(be.conj(vec1), vec2)

def ket_bra(vec: Array, backend: str = "default") -> Array:
    r"""
    Compute ket-bra (outer product) of a vector.
    
    |Psi ><Psi | = Psi  Psi^dagger 
    
    Parameters
    ----------
    vec : array-like, shape (N,)
        Vector.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Outer product matrix, shape (N, N).
    """
    be = get_backend(backend)
    return be.outer(vec, be.conj(vec))

def bra_ket(vec: Array, backend: str = "default") -> Array:
    r"""
    Compute bra-ket (inner product) of a vector with itself.
    
    <Psi |Psi > = Psi dagger  \cdot  Psi  = ||Psi ||²
    
    Parameters
    ----------
    vec : array-like, shape (N,)
        Vector.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    scalar
        Squared norm (always real and non-negative).
    """
    be = get_backend(backend)
    return be.real(be.dot(be.conj(vec), vec))

def overlap(
        a           : Array,
        O           : Array,
        b           : Optional[Array] = None,
        backend     : str = "default") -> Array:
    r"""
    Compute matrix element <a|O|b>.
    
    Supports:
      - 1D vectors: returns scalar
      - 2D matrices (columns are states): returns matrix of overlaps
      - Mixed dimensions: returns vector
    
    Parameters
    ----------
    a : array-like, shape (dim,) or (dim, n)
        Left states. Columns are individual states if 2D.
    O : array-like or sparse, shape (dim, dim)
        Operator matrix.
    b : array-like, shape (dim,) or (dim, m), optional
        Right states. If None, defaults to a.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    scalar, 1D or 2D Array
        <a|O|b>. Shape depends on input dimensions.
        
    Examples
    --------
    >>> # Single pair of vectors: scalar result
    >>> s = overlap(psi, H, phi)  # shape: (), dtype: complex
    
    >>> # Multiple right states: vector result
    >>> v = overlap(psi, H, phi_array)  # shape: (n,), dtype: complex
    
    >>> # Multiple left and right: matrix result
    >>> M = overlap(psi_array, H, phi_array)  # shape: (m, n), dtype: complex
    """
    be = get_backend(backend)
    
    if b is None:
        b = a
    
    # Convert to 2D column matrices
    if a.ndim == 1:
        a_mat = a[:, None]
    else:
        a_mat = a
    
    if b.ndim == 1:
        b_mat = b[:, None]
    else:
        b_mat = b
    
    # Apply operator
    Ob  = O @ b_mat
    
    # Compute overlaps: adagger  O b
    res = a_mat.conj().T @ Ob
    
    # Squeeze trivial dimensions
    if res.shape == (1, 1):
        return res[0, 0]
    if res.shape[0] == 1:
        return res[0, :]
    if res.shape[1] == 1:
        return res[:, 0]
    
    return res

def overlap_diagonal(
        a       : Array,
        O       : Array,
        b       : Optional[Array] = None,
        backend : str = "default") -> Array:
    r"""
    Compute only diagonal elements <a_i|O|b_i>.
    
    More efficient than full overlap() when only diagonals needed.
    
    Parameters
    ----------
    a : array-like, shape (dim,) or (dim, n)
        Left states (columns are states if 2D).
    O : array-like, shape (dim, dim)
        Operator matrix.
    b : array-like, shape (dim,) or (dim, n), optional
        Right states. If None, defaults to a.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    scalar or 1D Array
        Diagonal elements <a_i|O|b_i>.
    """
    be = get_backend(backend)
    
    if b is None:
        b = a
    
    if a.ndim == 1:
        # Scalar case
        return inner(a, O @ b, backend)
    
    # Matrix case: extract diagonals
    a_mat = be.atleast_2d(a)
    b_mat = be.atleast_2d(b)
    
    if a_mat.shape != b_mat.shape:
        raise ValueError("a and b must have same shape for diagonal overlap")
    
    # Apply operator to each column
    Ob = O @ b_mat
    
    # Diagonal: a[i]dagger  O b[i]
    return be.einsum('ik,ki->i', be.conj(a_mat), Ob)

# =============================================================================
# Matrix Properties
# =============================================================================

def trace(matrix: Array, backend: str = "default") -> Any:
    r"""
    Compute matrix trace.
    
    Tr(A) = Sum_i A_ii
    
    Parameters
    ----------
    matrix : array-like, shape (N, N)
        Matrix to compute trace of.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    scalar
        Trace of matrix.
    """
    be = get_backend(backend)
    return be.trace(matrix)

def hilbert_schmidt_norm(matrix: Array, backend: str = "default") -> Any:
    r"""
    Compute Hilbert-Schmidt norm of matrix.
    
    ||A||_HS = √(Tr(Adagger  A))
    
    Parameters
    ----------
    matrix : array-like, shape (N, N)
        Matrix to compute norm of.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    scalar (real)
        Hilbert-Schmidt norm.
    """
    be = get_backend(backend)
    A = be.asarray(matrix, dtype=be.complex128)
    return be.sqrt(be.trace(be.conj(A) @ A))

def frobenius_norm(matrix: Array, backend: str = "default") -> Any:
    r"""
    Compute Frobenius norm of matrix.
    
    ||A||_F = √(Sum_ij |A_ij|²)
    
    Parameters
    ----------
    matrix : array-like, shape (N, M)
        Matrix to compute norm of.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    scalar (real)
        Frobenius norm.
    """
    be = get_backend(backend)
    return be.linalg.norm(matrix, 'fro')

# =============================================================================
# Matrix Creation
# =============================================================================

def identity(
        n       : int,
        dtype   : Optional[Union[str, np.dtype]] = None,
        backend : str = "default") -> Array:
    r"""
    Create identity matrix I_n.
    
    Parameters
    ----------
    n : int
        Size of identity matrix.
    dtype : data-type, optional
        Element data type (default: backend default).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Identity matrix, shape (n, n).
    """
    be = get_backend(backend)
    if dtype is None:
        dtype = be.float64
    return be.eye(n, dtype=dtype)

def identity_sparse(
        n       : int,
        dtype   : Optional[Union[str, np.dtype]] = None,
        backend : str = "default") -> Union[scipy.sparse.csr_matrix, "BCOO"]:
    r"""
    Create sparse identity matrix.
    
    Parameters
    ----------
    n : int
        Size of identity matrix.
    dtype : data-type, optional
        Element data type (default: float64).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    sparse matrix
        Sparse identity: scipy.sparse.csr_matrix (numpy) or BCOO (jax).
    """
    if dtype is None:
        dtype = np.float64
    
    if backend == "jax" and JAX_AVAILABLE:
        indices = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1, dtype=jnp.int32)
        data = jnp.ones(n, dtype=dtype)
        return BCOO((data, indices), shape=(n, n))
    else:
        return sp.sparse.eye(n, dtype=dtype)

# =============================================================================
# Backend Format Conversion
# =============================================================================

def to_dense(
        matrix  : Array,
        backend : str = "default") -> Array:
    r"""
    Convert sparse or other formats to dense array.
    
    Parameters
    ----------
    matrix : array-like
        Matrix to convert.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Dense array.
    """
    be = get_backend(backend)
    
    if sp.sparse.issparse(matrix):
        return matrix.toarray()
    elif JAX_AVAILABLE and isinstance(matrix, BCOO):
        return matrix.todense()
    else:
        return be.asarray(matrix)

def to_sparse(
        matrix  : Array,
        backend : str = "default",
        format  : str = "csr") -> Union[scipy.sparse.csr_matrix, "BCOO"]:
    r"""
    Convert dense matrix to sparse format.
    
    Parameters
    ----------
    matrix : array-like, shape (N, M)
        Matrix to convert.
    backend : str, optional
        Numerical backend (default: 'default').
    format : str, optional
        Sparse format: 'csr', 'csc', 'coo', 'bsr' (numpy)
        or 'bcoo' (jax). Default: 'csr'.
        
    Returns
    -------
    sparse matrix
        Sparse matrix in specified format.
    """
    if backend == "jax" and JAX_AVAILABLE and format == "bcoo":
        if isinstance(matrix, BCOO):
            return matrix
        else:
            return BCOO.fromdense(matrix)
    else:
        if sp.sparse.issparse(matrix):
            return matrix.asformat(format)
        else:
            return sp.sparse.csr_matrix(matrix, format=format)

# =============================================================================
# Eigendecomposition
# =============================================================================

def eig(matrix  : Array,
        backend : str = "default",
        **kwargs) -> Tuple[Array, Array]:
    r"""
    Eigendecomposition of general matrix.
    
    Computes all eigenvalues and eigenvectors.
    
    Parameters
    ----------
    matrix : array-like, shape (N, N)
        Matrix to diagonalize.
    backend : str, optional
        Numerical backend (default: 'default').
    **kwargs
        Additional arguments (backend-specific).
        
    Returns
    -------
    eigenvalues : Array, shape (N,)
        Eigenvalues.
    eigenvectors : Array, shape (N, N)
        Eigenvectors as columns.
    """
    be = get_backend(backend)
    
    if backend == "jax" and JAX_AVAILABLE:
        evals, evecs = jnp.linalg.eig(matrix)
    else:
        evals, evecs = np.linalg.eig(matrix)
    
    return evals, evecs

def eigh(
        matrix  : Array,
        backend : str = "default",
        **kwargs) -> Tuple[Array, Array]:
    r"""
    Eigendecomposition of Hermitian matrix (dense).
    
    More stable than eig() for Hermitian/symmetric matrices.
    
    Parameters
    ----------
    matrix : array-like, shape (N, N)
        Hermitian matrix to diagonalize.
    backend : str, optional
        Numerical backend (default: 'default').
    **kwargs
        Additional arguments (backend-specific).
        
    Returns
    -------
    eigenvalues : Array, shape (N,), real
        Eigenvalues in ascending order.
    eigenvectors : Array, shape (N, N)
        Eigenvectors as columns.
    """
    be = get_backend(backend)
    
    if backend == "jax" and JAX_AVAILABLE:
        evals, evecs = jnp.linalg.eigh(matrix)
    else:
        evals, evecs = np.linalg.eigh(matrix)
    
    return evals, evecs

def eigsh(
        matrix  : Array,
        k       : int = 6,
        which   : Literal['smallest', 'largest'] = 'smallest',
        backend : str = "default",
        **kwargs
) -> Tuple[Array, Array]:
    r"""
    Partial eigendecomposition of Hermitian matrix (sparse).
    
    Computes k extremal eigenvalues and eigenvectors.
    Uses SciPy sparse methods.
    
    Parameters
    ----------
    matrix : array-like, shape (N, N)
        Hermitian matrix to diagonalize.
    k : int, optional
        Number of eigenvalues to compute (default: 6).
    which : {'smallest', 'largest'}, optional
        Which eigenvalues to compute (default: 'smallest').
    backend : str, optional
        Numerical backend (default: 'default').
    **kwargs
        Additional arguments passed to scipy.sparse.linalg.eigsh.
        
    Returns
    -------
    eigenvalues : Array, shape (k,)
        k extremal eigenvalues.
    eigenvectors : Array, shape (N, k)
        Corresponding eigenvectors as columns.
    """
    # Convert to scipy's which convention
    which_map = {'smallest': 'SA', 'largest': 'LA'}
    which_sp = which_map.get(which, 'SA')
    
    evals, evecs = sp.sparse.linalg.eigsh(
        matrix,
        k=k,
        which=which_sp,
        **kwargs
    )
    
    return evals, evecs

# =============================================================================
# State Manipulation
# =============================================================================

def givens_rotation(
        V       : Array,
        i       : int,
        j       : int,
        theta   : float,
        backend : str = "default") -> Array:
    r"""
    Apply Givens rotation to matrix or vector.
    
    Rotates the (i,j) plane by angle θ.
    
    Parameters
    ----------
    V : array-like
        Matrix or vector to rotate.
    i, j : int
        Indices of rotation plane.
    theta : float
        Rotation angle in radians.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Rotated matrix/vector.
    """
    be          = get_backend(backend)
    V_rot       = be.array(V, copy=True)

    c           = be.cos(theta)
    s           = be.sin(theta)
    
    # Apply rotation to columns i and j
    v_i         = V_rot[:, i].copy()
    v_j         = V_rot[:, j].copy()
    
    V_rot[:, i] = c * v_i - s * v_j
    V_rot[:, j] = s * v_i + c * v_j
    
    return V_rot

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Basis transformations
    'change_basis',
    'change_basis_matrix',
    # Products
    'outer',
    'kron',
    'kron_sparse',
    # Inner products and overlaps
    'inner',
    'ket_bra',
    'bra_ket',
    'overlap',
    'overlap_diagonal',
    # Matrix properties
    'trace',
    'hilbert_schmidt_norm',
    'frobenius_norm',
    # Matrix creation
    'identity',
    'identity_sparse',
    # Format conversion
    'to_dense',
    'to_sparse',
    # Eigendecomposition
    'eig',
    'eigh',
    'eigsh',
    # State manipulation
    'givens_rotation',
]

# =============================================================================
# End of File
# =============================================================================