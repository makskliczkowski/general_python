'''
This module provides various linear algebra operations, including matrix transformations,
outer and Kronecker products, inner products, and matrix diagonalization.
It supports both dense and sparse matrices, and can operate with different numerical backends
such as NumPy and JAX.
'''

import numpy as np
import scipy as sp
from numba import njit
from functools import partial

from ..algebra.utils import maybe_jit, get_backend, JIT, Array

# -----------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    JAX_AVAILABLE = True
except ImportError:
    jax = jnp = jsp = None
    JAX_AVAILABLE = False

# -----------------------------------------------------------------
#! Matrix operations
# -----------------------------------------------------------------

@maybe_jit
def change_basis(unitary_matrix: Array, state_vector: Array, backend="default"):
    """
    Transforms a state vector to a new basis using a unitary matrix U.
    $V' = U V$
    
    Parameters:
        - unitary_matrix : array-like, shape (N, N)
        The unitary matrix defining the new basis.
        - state_vector : array-like, shape (N,) or (N, 1)
        The state vector to be transformed.
        - backend : module (optional) - string 'default', 'np', 'jnp'
        The numerical backend to use (default is jnp if JAX is available, otherwise np).
    
    Returns:
        transformed_state : array-like
            The state vector expressed in the new basis.
    """
    backend         =   get_backend(backend)
    
    # Convert inputs to arrays using the provided backend.
    U               = backend.asarray(unitary_matrix)
    vec             = backend.asarray(state_vector)
    
    # Determine a common data type.
    common_dtype    = backend.result_type(U, vec)
    U               = U.astype(common_dtype, copy=False)
    vec             = vec.astype(common_dtype, copy=False)
    
    # Perform the basis transformation.
    transformed_state = U @ vec
    return transformed_state

# -----------------------------------------------------------------

@maybe_jit
def _change_basis_matrix(unitary_matrix, matrix, backend = "default"):
    backend = get_backend(backend)
    
    # Convert inputs to arrays using the provided backend.
    U       = backend.asarray(unitary_matrix)
    A       = backend.asarray(matrix)
    
    # Promote the inputs to a common data type.
    common_dtype    = backend.result_type(U, A)
    U               = U.astype(common_dtype, copy=False)
    A               = A.astype(common_dtype, copy=False)
    
    # Compute the conjugate transpose of U (which equals the transpose if U is real).
    U_H             = backend.conj(U).T
    
    return U @ A @ U_H

@maybe_jit
def _change_basis_matrix_back(unitary_matrix, matrix, backend = "default"):
    backend = get_backend(backend)
    
    # Convert inputs to arrays using the provided backend.
    U       = backend.asarray(unitary_matrix)
    A       = backend.asarray(matrix)
    
    # Promote the inputs to a common data type.
    common_dtype    = backend.result_type(U, A)
    U               = U.astype(common_dtype, copy=False)
    A               = A.astype(common_dtype, copy=False)
    
    # Compute the conjugate transpose of U (which equals the transpose if U is real).
    U_H             = backend.conj(U).T
    
    return U_H @ A @ U

def change_basis_matrix(unitary_matrix, matrix, back = False, backend = "default"):
    """
    Change the basis of a matrix using a unitary transformation.
    This function applies a change-of-basis transformation to the given matrix using the
    specified unitary matrix. The transformation can be performed in both the forward and
    reverse directions depending on the 'back' flag.
    Parameters:
        unitary_matrix (array-like): A unitary matrix used for the change-of-basis transformation.
        matrix (array-like): The matrix whose basis is to be transformed.
        back (bool, optional): Flag indicating the direction of the transformation.
            - If False (default), computes U * A * Uᴴ (or U * A * Uᵀ if U is real).
            - If True, computes Uᴴ * A * U to reverse the transformation.
        backend (module or object, optional): An object providing array operations such as
            asarray, conj, and array transposition. This parameter should behave similarly to
            NumPy's API. The default value "default" should be replaced with an actual backend.
    Returns:
        array-like: The matrix after applying the change-of-basis transformation.
    Notes:
        - The function first converts the inputs to arrays using the provided backend's asarray.
        - It ensures that both inputs have a common data type, facilitating consistent computations.
        - For complex matrices, the conjugate transpose is computed; otherwise, it is equivalent
        to the simple transpose.
    """
    
    if not back:
        return _change_basis_matrix(unitary_matrix, matrix, backend=backend)
    return _change_basis_matrix_back(unitary_matrix, matrix, backend=backend)
# -----------------------------------------------------------------

@maybe_jit
def outer(A, B, backend="default"):
    """
    Computes the outer product of two vectors.
    https://numpy.org/doc/stable/reference/generated/numpy.outer.html
    Parameters
    ----------
    A : array-like, shape (N,)
        The first vector.
    B : array-like, shape (M,)
        The second vector.
    backend : module, optional
        The numerical backend to use (default is jnp if JAX is available, otherwise np).
    
    Returns
    -------
    outer_product : array-like, shape (N, M)
        The outer product of the two vectors.
    """
    backend = get_backend(backend)
    return backend.outer(A, B)

# -----------------------------------------------------------------

@maybe_jit
def kron(A, B, backend="default"):
    """
    Computes the Kronecker product of two matrices.
    https://numpy.org/doc/stable/reference/generated/numpy.kron.html
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
    backend = get_backend(backend)
    return backend.kron(A, B)

# -----------------------------------------------------------------

@maybe_jit
def ket_bra(vec: Array, backend="default"):
    """Computes the ket-bra (outer product) of a vector."""
    backend = get_backend(backend)
    return outer(vec, vec, backend=backend)

@maybe_jit
def bra_ket(vec: Array, backend="default"):
    """Computes the bra-ket (outer product) of a vector."""
    backend = get_backend(backend)
    return backend.dot(vec.T, vec)

@maybe_jit
def inner(vec1: Array, vec2: Array, backend="default"):
    """Computes the inner product of two vectors."""
    backend = get_backend(backend)
    return backend.dot(vec1, vec2)

# -----------------------------------------------------------------

# @njit
def act_np(mat, x):
    """
    Apply the transformation given by mat on the state(s) x using standard matrix multiplication.
    (This function assumes mat is a dense NumPy array.)
    """
    return np.matmul(mat, x)

def _act_dense_or_sparse_np(mat, x):
    '''
    '''
    # If mat is sparse, use its built-in multiplication (which is in object mode)
    if sp.sparse.issparse(mat):
        return mat @ x
    return act_np(mat, x)

# @maybe_jit
def act(mat, x, backend="default"):
    """
    Apply the transformation given by mat on the state(s) x.
    
    Parameters:
        mat : 2D array representing the transformation operator.
        x   : Either a 1D state vector or a 2D array whose columns are state vectors.
    
    Returns:
        The transformed state(s), computed via standard matrix multiplication.
    """
    backend = get_backend(backend)
    if backend == np or not JAX_AVAILABLE:
        return _act_dense_or_sparse_np(mat, x)
    return mat @ x

def overlap_jax(a: Array,
                O: Array,   # or any object supporting O @ x
                b: Array = None) -> Array:
    a_mat   = jnp.atleast_2d(a)  # (dim, n)
    b_mat   = jnp.atleast_2d(b) if b is not None else a_mat  # (dim, m)
    Ob      = O @ b_mat  # (dim, m) or (dim, 1)
    res     = jnp.matmul(a_mat.conj().T, Ob)  # (1, m), or (n, 1), or (1,1)
    return res

def overlap(a: Array, 
            O: Array,   # or any object supporting O @ x
            b: Array = None) -> Array:
    """
    Compute <a|O|b> for vectors or matrices of Column states.

    Parameters
    ----------
    a : array_like
        Shape (dim,) or (dim, n).  Column states |a_i> are the columns of `a`.
    O : array_like or sparse matrix
        Shape (dim, dim).  The operator.
    b : array_like
        Shape (dim,) or (dim, m).  Column states |b_j> are the columns of `b`.

    Returns
    -------
    overlaps : scalar, or ndarray
        If both `a` and `b` are 1D, returns a single complex scalar <a|O|b>.
        If one argument is 1D and the other is 2D, returns a 1D array of length n or m.
        If both are 2D, returns an (n \times m) matrix whose (i,j) entry is <a_i|O|b_j>.
    """
    # ensure 2D column‐matrix form
    if b is None:
        b = a
    
    if a.ndim == 1:
        a_mat = a[:, None]        # (dim, 1)
    else:
        a_mat = a                 # (dim, n)

    if b.ndim == 1:
        b_mat = b[:, None]        # (dim, 1)
    else:
        b_mat = b                 # (dim, m)

    # apply operator
    # Ob = np.matmul(O, b_mat)      # (dim, m) or (dim, 1)
    Ob = (O @ b_mat)                # (dim, m) or (dim, 1)

    # compute overlaps: a\dag (O b)
    # res = np.matmul(a_mat.conj().T, Ob)  # (1, m), or (n, 1), or (1,1)
    res = a_mat.conj().T @ Ob     # (n, m), or (1, m), (n, 1), or (1,1)

    # squeeze trivial dimensions
    if res.shape == (1, 1):
        return res[0, 0]
    if res.shape[0] == 1:
        return res[0, :]
    if res.shape[1] == 1:
        return res[:, 0]
    return res

# -----------------------------------------------------------------

# @maybe_jit
def overlap_diag(a  : Array,
                O   : Array,
                b   : Array = None) -> Array:
    """
    Compute only the diagonal elements <a_i|O|b_i> for column-states.

    If `a` and `b` are both 1D (shape (dim,)), returns the scalar
        <a|O|b>.

    If `a` and `b` are both 2D with shape (dim, n), returns a 1D array `r`
    of length n with
        r[i] = <a_i|O|b_i> = ∑_k conj(a[k,i]) \cdot  (O @ b[:,i])[k].

    If `b` is None, it defaults to `a`, giving <a_i|O|a_i> for each i.

    """
    return overlap(a, O, b).diagonal()
    if b is None:
        b = a
    
    # ensure column‐matrix form
    if a.ndim == 1:
        a_mat = a[:, None]       # (dim, 1)
    else:
        a_mat = a                # (dim, n)

    if b is None:
        b_mat = a_mat
    elif b.ndim == 1:
        b_mat = b[:, None]       # (dim, 1)
    else:
        b_mat = b                # (dim, n)

    if a_mat.shape != b_mat.shape:
        raise ValueError("`a` and `b` must have the same shape when both are matrices.")

    # apply operator to each column of b_mat
    Ob = O @ b_mat if hasattr(O, "__matmul__") else O.dot(b_mat)  # (dim, n)

    # elementwise multiply and sum over basis index: <a_i|O|b_i>
    diag_r = np.einsum('ik,ki->i', np.conj(a_mat), Ob)  # (n,)

    # unwrap scalar case
    if diag_r.size == 1:
        return diag_r[0]
    return diag_r

# -----------------------------------------------------------------
#! Matrix properties
# -----------------------------------------------------------------

def __trace(matrix, backend):
    '''Computes the trace of a matrix.'''
    return backend.trace(matrix)

def trace(matrix, backend = "default"):
    """
    Computes the trace of a matrix.
    The trace of a matrix A is defined as the sum of its diagonal elements.
    Parameters:
        matrix (array-like): The matrix for which to compute the trace.
        backend (module or object, optional): An object providing array operations such as
            asarrays
        and array transposition. This parameter should behave similarly to NumPy's API.
        The default value "default" should be replaced with an actual backend.
    Returns:    
        float: The trace of the input matrix.
    """
    backend = get_backend(backend)
    return __trace(matrix, backend)

def __hs_norm(matrix, backend):
    '''Computes the Hilbert-Schmidt norm of a matrix.'''
    return backend.sqrt(backend.trace(backend.dot(backend.conj(matrix), matrix)))

def hilbert_schmidt_norm(matrix, backend = "default"):
    """
    Computes the Hilbert-Schmidt norm of a matrix.
    The Hilbert-Schmidt norm of a matrix A is defined as:
    ||A||_HS = sqrt(Tr(A^H A))
    Parameters:
        matrix (array-like): The matrix for which to compute the Hilbert-Schmidt norm.
        backend (module or object, optional): An object providing array operations such as
            asarray
        and array transposition. This parameter should behave similarly to NumPy's API.
        The default value "default" should be replaced with an actual backend.
    Returns:
    
        float: The Hilbert-Schmidt norm of the input matrix.
    """
    backend = get_backend(backend)
    return __hs_norm(matrix, backend)

# -----------------------------------------------------------------
#! Matrix creation
# -----------------------------------------------------------------

def __identity(n, backend, dtype):
    '''Creates an identity matrix of size n x n.'''
    return backend.eye(n, dtype=dtype)

def identity(n : int, dtype = None, backend = "default"):
    """
    Creates an identity matrix of size n x n.
    Parameters:
        n (int): 
            The size of the identity matrix.
        backend (module or object, optional): An object providing array operations such as
            asarray, eye, and array transposition. This parameter should behave similarly to
            NumPy's API. The default value "default" should be replaced with an actual backend.
        dtype (data-type, optional): The data type of the matrix elements.
    Returns:
        array-like: The identity matrix of size n x n.
    """
    backend = get_backend(backend)
    return __identity(n, backend, dtype)

# -----------------------------------------------------------------
#! Transformation
# -----------------------------------------------------------------

def transform_backend(x, is_sparse: bool = False, backend="default"):
    """
    Transforms an array-like object 'x' to the appropriate backend format.

    Parameters:
        x (array-like): The input array (NumPy array, SciPy sparse matrix, JAX array, etc.).
        is_sparse (bool, optional): If True, attempts to convert to a sparse format in the target backend
                                    if applicable (csr_matrix for NumPy, BCOO for JAX if available).
                                    If False, converts to a dense array format in the target backend.
                                    Defaults to False.
        backend (str or module, optional):  "numpy", "jax", "default", or backend module (np, jnp).
                                            Defaults to "default", which resolves to JAX if available, NumPy otherwise.

    Returns:
        array-like: The transformed array in the specified backend format.
    """
    target_backend = get_backend(backend)
    
    if target_backend == np or not JAX_AVAILABLE:
        if is_sparse:
            if not sp.sparse.isspmatrix(x):
                return sp.sparse.csr_matrix(x) # Convert to CSR if sparse is requested and not already sparse
            else:
                return x                        # Already sparse, return as is
        else:
            if sp.sparse.isspmatrix(x):
                return x.toarray()              # Convert to dense NumPy array if sparse input and dense output requested
            else:
                return np.array(x)              # Ensure it's a NumPy array if dense is requested
    # Fall back to JAX if available
    if is_sparse:
        if sp.sparse.isspmatrix(x):
            coo = x.tocoo()
            # Convert SciPy sparse to JAX BCOO format
            return jax.experimental.sparse.BCOO((coo.data, np.array([coo.row, coo.col]).T), shape=coo.shape) 
        return x
    # Fall back to JAX if available - for not sparse
    if sp.sparse.isspmatrix(x):
        return jnp.array(x.toarray())           # Convert to dense JAX array if sparse input and dense output requested
    return jnp.array(x)                         # Convert to dense JAX array if dense input and dense output requested

# -----------------------------------------------------------------
#! Diagonalization
# -----------------------------------------------------------------

if JAX_AVAILABLE:

    @JIT
    def _eig_jax(matrix):
        '''Computes the eigenvalues and eigenvectors of a matrix.'''
        return jnp.linalg.eigh(matrix)
    
    def _eigsh_transfrom_jax(matrix):
        ''' Computes the eigenvalues and eigenvectors of a matrix.'''    
        matrix_np = np.array(matrix) if isinstance(matrix, jnp.ndarray) else matrix
        if not sp.sparse.isspmatrix(matrix_np):
            return sp.sparse.csr_matrix(matrix_np)
        return matrix_np
    
    def _eig_sparse_jax(matrix):
        '''Computes the eigenvalues and eigenvectors of a matrix.'''
            
        matrix_sparse       = _eigsh_transfrom_jax(matrix)
        evals_np, evecs_np  = sp.sparse.linalg.eigsh(matrix_sparse)
        return jnp.array(evals_np), jnp.array(evecs_np) # Convert results to JAX arrays

    def _eig_lanczos_jax(matrix, k=6, which='SA'):
        '''Computes the eigenvalues and eigenvectors of a matrix using the Lanczos algorithm.'''
        matrix_sparse       = _eigsh_transfrom_jax(matrix)
        evals_np, evecs_np  = sp.sparse.linalg.eigsh(matrix_sparse, k=k, which=which)
        return jnp.array(evals_np), jnp.array(evecs_np) 

    def _eig_shift_inv_jax(matrix, k=6, sigma=0.0, which='LM', mode='normal'):
        '''Computes the eigenvalues and eigenvectors of a matrix using the shift-invert method.'''
        matrix_sparse       = _eigsh_transfrom_jax(matrix)
        evals_np, evecs_np  = sp.sparse.linalg.eigsh(matrix_sparse, k=k, sigma=sigma, which=which, mode=mode)
        return jnp.array(evals_np), jnp.array(evecs_np)

def _eig_np(matrix):
    '''Computes the eigenvalues and eigenvectors of a matrix.'''
    return sp.linalg.eigh(matrix)

def _eig_sparse_np(matrix):
    '''Computes the eigenvalues and eigenvectors of a matrix.'''
    return sp.sparse.linalg.eigsh(matrix)

def _eig_lanczos_np(matrix, k=6, which='SA'):
    '''Computes the eigenvalues and eigenvectors of a matrix using the Lanczos algorithm.'''
    return sp.sparse.linalg.eigsh(matrix, k=k, which=which)

def _eig_shift_inv_np(matrix, k=6, sigma=0.0, which='LM', mode='normal'):
    '''Computes the eigenvalues and eigenvectors of a matrix using the shift-invert method.'''
    return sp.sparse.linalg.eigsh(matrix, k=k, sigma=sigma, which=which, mode=mode)

def eigh(matrix, backend="default", **kwargs):
    """
    Diagonalizes a Hermitian matrix.
    This function computes the eigenvalues and eigenvectors of a Hermitian matrix.
    Parameters:
        matrix (array-like): The Hermitian matrix to diagonalize.
        backend (module or object, optional): An object providing array operations such as
            asarrays
        and array transposition. This parameter should behave similarly to NumPy's API.
        The default value "default" should be replaced with an actual backend.
    Returns:
        tuple: A tuple containing the eigenvalues and eigenvectors of the input matrix.
    """
    backend = get_backend(backend)
    
    if backend == np or not JAX_AVAILABLE:
        return _eig_np(matrix)
    return _eig_jax(matrix)

def eigsh(matrix, backend="default", **kwargs):
    """
    Diagonalizes a Hermitian matrix.
    This function computes the eigenvalues and eigenvectors of a Hermitian matrix.
    Parameters:
        matrix (array-like):
            The Hermitian matrix to diagonalize.
        backend (module or object, optional):
            An object providing array operations such as
            asarrays
        and array transposition. This parameter should behave similarly to NumPy's API.
        The default value "default" should be replaced with an actual backend.
    Returns:
        tuple: A tuple containing the eigenvalues and eigenvectors of the input matrix.
    """
    backend = get_backend(backend)
    method  = kwargs.get("method", "lanczos")
    if method == "lanczos":
        k       = kwargs.get("k", 6)
        which   = kwargs.get("which", "SA")
        if backend == np or not JAX_AVAILABLE:
            return _eig_lanczos_np(matrix, k=k, which=which)
        return _eig_lanczos_jax(matrix, k=k, which=which)
    elif method == "shift-invert":
        k       = kwargs.get("k", 6)
        sigma   = kwargs.get("sigma", 0.0)
        which   = kwargs.get("which", "LM")
        mode    = kwargs.get("mode", "normal")
        if backend == np or not JAX_AVAILABLE:
            return _eig_shift_inv_np(matrix, k=k, sigma=sigma, which=which, mode=mode)
        return _eig_shift_inv_jax(matrix, k=k, sigma=sigma, which=which, mode=mode)
    if backend == np or not JAX_AVAILABLE:
        return _eig_np(matrix.todense())
    return _eig_jax(matrix.todense())

# ------------------------------------------------------------------
#! State manipulation
# ------------------------------------------------------------------

if JAX_AVAILABLE:
    
    @partial(jax.jit, static_argnums=(5,))
    def _apply_complex_givens_rotation_jax(
            V     : jnp.ndarray, 
            i     : int, 
            j     : int, 
            theta : float, 
            phi   : float, 
            dim   : int) -> jnp.ndarray:
        """
        Apply a complex Givens rotation on V of size (dim x dim).

        Rotation matrix R acts on the (i, j) subspace as:
        R_ii = cos(theta)
        R_jj = cos(theta)
        R_ij = -sin(theta) * exp(i * phi)
        R_ji = sin(theta) * exp(-i * phi)

        Parameters
        ----------
        V     : (dim, dim) complex ndarray
            Unitary matrix to rotate.
        i, j  : int
            Indices to rotate between.
        theta : float
            Rotation angle in radians.
        phi   : float
            Phase angle in radians.
        dim   : int
            Dimension of the unitary matrix.

        Returns
        -------
        jnp.ndarray
            Rotated matrix V_new = V @ R
        """
        R = jnp.eye(dim, dtype=jnp.complex64)
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        exp_i_phi = jnp.exp(1j * phi)
        exp_neg_i_phi = jnp.exp(-1j * phi)

        R = R.at[i, i].set(c)
        R = R.at[j, j].set(c)
        R = R.at[i, j].set(-s * exp_i_phi)
        R = R.at[j, i].set(s * exp_neg_i_phi)

        return V @ R

def _apply_givens_rotation_np(V: np.ndarray, i: int, j: int, theta: float, M: int = 2):
    """
    Applies a Givens rotation to a matrix or vector.
    
    Parameters:
        V (array-like): The matrix or vector to which the Givens rotation is applied.
        i (int): The index of the first row/column to rotate.
        j (int): The index of the second row/column to rotate.
        theta (float): The angle of rotation in radians.
        M (int, optional): The size of the Givens rotation matrix. Default is 2.
    
    Returns:
        array-like: The transformed matrix or vector after applying the Givens rotation.
    """
    R       = np.eye(M)                        # Identity matrix of size MxM 
    c, s    = np.cos(theta), np.sin(theta)    # Cosine and sine of the rotation angle 
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    # Apply the Givens rotation to the matrix/vector V
    return V @ R

def apply_givens_rotation(V: Array, i: int, j: int, theta: float, M: int = 2) -> Array:
    """
    Applies a Givens rotation to a matrix or vector.
    
    Parameters:
        V (array-like): The matrix or vector to which the Givens rotation is applied.
        i (int): The index of the first row/column to rotate.
        j (int): The index of the second row/column to rotate.
        theta (float): The angle of rotation in radians.
        M (int, optional): The size of the Givens rotation matrix. Default is 2.
    
    Returns:
        array-like: The transformed matrix or vector after applying the Givens rotation.
    """
    if isinstance(V, jnp.ndarray) and JAX_AVAILABLE:
        return _apply_givens_rotation_jax(V, i, j, theta, M)
    return _apply_givens_rotation_np(V, i, j, theta, M)

# ------------------------------------------------------------------
