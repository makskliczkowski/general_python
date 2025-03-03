from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from numba import njit

from general_python.algebra.utils import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend, JIT
import general_python.algebra.linalg_sparse as sparse 

if _JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

# -----------------------------------------------------------------
#! Matrix operations
# -----------------------------------------------------------------

@maybe_jit
def change_basis(unitary_matrix: 'array-like', state_vector: 'array-like', backend="default"):
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
def ket_bra(vec: 'array-like', backend="default"):
    """Computes the ket-bra (outer product) of a vector."""
    backend = get_backend(backend)
    return outer(vec, vec, backend=backend)

@maybe_jit
def bra_ket(vec: 'array-like', backend="default"):
    """Computes the bra-ket (outer product) of a vector."""
    backend = get_backend(backend)
    return backend.dot(vec.T, vec)

@maybe_jit
def inner(vec1: 'array-like', vec2: 'array-like', backend="default"):
    """Computes the inner product of two vectors."""
    backend = get_backend(backend)
    return backend.dot(vec1, vec2)

# -----------------------------------------------------------------

@njit
def act_np(mat, x):
    """
    Apply the transformation given by mat on the state(s) x using standard matrix multiplication.
    (This function assumes mat is a dense NumPy array.)
    """
    return np.matmul(mat, x)

def act_dense_or_sparse_np(mat, x):
    # If mat is sparse, use its built-in multiplication (which is in object mode)
    if sp.sparse.issparse(mat):
        return mat * x
    else:
        return act_np(mat, x)

@maybe_jit
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
    if backend == np or not _JAX_AVAILABLE:
        return act_dense_or_sparse_np(mat, x)
    return mat * x

# @njit
def overlap_np(a, b, mat = None):
    """
    Compute the quantum overlap <a|mat|b> where:
        - a and b can each be a state vector (1D) or a matrix (with columns as states),
        - in the matrix case, pairwise overlaps (each to each) are computed.
      
    The computation is performed via:
        overlap = (a^\dagger) dot (act(mat, b))
        
    Parameters:
        a      : Bra state(s) as a 1D vector or a 2D array (columns are states).
        mat    : 2D transformation matrix.
        b      : Ket state(s) as a 1D vector or a 2D array (columns are states).
    
    Returns:
        If both a and b are vectors: a scalar.
        If one is a matrix and the other a vector: a vector with the overlap of each column.
        If both are matrices: a matrix with the overlap for each bra/ket combination.
    """
    # Compute the transformed b
    transformed_b   = act(mat, b) if mat is not None else b
    # Convert inputs to backend arrays for uniform handling.
    # Case 1: Both a and tb_arr are vectors.
    if len(a.shape) and len(transformed_b.shape) == 1:
        return np.vdot(a, transformed_b)
    # Case 2: Both a and tb_arr are matrices (assumed: columns are states).
    elif len(a.shape) == 2 and len(transformed_b.shape) == 2:
        # Pairwise overlap: (a^\dagger) @ tb_arr gives a matrix whose (i,j) element is <a_i|mat|b_j>.
        return np.matmul(a.conj().T, transformed_b)
    # Case 3: a is a matrix and tb_arr is a vector.
    elif len(a.shape) == 2 and len(transformed_b.shape) == 1:
        # Return the overlap of each column in a with the vector tb_arr.
        return np.matmul(a.conj().T, transformed_b)
    # Case 4: a is a vector and tb_arr is a matrix.
    elif len(a.shape) == 1 and len(transformed_b.shape) == 2:
        # Treat a as a bra vector; compute its overlap with each column of tb_arr.
        # Expand a to a row vector and multiply.
        return np.matmul(a.conj()[None, :], transformed_b)[0]
    else:
        raise ValueError("Invalid dimensions for states a and b.")

@maybe_jit
def overlap(a, b, mat, backend="default"):
    """
    Compute the quantum overlap <a|mat|b> where:
        - a and b can each be a state vector (1D) or a matrix (with columns as states),
        - in the matrix case, pairwise overlaps (each to each) are computed.
    
    The computation is performed via:
        overlap = (a^\dagger) dot (act(mat, b))
        
    Parameters:
        a      : Bra state(s) as a 1D vector or a 2D array (columns are states).
        mat    : 2D transformation matrix.
        b      : Ket state(s) as a 1D vector or a 2D array (columns are states).
    
    Returns:
        If both a and b are vectors: a scalar.
        If one is a matrix and the other a vector: a vector with the overlap of each column.
        If both are matrices: a matrix with the overlap for each bra/ket combination.
    """
    backend         = get_backend(backend)
    if backend == np or not _JAX_AVAILABLE:
        if sp.sparse.issparse(mat):
            b = act(mat, b, backend=backend)
            return overlap_np(a, b)
        return overlap_np(a, b, mat)
    
    # Compute the transformed b
    transformed_b   = act(mat, b, backend=backend) if mat is not None else b
    # Convert inputs to backend arrays for uniform handling.
    a_arr           = backend.asarray(a)
    tb_arr          = backend.asarray(transformed_b)
    
    # Case 1: Both a and tb_arr are vectors.
    if a_arr.ndim == 1 and tb_arr.ndim == 1:
        return backend.vdot(a_arr, tb_arr)
    
    # Case 2: Both a and tb_arr are matrices (assumed: columns are states).
    elif a_arr.ndim == 2 and tb_arr.ndim == 2:
        # Pairwise overlap: (a^\dagger) @ tb_arr gives a matrix whose (i,j) element is <a_i|mat|b_j>.
        return backend.matmul(a_arr.conj().T, tb_arr)
    
    # Case 3: a is a matrix and tb_arr is a vector.
    elif a_arr.ndim == 2 and tb_arr.ndim == 1:
        # Return the overlap of each column in a with the vector tb_arr.
        return backend.matmul(a_arr.conj().T, tb_arr)
    
    # Case 4: a is a vector and tb_arr is a matrix.
    elif a_arr.ndim == 1 and tb_arr.ndim == 2:
        # Treat a as a bra vector; compute its overlap with each column of tb_arr.
        # Expand a to a row vector and multiply.
        return backend.matmul(a_arr.conj()[None, :], tb_arr)[0]
    else:
        raise ValueError("Invalid dimensions for states a and b.")

@maybe_jit
def overlap_diag(a, mat, b, backend="default"):
    """
    Compute only the diagonal overlaps <a_i|mat|b_i> for each state, where
    a and b are either:
        - Vectors (1D): returns the scalar overlap.
        - Matrices (2D with states as columns): returns a 1D array with the overlap for each corresponding pair.

    Parameters:
        a      : Bra state(s) as a 1D vector or 2D matrix.
        mat    : 2D transformation matrix.
        b      : Ket state(s) as a 1D vector or 2D matrix.

    Returns:
        A scalar if a and b are vectors, or a 1D array of diagonal overlaps if a and b are matrices.
    """
    backend         = get_backend(backend)
    # Transform the ket(s)
    transformed_b   = act(mat, b, backend=backend)
    a_arr           = backend.asarray(a)
    tb_arr          = backend.asarray(transformed_b)

    if a_arr.ndim == 1 and tb_arr.ndim == 1:
        return backend.vdot(a_arr, tb_arr)
    elif a_arr.ndim == 2 and tb_arr.ndim == 2:
        if a_arr.shape[1] != tb_arr.shape[1]:
            raise ValueError("For diagonal overlap, a and transformed_b must have the same number of columns.")
        # Compute diagonal elements: for each column, sum over the elementwise product.
        return backend.sum(a_arr.conj() * tb_arr, axis=0)
    else:
        raise ValueError("Diagonal overlap is defined only for both vectors or both matrices.")

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
    
    if target_backend == np or not _JAX_AVAILABLE:
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

if _JAX_AVAILABLE:

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

def eigh(matrix, backend="default"):
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
    
    if backend == np or not _JAX_AVAILABLE:
        return _eig_np(matrix)
    return _eig_jax(matrix)    

def eigsh(matrix, method, backend="default", **kwargs):
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
    
    if method == "lanczos":
        k       = kwargs.get("k", 6)
        which   = kwargs.get("which", "SA")
        if backend == np or not _JAX_AVAILABLE:
            return _eig_lanczos_np(matrix, k=k, which=which)
        return _eig_lanczos_jax(matrix, k=k, which=which)
    elif method == "shift-invert":
        k       = kwargs.get("k", 6)
        sigma   = kwargs.get("sigma", 0.0)
        which   = kwargs.get("which", "LM")
        mode    = kwargs.get("mode", "normal")
        if backend == np or not _JAX_AVAILABLE:
            return _eig_shift_inv_np(matrix, k=k, sigma=sigma, which=which, mode=mode)
        return _eig_shift_inv_jax(matrix, k=k, sigma=sigma, which=which, mode=mode)
    if backend == np or not _JAX_AVAILABLE:
        return _eig_np(matrix.todense())
    return _eig_jax(matrix.todense())
    
# ------------------------------------------------------------------