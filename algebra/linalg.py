from abc import ABC, abstractmethod
from .__utils__ import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend as __backend

# -----------------------------------------------------------------

# MATRIX OPERATIONS

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
    backend         =   __backend(backend)
    
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
    backend = __backend(backend)
    
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
    backend = __backend(backend)
    
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
    backend = __backend(backend)
    return backend.outer(A, B)

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
    backend = __backend(backend)
    return backend.kron(A, B)

# -----------------------------------------------------------------

@maybe_jit
def ket_bra(vec: 'array-like', backend="default"):
    """Computes the ket-bra (outer product) of a vector."""
    backend = __backend(backend)
    return outer(vec, vec, backend=backend)

@maybe_jit
def bra_ket(vec: 'array-like', backend="default"):
    """Computes the bra-ket (outer product) of a vector."""
    backend = __backend(backend)
    return backend.dot(vec.T, vec)

@maybe_jit
def inner(vec1: 'array-like', vec2: 'array-like', backend="default"):
    """Computes the inner product of two vectors."""
    backend = __backend(backend)
    return backend.dot(vec1, vec2)

# -----------------------------------------------------------------