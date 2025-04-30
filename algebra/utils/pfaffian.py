'''
file        : QES/Algebra/Linalg/pfaffian.py
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
date        : 2025-04-29

Algorithms for computing the Pfaffian of a matrix.
'''


import numpy as np
import scipy
import warnings
import numba
from enum import Enum, unique, TypeVar

from typing import Optional, Union, Tuple
from general_python.algebra.utils import JAX_AVAILABLE

#! jax
import general_python.algebra.linalg.pfaffian_jax as jnp

################################################################################################################

_ZERO_TOL       = 1e-15

@unique
class PfaffianAlgorithms(Enum):
    """
    Enum for the available Pfaffian algorithms.
    """
    Recursive   = 0
    ParlettReid = 1     # Placeholder
    Householder = 2     # Placeholder (often related to Hessenberg/Tridiagonal)
    Schur       = 3     # Placeholder
    Hessenberg  = 4     # Placeholder (often related to Householder/Tridiagonal)

################################################################################################################

@numba.njit(cache=True)
def _check_skew_symmetric_numba(A, tol=1e-9):
    """
    Checks if a matrix A is skew-symmetric within a tolerance using Numba.
    A matrix is skew-symmetric if A^T = -A.
    Parameters:
        A (np.ndarray):
            The matrix to check.
        tol (float):
            The tolerance for checking skew-symmetry.
    Returns:
        bool: True if A is skew-symmetric, False otherwise.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    return np.allclose(A, -A.T, atol=tol)

################################################################################################################

class Pfaffian:
    """
    Provides Pfaffian and related calculations using Numba JIT.
    """

    ############################################################################################################
    #! Recursive Pfaffian
    ############################################################################################################
    
    @staticmethod
    @numba.jit("f8(f8[:,:], i8)", nopython=True, cache=True)
    def _pfaffian_recursive(A, N):
        """
        Calculates the Pfaffian using the recursive definition (Numba JIT).
        WARNING: 
            This algorithm has O(N!) complexity and is very slow for N > 10-12.
        """
        if N == 0:
            return 1.0
        if N % 2 != 0:
            # Pfaffian is zero for odd-dimensional matrices
            return 0.0

        # Base case handled by the loop structure for N=2
        if N == 2:
            return A[0, 1]

        pf_sum          = 0.0
        fixed_row_col   = 0

        # Slicing and creating submatrices in Numba requires care.
        # We need indices for the submatrix.
        all_indices     = np.arange(N, dtype=np.int32)
        
        # Iterate through columns j > fixed_row_col (i.e., j from 1 to N-1)
        for j in range(1, N):
            # Calculate the sign (-1)^(j-1) since we index from 0
            # The formula often uses 1-based indexing: sum (-1)^(j+1-1) * A[0,j] * Pf(A_0j)
            # For 0-based: sum (-1)^j * A[0,j] * Pf(A_0j) ? No, (-1)^(j-1) relative to first elt
            sign = -1.0 if (j - 1) % 2 != 0 else 1.0 # (-1)^((j+1)-1-1) = (-1)^(j-1)

            # Create the indices for the minor matrix A_0j
            # Exclude row/col 0 and row/col j
            mask                = np.ones(N, dtype=np.bool_)
            mask[fixed_row_col] = False
            mask[j]             = False
            minor_indices       = all_indices[mask]

            # Create the minor matrix A_0j
            # Numba works better with explicit loops for slicing sometimes
            N_minor = N - 2
            if N_minor > 0 :
                A_minor = np.empty((N_minor, N_minor), dtype=A.dtype)
                for row_idx_minor, row_idx_orig in enumerate(minor_indices):
                    for col_idx_minor, col_idx_orig in enumerate(minor_indices):
                        A_minor[row_idx_minor, col_idx_minor] = A[row_idx_orig, col_idx_orig]
                
                # Recursive call for the minor matrix
                pf_minor        =   PfaffianNumba._pfaffian_recursive_numba(A_minor, N_minor)
                pf_sum          +=  sign * A[fixed_row_col, j] * pf_minor
            elif N_minor == 0:
                # Base case for the subproblem Pf({}) = 1
                pf_minor        =   1.0
                pf_sum          +=  sign * A[fixed_row_col, j] * pf_minor
        return pf_sum

    ############################################################################################################
    #! Hessenberg Pfaffian
    ############################################################################################################
    
    @staticmethod
    def _pfaffian_hessenberg(A, N):
        """
        Calculates the Pfaffian using the Hessenberg form (Numba JIT).
        This algorithm is more efficient than the recursive one.
        
        Parameters:
            A (np.ndarray):
                The skew-symmetric matrix.
            N (int):
                The size of the matrix.
        Note: 
            The algorithm is not jitted as of the usage of scipy.linalg.
            Using the fact that for an arbitrary skew-symmetric matrix,
            the pfaffian Pf(B A B^T ) = det(B)Pf(A)
        """

        if N==0:
            return A.dtype.type(1.0)
        
        if N % 2 != 0:
            # Pfaffian is zero for odd-dimensional matrices
            return A.dtype.type(0.0)
        
        # For skew-symmetric A, H should be tridiagonal skew-symmetric. Take the upper triangular part
        H, Q = scipy.linalg.hessenberg(A, calc_q=True)

        # Check if H is indeed tridiagonal (optional, for verification)
        # tol = 1e-9 * np.max(np.abs(H)) if N > 0 else 0
        # lower_check = np.all(np.abs(np.tril(H, k=-2)) < tol)
        # upper_check = np.all(np.abs(np.triu(H, k=2)) < tol)
        # if not (lower_check and upper_check):
        #     warnings.warn("Hessenberg matrix is not tridiagonal, result might be inaccurate.")

        q_det = np.linalg.det(Q)
        
        # The Pfaffian formula for tridiagonal uses the super-diagonal elements
        # Pf(H) = H[0,1] * H[2,3] * ... * H[N-2, N-1]
        # Product of H.diag(1) elements at indices 0, 2, 4, ...
        super_diag  = np.diag(H, k=1)
        pf_H        = np.prod(super_diag[::2])
        # Pf(A) = Pf(Q H Q.T) = det(Q) * Pf(H)
        return q_det * pf_H
    
    ############################################################################################################
    #! Schur Pfaffian
    ############################################################################################################
    
    @staticmethod
    # This function calls scipy, so it cannot be fully jitted with nopython=True
    def _pfaffian_schur(A, N):
        """
        Pfaffian via Schur decomposition (uses Scipy).
        Parameters:
            A (np.ndarray):
                The skew-symmetric matrix.
            N (int):
                The size of the matrix.
        Returns:
            The Pfaffian of the matrix.
        Note:
            This algorithm is not jitted as of the usage of scipy.linalg.
        """
        
        if N == 0:
            return A.dtype.type(1.0)
        if N % 2 != 0:
            return A.dtype.type(0.0)

        # Scipy's schur returns T (Schur form) and Z (unitary matrix)
        # For skew-symmetric A, T should be block diagonal with 2x2 blocks [[0, b], [-b, 0]]
        # or zero diagonal elements for real matrices.
        # For complex matrices, T is upper triangular with eigenvalues on diagonal (which are purely imaginary or 0).
        if np.iscomplexobj(A):
            T, Z = scipy.linalg.schur(A, output='complex')
        else:
            T, Z = scipy.linalg.schur(A, output='real')

        # get the determinant of Z
        z_det = np.linalg.det(Z)

        # The Pfaffian formula for the Schur form T (skew-symmetric block diagonal)
        # is the product of the super-diagonal elements of the 2x2 blocks.
        # Pf(T) = T[0,1] * T[2,3] * ... * T[N-2, N-1]
        super_diag  = np.diag(T, k=1)
        pf_T        = np.prod(super_diag[::2]) # Take every second element starting from index 0

        # Pf(A) = Pf(Z T Z^H) = det(Z) * Pf(T)
        return z_det * pf_T

    ############################################################################################################
    #! Pfaffian via Parlett-Reid 
    ############################################################################################################
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _pfaffian_parlett_reid(A_in, N):
        """
        Internal Numba JIT:
        Pfaffian via Parlett-Reid algorithm.
        Handles real and complex types. Modifies a copy of the input.
        
        Parameters:
            A_in (np.ndarray):
                The skew-symmetric matrix.
            N (int):
                The size of the matrix.
        Returns:
            The Pfaffian of the matrix.
        """
        if N == 0:
            return A_in.dtype.type(1.0)
        if N % 2 != 0:
            return A_in.dtype.type(0.0)

        A               = A_in.copy()
        pfaffian_val    = A.dtype.type(1.0)

        for k in range(0, N - 1, 2):
            #! Pivoting
            # Find largest element in magnitude in column k below row k + 1
            # A.col(k).subvec(k + 1, N - 1) -> A[k+1:N, k]
            col_slice   = A[k + 1:, k]

            kp_offset   = np.argmax(np.abs(col_slice))
            kp          = (k + 1) + kp_offset # Actual row index in A

            # Check if pivoting is needed
            if kp != k + 1:
                # Swap rows k+1 and kp - equivalent to Arma::swap_rows
                temp_row    = A[k + 1, :].copy()
                A[k + 1, :] = A[kp, :]
                A[kp, :]    = temp_row

                # Swap columns k+1 and kp - equivalent to Arma::swap_cols
                temp_col    = A[:, k + 1].copy()
                A[:, k + 1] = A[:, kp]
                A[:, kp]    = temp_col

                pfaffian_val *= -1.0 # Each swap pair negates the pfaffian

            #! Elimination 
            pivot_val   = A[k + 1, k] # This is -A[k, k+1] due to skew-symmetry

            # Check for zero pivot (within tolerance)
            if np.abs(pivot_val) < _ZERO_TOL:
                return A.dtype.type(0.0)

            # Pfaffian update: Multiply by A[k, k+1]
            pfaffian_val    *= A[k, k + 1]

            # Check if there's a submatrix to update
            if k + 2 < N:
                # Gauss vector tau = A[k, k+2:] / A[k, k+1]
                # Need to handle potential division by zero, but already checked pivot
                # Actually uses A[k+1, k+2:] / A[k+1, k] which is numerically better maybe?

                tau                 = A[k, k + 2:] / A[k, k + 1]
                col_k_plus_1_sub    = A[k + 2:, k + 1]

                # Let's simulate the update: A' = L A L^T. The update part is related to L_sub * A_sub * L_sub^T
                # Maybe the C++ code has a typo or unusual definition?
                # Let's stick to the literal translation: term = outer(tau, col), update = term - term.T
                col_times_row       = np.outer(col_k_plus_1_sub, tau)
                row_times_col       = np.outer(tau, col_k_plus_1_sub)
                # Directly update the submatrix in place without using subMat_view
                for i in range(N - k - 2):
                    for j in range(N - k - 2):
                        A[k + 2 + i, k + 2 + j] += row_times_col[i, j] - col_times_row[i, j]

        return pfaffian_val

    ############################################################################################################
    #! Cayley's formula
    ############################################################################################################
    
    @staticmethod
    @numba.jit("f8(f8, f8[:], f8[:])", nopython=True, cache=True)
    def _cayleys_formula(_pffA, _Ainv_row, _updRow):
        """
        Internal Numba implementation for Cayley's identity.
        Update the Pfaffian of a skew-symmetric matrix after a row and column update.
        Using the Cayley identity, the Pfaffian of a skew-symmetric matrix can be updated
        after a single row and single column update:
        
        A'.row[k] = A.row[k] + updRow
        A'.col[k] = A.col[k] + updRow
        
        The updated Pfaffian is calculated as:
        
        P'(A) = -P(A) * dot(Ainv_row, updRow)
        
        Parameters:
            _pffA (float):
                The Pfaffian of the original matrix A.
            _Ainv_row (np.ndarray):
                The k-th row of the inverse of A (A^{-1}).
            _updRow (np.ndarray):
                The k-th row of the update matrix delta
                (where A' = A + delta). Usually related to
                the change in the k-th row/column of A.
        Returns:
            A new Pfaffian value after the update.
        Raises:
            ValueError: If the input arrays are not 1D or have incompatible shapes.
        """
        
        # Ensure dot product inputs are 1D arrays
        if _Ainv_row.ndim != 1 or _updRow.ndim != 1:
            return np.nan

        if _Ainv_row.shape[0] != _updRow.shape[0]:
            return np.nan

        # Calculate dot product, do it in the logarithmic space
        # to avoid overflow/underflow issues
        
        _log_p          = np.log(_pffA)    
        _log_dot        = np.log(np.dot(_Ainv_row, _updRow))
        return -np.exp(_log_p + _log_dot)
    
        # Alternative direct calculation (not in log space)
        dot_product     = np.dot(_Ainv_row, _updRow)
        return -_pffA * dot_product

    @staticmethod
    @numba.jit("f8[:,:](f8[:,:], i8, f8[:])", nopython=True, cache=True)
    def _scherman_morrison_skew(Ainv, updIdx, updRow):
        """
        Internal Numba implementation for Sherman-Morrison update for skew-symmetric matrices.
        Assumes Ainv is the inverse of a skew-symmetric matrix.
        """
        N = Ainv.shape[0]
        if N == 0:
            return np.empty((0, 0), dtype=Ainv.dtype)
        
        if updRow.shape[0] != N:
            # Cannot raise error easily
            # Return invalid matrix?
            return np.full((N, N), np.nan)

        out = Ainv.copy()

        # Precompute dot products: dots = Ainv @ updRow (as column vector)
        # Numba needs explicit loops for
        dots = np.zeros(N, dtype=Ainv.dtype)
        for i in range(N):
            for j in range(N):
                dots[i] += Ainv[i, j] * updRow[j]

        # Precompute inverse of the critical dot product
        dot_k           = dots[updIdx]
        # Ensure no division by zero
        dotProductInv   = 1.0 / np.clip(dot_k, 1e-10, 1e10)

        # Update the matrix using the formula
        for i in range(N):
            d_i_alpha   = 1.0 if i == updIdx else 0.0
            Ainv_k_i    = Ainv[updIdx, i]

            for j in range(N):
                d_j_alpha   = 1.0 if j == updIdx else 0.0
                Ainv_k_j    = Ainv[updIdx, j]

                # Original SM-like update term
                update_term = dotProductInv * ((d_i_alpha - dots[i]) * Ainv_k_j +
                                               (dots[j] - d_j_alpha) * Ainv_k_i)
                out[i, j]   += update_term

                #! Sign flip for the updated row/column - WHY?
                if d_i_alpha > 0.5 or d_j_alpha > 0.5: # Use > 0.5 for float comparison safety
                    out[i, j] *= -1.0
        return out

##################################################################################################################
