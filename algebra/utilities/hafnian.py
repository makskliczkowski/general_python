'''
file        : QES/Algebra/Linalg/pfaffian.py
author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
date        : 2025-04-29

Algorithms for computing the Hafnian of a matrix.
'''


import numpy as np
import scipy
import warnings
import numba
from enum import Enum, unique

from typing import Optional, Union, Tuple
from ..utils import JAX_AVAILABLE

#! jax
from ..utilities import hafnian_jax as jnp

#############################################################################
#! Hafnians (bosonic Gaussians)
#############################################################################

class Hafnian:
    """
    Lightweight Hafnian utilities (Numba JIT).
    """

    # ------------------------------------------------------------------
    #! a) naive recursion
    # ------------------------------------------------------------------
    
    @classmethod
    @numba.jit(nopython=True, cache=True)
    def _hafnian_recursive(cls, A):
        """
        Recursively computes the hafnian of a symmetric matrix A.
        WARNING: Exponential complexity, practical only for small n (n < 12).
        """
        n = A.shape[0]
        if n == 0:
            # Base case: hafnian of 0x0 matrix is 1
            return 1.0 + 0.0j

        res = 0.0 + 0.0j
        # Fix the first row (row 0), pair it with every column j > 0
        for j in range(1, n):
            # Build the (n-2)x(n-2) submatrix by removing rows/cols 0 and j
            sub = np.empty((n - 2, n - 2), dtype=A.dtype)
            si = 0  # submatrix row index
            for r in range(1, n):
                if r == j:
                    continue
                sj = 0  # submatrix col index
                for c in range(1, n):
                    if c == j:
                        continue
                    sub[si, sj] = A[r, c]
                    sj += 1
                si += 1
            # Recursive call: multiply A[0, j] by hafnian of submatrix
            res += A[0, j] * cls._hafnian_recursive(sub)
        return res

    # ------------------------------------------------------------------
    #! b) Gray-code loop (Clifford–Tonucci)  --  O(2^m m²)
    # ------------------------------------------------------------------
    @classmethod
    @numba.jit(nopython=True, cache=True)
    def _hafnian_gray(cls, A):
        """
        Computes the hafnian of a symmetric matrix A using the Gray code loop
        (Clifford–Tonucci algorithm). Complexity: O(2^m * m^2), where n=2m.

        Reference:
        - Clifford, P., & Tonucci, R. (2020). "A fast recursive algorithm for the hafnian of a general complex matrix".
        https://arxiv.org/abs/2102.08387

        Parameters
        ----------
        A : np.ndarray
            Symmetric matrix of even dimension (n x n), n=2m.

        Returns
        -------
        complex
            The hafnian of matrix A.
        """
        m = A.shape[0] // 2  # Number of pairs (n = 2m)
        if m == 0:
            return 1.0 + 0.0j  # Hafnian of 0x0 matrix is 1

        idx = np.arange(2 * m, dtype=np.int64)
        h = 0.0 + 0.0j
        mask = 0

        # Iterate over all possible pairings using Gray code
        for _ in range(1 << (2 * m - 1)):
            # Find the least unset bit (first available index)
            least = (~mask) & -~mask
            bit = least.bit_length() - 1
            mask ^= least  # Mark this bit as used

            if bit & 1:
                # Odd index: try pairing with all higher indices
                i = bit
                for j in range(i + 1, 2 * m):
                    if (mask >> j) & 1:
                        # If j-th bit is set, pair (i, j) and recurse
                        h += A[i, j] * cls._prod_cached(A, mask ^ (1 << j))
            else:
                # Even index: contribution already counted in previous iterations
                pass

        return h

    @classmethod
    @numba.jit(nopython=True, cache=True, inline='always')
    def _prod_cached(cls, A, mask):
        """
        Recursively computes the sum over all perfect matchings encoded by the bit-mask.

        Parameters
        ----------
        A : np.ndarray
            Symmetric matrix (n x n), assumed float64 dtype.
        mask : int
            Bit-mask indicating which indices are available for pairing.
            Each bit set to 1 means the corresponding index is available.

        Returns
        -------
        complex
            The sum of products of A[i, j] over all possible pairings
            determined by the current mask.
        """
        if mask == 0:
            # Base case: no indices left to pair, return 1 (empty product)
            return 1.0 + 0.0j

        # Find the lowest set bit (first available index)
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        mask ^= lsb  # Mark index i as used

        acc = 0.0 + 0.0j
        # Try pairing i with every other available index j > i
        while mask:
            lsb2 = mask & -mask
            j = lsb2.bit_length() - 1
            mask ^= lsb2  # Mark index j as used
            # Multiply A[i, j] by the result of pairing the rest
            acc += A[i, j] * cls._prod_cached(A, mask)
        return acc

# ----------------------------------------------------------------------
#! Public dispatcher
# ----------------------------------------------------------------------

def hafnian(A: np.ndarray, method: Optional[str] = None) -> complex:
    """
    Computes the hafnian of a symmetric matrix A.

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix (n x n), should be float64 or complex128.
    method : {"gray", "recursive", None}, optional
        Algorithm to use:
            - "gray": Gray code loop (fast for n <= 20)
            - "recursive": Naive recursion (very slow, only for small n)
            - None: Chooses automatically ("gray" if n <= 20, else "recursive")

    Returns
    -------
    complex
        The hafnian of matrix A.

    Notes
    -----
    - For odd n, the hafnian is always zero.
    - For n == 0, returns 1 by convention.
    - No type promotion: input should be float64 or complex128.
    """
    n = A.shape[0]
    if n & 1:
        # Odd dimension: hafnian is zero by definition
        return 0.0
    if n == 0:
        # Hafnian of 0x0 matrix is 1
        return 1.0
    # Choose method automatically if not specified
    if method is None:
        method = "gray" if n <= 20 else "recursive"
    if method == "gray":
        # Use Gray code loop (efficient for small n)
        return Hafnian._hafnian_gray(A)
    if method == "recursive":
        # Use naive recursion (very slow, only for very small n)
        return Hafnian._hafnian_recursive(A)
    raise ValueError("method must be 'gray', 'recursive', or None")