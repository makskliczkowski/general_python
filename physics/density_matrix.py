'''
file    : general_python/physics/density_matrix.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl

This module contains functions for manipulating and analyzing density matrices in quantum mechanics.
'''

import numpy as np
try:
    import numba
except ImportError as e:
    numba   = None
    
from scipy.linalg import svd, eigh
from typing import Callable, Tuple, Union, List, Optional

try:
    from ..algebra.utils import Array
except ImportError as e:
    raise ImportError("Problem with importing utilities. Check if general_python is installed properly") from e

###############################################################################
#! Helper functions
###############################################################################

def _split_dims(state : Array, size_a: int, L: int | None) -> Tuple[int,int]:
    """
    Splits the dimensions of a quantum state vector into two subsystems.
    Args:
        state (Array):
            The quantum state vector to be split.
        size_a (int):
            The number of qubits (or log2 dimension) in subsystem A.
        L (int | None):
            The total number of qubits (or log2 dimension) in the system. 
            If None, it is inferred from the size of `state`.
    Returns:
        Tuple[int, int]:
            A tuple (dimA, dimB) where dimA is the dimension of subsystem A,
            and dimB is the dimension of subsystem B.
    Raises:
        ValueError:
            If the size of `state` is incompatible with the specified `size_a` and `L`.
    """

    if L is None:
        L = int(np.log2(state.size))
        
    dimA    = 1 << size_a
    dimB    = 1 << (L - size_a)
    if state.size != dimA * dimB:
        raise ValueError("state length incompatible with size_a and L")
    return dimA, dimB

###############################################################################

def mask_subsystem(va: Union[int, np.ndarray, List[int]], ns: int, local_dim: int=2, contiguous: bool=False) -> Tuple[Union[int, np.ndarray], Union[np.ndarray, None]]:
    """
    Process the subsystem specification `va` to determine if it is contiguous and to extract the relevant site indices.
    Args:
        va (Union[int, np.ndarray, List[int]]):
            - can be a binary mask (integer) indicating the occupied sites as bits (e.g., 0b101 for sites 0 and 2),
            - can be an array-like of site indices (e.g., [0, 2]),
            - can be an integer specifying the number of contiguous sites starting from 0 (e.g., 3 for sites [0, 1, 2]).
        
        ns (int):
            The total number of sites in the system.
        local_dim (int, optional):
            The local Hilbert space dimension per site (default: 2 for qubits).

    Returns:
        Tuple[Union[int, np.ndarray], Union[np.ndarray, None]]:
            A tuple containing:
                - The number of sites in subsystem A if `va` is an integer, or the array of site indices if `va` is an array-like.
                - If `va` is an integer and contiguous, returns None for the second element. Otherwise, returns an array of site indices in subsystem B.    
    """
    if isinstance(va, (int, np.integer)):
        if contiguous:
            order_a = tuple(range(va))
            order_b = tuple(range(va, ns))
            return (va, ns - va), order_a + order_b
        else:
            sites_a = np.where([(va >> i) & 1 for i in range(ns)])[0]
            sites_b = np.setdiff1d(np.arange(ns), sites_a)
            order_a = tuple(sites_a)
            order_b = tuple(sites_b)
            return (len(sites_a), len(sites_b)), order_a + order_b            
    else:
        sites_a = np.sort(np.asarray(va, dtype=np.int64))
        sites_b = np.setdiff1d(np.arange(ns), sites_a)
        order_a = tuple(sites_a)
        order_b = tuple(sites_b)
        return (len(sites_a), len(sites_b)), order_a + order_b
    
###############################################################################

#! Numpy

def rho_numpy(state : Array, dimA: int, dimB: int) -> Array:
    r"""
    Computes the reduced density matrix \rho from a pure state vector using NumPy.

    Given a state vector representing a bipartite quantum system of dimensions (dimA, dimB),
    this function reshapes the state into a matrix of shape (dimA, dimB) using column-major
    order (Fortran order), and then computes the density matrix :math:`\rho = \psi \psi^\dagger`.

    Backend
    -------
    **NumPy Only**. This function expects and returns NumPy arrays.
    For JAX-compatible versions, see ``physics.density_matrix_jax``.

    Args:
        state (Array):
            The state vector of the composite quantum system.
            Shape: ``(dimA * dimB,)``.
        dimA (int):
            Dimension of subsystem A.
        dimB (int):
            Dimension of subsystem B.

    Returns:
        Array:
            The reduced density matrix of subsystem A.
            Shape: ``(dimA, dimA)``.
    """
    
    # reshape as (dimA, dimB) and call BLAS gemm \rho = \psi  \psi \dag
    if dimA <= dimB:
        psi = state.reshape(dimA, dimB, order="F") # column-major
    else:
        psi = state.reshape(dimB, dimA, order="F") # column-major
    return psi @ psi.conj().T

def rho_mask(state : np.ndarray,
        extract_a   : Callable[[int], int],
        extract_b   : Callable[[int], int],
        size_a      : int,
        size        : int,
        tol         : float = 1e-12):
    """
    Constructs a reshaped wavefunction (psi) from a given quantum state vector by mapping indices using provided extraction functions.

    This function is typically used to prepare a state for partial trace or reduced density matrix calculations by reshaping the state vector into a 2D array according to subsystem partitions.

    Args:
        state (np.ndarray):
            The input quantum state vector (1D complex array).
        extract_a (Callable[[int], int]):
            Function to extract the subsystem-A index from a basis index.
        extract_b (Callable[[int], int]):
            Function to extract the subsystem-B index from a basis index.
        size_a (int):
            Number of qubits (or bits) in subsystem A.
        size (int):
            Total number of qubits (or bits) in the system.
        tol (float, optional):
            Amplitude threshold below which values are ignored (default: 1e-12).

    Returns:
        np.ndarray:
            A 2D array of shape (2**size_a, 2**(size - size_a)) representing the reshaped wavefunction.
    """
    size_b  = size - size_a
    dA      = 1 << size_a
    dB      = 1 << size_b
    psi     = np.zeros((dA, dB), dtype=state.dtype)
    for i in range(state.size):
        amp             = state[i]
        if np.abs(amp) <= tol:        
            continue
        iA              = extract_a(i)
        iB              = extract_b(i)
        psi[iA, iB]    += amp
    return psi

def schmidt_numpy( state : Array,
                    dimA            : int,
                    dimB            : int,
                    eig             : bool,
                    *,
                    return_square   : bool = True           # shall return squared Schmidt coefficients to be consistent with density matrix eigenvalues?
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    r"""
    Schmidt decomposition for a bipartite pure state |\psi > \in H_A \otimes H_B.

    Parameters
    ----------
    state : (dA*dB,)
        State vector of the full system.
    dimA : int
        Dimension of subsystem A.
    dimB : int
        Dimension of subsystem B.
    eig : bool
        If True, use eigenvalue decomposition of the reduced density matrix; 
        if False, use singular value decomposition (SVD).
    return_square : bool, optional
        If True, return squared Schmidt coefficients (eigenvalues of the reduced density matrix).
        
    Note
    -----
    - If `eig=True`, we diagonalize the reduced density matrix of the smaller subsystem.
    - If `eig=False`, we take SVD of the reshaped state and square singular values.
    - The dimensions are reshaped with order="F", treated as consequtive columns. This corresponds to 
    subsystems A being consecutive qubits 0,1,...,size_a-1 in the full system of L=log2(dimA*dimB) qubits.

    Returns
    -------
    vals : (r,)
        Schmidt weights (squared coefficients = eigenvalues of the reduced density matrix),
        sorted in descending order.
    vecs : (d_small, r)
        Schmidt vectors of the *smaller* subsystem (A if dimA ≤ dimB, else B),
        columns ordered consistently with `vals`.
    rho  : (d_small, d_small) | None
        Reduced density matrix of the smaller subsystem (if `eig=True`).
    Notes
    -----
    - The input state is reshaped with order="F".
    - If `eig=True`, we diagonalize the reduced density matrix of the smaller subsystem.
    - If `eig=False`, we take SVD of the reshaped state and square singular values.
    """
    d_small         = dimA if dimA <= dimB else dimB
    d_large         = dimB if dimA <= dimB else dimA

    psi             = state.reshape(d_small, d_large, order="F")
    rho             = None
    if eig:
        # Reduced density matrix of the smaller subsystem
        rho         = psi @ psi.conj().T
        vals, vecs  = np.linalg.eigh(rho)                   # ascending eigenvalues
        vals        = np.clip(np.real(vals), 0.0, None)     # numerical safety
        vals        = vals[::-1]                            # descending
        vecs        = vecs[:, ::-1]
    else:
        # SVD: psi = U S V\dag ; U are Schmidt vectors of the smaller subsystem
        vecs, s, _  = np.linalg.svd(psi, full_matrices=False)
        vals        = np.real(s * s) if return_square else np.real(s)
    return vals, vecs, rho

# -----------------------------------------------------------------------------

def rho_single_site(psi, site, ns):
    """
    Returns the 2x2 reduced density matrix for site `site` of state psi.
    Args:
        psi (Array):
            The quantum state vector of the full system.
        site (int):
            The site index for which the reduced density matrix is computed.
        ns (int):
            The total number of sites (qubits) in the system.
    Returns:
        Array:
            The 2x2 reduced density matrix for the specified site.
    Example:
        >>> psi = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)  # 3-qubit GHZ state
        >>> site = 1
        >>> ns = 3
        >>> rho = rho_single_site(psi, site, ns)
        >>> print(rho)
        [[0.5 0. ]
         [0.  0.5]]
    """
    
    # Map a basis index k to its bit at `site`
    extract_a = lambda k: (k >> site) & 1 # Example k=1, ns=3, site=1 -> (0b001 >> 1) & 1 = 0b000 & 1 = 0

    # Map k to all the other bits except `site`
    def extract_b(k):
        low  = k & ((1 << site) - 1)
        high = k >> (site + 1)
        return (high << site) | low

    psi_mat = rho_mask(
        state      = psi,
        extract_a  = extract_a,
        extract_b  = extract_b,
        size_a     = 1,
        size       = ns,
        tol        = 1e-12,
    )

    rho = psi_mat @ psi_mat.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    return rho

def rho_two_sites(psi, site_i, site_j, ns):
    """
    Returns the 4x4 reduced density matrix for sites (i,j).
    """

    # Sort so i < j
    if site_j < site_i:
        site_i, site_j = site_j, site_i

    # Extract two bits
    def extract_a(k):
        b0 = (k >> site_i) & 1
        b1 = (k >> site_j) & 1
        return (b1 << 1) | b0 # little-endian (00,01,10,11)

    # Extract remaining bits
    def extract_b(k):
        # remove bit j:
        low_j  = k & ((1 << site_j) - 1)
        high_j = k >> (site_j + 1)
        k1     = (high_j << site_j) | low_j

        # remove bit i:
        low_i  = k1 & ((1 << site_i) - 1)
        high_i = k1 >> (site_i + 1)
        k2     = (high_i << site_i) | low_i
        return k2

    psi_mat = rho_mask(
        state      = psi,
        extract_a  = extract_a,
        extract_b  = extract_b,
        size_a     = 2,
        size       = ns,
        tol        = 1e-12,
    )
    rho     = psi_mat @ psi_mat.conj().T
    rho     = 0.5 * (rho + rho.conj().T)
    return rho

def rho_spectrum(rho: np.ndarray, eps: float = 1e-13):
    '''
    Diagonalize the density matrix
    '''
    w = np.linalg.eigvalsh(rho)
    w = np.clip(w.real, 0.0, 1.0)
    w = w[w > eps]
    return w
    
# -----------------------------------------------------------------------------

if numba:
    
    # @numba.njit
    def rho_numba(state: Array, dimA: int, dimB: int) -> Array:
        psi = state.reshape(dimA, dimB, order="F")
        return psi @ psi.conj().T

    # @numba.njit(cache=True)
    def rho_numba_mask(state    : np.ndarray,
                    order       : tuple,
                    size_a      : int) -> Array:
        """
        Reshapes and reorders a quantum state vector for subsystem partitioning.
        This function takes a 1D quantum state vector and reshapes it into a multi-dimensional array,
        then transposes and flattens it to produce a 2D array suitable for partial trace or reduced
        density matrix calculations. The partitioning is determined by the `order` and `size_a` arguments.
        Args:
            state (np.ndarray):
                The input quantum state vector as a 1D complex-valued array of length 2**N, where N is the number of qubits.
            order (tuple):
                A tuple specifying the new order of qubits after partitioning. The first `size_a` elements correspond to subsystem A.
            size_a (int):
                The number of qubits in subsystem A.
        Returns:
            Array:
                A 2D array of shape (2**size_a, 2**(N - size_a)), where N = len(order), representing the reshaped and reordered wavefunction.
        """
        psi_nd      = state.reshape((2, ) * len(order), order='F') # no copy, reshape to (2, 2, ..., 2)
        dA          = 1 << size_a
        return psi_nd.transpose(order).reshape(dA, -1)
        
                    # extract_a   : Callable[[int], int],
                    # extract_b   : Callable[[int], int],
                    # size_a      : int,
                    # size        : int,
                    # tol         : float = 1e-14):
        # """
        # Constructs a reshaped wavefunction (psi) from a given quantum state vector by mapping indices using provided extraction functions.

        # This function is typically used to prepare a state for partial trace or reduced density matrix calculations by reshaping the state vector into a 2D array according to subsystem partitions.

        # Args:
        #     state (np.ndarray):
        #         The input quantum state vector (1D complex array).
        #     extract_a (Callable[[int], int]):
        #         Function to extract the subsystem-A index from a basis index.
        #     extract_b (Callable[[int], int]):
        #         Function to extract the subsystem-B index from a basis index.
        #     size_a (int):
        #         Number of qubits (or bits) in subsystem A.
        #     size (int):
        #         Total number of qubits (or bits) in the system.
        #     tol (float, optional):
        #         Amplitude threshold below which values are ignored (default: 1e-12).

        # Returns:
        #     np.ndarray:
        #         A 2D array of shape (2**size_a, 2**(size - size_a)) representing the reshaped wavefunction.
        # """
        
        # psi_nd  = state.reshape((2,) * size) # no copy, reshape to (2, 2, ..., 2)
        
        
        
        # size_b  = size - size_a
        # dA      = 1 << size_a
        # dB      = 1 << size_b
        # psi     = np.zeros((dA, dB), dtype=state.dtype)
        # for i in numba.prange(state.size):
        #     i               = np.int64(i)
        #     amp             = state[i]
        #     if np.abs(amp) <= tol:        
        #         continue
        #     iA              = extract_a(i)
        #     iB              = extract_b(i)
        #     psi[iA, iB]    += amp
        # return psi

    # @numba.njit(cache=True, fastmath=True)
    def schmidt_numba(  psi     : Array,
                        dimA    : int,
                        dimB    : int,
                        eig     : bool) -> Tuple[np.ndarray, np.ndarray]:
        if eig:
            if dimA <= dimB:
                rho         = psi @ psi.conj().T
                vals, vecs  = np.linalg.eigh(rho)
            else:
                sigma       = psi.conj().T @ psi
                vals, V     = np.linalg.eigh(sigma)
                vecs        = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
        else:
            vecs, s, _      = np.linalg.svd(psi, full_matrices=False)
            vals            = s * s
        return vals[::-1], vecs[:, ::-1]

    @numba.njit(cache=True, fastmath=True)
    def schmidt_numba_mask(psi      : np.ndarray,
                        order       : tuple,
                        size_a      : int,
                        eig         : bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Schmidt decomposition of a bipartite quantum state using Numba.
        Parameters
        ----------
        state : np.ndarray
            The input state vector representing the bipartite quantum system.
        order : tuple
            A tuple specifying the new order of qubits after partitioning. The first `size_a` elements correspond to subsystem A.
        size_a : int
            Dimension of subsystem A.
        eig : bool
            If True, use eigenvalue decomposition of the reduced density matrix; 
            if False, use singular value decomposition (SVD).
        Returns
        -------
        vals : np.ndarray
            The Schmidt coefficients (squared singular values or eigenvalues), sorted in descending order.
        vecs : np.ndarray
            The corresponding Schmidt vectors, columns ordered according to descending Schmidt coefficients.
        """
        
        if eig:
            if size_a <= len(order) - size_a:
                rho         = psi @ psi.conj().T
                vals, vecs  = np.linalg.eigh(rho)
            else:
                sigma       = psi.conj().T @ psi
                vals, V     = np.linalg.eigh(sigma)
                vecs        = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
        else:
            vecs, s, _      = np.linalg.svd(psi, full_matrices=False)
            vals            = np.square(s)
        return vals[::-1], vecs[:, ::-1]
    
    # @numba.njit
    # def schmidt_numba_mask(state    : np.ndarray,
    #                     extract_a   : Callable[[int], int],
    #                     extract_b   : Callable[[int], int],
    #                     size_a      : int,
    #                     size        : int,
    #                     eig         : bool) -> Tuple[np.ndarray, np.ndarray]:
    #     psi = rho_numba_mask(state, extract_a, extract_b, size_a, size)
    #     if eig:
    #         if size_a <= size - size_a:
    #             rho         = psi @ psi.conj().T
    #             vals, vecs  = np.linalg.eigh(rho)
    #         else:
    #             sigma       = psi.conj().T @ psi
    #             vals, V     = np.linalg.eigh(sigma)
    #             vecs        = psi @ V / np.sqrt(np.maximum(vals, 1e-30))
    #     else:
    #         vecs, s, _      = np.linalg.svd(psi, full_matrices=True)
    #         vals            = s * s
    #     return vals[::-1], vecs[:, ::-1]
    
else:
    rho_numba = rho_numba_mask = schmidt_numba = None

# -----------------------------------------------------------------------------

#! JAX if available
try:
    from ..physics import density_matrix_jax as jnp
except ImportError:
    jnp = np
    
# ###############################################################################
#! End of file
###############################################################################