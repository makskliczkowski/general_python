r'''
This module contains functions for manipulating and analyzing density matrices in quantum mechanics.
It provides optimized implementations using NumPy and Numba for computing reduced density matrices,
Schmidt decompositions, and entanglement spectra.

QES Convention:
- State vector index i = s0 + d*s1 + d^2*s2 + ... (Little-endian / Fortran order)
- Subsystem A site order: [a0, a1, a2, ...] -> Row index I = sa0 + d*sa1 + ...
- Subsystem B site order: [b0, b1, b2, ...] -> Col index J = sb0 + d*sb1 + ...
where d is the local_dim.

Fermionic Systems:
For fermionic systems mapped via Jordan-Wigner transformation, non-local string operators
create additional correlations between non-contiguous subsystem sites. The `fermionic=True`
flag applies sign corrections that account for the fermionic exchange statistics when
reordering sites. This ensures correct reduced density matrices for arbitrary subsystem
geometries.

--------------------------------
Author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
version     : 2.1
copyright   : (c) 2026 by Maksymilian Kliczkowski. All rights reserved.
--------------------------------
'''

import  numpy as np
import  scipy.linalg as la
from    typing import Tuple, Union, List, Optional, Any

# -----------------------------------------------------------------------------
#! Fermionic Sign Correction
# -----------------------------------------------------------------------------

def _fermionic_parity_signs_fast(ns: int, order: Tuple[int, ...]) -> np.ndarray:
    """
    Optimized version of fermionic parity sign computation using bit manipulation.
    
    Computes signs for all 2^ns basis states in O(2^ns * k^2) where k is the
    number of inverted pairs in the permutation.
    
    Parameters
    ----------
    ns : int
        Total number of sites.
    order : Tuple[int, ...]
        Target permutation order.
    
    Returns
    -------
    np.ndarray
        Sign array of shape (2^ns,) with values +1.0 or -1.0.
    """
    dim = 1 << ns
    
    # Compute inverse permutation
    inv_order = np.empty(ns, dtype=np.int64)
    for new_pos, old_site in enumerate(order):
        inv_order[old_site] = new_pos
    
    # Find all inverted pairs (i < j but inv_order[i] > inv_order[j])
    inverted_pairs = []
    for i in range(ns):
        for j in range(i + 1, ns):
            if inv_order[i] > inv_order[j]:
                inverted_pairs.append((i, j))
    
    if not inverted_pairs:
        # No inversions, all signs are +1
        return np.ones(dim, dtype=np.float64)
    
    # Compute total parity for each basis state
    # parity = sum over inverted pairs of (both_occupied)
    states = np.arange(dim, dtype=np.uint64)
    parity = np.zeros(dim, dtype=np.int64)
    
    for i, j in inverted_pairs:
        mask_i = np.uint64(1 << i)
        mask_j = np.uint64(1 << j)
        both_occ = ((states & mask_i) != 0) & ((states & mask_j) != 0)
        parity += both_occ.astype(np.int64)
    
    # Sign is (-1)^parity
    return np.where(parity & 1, -1.0, 1.0)


# -----------------------------------------------------------------------------
#! Helper functions
# -----------------------------------------------------------------------------

def mask_subsystem(
    va          : Union[int, np.ndarray, List[int]], 
    ns          : int, 
    local_dim   : int = 2, 
    contiguous  : bool = False
) -> Tuple[Tuple[int, int], Tuple[int, ...]]:
    r"""
    Process the subsystem specification to extract site indices and the permutation order.
    The order tuple specifies how to permute the state vector to bring subsystem A sites to the front.
    
    Parameters
    ----------
    va : Union[int, np.ndarray, List[int]]
        Subsystem specification. Can be:
        - An integer (if contiguous=True) specifying the number of contiguous sites in A starting from site 0.
        - A bitmask integer where bits set to 1 indicate sites in A (if contiguous=False).
        - A list or array of site indices in A.
    ns : int
        Total number of sites in the system.
    local_dim : int
        Local Hilbert space dimension (default is 2 for qubits).
    contiguous : bool
        If True, treat va as the number of contiguous sites in A starting from site 0. If False, treat va as a bitmask or list of site indices.
    """
    if isinstance(va, (int, np.integer)):
        if contiguous:
            sites_a = np.arange(va, dtype=np.int64)
        else:
            # Treat as bitmask - vectorized extraction
            bits    = np.arange(ns, dtype=np.uint64)
            mask    = ((va >> bits) & 1) != 0
            sites_a = np.where(mask)[0]
    else:
        sites_a = np.sort(np.asarray(va, dtype=np.int64))
    
    # local dimension d = 2 for qubits, so size of subsystem A is d^|A| and B is d^(ns-|A|)
    if local_dim != 2:
        raise NotImplementedError("Currently only local_dim=2 is supported in mask_subsystem.")
    
    sites_b = np.setdiff1d(np.arange(ns), sites_a)
    order   = tuple(sites_a) + tuple(sites_b)
    return (len(sites_a), len(sites_b)), order

# -----------------------------------------------------------------------------
#! Core RDM Functions
# -----------------------------------------------------------------------------

def psi_numpy(
    state       : np.ndarray, 
    order       : Tuple[int, ...], 
    size_a      : int, 
    ns          : int, 
    local_dim   : int = 2,
    fermionic   : bool = False
) -> np.ndarray:
    """
    Reshape and reorder a quantum state vector into a matrix Psi_{A,B} using NumPy.
    This representation is used to compute the reduced density matrix rho_A = Psi @ Psi^dagger.
    
    Parameters
    ----------
    state : np.ndarray
        The input state vector of shape (local_dim**ns,).
    order : Tuple[int, ...]
        The permutation order of sites to bring subsystem A sites to the front.
    size_a : int
        The number of sites in subsystem A.
    ns : int
        Total number of sites in the system.
    local_dim : int
        Local Hilbert space dimension (default is 2 for qubits).
    fermionic : bool
        If True, apply fermionic sign corrections for site permutation.
        This accounts for the anticommutation of fermionic operators when
        reordering sites, essential for correct RDMs of non-contiguous
        subsystems in fermionic systems mapped via Jordan-Wigner.
    
    Returns
    -------
    np.ndarray
        Reshaped state matrix Psi of shape (dA, dB) where dA = local_dim^size_a
        and dB = local_dim^(ns - size_a).
    
    Notes
    -----
    For fermionic systems (fermionic=True):
    
    When we permute the site ordering, fermionic operators anticommute:
        c_i c_j = -c_j c_i
    
    For a basis state |n_0, n_1, ..., n_{ns-1}> represented as:
        c_0^{n_0} c_1^{n_1} ... c_{ns-1}^{n_{ns-1}} |vacuum>
    
    Reordering the sites requires swapping creation operators, each swap
    of two occupied sites contributes a factor of -1.
    
    The total sign is (-1)^{number of inversions in occupied sites}.
    """
    dA          = local_dim**size_a
    dB          = local_dim**(ns - size_a)

    if fermionic and local_dim != 2:
        raise NotImplementedError("Fermionic mode only supports local_dim=2 (qubits/fermions).")

    # Apply fermionic sign correction if needed
    if fermionic and order is not None and order != tuple(range(ns)):
        signs = _fermionic_parity_signs_fast(ns, order)
        state = state * signs # Element-wise multiplication
    
    if order is None or order == tuple(range(ns)):
        # Fast path: no permutation needed
        psi         = state.reshape(dA, dB, order="F")
    else:
        # Check if order is contiguous (a0, a1, ..., aK, b0, b1, ...) -> can avoid full ND reshape
        is_contiguous_a = order[:size_a] == tuple(range(size_a))
        is_contiguous_b = order[size_a:] == tuple(range(size_a, ns))
        
        if is_contiguous_a and is_contiguous_b:
            # Order is trivial after all - just reshape
            psi         = state.reshape(dA, dB, order="F")
        else:
            # General permutation: reshape ND, transpose, reshape back
            psi_nd      = state.reshape((local_dim,) * ns, order="F")
            psi_perm    = psi_nd.transpose(order)
            psi         = psi_perm.reshape(dA, dB, order="F")
    
    return psi

def rho_numpy(
    state       : np.ndarray, 
    size_a      : int, 
    ns          : int, 
    local_dim   : int = 2,
    order       : Optional[Tuple[int, ...]] = None,
    fermionic   : bool = False
) -> np.ndarray:
    """
    Compute reduced density matrix using NumPy with Fortran-order convention.
    Works for any local_dim (fermionic mode requires local_dim=2).
    
    Parameters
    ----------
    state : np.ndarray
        The input state vector of shape (local_dim**ns,).
    size_a : int
        The number of sites in subsystem A.
    ns : int
        Total number of sites in the system.
    local_dim : int
        Local Hilbert space dimension (default is 2 for qubits).
    order : Optional[Tuple[int, ...]]
        The permutation order of sites to bring subsystem A sites to the front.
        If None, assumes natural order.
    fermionic : bool
        If True, apply fermionic sign corrections. See psi_numpy for details.
    
    Returns
    -------
    np.ndarray
        Reduced density matrix rho_A of shape (dA, dA).
    """
    psi = psi_numpy(state, order, size_a, ns, local_dim, fermionic=fermionic)
    return psi @ psi.conj().T

# -----------------------------------------------------------------------------
#! High-level API
# -----------------------------------------------------------------------------

def rho(
    state       : np.ndarray, 
    va          : Union[int, List[int], np.ndarray], 
    ns          : Optional[int] = None, 
    local_dim   : int           = 2, 
    contiguous  : bool          = False,
    fermionic   : bool          = False,
    *,
    la          : Optional[int] = None,
    order       : Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """
    Compute the reduced density matrix rho_A of a subsystem A.
    
    Parameters
    ----------
    state : np.ndarray
        The input state vector of shape (local_dim**ns,).
    va : Union[int, List[int], np.ndarray]
        Subsystem specification. Can be:
        - An integer: if contiguous=True, number of sites in A starting from site 0.
                     if contiguous=False, bitmask where bit i=1 means site i is in A.
        - A list/array of site indices in subsystem A (any geometry).
    ns : Optional[int]
        Total number of sites. If None, inferred from state size.
    local_dim : int
        Local Hilbert space dimension (default 2 for qubits/fermions).
    contiguous : bool
        If True, treat integer va as number of contiguous sites from 0.
    fermionic : bool
        If True, apply fermionic sign corrections for non-contiguous subsystems.
        Essential for correct entanglement entropy of Jordan-Wigner mapped fermions.
        
        For fermionic systems, permuting sites requires accounting for the
        anticommutation of creation operators. Each pair of occupied sites
        that gets inverted in the permutation contributes a factor of -1.
        
        Use fermionic=True when:
        - Computing RDM of non-contiguous subsystem in a fermionic system
        - Working with Slater determinants or their superpositions
        - Comparing with correlation matrix entropy results
        
        For contiguous subsystems (sites 0,1,...,k-1), the fermionic flag
        has no effect since no site permutation is needed.
    la : Optional[int]
        Deprecated alias for specifying contiguous subsystem size.
    order : Optional[Tuple[int, ...]]
        Explicit site permutation order. If provided, overrides va.
    
    Returns
    -------
    np.ndarray
        Reduced density matrix rho_A of shape (dA, dA) where dA = local_dim^|A|.
    
    Examples
    --------
    >>> # Contiguous subsystem (first 3 sites)
    >>> rho_A = rho(psi, va=3, ns=8, contiguous=True)
    
    >>> # Non-contiguous subsystem [0, 2, 4] for fermions
    >>> rho_A = rho(psi, va=[0, 2, 4], ns=8, fermionic=True)
    
    >>> # Bitmask specification: sites 0 and 2 (binary 101 = 5)
    >>> rho_A = rho(psi, va=5, ns=4)
    """
    if ns is None:
        ns = int(np.round(np.log(state.size) / np.log(local_dim)))
    
    if la is not None and contiguous:
        size_a  = la

    if order is not None:
        return rho_numpy(state, size_a, ns, local_dim, order, fermionic=fermionic)
    
    (size_a, size_b), order = mask_subsystem(va, ns, local_dim, contiguous)
    return rho_numpy(state, size_a, ns, local_dim, order, fermionic=fermionic)

def schmidt(
    state       : np.ndarray, 
    va          : Union[int, List[int], np.ndarray], 
    ns          : Optional[int] = None, 
    local_dim   : int = 2,
    contiguous  : bool = False,
    fermionic   : bool = False,
    eig         : bool = False,
    square      : bool = True,
    *,
    sub_size    : Optional[int] = None, 
    order       : Optional[Tuple[int, ...]] = None,
    return_vecs : bool = True
) -> Tuple[np.ndarray, Any]:
    """
    Compute the Schmidt decomposition of a state vector.
    
    For a bipartition of the system into subsystems A and B, the Schmidt
    decomposition expresses the state as:
        |psi> = sum_k lambda_k |phi_k>_A |chi_k>_B
    
    Parameters
    ----------
    state : np.ndarray
        The input state vector of shape (local_dim**ns,).
    va : Union[int, List[int], np.ndarray]
        Subsystem A specification (see rho() for details).
    ns : Optional[int]
        Total number of sites. If None, inferred from state size.
    local_dim : int
        Local Hilbert space dimension (default 2).
    contiguous : bool
        If True, treat integer va as number of contiguous sites.
    fermionic : bool
        If True, apply fermionic sign corrections for site permutation.
        See rho() for detailed explanation.
    eig : bool
        If True, use RDM eigendecomposition instead of SVD.
        SVD (default) is generally faster and more numerically stable.
    square : bool
        If True (default), return squared singular values (= RDM eigenvalues).
        If False, return singular values directly.
    sub_size : Optional[int]
        Deprecated alias for contiguous subsystem size.
    order : Optional[Tuple[int, ...]]
        Explicit site permutation order.
    return_vecs : bool
        If True, return Schmidt vectors along with values.
    
    Returns
    -------
    If return_vecs=True:
        Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]
        (schmidt_values, (U, Vh), psi_matrix)
        
        For SVD path: U[i,k] are left singular vectors, Vh[k,j] are right.
        For eig path: (eigenvectors, None), rho_A
        
    If return_vecs=False:
        np.ndarray
        Schmidt values only (squared singular values if square=True).
    
    Examples
    --------
    >>> # Get Schmidt spectrum for fermionic non-contiguous subsystem
    >>> s_sq = schmidt(psi, va=[0, 2, 4], ns=8, fermionic=True, return_vecs=False)
    >>> entropy = -np.sum(s_sq * np.log(s_sq + 1e-15))
    """
    if ns is None:
        ns = int(np.round(np.log(state.size) / np.log(local_dim)))
    
    if sub_size is not None and contiguous:
        size_a  = sub_size
        order   = None
    
    # If order is provided, it overrides contiguous and sub_size
    if order is not None:
        if 'size_a' not in locals():
            size_a = order.index(max(order) + 1) # Find where subsystem A ends in the order
        size_b = ns - size_a
    else:
        (size_a, size_b), order = mask_subsystem(va, ns, local_dim, contiguous)

    if not eig:
        # SVD path: fast, avoids RDM computation
        psi_mat = psi_numpy(state, order, size_a, ns, local_dim, fermionic=fermionic)
        if return_vecs:
            u, s, vh    = la.svd(psi_mat, full_matrices=False)
            return s**2 if square else s, (u, vh), psi_mat
        else:
            # For values only, use more efficient SVD call (no full decomposition)
            s           = la.svdvals(psi_mat)
            return s**2 if square else s
    else:
        rho_A = rho(state, va, ns, local_dim, contiguous, fermionic=fermionic)
        if return_vecs:
            vals, vecs  = np.linalg.eigh(rho_A)
            vals        = np.clip(vals, 0.0, 1.0)
            idx         = np.argsort(vals)[::-1]
            return vals[idx], vecs[:, idx], rho_A
        else:
            # Values-only path: NumPy's Hermitian eigensolver is markedly faster than SciPy's wrapper
            w           = np.linalg.eigvalsh(rho_A)
            w           = np.clip(w, 0.0, 1.0)
            return np.sort(w[w > 1e-15])[::-1]

def rho_spectrum(rho_mat: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Compute the eigenvalue spectrum of a density matrix.
    
    Parameters
    ----------
    rho_mat : np.ndarray
        Density matrix (Hermitian, positive semi-definite).
    eps : float
        Threshold for filtering small eigenvalues.
    
    Returns
    -------
    np.ndarray
        Sorted eigenvalues (descending) above threshold eps.
    """
    w = np.linalg.eigvalsh(rho_mat)
    w = np.clip(w, 0.0, 1.0)
    return np.sort(w[w > eps])[::-1]

# -----------------------------------------------------------------------------
#! Specialized site functions
# -----------------------------------------------------------------------------

def rho_single_site(
    state       : np.ndarray, 
    site        : int, 
    ns          : int, 
    local_dim   : int = 2,
    fermionic   : bool = False
) -> np.ndarray:
    """
    Compute the single-site reduced density matrix.
    
    Parameters
    ----------
    state : np.ndarray
        Many-body state vector.
    site : int
        Site index to trace out all others except this site.
    ns : int
        Total number of sites.
    local_dim : int
        Local Hilbert space dimension (default 2).
    fermionic : bool
        If True, apply fermionic sign corrections.
    
    Returns
    -------
    np.ndarray
        Single-site RDM of shape (local_dim, local_dim).
    """
    return rho(state, va=[site], ns=ns, local_dim=local_dim, fermionic=fermionic)

def rho_two_sites(
    state       : np.ndarray, 
    site_i      : int, 
    site_j      : int, 
    ns          : int, 
    local_dim   : int = 2,
    fermionic   : bool = False
) -> np.ndarray:
    """
    Compute the two-site reduced density matrix.
    
    Parameters
    ----------
    state : np.ndarray
        Many-body state vector.
    site_i, site_j : int
        Site indices for the two-site subsystem.
    ns : int
        Total number of sites.
    local_dim : int
        Local Hilbert space dimension (default 2).
    fermionic : bool
        If True, apply fermionic sign corrections.
        Important when site_i and site_j are non-adjacent.
    
    Returns
    -------
    np.ndarray
        Two-site RDM of shape (local_dim^2, local_dim^2).
    """
    return rho(state, va=[site_i, site_j], ns=ns, local_dim=local_dim, fermionic=fermionic)


def fermionic_entanglement_entropy(
    state       : np.ndarray,
    subsystem   : Union[int, List[int], np.ndarray],
    ns          : Optional[int] = None,
    local_dim   : int = 2
) -> float:
    """
    Compute fermionic entanglement entropy using sign-corrected RDM.
    
    This is the correct way to compute entanglement entropy for fermionic
    systems (Slater determinants and their superpositions) for arbitrary
    subsystem geometries.
    
    Parameters
    ----------
    state : np.ndarray
        Many-body state vector of fermionic system.
    subsystem : Union[int, List[int], np.ndarray]
        If int: contiguous subsystem of first `subsystem` sites.
        If array: indices of sites in subsystem A (any geometry).
    ns : Optional[int]
        Total number of sites. If None, inferred from state size.
    local_dim : int
        Local Hilbert space dimension (default 2).
    
    Returns
    -------
    float
        Von Neumann entanglement entropy S = -Tr(rho_A log rho_A).
    
    Notes
    -----
    For Gaussian states (single Slater determinant), this should match
    the correlation matrix method. For superpositions of Slater determinants,
    the correlation matrix method is not applicable, but this method works.
    
    Examples
    --------
    >>> # Entropy of non-contiguous subsystem [0, 2, 4]
    >>> S = fermionic_entanglement_entropy(psi, [0, 2, 4], ns=8)
    """
    if isinstance(subsystem, int):
        # Contiguous: no sign correction needed
        rho_A = rho(state, va=subsystem, ns=ns, local_dim=local_dim, contiguous=True)
    else:
        # Non-contiguous: use fermionic sign correction
        rho_A = rho(state, va=subsystem, ns=ns, local_dim=local_dim, fermionic=True)
    
    # Compute von Neumann entropy
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = np.clip(eigenvalues, 1e-15, 1.0)
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


# -----------------------------------------------------------------------------
#! End of file
# -----------------------------------------------------------------------------
