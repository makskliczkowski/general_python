r'''
This module contains functions for manipulating and analyzing density matrices in quantum mechanics.
It provides optimized implementations using NumPy and Numba for computing reduced density matrices,
Schmidt decompositions, and entanglement spectra.

QES Convention:
- State vector index i = s0 + d*s1 + d^2*s2 + ... (Little-endian / Fortran order)
- Subsystem A site order: [a0, a1, a2, ...] -> Row index I = sa0 + d*sa1 + ...
- Subsystem B site order: [b0, b1, b2, ...] -> Col index J = sb0 + d*sb1 + ...
where d is the local_dim.

--------------------------------
Author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
version     : 2.0
copyright   : (c) 2026 by Maksymilian Kliczkowski. All rights reserved.
--------------------------------
'''

import  numpy as np
import  scipy.linalg as la
from    typing import Tuple, Union, List, Optional, Any

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
    local_dim   : int = 2
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
    """
    dA          = local_dim**size_a
    dB          = local_dim**(ns - size_a)

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
    order       : Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """
    Compute reduced density matrix using NumPy with Fortran-order convention.
    Works for any local_dim.
    
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
        The permutation order of sites to bring subsystem A sites to the front. If None, assumes natural order.
    """
    psi = psi_numpy(state, order, size_a, ns, local_dim)
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
) -> np.ndarray:
    """
    Compute the reduced density matrix rho_A of a subsystem A.
    Supports arbitrary local_dim.
    """
    if ns is None:
        ns = int(np.round(np.log(state.size) / np.log(local_dim)))
    
    (size_a, size_b), order = mask_subsystem(va, ns, local_dim, contiguous)
    return rho_numpy(state, size_a, ns, local_dim, order)

def schmidt(
    state       : np.ndarray, 
    va          : Union[int, List[int], np.ndarray], 
    ns          : Optional[int] = None, 
    local_dim   : int = 2,
    contiguous  : bool = False,
    eig         : bool = False,
    square      : bool = True,
    *,
    return_vecs : bool = True
) -> Tuple[np.ndarray, Any]:
    """
    Compute the Schmidt decomposition of a state vector.
    SVD path (default, eig=False) is faster for values-only; best overall for most cases.
    RDM+eig path is useful only if you already have an RDM and want eigenvectors.
    """
    if ns is None:
        ns = int(np.round(np.log(state.size) / np.log(local_dim)))
            
    (size_a, size_b), order = mask_subsystem(va, ns, local_dim, contiguous)

    if not eig:
        # SVD path: fast, avoids RDM computation
        psi_mat = psi_numpy(state, order, size_a, ns, local_dim)
        if return_vecs:
            u, s, vh    = la.svd(psi_mat, full_matrices=False)
            return s**2 if square else s, (u, vh), psi_mat
        else:
            # For values only, use more efficient SVD call (no full decomposition)
            s           = la.svdvals(psi_mat)
            return s**2 if square else s
    else:
        # RDM+eigenvalue path: slower but useful if RDM is already cached
        # Note: For typical use, SVD path is recommended for speed.
        rho_A = rho(state, va, ns, local_dim, contiguous)
        if return_vecs:
            vals, vecs  = la.eigh(rho_A)
            vals        = np.clip(vals, 0.0, 1.0)
            idx         = np.argsort(vals)[::-1]
            return vals[idx], vecs[:, idx], rho_A
        else:
            w           = la.eigvalsh(rho_A)
            w           = np.clip(w, 0.0, 1.0)
            return np.sort(w[w > 1e-15])[::-1]

def rho_spectrum(rho_mat: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Compute the eigenvalue spectrum of a density matrix.
    """
    w = la.eigvalsh(rho_mat)
    w = np.clip(w, 0.0, 1.0)
    return np.sort(w[w > eps])[::-1]

# -----------------------------------------------------------------------------
#! Specialized site functions
# -----------------------------------------------------------------------------

def rho_single_site(state: np.ndarray, site: int, ns: int, local_dim: int = 2) -> np.ndarray:
    return rho(state, va=[site], ns=ns, local_dim=local_dim)

def rho_two_sites(state: np.ndarray, site_i: int, site_j: int, ns: int, local_dim: int = 2) -> np.ndarray:
    return rho(state, va=[site_i, site_j], ns=ns, local_dim=local_dim)

# -----------------------------------------------------------------------------
#! End of file
# -----------------------------------------------------------------------------
