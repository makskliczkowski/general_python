r'''
JAX-optimized reduced density matrix and Schmidt decomposition.
Mirrors the NumPy API in density_matrix.py with full fermionic support.

This module provides JAX-accelerated implementations of:
- Reduced density matrix computation (rho)
- Schmidt decomposition (schmidt)
- Single-site and two-site RDMs
- Fermionic sign corrections for non-contiguous subsystems

QES Convention:
- State vector index i = s0 + d*s1 + d^2*s2 + ... (Little-endian / Fortran order)
- Subsystem A site order: [a0, a1, a2, ...] -> Row index I = sa0 + d*sa1 + ...
- Subsystem B site order: [b0, b1, b2, ...] -> Col index J = sb0 + d*sb1 + ...

Fermionic Systems:
For fermionic systems mapped via Jordan-Wigner transformation, non-local string
operators create additional correlations between non-contiguous subsystem sites.
The `fermionic=True` flag applies sign corrections that account for fermionic
exchange statistics when reordering sites.

--------------------------------
Author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
version     : 2.1
copyright   : (c) 2026 by Maksymilian Kliczkowski. All rights reserved.
file        : general_python/physics/density_matrix_jax.py
--------------------------------
'''

from __future__ import annotations

from functools  import lru_cache, partial
from typing     import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import  jax
    import  jax.numpy as jnp
    from    jax import jit
    JAX_AVAILABLE   = True
except ImportError:
    jax             = None
    jnp             = None
    jit             = None
    JAX_AVAILABLE   = False

from .density_matrix import mask_subsystem, _fermionic_parity_signs_fast

# -----------------------------------------------------------------------------
#! Fermionic Sign Correction (JAX)
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @lru_cache(maxsize=256)
    def _fermionic_signs_cached(ns: int, order: Tuple[int, ...]) -> np.ndarray:
        """
        Cached fermionic parity signs (computed once per configuration).
        Returns NumPy array to be converted to JAX inside JIT.
        """
        return _fermionic_parity_signs_fast(ns, order)

# -----------------------------------------------------------------------------
#! Low-level JAX kernels
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @lru_cache(maxsize=256) # Cache gather indices for up to 256 different subsystem configurations (va, ns, local_dim, order)
    def _psi_take_indices(
        ns          : int,
        size_a      : int,
        local_dim   : int,
        order       : Tuple[int, ...]
    ) -> np.ndarray:
        """
        Precompute gather indices such that:
        psi_mat.reshape(-1, order='C') == state[idx]
        using the 'density_matrix' little-endian site convention.
        
        Parameters
        ----------
        ns : int
            Total number of sites in the system.
        size_a : int
            Number of sites in subsystem A.
        local_dim : int
            Local Hilbert space dimension (e.g., 2 for qubits).
        order : Tuple[int, ...]
            Permutation of site indices that brings subsystem A sites to the front.
            
        Returns
        -------
        np.ndarray
            Array of indices to gather from the state vector to reshape into Psi_{A,B}.
        """
        dA          = local_dim ** size_a
        dB          = local_dim ** (ns - size_a)
        idx         = np.empty(dA * dB, dtype=np.int64)
        sites_a     = order[:size_a]
        sites_b     = order[size_a:]
        pow_site    = np.asarray([local_dim ** i for i in range(ns)], dtype=np.int64)

        pos = 0
        for a in range(dA):
            ta          = a
            digits_a    = np.empty(size_a, dtype=np.int64)
            for k in range(size_a):
                digits_a[k] = ta % local_dim
                ta          //= local_dim

            for b in range(dB):
                tb = b
                i = 0
                for k, site in enumerate(sites_a):
                    i       +=  int(digits_a[k]) * int(pow_site[site])
                for k, site in enumerate(sites_b):
                    digit   =   tb % local_dim
                    tb      //= local_dim
                    i       +=  int(digit) * int(pow_site[site])
                idx[pos] = i
                pos     += 1
        return idx

    @partial(jit, static_argnums=(1, 2, 3, 4))
    def psi_jax(
        state       : "jnp.ndarray",
        size_a      : int,
        ns          : int,
        local_dim   : int = 2,
        order       : Optional[Tuple[int, ...]] = None,
    ) -> "jnp.ndarray":
        """
        Reshape/reorder a state into Psi_{A,B} using QES little-endian convention.
        
        Parameters
        ----------
        state : jnp.ndarray
            The input state vector of shape (local_dim**ns,).
        size_a : int
            The number of sites in subsystem A.
        ns : int
            Total number of sites in the system.
        local_dim : int
            Local Hilbert space dimension (default is 2 for qubits).
        order : Optional[Tuple[int, ...]]
            The permutation order of sites to bring subsystem A sites to the front.
            
        Returns
        -------
        jnp.ndarray
            Reshaped state matrix Psi of shape (dA, dB) where dA = local_dim^size_a
            and dB = local_dim^(ns - size_a).
            
        Notes
        -----
        This is the JAX-JIT compiled kernel. For fermionic sign corrections,
        use the high-level `rho()` or `schmidt()` functions with `fermionic=True`.
        """
        dA      = local_dim ** size_a
        dB      = local_dim ** (ns - size_a)
        ord_t   = tuple(range(ns)) if order is None else tuple(order)
        idx     = jnp.asarray(_psi_take_indices(ns, size_a, local_dim, ord_t), dtype=jnp.int32)
        flat    = jnp.take(state, idx)
        return jnp.reshape(flat, (dA, dB))

    @partial(jit, static_argnums=(1, 2, 3, 4))
    def rho_jax(
        state       : "jnp.ndarray",
        size_a      : int,
        ns          : int,
        local_dim   : int = 2,
        order       : Optional[Tuple[int, ...]] = None,
    ) -> "jnp.ndarray":
        """
        Reduced density matrix rho_A from a pure-state vector.
        """
        psi = psi_jax(state, size_a, ns, local_dim, order)
        return psi @ jnp.conj(psi).T

else:
    psi_jax = None
    rho_jax = None

# -----------------------------------------------------------------------------
#! High-level API (NumPy-parity)
# -----------------------------------------------------------------------------

def rho(
    state       : Any,
    va          : Union[int, List[int], np.ndarray],
    ns          : Optional[int] = None,
    local_dim   : int = 2,
    contiguous  : bool = False,
    fermionic   : bool = False,
) -> Any:
    """
    Compute reduced density matrix rho_A of subsystem A (JAX backend).
    
    Parameters
    ----------
    state : array-like
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
        
    Returns
    -------
    jnp.ndarray
        Reduced density matrix rho_A of shape (dA, dA) where dA = local_dim^|A|.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available.")

    state_jax = jnp.asarray(state)
    if ns is None:
        ns = int(np.round(np.log(state_jax.size) / np.log(local_dim)))

    (size_a, _), order = mask_subsystem(va, ns, local_dim, contiguous)
    
    # Apply fermionic sign correction if needed
    if fermionic and local_dim == 2 and order != tuple(range(ns)):
        signs = jnp.asarray(_fermionic_signs_cached(ns, order))
        state_jax = state_jax * signs
    
    return rho_jax(state_jax, size_a, ns, local_dim, order)

def schmidt(
    state       : Any,
    va          : Union[int, List[int], np.ndarray],
    ns          : Optional[int] = None,
    local_dim   : int = 2,
    contiguous  : bool = False,
    fermionic   : bool = False,
    eig         : bool = False,
    square      : bool = True,
    *,
    return_vecs : bool = True,
):
    """
    Schmidt decomposition (JAX backend), API-compatible with density_matrix.schmidt.
    
    For a bipartition of the system into subsystems A and B, the Schmidt
    decomposition expresses the state as:
        |psi> = sum_k lambda_k |phi_k>_A |chi_k>_B
    
    Parameters
    ----------
    state : array-like
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
    eig : bool
        If True, use RDM eigendecomposition instead of SVD.
    square : bool
        If True (default), return squared singular values (= RDM eigenvalues).
    return_vecs : bool
        If True, return Schmidt vectors along with values.
    
    Returns
    -------
    If return_vecs=True:
        Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
        
    If return_vecs=False:
        jnp.ndarray of Schmidt values.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available.")

    state_jax = jnp.asarray(state)
    if ns is None:
        ns = int(np.round(np.log(state_jax.size) / np.log(local_dim)))

    (size_a, _), order = mask_subsystem(va, ns, local_dim, contiguous)
    
    # Apply fermionic sign correction if needed
    if fermionic and local_dim == 2 and order != tuple(range(ns)):
        signs       = jnp.asarray(_fermionic_signs_cached(ns, order))
        state_jax   = state_jax * signs

    psi_mat = psi_jax(state_jax, size_a, ns, local_dim, order)
    if not eig:
        u, s, vh    = jnp.linalg.svd(psi_mat, full_matrices=False)
        vals        = s ** 2 if square else s
        return (vals, (u, vh)) if return_vecs else vals

    rho_A = psi_mat @ jnp.conj(psi_mat).T
    if return_vecs:
        vals, vecs  = jnp.linalg.eigh(rho_A)
        vals        = jnp.clip(vals, 0.0, 1.0)
        idx         = jnp.argsort(vals)[::-1]
        return vals[idx], vecs[:, idx]

    w = jnp.linalg.eigvalsh(rho_A)
    w = jnp.clip(w, 0.0, 1.0)
    w = jnp.sort(w)[::-1]
    return w[w > 1e-15]

def rho_spectrum(rho_mat: Any, eps: float = 1e-15):
    """
    Eigenvalue spectrum of a density matrix (descending, clipped to [0,1]).
    
    Parameters
    ----------
    rho_mat : array-like
        Density matrix (Hermitian, positive semi-definite).
    eps : float
        Threshold for filtering small eigenvalues.
    
    Returns
    -------
    jnp.ndarray
        Sorted eigenvalues (descending) above threshold eps.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available.")

    w = jnp.linalg.eigvalsh(jnp.asarray(rho_mat))
    w = jnp.clip(w, 0.0, 1.0)
    w = jnp.sort(w)[::-1]
    return w[w > eps]

def rho_single_site(
    state       : Any, 
    site        : int, 
    ns          : int, 
    local_dim   : int = 2,
    fermionic   : bool = False
):
    """
    Compute the single-site reduced density matrix.
    
    Parameters
    ----------
    state : array-like
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
    jnp.ndarray
        Single-site RDM of shape (local_dim, local_dim).
    """
    return rho(state, va=[site], ns=ns, local_dim=local_dim, fermionic=fermionic)

def rho_two_sites(
    state       : Any, 
    site_i      : int, 
    site_j      : int, 
    ns          : int, 
    local_dim   : int = 2,
    fermionic   : bool = False
):
    """
    Compute the two-site reduced density matrix.
    
    Parameters
    ----------
    state : array-like
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
    jnp.ndarray
        Two-site RDM of shape (local_dim^2, local_dim^2).
    """
    return rho(state, va=[site_i, site_j], ns=ns, local_dim=local_dim, fermionic=fermionic)

# -----------------------------------------------------------------------------
#! End of file
# -----------------------------------------------------------------------------
