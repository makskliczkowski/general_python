'''
JAX-optimized reduced density matrix and Schmidt decomposition.
Mirrors the NumPy API in density_matrix.py.

--------------------------------
Author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
--------------------------------
'''

from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    jit = None
    JAX_AVAILABLE = False

from .density_matrix import mask_subsystem

# -----------------------------------------------------------------------------
#! Low-level JAX kernels
# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @lru_cache(maxsize=256)
    def _psi_take_indices(
        ns: int,
        size_a: int,
        local_dim: int,
        order: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Precompute gather indices such that:
        psi_mat.reshape(-1, order='C') == state[idx]
        using QES little-endian site convention.
        """
        dA = local_dim ** size_a
        dB = local_dim ** (ns - size_a)
        idx = np.empty(dA * dB, dtype=np.int64)
        sites_a = order[:size_a]
        sites_b = order[size_a:]
        pow_site = np.asarray([local_dim ** i for i in range(ns)], dtype=np.int64)

        pos = 0
        for a in range(dA):
            ta = a
            digits_a = np.empty(size_a, dtype=np.int64)
            for k in range(size_a):
                digits_a[k] = ta % local_dim
                ta //= local_dim

            for b in range(dB):
                tb = b
                i = 0
                for k, site in enumerate(sites_a):
                    i += int(digits_a[k]) * int(pow_site[site])
                for k, site in enumerate(sites_b):
                    digit = tb % local_dim
                    tb //= local_dim
                    i += int(digit) * int(pow_site[site])
                idx[pos] = i
                pos += 1
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
) -> Any:
    """
    Compute reduced density matrix rho_A of subsystem A (JAX backend).
    Signature matches density_matrix.rho.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available.")

    state_jax = jnp.asarray(state)
    if ns is None:
        ns = int(np.round(np.log(state_jax.size) / np.log(local_dim)))

    (size_a, _), order = mask_subsystem(va, ns, local_dim, contiguous)
    return rho_jax(state_jax, size_a, ns, local_dim, order)

def schmidt(
    state       : Any,
    va          : Union[int, List[int], np.ndarray],
    ns          : Optional[int] = None,
    local_dim   : int = 2,
    contiguous  : bool = False,
    eig         : bool = False,
    square      : bool = True,
    *,
    return_vecs : bool = True,
):
    """
    Schmidt decomposition (JAX backend), API-compatible with density_matrix.schmidt.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available.")

    state_jax = jnp.asarray(state)
    if ns is None:
        ns = int(np.round(np.log(state_jax.size) / np.log(local_dim)))

    (size_a, _), order = mask_subsystem(va, ns, local_dim, contiguous)

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
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available.")

    w = jnp.linalg.eigvalsh(jnp.asarray(rho_mat))
    w = jnp.clip(w, 0.0, 1.0)
    w = jnp.sort(w)[::-1]
    return w[w > eps]

def rho_single_site(state: Any, site: int, ns: int, local_dim: int = 2):
    return rho(state, va=[site], ns=ns, local_dim=local_dim)

def rho_two_sites(state: Any, site_i: int, site_j: int, ns: int, local_dim: int = 2):
    return rho(state, va=[site_i, site_j], ns=ns, local_dim=local_dim)

# -----------------------------------------------------------------------------
#! End of file
# -----------------------------------------------------------------------------
