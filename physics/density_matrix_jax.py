'''
JAX-optimized reduced density matrix and Schmidt decomposition.
Mirrors the NumPy API for consistency across backends.

--------------------------------
Author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
--------------------------------
'''

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    jax = jnp = None
    JAX_AVAILABLE = False

from typing import Optional, Union, Tuple, List
from functools import partial

# -----------------------------------------------------------------------------

if JAX_AVAILABLE:

    @partial(jit, static_argnums=(1, 2, 3))
    def rho_jax(state: jnp.ndarray, size_a: int, ns: int, order: Optional[Tuple[int, ...]] = None):
        """
        Compute RDM using JAX.
        """
        if order is None or order == tuple(range(ns)):
            # Contiguous
            dA = 1 << size_a
            dB = 1 << (ns - size_a)
            # JAX uses C-order, but QES uses F-order convention for qubits.
            # To mimic F-order reshape(dA, dB) in JAX:
            # We reshape to (2, 2, ..., 2) and transpose to reversed order.
            psi_nd = jnp.reshape(state, (2,) * ns)
            # Reversed for F-order equivalent
            psi_mat = jnp.reshape(jnp.transpose(psi_nd, tuple(range(ns))[::-1]), (dA, dB), order='C')
        else:
            # Masked
            psi_nd = jnp.reshape(state, (2,) * ns)
            # Map QES site indices to JAX axes (reversed)
            perm = tuple(ns - 1 - i for i in order)
            psi_perm = jnp.transpose(psi_nd, perm)
            dA = 1 << size_a
            psi_mat = jnp.reshape(psi_perm, (dA, -1))
            
        return psi_mat @ jnp.conj(psi_mat).T

    @partial(jit, static_argnums=(1, 2, 3, 4, 5))
    def schmidt_jax(
        state       : jnp.ndarray, 
        size_a      : int, 
        ns          : int, 
        order       : Optional[Tuple[int, ...]] = None,
        eig         : bool = False,
        square      : bool = False
    ):
        """
        Schmidt decomposition using JAX.
        """
        # Reshape logic similar to rho_jax
        psi_nd = jnp.reshape(state, (2,) * ns)
        if order is None:
            order = tuple(range(ns))
            
        perm = tuple(ns - 1 - i for i in order)
        psi_mat = jnp.reshape(jnp.transpose(psi_nd, perm), (1 << size_a, -1))
        
        if eig:
            # RDM approach
            rho_A       = psi_mat @ jnp.conj(psi_mat).T
            vals, vecs  = jnp.linalg.eigh(rho_A)
            vals        = jnp.clip(vals, 0.0, 1.0)
            return vals[::-1], vecs[:, ::-1]
        else:
            # SVD approach
            u, s, vh    = jnp.linalg.svd(psi_mat, full_matrices=False)
            vals        = s**2 if square else s
            return vals, (u, vh)

else:
    rho_jax = schmidt_jax = None

# -----------------------------------------------------------------------------
#! High-level JAX-dispatch API
# -----------------------------------------------------------------------------

def rho(
    state       : Any, 
    va          : Union[int, List[int], np.ndarray], 
    ns          : Optional[int] = None, 
    local_dim   : int           = 2, 
    contiguous  : bool          = False
) -> Any:
    """
    High-level JAX-specific RDM function.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available.")
        
    if ns is None:
        ns = int(jnp.round(jnp.log(state.size) / jnp.log(local_dim)))
    
    # We can't import mask_subsystem from density_matrix here easily due to circular/backend deps,
    # but we can implement a JAX-friendly version or use the numpy one.
    import numpy as np
    from .density_matrix import mask_subsystem
    (size_a, size_b), order = mask_subsystem(va, ns, local_dim, contiguous)
    
    return rho_jax(state, size_a, ns, order)

# -----------------------------------------------------------------------------
#! End of file
# -----------------------------------------------------------------------------
