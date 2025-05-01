'''
file    : QES/general_python/physics/density_matrix_jax.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-05-01
This module provides a function to compute the reduced density matrix
'''
from general_python.algebra.utils import JAX_AVAILABLE, Array
from typing import Optional, Union
from functools import partial

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    
    @jax.jit
    def r_dens_mat(state    : jnp.ndarray,
                    A_size  : int,
                    L       : int | None = None) -> jnp.ndarray:
        if L is None:
            L = int(jnp.log2(state.size))
        dimA    = 1 << A_size
        dimB    = 1 << (L - A_size)
        psi     = jnp.reshape(state, (dimA, dimB))
        return psi @ jnp.conj(psi).T
    
    # --------------------------------------------------------------------
    
    @partial(jax.jit, static_argnames=("La", "L", "use_eigh"))
    def r_dens_mat_schmidt(state        : jnp.ndarray,
                            La          : int,
                            L           : int | None = None,
                            use_eigh    : bool = True):
        """
        JAX version of r_dens_mat_schmidt:
        returns (eigvals, U) — both on-device.

        * jit-compiled on first call;
        * reshaping is free (view);
        * chooses the smaller Hermitian block for `eigh`, same as NumPy code.
        """
        if L is None:
            L = jnp.int32(jnp.log2(state.size))

        dimA = 1 << La
        dimB = 1 << (L - La)

        # logic identical to NumPy path, but pure JAX ops
        _eig_val = jnp.reshape(state, (dimA, dimB))

        if use_eigh:
            rho = jnp.where(dimA <= dimB,
                        _eig_val @ jnp.conj(_eig_val).T,    # dimA×dimA
                        jnp.conj(_eig_val).T @ _eig_val)    # dimB×dimB
            eigvals, U = jnp.linalg.eigh(rho)
        else:
            U, s, _ = jnp.linalg.svd(_eig_val, full_matrices=False)
            eigvals = s * s
        return eigvals, U
    
    # --------------------------------------------------------------------
    
else:
    reduced_density_matrix_jax = None # JAX not installed
    r_dens_mat_schmidt_jax = None
    
# --------------------------------------------------------------------