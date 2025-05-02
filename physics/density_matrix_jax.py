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
    
    @partial(jax.jit, static_argnums=(1, 2))
    def rho(state, dimA: int, dimB: int):
        psi = jnp.reshape(state, (dimA, dimB), order="F")
        return psi @ jnp.conj(psi).T

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def schmidt(state, dimA: int, dimB: int, use_eig: bool):
        psi = jnp.reshape(state, (dimA, dimB), order="F")
        if use_eig:
            if dimA <= dimB:
                rho         = psi @ jnp.conj(psi).T
                vals, vecs  = jnp.linalg.eigh(rho)
            else:
                sigma   = jnp.conj(psi).T @ psi
                vals, V = jnp.linalg.eigh(sigma)
                vecs    = psi @ V / jnp.sqrt(jnp.maximum(vals, 1e-30))
        else:
            vecs, s, _  = jnp.linalg.svd(psi, full_matrices=False)
            vals        = s * s
        # flip for descending order
        return vals[::-1], vecs[:, ::-1]
    
else:
    schmidt = rho = None
    
# --------------------------------------------------------------------