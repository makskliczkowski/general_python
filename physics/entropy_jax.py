'''
file    : QES/general_python/physics/entropy_jax.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-05-01

'''

from general_python.algebra.utils import JAX_AVAILABLE, Array
from typing import Optional, Union
from functools import partial

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    
    @jax.jit
    def _eigvals_jax(rho: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.eigvalsh(rho)
    
    # --------------------------------------------------------------------
    
    @jax.jit
    def _clean_probs_jax(p: jnp.ndarray, eps: float = 1e-15):
        q  = jnp.where(p < eps, 0.0, p)
        s  = jnp.sum(q)
        return jnp.where(jnp.abs(s - 1.0) > 1e-12, q / s, q)

    # --------------------------------------------------------------------

    @jax.jit
    def vn_entropy_jax(lam, base=jnp.e):
        lam = _clean_probs_jax(lam)
        log = jnp.log(lam + 1e-30)
        if base != jnp.e:
            log /= jnp.log(base)
        return -jnp.vdot(lam, log).real

    # --------------------------------------------------------------------

    @jax.jit
    def renyi_entropy_jax(lam, q: float, base=jnp.e):
        lam = _clean_probs_jax(lam)
        if q == 1.0:
            return vn_entropy_jax(lam, base)
        s   = jnp.sum(lam ** q)
        log = jnp.log(s)
        if base != jnp.e:
            log /= jnp.log(base)
        return log / (1.0 - q)

    # --------------------------------------------------------------------

    @jax.jit
    def tsallis_entropy_jax(lam, q: float):
        lam = _clean_probs_jax(lam)
        if q == 1.0:
            return vn_entropy_jax(lam)
        return (1.0 - jnp.sum(lam ** q)) / (q - 1.0)
    
    # --------------------------------------------------------------------
