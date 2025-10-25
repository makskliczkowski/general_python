'''
file    : QES/general_python/physics/density_matrix_jax.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-05-01
This module provides a function to compute the reduced density matrix
using JAX. It enables to optimize the computation of the reduced
density matrix for quantum states, particularly useful in quantum
entanglement and quantum information tasks.
'''

try:
    import jax
    from jax import numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = jnp = None
    JAX_AVAILABLE = False

from typing import Optional, Union
from functools import partial

# --------------------------------------------------------------------

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    
    @partial(jax.jit, static_argnums=(1, 2))
    def rho_jax(state, dimA: int, dimB: int):
        psi = jnp.reshape(state, (dimA, dimB), order="F")
        return psi @ jnp.conj(psi).T

    @partial(jax.jit, static_argnums=(1, 2))
    def rho_mask_jax(state  : jnp.ndarray,
                    order   : tuple[int, ...],
                    size_a  : int) -> jnp.ndarray:
        r"""
        Reshape-and-permute a pure \(N\)-qubit state
        \(|\psi\rangle\in\mathbb C^{2^N}\)
        into a matrix  
        \(\psi_{A,B}\in\mathbb C^{2^{|A|}\times 2^{|B|}}\)
        that is ready for a partial-trace / reduced-density-matrix
        calculation with respect to subsystem *A* (first ``size_a`` qubits in
        ``order``).

        Parameters
        ----------
        state : jnp.ndarray
            Flat state vector, length \(2^N\).
        order : tuple[int]
            Permutation of qubit indices; the *first* ``size_a`` entries
            define subsystem *A*.
        size_a : int
            \(|A|\) - number of qubits in subsystem *A*.

        Returns
        -------
        jnp.ndarray
            Array of shape \((2^{|A|},\,2^{N-|A|})\).
        """
        N = len(order)                          # total number of qubits

        # reshape: 1-D -> N-D tensor (row-major)
        psi_nd = jnp.reshape(state, (2,) * N)   # JAX supports only row-major

        # mimic Fortran-order semantics
        #   For a (2,2,…,2) tensor, Fortran layout is equivalent to
        #   row-major layout with *reversed* axis numbering.
        perm = tuple(N - 1 - o for o in order)  # map Fortran axes -> C axes

        # reorder qubits and flatten back to matrix
        psi_perm = jnp.transpose(psi_nd, perm)
        dA       = 1 << size_a                  # 2**size_a
        return jnp.reshape(psi_perm, (dA, -1))

    @partial(jax.jit, static_argnums=(1, 2, 3))
    def schmidt_jax(state, dimA: int, dimB: int, use_eig: bool):
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
    
    @partial(jax.jit, static_argnums=(1, 2, 3))
    def schmidt_mask_jax(state: jnp.array, order: tuple, size_a: int, eig: bool = False):
        """
        Computes the Schmidt decomposition for a given state and mask.
        
        Args:
            state (Array): The input state vector.
            order (int): The order of the mask.
            size_a (int): The size of the first subsystem.
        
        Returns:
            Array: The Schmidt decomposition.
        """
        # Reshape the state vector into a matrix
        psi_nd  = jnp.reshape(state, (2,) * len(order), order="F")
        dA      = 1 << size_a
        psi     = psi_nd.transpose(order).reshape((dA, -1))
        if eig:
            if size_a <= len(order) - size_a:
                rho         = psi @ jnp.conj(psi).T
                vals, vecs  = jnp.linalg.eigh(rho)
            else:
                sigma       = jnp.conj(psi).T @ psi
                vals, V     = jnp.linalg.eigh(sigma)
                vecs        = psi @ V / jnp.sqrt(jnp.maximum(vals, 1e-30))
        else:
            vecs, s, _      = jnp.linalg.svd(psi, full_matrices=False)
            vals            = s * s
        return vals[::-1], vecs[:, ::-1]
    
else:
    schmidt_jax = rho_jax = schmidt_mask_jax = rho_mask_jax = None
    
# --------------------------------------------------------------------