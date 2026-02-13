'''
file    : general_python/physics/entropy_jax.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-05-01

'''

try:
    import jax
    from jax import numpy as jnp
    JAX_AVAILABLE   = True
    Array           = jnp.ndarray
except ImportError:
    JAX_AVAILABLE   = False
    Array           = None
    
from typing     import Optional, Union
from functools  import partial
from enum       import Enum, unique
from math       import log2

# ------------------------------------------------------------------------

_EPS = 1e-10

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

    @partial(jax.jit, static_argnames=("base",))
    def vn_entropy_jax(lam: jnp.ndarray, base: float = jnp.e) -> float:
        r"""
        Von Neumann (Shannon) entropy

        \[
            S = -\sum_i p_i \log_b p_i ,
            \qquad  p_i \ge 0,\; \sum_i p_i = 1 .
        \]

        Parameters
        ----------
        lam  : jnp.ndarray
            Probability vector.
        base : float, optional
            Logarithm base \(b\).  Default = \(e\).

        Returns
        -------
        float
            \(S\).
        """
        lam      = _clean_probs_jax(lam)
        logp     = jnp.log(lam + _EPS)
        if base != jnp.e:
            logp /= jnp.log(base)
        return -jnp.vdot(lam, logp)

    @partial(jax.jit, static_argnames=("base",))
    def renyi_entropy_jax(lam: jnp.ndarray, q: float, base: float = jnp.e) -> float:
        r"""
        Rényi entropy of order \(q\neq 1\)

        \[
            S_q = \frac{\log_b \!\bigl(\sum_i p_i^{\,q}\bigr)}{1-q},
            \qquad  q>0 .
        \]

        For \(q\to1\) it reduces to the von Neumann entropy.

        Parameters
        ----------
        lam  : jnp.ndarray
        q    : float
        base : float, optional

        Returns
        -------
        float
        """
        lam = _clean_probs_jax(lam)

        def _vn(_: None) -> float:
            return vn_entropy_jax(lam, base)

        def _generic(_: None) -> float:
            s       = jnp.sum(lam ** q)
            log_s   = jnp.log(s)
            if base != jnp.e:
                log_s   /= jnp.log(base)
            return log_s / (1.0 - q)

        return jax.lax.cond(jnp.isclose(q, 1.0), _vn, _generic, operand=None)

    @jax.jit
    def tsallis_entropy_jax(lam: jnp.ndarray, q: float) -> float:
        r"""
        Tsallis entropy

        \[
            S_q^{\mathrm{T}} = \frac{1 - \sum_i p_i^{\,q}}{q-1},
            \qquad  q>0 .
        \]

        Parameters
        ----------
        lam : jnp.ndarray
        q   : float

        Returns
        -------
        float
        """
        lam = _clean_probs_jax(lam)

        def _vn(_: None) -> float:
            return vn_entropy_jax(lam)

        def _generic(_: None) -> float:
            return (1.0 - jnp.sum(lam ** q)) / (q - 1.0)

        return jax.lax.cond(jnp.isclose(q, 1.0), _vn, _generic, operand=None)


    @partial(jax.jit, static_argnames=("base",))
    def sp_correlation_entropy_jax(lam: jnp.ndarray, q: float, base: float = jnp.e) -> float:
        r"""
        Compute the single-particle correlation entropy for a set of eigenvalues.

        This function calculates either the von Neumann entropy (for q=1) or the Rényi entropy (for q\neq 1)
        associated with the eigenvalues of a correlation matrix. The eigenvalues \lambda are assumed to be in [-1, 1].
        The probabilities are defined as \( p = \frac{1}{2}(1 + \lambda) \).

        Formulas:
            - For q = 1 (von Neumann entropy):
                \( S = -\sum_i \left[ p_i \log_b p_i + (1 - p_i) \log_b (1 - p_i) \right] \)
            - For q \neq  1 (Rényi entropy):
                \( S_q = \frac{1}{1 - q} \sum_i \log_b \left( p_i^q + (1 - p_i)^q \right) \)

        lam : jnp.ndarray
            Array of correlation-matrix eigenvalues (\(\lambda\)), each in the interval [-1, 1].
        q : float
            Entropy order parameter. Use q=1 for von Neumann entropy, q\neq 1 for Rényi entropy.
            Logarithm base for entropy calculation (default: natural logarithm, e).

            The computed entropy value.

        Notes
        -----
        - The function is numerically stable for probabilities close to 0 or 1.
        - For q=1, the result is the standard von Neumann entropy.
        - For q\neq 1, the result is the Rényi entropy of order q.
        """
        log_base = jnp.log(base)
        p        = 0.5 * (1.0 + lam)
        pm       = 1.0 - p

        def _vn(_: None) -> float:
            ent  = -jnp.sum(p  * jnp.log(p  + _EPS) +
                            pm * jnp.log(pm + _EPS))
            if base != jnp.e:
                ent /= log_base
            return ent

        def _renyi(_: None) -> float:
            s = jnp.sum(jnp.log(p ** q + pm ** q))
            return s / ((1.0 - q) * log_base)

        return jax.lax.cond(jnp.isclose(q, 1.0), _vn, _renyi, operand=None)

    # ------------------------------------------------------------------------
else:
    vn_entropy_jax = None
    renyi_entropy_jax = None
    tsallis_entropy_jax = None
    sp_correlation_entropy_jax = None
    
# ------------------------------------------------------------------------
    