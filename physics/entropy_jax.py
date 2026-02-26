'''
JAX entropy kernels mirroring entropy.py behavior.

--------------------------------
Author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
--------------------------------
'''

from __future__ import annotations

from functools import partial
from typing import Optional

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    Array = jnp.ndarray
except ImportError:
    jax             = None
    jnp             = None
    JAX_AVAILABLE   = False
    Array           = None

# ------------------------------------------------------------------------

_EPS = 1e-15

def _validate_log_base(base: Optional[float]) -> Optional[float]:
    if base is None:
        return None
    b = float(base)
    if b <= 0.0 or b == 1.0:
        raise ValueError(f"Invalid logarithm base={base}. Base must be positive and different from 1.")
    return b

if JAX_AVAILABLE:

    @jax.jit
    def _eigvals_jax(rho: "jnp.ndarray") -> "jnp.ndarray":
        return jnp.linalg.eigvalsh(rho)

    @jax.jit
    def _clean_probs_jax(p: "jnp.ndarray", eps: float = _EPS) -> "jnp.ndarray":
        q = jnp.where(jnp.real(p) < eps, 0.0, jnp.real(p))
        s = jnp.sum(q)

        def _renorm(_: None):
            return jnp.where(jnp.abs(s - 1.0) > 1e-14, q / s, q)

        def _zeros(_: None):
            return q

        return jax.lax.cond(s > 0.0, _renorm, _zeros, operand=None)

    @partial(jax.jit, static_argnames=("base",))
    def vn_entropy_jax(lam: "jnp.ndarray", base: Optional[float] = None) -> float:
        """
        Von Neumann entropy.
        """
        lam     = _clean_probs_jax(jnp.asarray(lam))
        logp    = jnp.log(jnp.clip(lam, _EPS, 1.0))
        ent     = -jnp.sum(lam * logp)
        if base is not None:
            ent = ent / jnp.log(base)
        return ent

    @partial(jax.jit, static_argnames=("base",))
    def renyi_entropy_jax(lam: "jnp.ndarray", q: float, base: Optional[float] = None) -> float:
        """
        Rényi entropy of order q.
        """
        lam = _clean_probs_jax(jnp.asarray(lam))

        def _vn(_: None):
            return vn_entropy_jax(lam, base=base)

        def _generic(_: None):
            s   = jnp.maximum(jnp.sum(lam ** q), _EPS)
            out = jnp.log(s) / (1.0 - q)
            if base is not None:
                out = out / jnp.log(base)
            return out

        return jax.lax.cond(jnp.isclose(q, 1.0), _vn, _generic, operand=None)

    @partial(jax.jit, static_argnames=("base",))
    def tsallis_entropy_jax(lam: "jnp.ndarray", q: float, base: Optional[float] = None) -> float:
        """
        Tsallis entropy.
        """
        lam = _clean_probs_jax(jnp.asarray(lam))

        def _vn(_: None):
            return vn_entropy_jax(lam, base=base)

        def _generic(_: None):
            return (1.0 - jnp.sum(lam ** q)) / (q - 1.0)

        return jax.lax.cond(jnp.isclose(q, 1.0), _vn, _generic, operand=None)

    @partial(jax.jit, static_argnames=("base",))
    def sp_correlation_entropy_jax(lam: "jnp.ndarray", q: float, base: Optional[float] = None) -> float:
        """
        Single-particle correlation entropy from correlation-matrix eigenvalues in [-1, 1].
        """
        lam = jnp.real(jnp.asarray(lam))
        p   = jnp.clip(0.5 * (1.0 + lam), 0.0, 1.0)
        pm  = jnp.clip(1.0 - p, 0.0, 1.0)

        def _vn(_: None):
            ent = -jnp.sum(p * jnp.log(jnp.clip(p, _EPS, 1.0)) + pm * jnp.log(jnp.clip(pm, _EPS, 1.0)))
            return ent

        def _renyi(_: None):
            s   = jnp.sum(jnp.log(jnp.clip(p ** q + pm ** q, _EPS, None)))
            out = s / (1.0 - q)
            if base is not None:
                out = out / jnp.log(base)
            return out

        return jax.lax.cond(jnp.isclose(q, 1.0), _vn, _renyi, operand=None)

    @partial(jax.jit, static_argnames=("threshold",))
    def information_entropy_jax(states: "jnp.ndarray", threshold: float = 1e-12):
        """
        Shannon information entropy per state column.
        """
        s       = jnp.asarray(states)
        is_1d   = s.ndim == 1
        if is_1d:
            s   = s.reshape((s.shape[0], 1))

        p       = jnp.abs(s) ** 2
        mask    = p > threshold
        contrib = jnp.where(mask, p * jnp.log(jnp.clip(p, _EPS, None)), 0.0)
        ent     = -jnp.sum(contrib, axis=0)
        return ent[0] if is_1d else ent

    @partial(jax.jit, static_argnames=("threshold", "square"))
    def participation_entropy_jax(
        states      : "jnp.ndarray",
        q           : float = 1.0,
        threshold   : float = 1e-12,
        square      : bool = False
    ):
        """
        Participation entropy per state column.
        """
        s       = jnp.asarray(states)
        is_1d   = s.ndim == 1
        if is_1d:
            s   = s.reshape((s.shape[0], 1))

        two_q   = 2.0 * q if square else q

        def _q1(_: None):
            p       = jnp.abs(s) ** two_q
            mask    = p > threshold
            contrib = jnp.where(mask, p * jnp.log(jnp.clip(p, _EPS, None)), 0.0)
            return -jnp.sum(contrib, axis=0)

        def _qne(_: None):
            p       = jnp.abs(s) ** q
            acc     = jnp.sum(jnp.where(p > threshold, p, 0.0), axis=0)
            return jnp.where(acc > 0.0, jnp.log(acc) / (1.0 - q), 0.0)

        out = jax.lax.cond(jnp.isclose(q, 1.0), _q1, _qne, operand=None)
        return out[0] if is_1d else out

else:
    _eigvals_jax                = None
    _clean_probs_jax            = None
    vn_entropy_jax              = None
    renyi_entropy_jax           = None
    tsallis_entropy_jax         = None
    sp_correlation_entropy_jax  = None
    information_entropy_jax     = None
    participation_entropy_jax   = None

# ------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------
