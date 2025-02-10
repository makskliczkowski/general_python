'''
This module provides linear algebra functions and utilities.
'''

# Import the required modules

import numpy as np
import numpy.random as nrn
import scipy as sp

try:
    import jax.numpy as jnp
    import jax.scipy as jsp
    import jax.random as jrn
    from jax import jit
    _JAX_AVAILABLE  = True
    _KEY            = jrn.PRNGKey(0)
except ImportError:
    print("JAX not available. Falling back to NumPy and SciPy.")
    _JAX_AVAILABLE  = False
    _KEY            = None

# Global default backend: use jax.numpy if available, else numpy.
DEFAULT_BACKEND = jnp if _JAX_AVAILABLE else np

# ---------------------------------------------------------------------

def get_backend(backend, random=False, seed=None, scipy=False):
    """
    Given a backend specifier, return the corresponding module.
    
    Parameters
    ----------
    backend : str or module or None
        - If a string:
            - "default" - returns the DEFAULT_BACKEND (JAX if available, else NumPy)
            - "np"      - requires NumPy
            - "jnp"     - requires JAX
        - If None or "default", returns the DEFAULT_BACKEND (JAX if available, else NumPy).
        - Otherwise, returns the backend as is.
    
    Returns
    -------
    module
        The backend module.
    """
    if isinstance(backend, str):
        if backend.lower() == "np" or backend.lower() == "numpy":
            if not random and not scipy:
                return np
            if not random and scipy:
                return np, sp
            if seed is not None:
                nrn.seed(seed)
            if scipy:
                return np, nrn, sp
            return np, nrn
        elif backend.lower() == "jnp" or backend.lower() == "jax":
            if _JAX_AVAILABLE:
                if not random and not scipy:
                    return jnp
                if not random and scipy:
                    return jnp, jsp
                if seed is not None:
                    _KEY = jrn.PRNGKey(seed)
                if scipy:
                    return jnp, jrn, jsp
                return jnp, jrn
            else:
                raise ValueError("JAX not available. Please choose a different backend.")
        elif backend.lower() == "default":
            if not random and not scipy:
                return DEFAULT_BACKEND
            elif not random and scipy:
                return DEFAULT_BACKEND, sp if _JAX_AVAILABLE else jsp
            if seed is not None:
                _KEY = nrn.seed(seed) if _JAX_AVAILABLE else None
            if scipy:
                return DEFAULT_BACKEND, (nrn, sp) if not _JAX_AVAILABLE else (DEFAULT_BACKEND, jsp)
            return DEFAULT_BACKEND, jrn if _JAX_AVAILABLE else nrn
        else:
            raise ValueError(f"Unsupported backend string: {backend}")
    elif backend is None:
        return get_backend("default", random=random, seed=seed, scipy=scipy)
    else:
        return get_backend("default", random=random, seed=seed, scipy=scipy)

# ---------------------------------------------------------------------

def maybe_jit(func):
    """
    Decorator that applies JAX JIT compilation if available.
    Marks 'backend' as static so that passing a module or string does not trigger tracing.
    
    Parameters
    ----------
    func : function
        The function to be compiled with JIT.

    Returns
    -------
    function
        The JIT-compiled function (if JAX is available).
    """
    if _JAX_AVAILABLE:
        return jit(func, static_argnames=("backend",))
    else:
        return func

# ---------------------------------------------------------------------