'''
This module provides linear algebra functions and utilities.
'''

# Import the required modules

import numpy as np

try:
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax import jit
    _JAX_AVAILABLE = True
except ImportError:
    print("JAX not available. Falling back to NumPy and SciPy.")
    import scipy as sp
    _JAX_AVAILABLE = False

# Global default backend: use jax.numpy if available, else numpy.
DEFAULT_BACKEND = jnp if _JAX_AVAILABLE else np

# ---------------------------------------------------------------------

def get_backend(backend):
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
        if backend.lower() == "np":
            return np
        elif backend.lower() == "jnp":
            if _JAX_AVAILABLE:
                return jnp
            else:
                raise ValueError("JAX is not available.")
        elif backend.lower() == "default":
            return DEFAULT_BACKEND
        else:
            raise ValueError(f"Unsupported backend string: {backend}")
    elif backend is None:
        return DEFAULT_BACKEND
    else:
        return backend

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