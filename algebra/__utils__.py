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

# ---------------------------------------------------------------------

# Global default backend: use jax.numpy if available, else numpy.
DEFAULT_BACKEND = jnp if _JAX_AVAILABLE else np

# ---------------------------------------------------------------------

def get_backend(backend, random=False, seed=None, scipy=False):
    """
    Return backend modules based on the provided specifier.
    
    Parameters
    ----------
    backend : str or module or None
        Backend specifier. If a string:
            - "default"         : Uses DEFAULT_BACKEND (JAX if available, else NumPy).
            - "np" or "numpy"   : Returns the NumPy backend.
            - "jnp" or "jax"    : Returns the JAX backend (if available).
        If None or not a string, "default" is assumed.
    random : bool, optional
        If True, include the random module. For JAX, also return a PRNG key (if seed is provided).
    seed : int, optional
        If provided, sets the seed for the random module.
    scipy : bool, optional
        If True, also return the associated SciPy module.
        
    Returns
    -------
    module or tuple
        If neither random nor scipy is requested, returns the main backend module.
        Otherwise, returns a tuple containing:
            (main_module, random_module, scipy_module)
        For the JAX backend, the "random_module" entry is itself a tuple: (jax.random, key) else (numpy.random, None).
        where key is a PRNGKey (or None if seed is not provided).
    """
    # Normalize the backend specifier.
    if isinstance(backend, str):
        b_str = backend.lower()
    else:
        b_str = "default"

    # Use default backend if "default" is specified.
    if b_str == "default":
        b_str = "jax" if _JAX_AVAILABLE else "np"

    # Handle NumPy backend.
    if b_str in ("np", "numpy"):
        main_module     = np
        rnd_module      = np.random if random else None
        scipy_module    = sp if scipy else None
        if seed is not None:
            np.random.seed(seed)
        ret             = [main_module]
        if random:
            ret.append((rnd_module, None))
        if scipy:
            ret.append(scipy_module)
        return tuple(ret) if len(ret) > 1 else main_module

    # Handle JAX backend.
    elif b_str in ("jnp", "jax"):
        if not _JAX_AVAILABLE:
            raise ValueError("JAX not available. Please choose a different backend.")
        main_module     =   jnp
        rnd_module      =   jrn if random else None
        scipy_module    =   jsp if scipy else None
        key             =   jrn.PRNGKey(seed) if (random and seed is not None) else None
        # For JAX, if random is requested, return a tuple (rnd_module, key)
        ret = [main_module]
        if random:
            ret.append((rnd_module, key))
        if scipy:
            ret.append(scipy_module)
        return tuple(ret) if len(ret) > 1 else main_module
    else:
        raise ValueError(f"Unsupported backend string: {backend}")

# ---------------------------------------------------------------------

def maybe_jit(func):
    """
    Decorator that applies JAX JIT compilation only when the backend is JAX.
    If the function is called with backend set to 'np' (or the NumPy module), then
    the original (non-jitted) function is used.
    
    The decorator assumes that the decorated function has a keyword argument
    called 'backend' (which is also marked as static during JIT compilation).
    """
    if not _JAX_AVAILABLE:
        # If JAX isn’t available, simply return the original function.
        return func

    # JIT compile the function with the backend as a static argument.
    jitted_func = jit(func, static_argnames=("backend",))
    
    # Create a wrapper whenever the function is taken as input.
    def wrapper(*args, **kwargs):
        # Look for the backend keyword argument.
        backend = kwargs.get("backend", None)
        if backend is None:
            return jitted_func(*args, **kwargs) # If no backend is provided, we assume the default is JAX.
        
        if isinstance(backend, str):            # If the backend is provided as a string, check its value.
            if backend.lower() in ("np", "numpy"):
                return func(*args, **kwargs)
            else:
                return jitted_func(*args, **kwargs)
        
        if backend is np:                       # If the backend is given as a module, check if it is numpy.
            return func(*args, **kwargs)
        else:
            return jitted_func(*args, **kwargs)
    
    return wrapper

# ---------------------------------------------------------------------