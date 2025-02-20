#general_python/algebra/ran_wrapper.py

'''
This module provides a wrapper for random number generation functions. The wrapper
allows for the use of different backends and provides a common interface for generating
random numbers. The module also provides a helper function to generate random test
matrices and vectors for testing solvers.
'''

import numpy as np
import numpy.random as npr
import scipy as sp
from enum import Enum, unique

# random matrices
from tenpy.linalg.random_matrix import COE, GUE, GOE, CRE, CUE

# -----------------------------------------------------------------------------

from typing import Union, Tuple, Optional
import warnings
from jax import jit

# -----------------------------------------------------------------------------

from typing import Optional, Callable
from general_python.algebra.utils import _JAX_AVAILABLE, DEFAULT_BACKEND, get_backend as __backend, _KEY

###############################################################################
#! Random number generator initialization
###############################################################################

def initialize(backend="default", seed: Optional[int] = None):
    ''' 
    Initializes the random number generator with a new seed.
    
    Parameters:
        backend : str (optional)
            The backend to use for random number generation.
        seed : int (optional)
            The seed to use for the random number generator.
            
    Returns:
        A tuple (rnd_module, key) where key is the PRNG key for JAX (or None for NumPy).
    '''
    module = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = module if isinstance(module, tuple) else (module, (None, None))
    if key is None:
        key = _KEY
    if rnd_module is None:
        if backend_mod is np:
            rnd_module = npr
        else:
            from jax import random as rnd_module
            key = rnd_module.PRNGKey(seed)
    return rnd_module, key

# -----------------------------------------------------------------------------

def set_global_seed(seed: int, backend: str = "default"):
    """
    Reinitialize the global random state.
    
    For NumPy, sets the seed using np.random.seed(seed).
    For JAX, updates the module-level _KEY.
    
    Parameters
    ----------
    seed : int
        The new seed value.
    backend : str, optional
        Backend specifier (default is "default").
    """
    global _KEY
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    if backend_mod is np:
        np.random.seed(seed)
    else:
        from jax import random as rnd_module
        _KEY = rnd_module.PRNGKey(seed)

###############################################################################
#! Random uniform distribution
###############################################################################

def __uniform_np(rng, low, high, size):
    ''' 
        Generate a random uniform array using NumPy. 
        Parameters:
            - rng : numpy.random.Generator if NumPy version >= 1.17, else numpy.random
                The random number generator.
            - low : float
                Lower bound of the distribution.
            - high : float
                Upper bound of the distribution.
            - size : tuple
                Shape of the output array.
        Returns:
            - array
                An array of the specified shape drawn from a uniform distribution.
            '''
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr

    return rng.uniform(low=low, high=high, size=size)

@jit
def __uniform_jax(rng_module, key, shape, minval, maxval, dtype):
    ''' Generate a random uniform array using JAX. '''
    return rng_module.uniform(key, shape=shape, minval=minval, maxval=maxval, dtype=dtype)

def uniform(shape: Union[tuple, int], backend="default", seed=None,
            minval=0.0, maxval=1.0, dtype=None):
    """
    Generate a random uniform array.

    Parameters
    ----------
    shape : tuple or int
        Shape of the output array.
    backend : str, optional
        Backend specifier (e.g. "default", "np", "jax").
    seed : int, optional
        Random seed.
    minval : float, optional
        Lower bound of the distribution.
    maxval : float, optional
        Upper bound of the distribution.
    dtype : data-type, optional
        Desired output dtype.

    Returns
    -------
    array
        An array of the specified shape drawn from a uniform distribution.
    """
    # Obtain the main module and its random component
    modules = __backend(backend, random=True, seed=seed)
    backend, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))

    if dtype is None:
        dtype = backend.float32
    if backend is np:
        return __uniform_np(rnd_module, minval, maxval, shape).astype(dtype)
    return __uniform_jax(rnd_module, key if key is not None else _KEY, shape, minval, maxval, dtype)

###############################################################################
#! Random normal distribution with specified mean and standard deviation
###############################################################################

def __normal_np(rng, shape, mean=0.0, std=1.0):
    ''' 
        Generate a random normal array using NumPy. 
        Parameters:
            - rng : numpy.random.Generator if NumPy version >= 1.17, else numpy.random
                The random number generator.    
            - shape : tuple
                Shape of the output array.
            - mean : float (optional)
                Mean of the distribution.
            - std : float (optional)
                Standard deviation of the distribution.
        Returns:
            - array
                An array of the specified shape drawn from a normal distribution.
    '''
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr

    return rng.normal(loc=mean, scale=std, size=shape)

@jit
def __normal_jax(rng_module, key, shape, dtype, mean, std):
    ''' 
        Generate a random normal array using JAX. 
        Parameters:
            - rng_module : module
                The JAX random module.
            - key : PRNGKey
                The random number generator key.
            - shape : tuple
                Shape of the output array.
            - dtype : data-type
                Desired output dtype.
            - mean : float
                Mean of the distribution.
            - std : float
                Standard deviation of the distribution.
        Returns:
            - array
                An array of the specified shape drawn from a normal distribution.        
    '''
    return rng_module.normal(key, shape=shape, dtype=dtype) * std + mean

def normal(shape, backend="default", seed=None, dtype=None, mean=0.0, std=1.0):
    """
    Generate a random normal array.

    Parameters
    ----------
    shape : tuple
        Shape of the output array.
    backend : str, optional
        Backend specifier.
    seed : int, optional
        Random seed.
    dtype : data-type, optional
        Desired output dtype.
    mean : float, optional
        Mean of the distribution.
    std : float, optional
        Standard deviation of the distribution.

    Returns
    -------
    array
        An array of the specified shape drawn from a normal distribution.
    """
    modules = __backend(backend, random=True, seed=seed)
    main_module, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    
    if dtype is None:
        dtype = main_module.float32  
    if main_module is np:
        return __normal_np(rnd_module, shape, mean, std).astype(dtype)
    return __normal_jax(rnd_module, key if key is not None else _KEY, shape, dtype, mean, std)

###############################################################################
#! Random exponential distribution
###############################################################################

def __exponential_np(rng, scale, size):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr

    return rng.exponential(scale=scale, size=size)

@jit
def __exponential_jax(rng_module, key, shape, scale, dtype):
    return rng_module.exponential(key, shape=shape, dtype=dtype) * scale

def exponential(shape, backend="default", seed: Optional[int] = None, scale=1.0, dtype=None):
    """
    Generate a random exponential array.
    """
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    if dtype is None:
        dtype = backend_mod.float32
    if backend_mod is np:
        return __exponential_np(rnd_module, scale, shape).astype(dtype)
    return __exponential_jax(rnd_module, key if key is not None else _KEY, shape, scale, dtype)

###############################################################################
#! Random poisson distribution
###############################################################################

def __poisson_np(rng, lam, size):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr

    return rng.poisson(lam=lam, size=size)

@jit
def __poisson_jax(rng_module, key, shape, lam, dtype):
    return rng_module.poisson(key, lam=lam, shape=shape, dtype=dtype)

def poisson(shape, backend="default", seed: Optional[int] = None, lam=1.0, dtype=None):
    """
    Generate a random poisson array.
    """
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    if dtype is None:
        # Use int32 by default for counts.
        dtype = backend_mod.int32 if backend_mod is not np else np.int32
    if backend_mod is np:
        return __poisson_np(rnd_module, lam, shape).astype(dtype)
    return __poisson_jax(rnd_module, key if key is not None else _KEY, shape, lam, dtype)

###############################################################################
#! Random gamma distribution
###############################################################################

def __gamma_np(rng, a, scale, size):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr
    return rng.gamma(shape=a, scale=scale, size=size)

@jit
def __gamma_jax(rng_module, key, shape, a, scale, dtype):
    return rng_module.gamma(key, a, shape=shape, dtype=dtype) * scale

def gamma(shape, backend="default", seed: Optional[int] = None, a=1.0, scale=1.0, dtype=None):
    """
    Generate a random gamma array.
    """
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    if dtype is None:
        dtype = backend_mod.float32
    if backend_mod is np:
        return __gamma_np(rnd_module, a, scale, shape).astype(dtype)
    return __gamma_jax(rnd_module, key if key is not None else _KEY, shape, a, scale, dtype)

###############################################################################
# Random beta distribution
###############################################################################

def __beta_np(rng, a, b, size):
    if rng is None:
        rng = npr.default_rng(None)
    return rng.beta(a, b, size=size)

@jit
def __beta_jax(rng_module, key, shape, a, b, dtype):
    try:
        return rng_module.beta(key, a, b, shape=shape, dtype=dtype)
    except AttributeError:
        # Fall back to the gamma approach: X ~ Gamma(a,1), Y ~ Gamma(b,1) => X/(X+Y) ~ Beta(a,b)
        from jax import random as jrand
        key1, key2 = jrand.split(key)
        x = rng_module.gamma(key1, a, shape=shape, dtype=dtype)
        y = rng_module.gamma(key2, b, shape=shape, dtype=dtype)
        return x / (x + y)

def beta(shape, backend="default", seed: Optional[int] = None, a=0.5, b=0.5, dtype=None):
    """
    Generate a random beta array.
    """
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    if dtype is None:
        dtype = backend_mod.float32
    if backend_mod is np:
        return __beta_np(rnd_module, a, b, shape).astype(dtype)
    return __beta_jax(rnd_module, key if key is not None else _KEY, shape, a, b, dtype)

###############################################################################
# Random randint distribution
###############################################################################

def __randint_np(rng, low, high, size):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr
    return rng.integers(low=low, high=high, size=size)

@jit
def __randint_jax(rng_module, key, shape, low, high, dtype):
    return rng_module.randint(key, shape=shape, minval=low, maxval=high, dtype=dtype)

def randint(low, high, shape, backend="default", seed: Optional[int] = None, dtype=None):
    """
    Generate a random integer array.
    
    Parameters
    ----------
    low : int
        Lower bound (inclusive).
    high : int
        Upper bound (exclusive).
    shape : tuple or int
        Shape of the output array.
    backend : str, optional
        Backend specifier (e.g., "default", "np", "jax").
    seed : int, optional
        Random seed.
    dtype : data-type, optional
        Desired output dtype.
    
    Returns
    -------
    array
        An array of random integers.
    """
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    if dtype is None:
        dtype = backend_mod.int32
    if backend_mod is np:
        return __randint_np(rnd_module, low, high, shape).astype(dtype)
    return __randint_jax(rnd_module, key if key is not None else _KEY, shape, low, high, dtype)

###############################################################################
# Random choice function (already provided)
###############################################################################

def __choice_np(rng, a, size, replace=True, p=None):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr
    return rng.choice(a, size=size, replace=replace, p=p)

@jit
def __choice_jax(rng_module, key, a, size, backend_mod):
    idx = rng_module.randint(key, shape=(size,) if isinstance(size, int) else size,
                            minval=0, maxval=a.shape[0])
    return backend_mod.take(a, idx)

def choice(a    :   'array-like',
        size    :   Union[Tuple, int],
        replace =   True,
        p       =   None,
        backend =   "default",
        seed    :   Optional[int] = None):
    """
    Randomly select elements from an array.
    
    For NumPy, uses np.random.choice.
    For JAX, uses jax.random.randint to generate indices, then selects using jnp.take.
    
    Parameters
    ----------
    a : array-like
        The input array.
    size : int or tuple
        Number of samples to draw.
    replace : bool, optional
        Whether sampling is with replacement.
    p : array-like, optional
        Probability distribution. (Not implemented for JAX.)
    backend : str, optional
        Backend specifier.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    array
        An array of randomly chosen elements.
    """
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    a = backend_mod.array(a)

    if backend_mod is np:
        return __choice_np(rnd_module, a, size, replace=replace, p=p)
    else:
        if key is None:
            key = _KEY
        if p is not None:
            warnings.warn("JAX does not support probability-weighted choice; ignoring p.")
        return __choice_jax(rnd_module, key, a, size, backend_mod)

###############################################################################
# Random shuffle functions
###############################################################################

def shuffle(a, backend="default", seed: Optional[int] = None):
    """
    Return a shuffled copy of the input array.

    Parameters
    ----------
    a : array-like
        The array to shuffle.
    backend : str, optional
        Backend specifier (e.g. "default", "np", "jax").
    seed : int, optional
        Random seed.

    Returns
    -------
    array
        A shuffled copy of the array.
    """
    # Obtain backend modules (main and random/key)
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    
    # Ensure a is an array of the proper type.
    a = backend_mod.array(a)
    
    if backend_mod is np:
        # NumPy: np.random.permutation returns a shuffled copy.
        return rnd_module.permutation(a)
    else:
        # JAX: use jax.random.permutation.
        if key is None:
            key = _KEY
        return rnd_module.permutation(key, a)

# -----------------------------------------------------------------------------

def shuffle_indices(n, backend="default", seed: Optional[int] = None):
    """
    Return a shuffled array of indices from 0 to n-1.

    Parameters
    ----------
    n : int
        The number of indices.
    backend : str, optional
        Backend specifier (e.g. "default", "np", "jax").
    seed : int, optional
        Random seed.

    Returns
    -------
    array
        An array containing a random permutation of the indices 0,...,n-1.
    """
    modules = __backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    
    if backend_mod is np:
        return rnd_module.permutation(n)
    else:
        if key is None:
            key = _KEY
        return rnd_module.permutation(key, n)

###############################################################################
# Matrices from random distributions
###############################################################################

@unique
class RMT(Enum):
    ''' Random matrix types. '''
    COE = "COE"
    GUE = "GUE"
    GOE = "GOE"
    CRE = "CRE"
    CUE = "CUE"

def random_matrix(shape, typek : RMT, backend="default", dtype=None):
    """
    Generate a random matrix from TenPy's random matrix distributions.

    Parameters
    ----------
    shape : tuple
        Shape of the output matrix.
    backend : str, optional
        Backend specifier.
    dtype : data-type, optional
        Desired output dtype.

    Returns
    -------
    array
        A random matrix of the specified shape.
    """
    if dtype is None:
        dtype = np.float64
    
    mat = None
    
    if typek == RMT.COE:
        mat = COE(shape)
    elif typek == RMT.GUE:
        mat = GUE(shape)
    elif typek == RMT.GOE:
        mat = GOE(shape)
    elif typek == RMT.CRE:
        mat = CRE(shape)
    elif typek == RMT.CUE:
        mat = CUE(shape)
    else:
        raise ValueError("Invalid random matrix type.")

    if not ((isinstance(backend, str) and (backend == 'np' or backend == 'numpy')) or backend == np):
        backend = __backend(backend)
        mat     = backend.array(mat, dtype=dtype)
    return mat

###############################################################################
#! Random vectors
###############################################################################

def random_vector(shape, typek: str, backend="default", dtype = None, separator = ';'):
    """
    Based on the type of random vector, generate a random vector.
    Types of random vectors:
        - "uniform" : Uniformly distributed random vector - 'r;first;last'
        - "normal"  : Normally distributed random vector. - 'n;mean;std'
    Parameters:
        - shape : tuple
            Shape of the output vector.
        - typek : str
            Type of the random vector.
        - backend : str
            Backend specifier.
        - dtype : data-type
            Desired output dtype.
        - separator : str
            Separator for the type string.    
    Returns:
        - array
            A random vector of the specified shape.
    """
    
    def __parse_type(typek):
        ''' Parse the type of random vector. '''
        val = typek.split(separator)
        
        if val[0] == 'r':
            return 'uniform', float(val[1]), float(val[2])
        elif val[0] == 'n':
            return 'normal', float(val[1]), float(val[2])
        else:
            raise ValueError("Invalid random vector type.")
        
    if dtype is None:
        dtype = np.float64
        
    typek, val1, val2 = __parse_type(typek)
    if typek == 'uniform':
        return uniform(shape, backend=backend, minval=val1, maxval=val2, dtype=dtype)
    elif typek == 'normal':
        return normal(shape, backend=backend, mean=val1, std=val2, dtype=dtype)
    else:
        raise ValueError("Invalid random vector type.")