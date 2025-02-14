#general_python/algebra/ran_wrapper.py

'''
This module provides a wrapper for random number generation functions. The wrapper
allows for the use of different backends and provides a common interface for generating
random numbers. The module also provides a helper function to generate random test
matrices and vectors for testing solvers.
'''

import numpy as np
import scipy as sp
import numpy.random as npr
from abc import ABC, abstractmethod
from typing import Union

# -----------------------------------------------------------------------------

from typing import Optional, Callable
from .utils import _JAX_AVAILABLE, DEFAULT_BACKEND, maybe_jit, get_backend as __backend, _KEY

# -----------------------------------------------------------------------------

def initialize(backend = "default", seed = None):
    ''' Initializes the random number generator with a new seed. '''
    module                      = __backend(backend, random=True, seed=seed)
    backend, (rnd_module, key)  = __backend(backend, random=True, seed=seed) if isinstance(module, tuple) else (module, (None, None))
    if key is None:
        key = _KEY
        
    if rnd_module is None:
        if backend is np:
            rnd_module = npr
        else:
            from jax import random as rnd_module
            rnd_module = rnd_module
            
    # initialize the random number generator
    rnd_module.seed(key)
    return rnd_module

###############################################################################
# Random uniform distribution
###############################################################################

def __uniform_np(low, high, size):
    ''' Generate a random uniform array using NumPy. '''
    rng = npr.default_rng()
    return rng.uniform(low=low, high=high, size=size)

@maybe_jit
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

    if backend is np:
        return __uniform_np(minval, maxval, shape).astype(dtype if dtype is not None else backend.float32)
    
    # JAX backend: use the provided key (or default) and default to float32 if dtype is not given.
    if dtype is None:
        dtype = backend.float32
    return __uniform_jax(rnd_module, key if key is not None else _KEY, shape, minval, maxval, dtype)

###############################################################################
# Random normal distribution with specified mean and standard deviation
###############################################################################

def __normal_np(shape, mean=0.0, std=1.0):
    ''' Generate a random normal array using NumPy. '''
    rng = npr.default_rng()
    return rng.normal(loc=mean, scale=std, size=shape)

@maybe_jit
def __normal_jax(rng_module, key, shape, dtype, mean, std):
    ''' Generate a random normal array using JAX. '''
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
    
    if main_module is np:
        return __normal_np(shape, mean, std).astype(dtype if dtype is not None else main_module.float32)
    
    if dtype is None:
        dtype = main_module.float32
    return __normal_jax(rnd_module, key if key is not None else _KEY, shape, dtype, mean, std)

###############################################################################
# Random integers
###############################################################################

def random_integers(low, high, shape, backend="default", seed=None, dtype=None):
    """
    Generate random integers.

    Parameters
    ----------
    low : int
        Lower bound (inclusive).
    high : int
        Upper bound (exclusive).
    shape : tuple
        Shape of the output array.
    backend : str, optional
        Backend specifier.
    seed : int, optional
        Random seed.
    dtype : data-type, optional
        Desired output dtype.

    Returns
    -------
    array
        An array of random integers.
    """
    modules = get_backend(backend, random=True, seed=seed)
    main_module, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    
    if main_module is np:
        arr = rnd_module.randint(low, high, size=shape)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    else:
        if key is None:
            key = DEFAULT_BACKEND_KEY
        if dtype is None:
            dtype = main_module.int32
        return rnd_module.randint(key, shape=shape, minval=low, maxval=high, dtype=dtype)

def random_choice(a, size, replace=True, p=None, backend="default", seed=None):
    """
    Randomly select elements from an array.

    Parameters
    ----------
    a : array-like
        The input array.
    size : int or tuple of ints
        Number of samples to draw.
    replace : bool, optional
        Whether sampling is done with replacement.
    p : array-like, optional
        Probabilities associated with each element in a.
    backend : str, optional
        Backend specifier.
    seed : int, optional
        Random seed.

    Returns
    -------
    array
        An array of randomly chosen elements.

    Raises
    ------
    NotImplementedError
        If probability weights (p) are provided for the JAX backend.
    """
    modules = get_backend(backend, random=True, seed=seed)
    main_module, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    a = main_module.array(a)
    
    if main_module is np:
        return np.random.choice(a, size=size, replace=replace, p=p)
    else:
        if key is None:
            key = DEFAULT_BACKEND_KEY
        if p is not None:
            raise NotImplementedError("JAX random.choice with probabilities is not implemented.")
        # Generate random indices and select entries.
        idx = rnd_module.randint(key, shape=(size,) if isinstance(size, int) else size, minval=0, maxval=a.shape[0])
        return main_module.take(a, idx)
