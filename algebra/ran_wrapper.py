#general_python/algebra/ran_wrapper.py

'''
This module provides a wrapper for random number generation functions. The wrapper
allows for the use of different backends and provides a common interface for generating
random numbers. The module also provides a helper function to generate random test
matrices and vectors for testing solvers.
'''

import numpy as np
import numpy.random as npr
import numba

import scipy as sp
from enum import Enum, unique

# random matrices
try:
    from tenpy.linalg.random_matrix import COE, GUE, GOE, CRE, CUE
except ImportError:
    # Mock random matrix classes for documentation generation
    class MockRandomMatrix:
        def __init__(self, *args, **kwargs):
            pass
    COE = GUE = GOE = CRE = CUE = MockRandomMatrix

# -----------------------------------------------------------------------------

from typing import Union, Tuple, Optional, Callable
from functools import partial
from general_python.algebra.utils import JAX_AVAILABLE, DEFAULT_BACKEND, get_backend

# Initialize JAX_RND_DEFAULT_KEY
JAX_RND_DEFAULT_KEY = None

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import random as random_jp
    
    # Initialize default JAX PRNG key
    JAX_RND_DEFAULT_KEY = random_jp.PRNGKey(42)  # Default seed
else:
    JAX_RND_DEFAULT_KEY = None

###############################################################################
#! Random number generator initialization
###############################################################################

def initialize(backend  : str           = "default",
                seed    : Optional[int] = None):
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
    module = get_backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = module if isinstance(module, tuple) else (module, (None, None))
    if key is None and JAX_AVAILABLE:
        key = JAX_RND_DEFAULT_KEY
    
    # goon
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
    For JAX, updates the module-level JAX_RND_DEFAULT_KEY.
    
    Parameters
    ----------
    seed : int
        The new seed value.
    backend : str, optional
        Backend specifier (default is "default").
    """
    
    global JAX_RND_DEFAULT_KEY
    modules                         = get_backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key)  = modules if isinstance(modules, tuple) else (modules, (None, None))
    if backend == "np" or backend == "numpy" or backend is np:
        np.random.seed(seed)
        if hasattr(npr, "default_rng"):
            # Reset the default_rng for newer NumPy versions
            npr.default_rng(seed)
    else:
        from jax import random as rnd_module
        JAX_RND_DEFAULT_KEY = rnd_module.PRNGKey(seed)

# -----------------------------------------------------------------------------

def default_dtype(backend="default"):
    """
    Get the default data type for the specified backend.
    
    Parameters
    ----------
    backend : str, optional
        Backend specifier (default is "default").
        
    Returns
    -------
    dtype
        The default data type for the specified backend.
    """
    if backend == 'np' or backend == 'numpy' or backend is np:
        return np.float64
    return jnp.float32

# -----------------------------------------------------------------------------

def handle_rng(rng = None, rng_k = None, backend="default", seed: Optional[int] = None):
    """
    Handle the random number generator and key for the specified backend.
    
    Parameters
    ----------
    rng : module, optional
        The random number generator module (default is None).
    rng_k : PRNGKey, optional
        The JAX PRNG key (default is None).
    backend : str, optional
        Backend specifier (default is "default").
    seed : int, optional
        Random seed (default is None).
        
    Returns
    -------
    tuple
        A tuple containing the random number generator module and the PRNG key.
    """
    if rng is None or (backend != 'np' and backend != 'numpy' and backend != np and rng_k is None):
        modules                     = get_backend(backend, random=True, seed=seed)
        backend_mod, (rng, rng_k)   = modules if isinstance(modules, tuple) else (modules, (None, None))
    return rng, rng_k

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

if JAX_AVAILABLE:
    @partial(jax.jit, static_argnames=('minval', 'maxval', 'shape'))
    def _uniform_jax(key, shape, minval, maxval):
        ''' Generate a random uniform array using JAX. '''
        return random_jp.uniform(key, shape=shape, minval=minval, maxval=maxval)
else:
    def _uniform_jax(key, shape, minval, maxval):
        return None
    
def uniform(shape   : Union[tuple, int] =   (1,),
            backend                     =   "default",
            seed    : Optional[int]     =   None,
            minval                      =   0.0,
            maxval                      =   1.0,
            dtype                       =   None,
            rng                         =   None,
            rng_k                       =   None):
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
    
    (rng, rng_k) = handle_rng(rng, rng_k, backend, seed)
    if dtype is None:
        dtype = default_dtype(backend)        
    if backend is np or backend == 'np' or backend == 'numpy':
        return __uniform_np(rng, minval, maxval, shape).astype(dtype)
    if rng_k is None:
        rng_k = JAX_RND_DEFAULT_KEY
    return _uniform_jax(rng_k, shape, minval, maxval).astype(dtype)

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

if JAX_AVAILABLE:
    @jax.jit
    def __normal_jax(key, shape, dtype, mean, std):
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
        return random_jp.normal(key, shape=shape, dtype=dtype) * std + mean
else:
    def __normal_jax(key, shape, dtype, mean, std):
        return None

def normal(shape,
        backend =   "default",
        seed    =   None,
        dtype   =   None,
        mean    =   0.0,
        std     =   1.0,
        rng     =   None,
        rng_k   =   None):
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
    rng, rng_k = handle_rng(rng, rng_k, backend, seed)
    if dtype is None:
        dtype = default_dtype(backend)
        
    if backend is np or backend == 'np' or backend == 'numpy':
        return __normal_np(rng, shape, mean, std).astype(dtype)
    if rng_k is None:
        rng_k = JAX_RND_DEFAULT_KEY
    return __normal_jax(rng_k, shape, dtype, mean, std)

###############################################################################
#! Random exponential distribution
###############################################################################

def __exponential_np(rng, scale, size):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr
    return rng.exponential(scale=scale, size=size)

if JAX_AVAILABLE:
    @jax.jit
    def __exponential_jax(key, shape, scale, dtype):
        return random_jp.exponential(key, shape=shape, dtype=dtype) * scale
else:
    def __exponential_jax(key, shape, scale, dtype):
        return None
    
def exponential(shape,
                backend="default",
                seed: Optional[int] = None,
                scale=1.0, dtype=None, rng=None, rng_k=None):
    """
    Generate a random exponential array.
    """
    (rng, rng_k) = handle_rng(rng, rng_k, backend, seed)
    
    if dtype is None:
        dtype = default_dtype(backend)
        
    if backend is np or backend == 'np' or backend == 'numpy':
        return __exponential_np(rng, scale, shape).astype(dtype)
    if rng_k is None:
        rng_k = JAX_RND_DEFAULT_KEY
    return __exponential_jax(rng, rng_k, shape, scale, dtype)

###############################################################################
#! Random poisson distribution
###############################################################################

def __poisson_np(rng, lam, size):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr
    return rng.poisson(lam=lam, size=size)

if JAX_AVAILABLE:
    @jax.jit
    def __poisson_jax(key, shape, lam, dtype):
        return random_jp.poisson(key, lam=lam, shape=shape, dtype=dtype)
else:
    def __poisson_jax(key, shape, lam, dtype):
        return None
    
def poisson(shape, backend="default",
            seed: Optional[int] = None,
            lam=1.0, dtype=None, rng=None, rng_k=None):
    """
    Generate a random poisson array.
    """
    (rng, rng_k) = handle_rng(None, None, backend, seed)
    if dtype is None:
        dtype = default_dtype(backend)
    if backend is np or backend == 'np' or backend == 'numpy':
        return __poisson_np(rng, lam, shape).astype(dtype)
    if rng_k is None:
        rng_k = JAX_RND_DEFAULT_KEY
    return __poisson_jax(rng, rng_k, shape, lam, dtype)

###############################################################################
#! Random gamma distribution
###############################################################################

def __gamma_np(rng, a, scale, size):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr
    return rng.gamma(shape=a, scale=scale, size=size)

if JAX_AVAILABLE:
    @jax.jit
    def __gamma_jax(key, shape, a, scale, dtype):
        return random_jp.gamma(key, a, shape=shape, dtype=dtype) * scale
else:
    def __gamma_jax(key, shape, a, scale, dtype):
        return None
    
def gamma(shape, backend="default",
        seed: Optional[int] = None,
        a=1.0, scale=1.0, dtype=None, rng=None, rng_k=None):
    """
    Generate a random gamma array.
    """
    (rng, rng_k) = handle_rng(None, None, backend, seed)
    if dtype is None:
        dtype = default_dtype(backend)
    if backend is np or backend == 'np' or backend == 'numpy':
        return __gamma_np(rng, a, scale, shape).astype(dtype)
    if rng_k is None:
        rng_k = JAX_RND_DEFAULT_KEY
    return __gamma_jax(rng, rng_k, shape, a, scale, dtype)
    
###############################################################################
# Random beta distribution
###############################################################################

def _beta_np(rng, a, b, size):
    if rng is None:
        rng = npr.default_rng(None)
    return rng.beta(a, b, size=size)

if JAX_AVAILABLE:
    @jax.jit
    def _beta_jax(key, shape, a, b, dtype):
        try:
            return random_jp.beta(key, a, b, shape=shape, dtype=dtype)
        except AttributeError as e:
            # Fall back to the gamma approach: X ~ Gamma(a,1), Y ~ Gamma(b,1) => X/(X+Y) ~ Beta(a,b)
            key1, key2 = random_jp.split(key)
            x = random_jp.gamma(key1, a, shape=shape, dtype=dtype)
            y = random_jp.gamma(key2, b, shape=shape, dtype=dtype)
            return x / (x + y)
else:
    def _beta_jax(key, shape, a, b, dtype):
        return None
    
def beta(shape, backend="default", seed: Optional[int] = None, a=0.5, b=0.5, dtype=None):
    """
    Generate a random beta array.
    """
    modules = get_backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    if dtype is None:
        dtype = backend_mod.float32
    if backend_mod is np:
        return _beta_np(rnd_module, a, b, shape).astype(dtype)
    if key is None:
        key = JAX_RND_DEFAULT_KEY
    return _beta_jax(key, shape, a, b, dtype)

###############################################################################
#! Random randint distribution
###############################################################################

@numba.njit
def randint_np(low, high, size):
    '''Create a random integer array using NumPy.'''
    # if np.__version__ >= '1.17':
        # if rng is None:
            # rng = npr.default_rng(None)    
            # return rng.integers(low=low, high=high, size=size)
        # return rng.integers(low=low, high=high, size=size)
    # return rng.randint(low=low, high=high, size=size)
    return np.random.randint(low=low, high=high, size=size)

if JAX_AVAILABLE:
    @partial(jax.jit, static_argnames=('shape',))
    def randint_jax(key, shape, low: int, high: int):
        """Efficient random integer generation in JAX."""
        return random_jp.randint(key, shape, minval=low, maxval=high)
else:
    def randint_jax(key, shape, low: int, high: int):
        return None
    
def randint(low, high, shape, backend="default", seed: Optional[int] = None, dtype=None, rng=None, rng_k=None):
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
    (rng, rng_k) = handle_rng(rng, rng_k, backend, seed)
    if dtype is None:
        dtype = default_dtype(backend)
    if backend is np or backend == 'np' or backend == 'numpy':
        return randint_np(low, high, shape).astype(dtype)
    if rng_k is None:
        rng_k = JAX_RND_DEFAULT_KEY
    return randint_jax(rng_k, shape, low, high).astype(dtype)

###############################################################################
#! Random choice function (already provided)
###############################################################################

def choice_np(rng, a, size, replace=True, p=None):
    if rng is None:
        rng = npr.default_rng(None) if hasattr(npr, "default_rng") else npr
    return rng.choice(a, size=size, replace=replace, p=p)

if JAX_AVAILABLE:

    @partial(jax.jit, static_argnames=('shape'))
    def choice_jax(key, a, shape):
        '''
        Randomly select elements from an array using JAX.
        Parameters
        ----------
        rng_module : module
            The JAX random module.
        key : PRNGKey
            The random number generator key.
        a : array-like
            The input array.
        size : int or tuple
            Number of samples to draw.
        backend_mod : module
            The backend module (e.g., jax.numpy).
        Returns
        -------
        '''
        a = jnp.asarray(a)
        return random_jp.choice(key, a, shape=shape, replace=True)
        # Generate random indices and select elements from the array)
        # idx = random_jp.randint(key, shape=(size,) if isinstance(size, int) else size,
                                # minval=0, maxval=a.shape[0])
        # return jnp.take(a, idx)

else:
    def choice_jax(key, a, shape):
        return None

def choice(a    :   'array-like',
        shape   :   Union[Tuple, int],
        replace =   True,
        p       =   None,
        rng     =   None,
        rng_k   =   None,
        backend =   DEFAULT_BACKEND):
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
    (rng, rng_k) = handle_rng(rng, rng_k, backend)
    if backend == 'np' or backend == 'numpy' or backend is np:
        return choice_np(rng, a, shape, replace=replace, p=p)
    if rng_k is None:
        rng_k = JAX_RND_DEFAULT_KEY
    return choice_jax(rng_k, a, shape)

###############################################################################
#! Random shuffle functions
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
    modules = get_backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    
    # Ensure a is an array of the proper type.
    a = backend_mod.array(a)
    
    if backend_mod is np:
        # NumPy: np.random.permutation returns a shuffled copy.
        return rnd_module.permutation(a)
    else:
        # JAX: use jax.random.permutation.
        if key is None:
            key = JAX_RND_DEFAULT_KEY
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
    modules = get_backend(backend, random=True, seed=seed)
    backend_mod, (rnd_module, key) = modules if isinstance(modules, tuple) else (modules, (None, None))
    
    if backend_mod is np:
        return rnd_module.permutation(n)
    else:
        if key is None:
            key = JAX_RND_DEFAULT_KEY
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
        a   = np.random.normal(0, 1, size=shape)
        mat = (a + a.T) / np.sqrt(2)
        # mat = GOE(shape)
    elif typek == RMT.CRE:
        mat = CRE(shape)
    elif typek == RMT.CUE:
        mat = CUE(shape)
    else:
        raise ValueError("Invalid random matrix type.")

    if not ((isinstance(backend, str) and (backend == 'np' or backend == 'numpy')) or backend == np):
        backend = get_backend(backend)
        mat     = backend.array(mat, dtype=dtype)
    return mat


def goe(shape, backend="default", dtype=None):
    ''' Generate a Gaussian Orthogonal Ensemble (GOE) matrix. '''
    return random_matrix(shape, typek=RMT.GOE, backend=backend, dtype=dtype)

def gue(shape, backend="default", dtype=None):
    ''' Generate a Gaussian Unitary Ensemble (GUE) matrix. '''
    return random_matrix(shape, typek=RMT.GUE, backend=backend, dtype=dtype)

def coe(shape, backend="default", dtype=None):
    ''' Generate a Circular Orthogonal Ensemble (COE) matrix. '''
    return random_matrix(shape, typek=RMT.COE, backend=backend, dtype=dtype)

def cue(shape, backend="default", dtype=None):
    ''' Generate a Circular Unitary Ensemble (CUE) matrix. '''
    return random_matrix(shape, typek=RMT.CUE, backend=backend, dtype=dtype)

def cre(shape, backend="default", dtype=None):
    ''' Generate a Circular Real Ensemble (CRE) matrix. '''
    return random_matrix(shape, typek=RMT.CRE, backend=backend, dtype=dtype)

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
    
##############################################################################