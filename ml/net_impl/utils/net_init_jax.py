'''
file:       : general_python/ml/net_impl/utils/net_init_jax.py
author      : Maksymilian Kliczkowski
email       :
date        : 2025-03-10

'''

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn


# from general_python utilities
from ....ml.net_impl.net_general import GeneralNet

######################################################################
#! HE INITIALIZATION
######################################################################

def complex_he_init(key, shape, dtype=jnp.complex64):
    """
    Custom initializer for complex weights using He initialization.
    Splits the random key to initialize the real and imaginary parts.
    """
    key1, key2  = random.split(key)
    real_init   = nn.initializers.variance_scaling(2.0, mode='fan_in', distribution='truncated_normal')
    imag_init   = nn.initializers.variance_scaling(2.0, mode='fan_in', distribution='truncated_normal')
    real_part   = real_init(key1, shape, jnp.float32)
    imag_part   = imag_init(key2, shape, jnp.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_he_init(key, shape, dtype=jnp.float32):
    """
    Custom initializer for real weights using He initialization.
    """
    he_init = nn.initializers.variance_scaling(2.0, mode='fan_in', distribution='truncated_normal')
    return he_init(key, shape, dtype)

######################################################################
#! UNIFORM INITIALIZATION
######################################################################

def complex_uniform_init(key, shape, dtype=jnp.complex64):
    """
    Custom initializer for complex weights using uniform distribution.
    Splits the random key to initialize the real and imaginary parts.
    """
    key1, key2  = random.split(key)
    real_init   = nn.initializers.uniform()
    imag_init   = nn.initializers.uniform()
    real_part   = real_init(key1, shape, jnp.float32)
    imag_part   = imag_init(key2, shape, jnp.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_uniform_init(key, shape, dtype=jnp.float32):
    """
    Custom initializer for real weights using uniform distribution.
    """
    uniform_init = nn.initializers.uniform()
    return uniform_init(key, shape, dtype)

#######################################################################
#! XAVIER INITIALIZATION
#######################################################################

def complex_xavier_init(key, shape, dtype=jnp.complex64):
    """
    Custom initializer for complex weights using Xavier initialization.
    Splits the random key to initialize the real and imaginary parts.
    """
    key1, key2  = random.split(key)
    real_init   = nn.initializers.xavier_uniform()
    imag_init   = nn.initializers.xavier_uniform()
    real_part   = real_init(key1, shape, jnp.float32)
    imag_part   = imag_init(key2, shape, jnp.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_xavier_init(key, shape, dtype=jnp.float32):
    '''
    Custom initializer for real weights using Xavier initialization.
    '''
    xavier_init = nn.initializers.xavier_uniform()
    return xavier_init(key, shape, dtype)

########################################################################
#! NORMAL INITIALIZATION
########################################################################

def complex_normal_init(key, shape, dtype=jnp.complex64):
    """
    Custom initializer for complex weights using normal distribution.
    Splits the random key to initialize the real and imaginary parts.
    """
    key1, key2  = random.split(key)
    real_init   = nn.initializers.normal()
    imag_init   = nn.initializers.normal()
    real_part   = real_init(key1, shape, jnp.float32)
    imag_part   = imag_init(key2, shape, jnp.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_normal_init(key, shape, dtype=jnp.float32):
    """
    Custom initializer for real weights using normal distribution.
    """
    normal_init = nn.initializers.normal()
    return normal_init(key, shape, dtype)

########################################################################
#! CONSTANT INITIALIZATION
########################################################################

def complex_constant_init(key, shape, dtype=jnp.complex64):
    """
    Custom initializer for complex weights using constant initialization.
    Splits the random key to initialize the real and imaginary parts.
    """
    key1, key2  = random.split(key)
    real_init   = nn.initializers.constant(0.0)
    imag_init   = nn.initializers.constant(0.0)
    real_part   = real_init(key1, shape, jnp.float32)
    imag_part   = imag_init(key2, shape, jnp.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_constant_init(key, shape, dtype=jnp.float32):
    """
    Custom initializer for real weights using constant initialization.
    """
    constant_init = nn.initializers.constant(0.0)
    return constant_init(key, shape, dtype)

########################################################################

def cplx_init(rng, shape, dtype):
    '''
    Custom initializer for complex weights.
    Splits the random key to initialize the real and imaginary parts.
    '''
    rng1, rng2  = jax.random.split(rng)
    unif        = jax.nn.initializers.uniform()
    return unif(rng1, shape, dtype=jnp.float32) + 1.j * unif(rng2, shape, dtype=jnp.float32)

def cplx_variance_scaling(scale, mode, distribution, dtype):
    """Returns a function that initializes complex weights."""
    # Define the actual initializer function signature expected by Flax
    def init(key, shape, param_dtype_ignored=None): # param_dtype is passed by Dense layer
        key_real, key_imag = random.split(key)
        # Get the Flax variance scaling initializer function
        real_init_func = nn.initializers.variance_scaling(scale, mode, distribution)
        imag_init_func = nn.initializers.variance_scaling(scale, mode, distribution)
        # Determine real dtype
        real_dtype = jnp.real(jnp.zeros(1, dtype=dtype)).dtype
        # Call the Flax initializers with the correct shape and REAL dtype
        real_part = real_init_func(key_real, shape, real_dtype)
        imag_part = imag_init_func(key_imag, shape, real_dtype)
        return (real_part + 1j * imag_part).astype(dtype) # Cast to final complex dtype
    return init # Return the initializer FUNCTION

def lecun_normal(dtype):
    """Returns the Lecun normal initializer function."""
    # Directly return the initializer function from Flax
    return nn.initializers.lecun_normal(dtype=dtype)

########################################################################
