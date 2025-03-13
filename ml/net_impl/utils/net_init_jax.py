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
from general_python.ml.net_impl.net_general import GeneralNet

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