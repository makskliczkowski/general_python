"""
file       : general_python/ml/net_impl/utils/net_init_numpy.py
author     : Maksymilian Kliczkowski
email      :
date       : 2025-03-10
"""

import numpy as np

######################################################################
#! HE INITIALIZATION
######################################################################

def complex_he_init(key, shape, dtype=np.complex64):
    """
    Custom initializer for complex weights using He initialization in NumPy.
    Splits the random key to initialize the real and imaginary parts.
    """
    rng1 = np.random.default_rng(key)
    rng2 = np.random.default_rng(key + 1 if key is not None else None)
    # For He initialization (variance scaling), use scale = sqrt(2/fan_in)
    fan_in = shape[0] if len(shape) > 0 else 1
    scale = np.sqrt(2.0 / fan_in)
    real_part = rng1.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32) * scale
    imag_part = rng2.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32) * scale
    return (real_part + 1j * imag_part).astype(dtype)

def real_he_init(key, shape, dtype=np.float32):
    """
    Custom initializer for real weights using He initialization in NumPy.
    """
    rng = np.random.default_rng(key)
    fan_in = shape[0] if len(shape) > 0 else 1
    scale = np.sqrt(2.0 / fan_in)
    return (rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32) * scale).astype(dtype)

######################################################################
#! UNIFORM INITIALIZATION
######################################################################

def complex_uniform_init(key, shape, dtype=np.complex64):
    """
    Custom initializer for complex weights using uniform distribution in NumPy.
    Splits the random key to initialize the real and imaginary parts.
    """
    rng1 = np.random.default_rng(key)
    rng2 = np.random.default_rng(key + 1 if key is not None else None)
    # Define a simple uniform range.
    low, high = -0.05, 0.05
    real_part = rng1.uniform(low=low, high=high, size=shape).astype(np.float32)
    imag_part = rng2.uniform(low=low, high=high, size=shape).astype(np.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_uniform_init(key, shape, dtype=np.float32):
    """
    Custom initializer for real weights using uniform distribution in NumPy.
    """
    rng = np.random.default_rng(key)
    low, high = -0.05, 0.05
    return rng.uniform(low=low, high=high, size=shape).astype(dtype)

######################################################################
#! XAVIER INITIALIZATION
######################################################################

def complex_xavier_init(key, shape, dtype=np.complex64):
    """
    Custom initializer for complex weights using Xavier initialization in NumPy.
    Splits the random key to initialize the real and imaginary parts.
    """
    rng1 = np.random.default_rng(key)
    rng2 = np.random.default_rng(key + 1 if key is not None else None)
    fan_in = shape[0] if len(shape) > 0 else 1
    fan_out = shape[1] if len(shape) > 1 else 1
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    real_part = rng1.uniform(low=-limit, high=limit, size=shape).astype(np.float32)
    imag_part = rng2.uniform(low=-limit, high=limit, size=shape).astype(np.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_xavier_init(key, shape, dtype=np.float32):
    """
    Custom initializer for real weights using Xavier initialization in NumPy.
    """
    rng = np.random.default_rng(key)
    fan_in = shape[0] if len(shape) > 0 else 1
    fan_out = shape[1] if len(shape) > 1 else 1
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(low=-limit, high=limit, size=shape).astype(dtype)

######################################################################
#! NORMAL INITIALIZATION
######################################################################

def complex_normal_init(key, shape, dtype=np.complex64):
    """
    Custom initializer for complex weights using normal distribution in NumPy.
    Splits the random key to initialize the real and imaginary parts.
    """
    rng1 = np.random.default_rng(key)
    rng2 = np.random.default_rng(key + 1 if key is not None else None)
    real_part = rng1.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)
    imag_part = rng2.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_normal_init(key, shape, dtype=np.float32):
    """
    Custom initializer for real weights using normal distribution in NumPy.
    """
    rng = np.random.default_rng(key)
    return rng.normal(loc=0.0, scale=1.0, size=shape).astype(dtype)

######################################################################
#! CONSTANT INITIALIZATION
######################################################################

def complex_constant_init(key, shape, dtype=np.complex64):
    """
    Custom initializer for complex weights using constant initialization in NumPy.
    Splits the random key to initialize the real and imaginary parts.
    Returns an array of zeros.
    """
    real_part = np.zeros(shape, dtype=np.float32)
    imag_part = np.zeros(shape, dtype=np.float32)
    return (real_part + 1j * imag_part).astype(dtype)

def real_constant_init(key, shape, dtype=np.float32):
    """
    Custom initializer for real weights using constant initialization in NumPy.
    Returns an array of zeros.
    """
    return np.zeros(shape, dtype=dtype)

######################################################################