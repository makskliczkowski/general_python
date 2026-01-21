'''
Binary manipulation functions using JAX.
Provides functions to convert integers to binary/spin representations,
and to flip bits in integers or arrays using JAX for performance.

-----------
File        : general_python/common/embedded/binary_jax.py
description : JAX-based binary manipulation utilities.
author      : Maksymilian Kliczkowski
email       : maxgrom97@gmail.com
version     : 1.0.0
'''

import os
from typing                     import Optional, Union, TYPE_CHECKING

try:
    import jax
    import jax.numpy            as jnp
    from functools              import partial
    DEFAULT_JP_INT_TYPE         = jnp.int64
    DEFAULT_JP_FLOAT_TYPE       = jnp.float64
    BACKEND_REPR                = float(os.environ.get("PY_BACKEND_REPR", "0.5"))
    BACKEND_DEF_SPIN            = bool(int(os.environ.get("PY_BACKEND_DEF_SPIN", "1")))
    JAX_AVAILABLE               = True
except ImportError:
    jax                         = None
    jnp                         = None
    partial                     = None
    JAX_AVAILABLE               = False

# --------------------------------------------------------------------------------------------------

if JAX_AVAILABLE:

    def reverse_byte(b : int):
        """
        Reverse the bits in a single byte (8 bits).
        """
        result = 0
        for i in range(8):
            if b & (1 << i):
                result |= 1 << (7 - i)
        return result

    # lookup tables - created once
    lookup_reversal_jax     = jnp.array([(1 << i) for i in range(63)], dtype = DEFAULT_JP_INT_TYPE)
    lookup_binary_power_jax = jnp.array([reverse_byte(i) for i in range(256)], dtype = DEFAULT_JP_INT_TYPE)

    # --------------------------------------------------------------------------------------------------
    #! Vector based functions - for extracting bits from a vector
    # --------------------------------------------------------------------------------------------------

    @jax.jit
    def check_int_traced_jax(n, k):
        """
        Check the i-th bit in an integer through JAX tracing.
        """
        return jnp.bitwise_and(n, jnp.left_shift(1, k)) > 0

    @jax.jit
    def check_arr_jax(n, k : int):
        """
        Checks if the k-th element of the input array `n` is greater than 0.
        """
        return n[k] > 0
    
    @partial(jax.jit, static_argnames=('size', 'dtype', 'value_true', 'value_false'))
    def int2base_jax(n              : int,
                    size            : int,
                    dtype                   = DEFAULT_JP_INT_TYPE,
                    value_true      : float = BACKEND_REPR,
                    value_false     : float = -BACKEND_REPR):
        """
        Convert an integer to a binary (or spin) representation using JAX.
        Optimized vectorized implementation.
        
        Args:
            n (int)             : The integer to convert.
            size (int)          : The number of bits in the representation.
            dtype               : The desired dtype for the output jax array.
            value_true          : The value to use for spin representation.
            value_false         : The value to use for spin representation.
        Returns:
            jnp.ndarray: The binary (or spin) representation of the integer.
        """
        # Vectorized bit extraction
        # idx: [size-1, size-2, ..., 0]
        idx     = jnp.arange(size - 1, -1, -1)
        # powers: [2^(size-1), ..., 1]
        powers  = jnp.left_shift(1, idx)
        
        # Check bits: (n & powers) != 0
        # This broadcasts n against the vector of powers
        bits    = jnp.bitwise_and(n, powers) != 0
        
        # Map to values
        return jnp.where(bits, value_true, value_false).astype(dtype)

    # --------------------------------------------------------------------------------------------------
    #! Flip bits in an integer or a vector
    # --------------------------------------------------------------------------------------------------

    @jax.jit
    def flip_all_array_spin(n : 'Array'):
        """
        Flip all bits in a representation of a state (Spin representation).
        """
        return -n

    # --------------------------------------------------------------------------------------------------
    #! Single bit flip
    # --------------------------------------------------------------------------------------------------

    @jax.jit
    def flip_array_jax_spin(n : 'Array', k : int):
        """
        Flip a single bit in a representation of a state (Spin representation).
        """
        return n.at[k].set(-n[k])

    @partial(jax.jit, static_argnames=('spin_value',))
    def flip_array_jax_nspin(n : 'Array', k : int, spin_value: float = 1.0):
        """
        Flip a single bit in a representation of a state (Binary 0/1 or 0/v representation).
        Uses arithmetic flip: new_val = spin_value - old_val.
        Assumes values are exactly 0 or spin_value.
        """
        return n.at[k].set(spin_value - n[k])
    
    def flip_array_jax(n : 'Array', k : int, spin : bool = BACKEND_DEF_SPIN, spin_value: float = BACKEND_REPR):
        """
        Flip a single bit in a JAX array.
        """
        if spin:
            return flip_array_jax_spin(n, k)
        return flip_array_jax_nspin(n, k, spin_value=spin_value)
    
    # Multi-index versions using vectorized operations
    
    @jax.jit
    def flip_array_jax_spin_multi(n: 'Array', ks: 'Array'):
        """
        Flip multiple spins in a JAX array (spin representation).
        """
        return n.at[ks].set(-n[ks])

    @partial(jax.jit, static_argnames=('spin_value',))
    def flip_array_jax_nspin_multi(n: 'Array', ks: 'Array', spin_value: float = 1.0):
        """
        Flip multiple bits in a JAX array (binary representation).
        """
        return n.at[ks].set(spin_value - n[ks])
    
    def flip_array_jax_multi(n: 'Array', ks: 'Array', spin: bool = BACKEND_DEF_SPIN, spin_value: float = BACKEND_REPR):
        """
        Flip multiple bits in a JAX array.
        """
        if spin:
            return flip_array_jax_spin_multi(n, ks)
        return flip_array_jax_nspin_multi(n, ks, spin_value=spin_value)
    
    @jax.jit
    def flip_int_traced_jax(n : int, k : int) -> jnp.ndarray:
        '''
        Internal helper function for flipping a single bit in an integer through JAX tracing.
        '''
        # If bit k is set (1), subtract power to make it 0.
        # If bit k is unset (0), add power to make it 1.
        # check_int_traced_jax returns boolean.
        # We need dynamic behavior for JIT.
        power = jnp.left_shift(1, k)
        # return n ^ power # XOR is the canonical flip for integers
        return jnp.bitwise_xor(n, power)
    
    @jax.jit
    def flip_int_traced_jax_multi(n: int, ks: 'Array') -> jnp.ndarray:
        """
        Flip multiple bits in an integer representation using JAX.
        """
        # Calculate mask of all bits to flip
        # 1 << ks
        powers  = jnp.left_shift(1, ks)
        # Safe way: reduce with bitwise_or over the powers
        mask    = jnp.sum(powers).astype(n.dtype) # sum works if no overlaps. bitwise_or reduction is safer.
        
        # Robust way using reduce:
        def xor_scan(carry, x):
            return jnp.bitwise_xor(carry, x), None
        
        final_n, _ = jax.lax.scan(xor_scan, n, powers)
        return final_n

else:
    flip_all_array_spin         = None
    flip_array_jax              = None
    flip_array_jax_multi        = None
    flip_int_traced_jax         = None
    flip_int_traced_jax_multi   = None
    check_arr_jax               = None
    int2base_jax                = None
    check_int_traced_jax        = None
    reverse_byte                = None

# --------------------------------------------------------------------------------------------------
#! End of binary_jax.py
# --------------------------------------------------------------------------------------------------