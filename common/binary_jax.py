

import time
from typing import List, Optional
from general_python.algebra.utils import _JAX_AVAILABLE, DEFAULT_INT_TYPE, DEFAULT_NP_FLOAT_TYPE

BACKEND_REPR       = 0.5
BACKEND_DEF_SPIN   = True

if _JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from general_python.algebra.utils import DEFAULT_JP_INT_TYPE, DEFAULT_JP_FLOAT_TYPE

    def reverse_byte(b : int):
        """
        Reverse the bits in a single byte (8 bits).

        Args:
            b (int): The byte to reverse.

        Returns:
            int: The byte with the bits reversed.
        """
        result = 0                          # Initialize the result.
        for i in range(8):                  # Loop over the bits in the byte.
            if b & (1 << i):                # Check if the i-th bit is set.
                result |= 1 << (7 - i)      # Set the corresponding bit in the result. 
        return result

    # lookup tables
    lookup_reversal_jax     = jnp.array([(1 << i) for i in range(63)], dtype = DEFAULT_JP_INT_TYPE)
    lookup_binary_power_jax = jnp.array([reverse_byte(i) for i in range(256)], dtype = DEFAULT_JP_INT_TYPE)

    # --------------------------------------------------------------------------------------------------
    #! Vector based functions - for extracting bits from a vector
    # --------------------------------------------------------------------------------------------------

    @jax.jit
    def check_int_traced_jax(n, k):
        """
        Check the i-th bit in an integer through JAX tracing.
        Parameters:
            n (array-like)  : The state to check.
            k (int)         : The index of the bit to check.
        Returns:
            bool: True if the k-th bit is set, False otherwise.
        """
        return jnp.bitwise_and(n, jnp.left_shift(1, k)) > 0

    @jax.jit
    def check_arr_jax(n, k : int):
        """
        Checks if the k-th element of the input array `n` is greater than 0.

        Args:
            n (array-like): The input array to check.
            k (int): The index of the element to check in the array `n`.

        Returns:
            bool: True if the k-th element of `n` is greater than 0, False otherwise.
        """
        return n[k] > 0
    
        k           = jnp.asarray(k)
        start_index = k.reshape(1,)
        sliced_val  = jax.lax.dynamic_slice(n, start_index, (1,))
        return sliced_val[0] > 0
    
    @jax.jit
    def int2base_jax(n              : int,
                    size            : int,
                    dtype                   = DEFAULT_JP_INT_TYPE,
                    value_true      : float = BACKEND_REPR,
                    value_false     : float = -BACKEND_REPR):
        """
        Convert an integer to a binary (or spin) representation using JAX.
        Args:
            n (int)             : The integer to convert.
            size (int)          : The number of bits in the representation.
            dtype               : The desired dtype for the output jax array.
            value_true          : The value to use for spin representation.
            value_false         : The value to use for spin representation.
        Returns:
            jnp.ndarray: The binary (or spin) representation of the integer.
        """
        bits = jnp.zeros(size, dtype=dtype)
        # Iterate from the most significant bit to the least.
        for pos in range(size - 1, -1, -1):
            bit     = check_int_traced_jax(n, pos)
            bits    = bits.at[size - 1 - pos].set(value_true if bit else value_false)
        return bits

    # --------------------------------------------------------------------------------------------------
    #! Flip bits in an integer or a vector
    # --------------------------------------------------------------------------------------------------

    @jax.jit
    def flip_all_array_spin(n : 'array-like'):
        """
        Flip all bits in a representation of a state.
        - This is a helper function for flip_all.
        Parameters:
            n (array-like)  : The state to flip.
        Note:
            The function is implemented for both integers and NumPy arrays (np or jnp).
        Returns:
            array-like      : The flipped state.
        """
        # arrays are assumed to be NumPy or JAX arrays
        return -n

    # --------------------------------------------------------------------------------------------------
    #! SINGLE BIT FLIP
    # --------------------------------------------------------------------------------------------------

    @jax.jit
    def flip_array_jax_spin(n : 'array-like', k : int):
        """
        Flip a single bit in a representation of a state.
        - This is a helper function for flip.
        Parameters:
            n (array-like)  : The state to flip.
            k (int)         : The index of the bit to flip.
        Returns:
            array-like      : The flipped state.
        """
        n = n.at[k].set(-n[k])
        return n

    @jax.jit
    def flip_array_jax_nspin(n : 'array-like', k : int):
        """
        Flip a single bit in a representation of a state.
        - This is a helper function for flip.
        Parameters:
            n (array-like)  : The state to flip.
            k (int)         : The index of the bit to flip.
        Returns:
            array-like      : The flipped state.    
        """
        update  =   (n[k] + 1) % 2
        n       =   n.at[k].set(update)
        return n
    
    def flip_array_jax(n : 'array-like', k : int,
                    spin : bool = BACKEND_DEF_SPIN):
        """
        Flip a single bit in a JAX array.
        Parameters:
            n (array-like)  : The state to flip.
            k (int)         : The index of the bit to flip.
            spin (bool)     : A flag to indicate whether to use spin values.
        Returns:
            array-like      : The flipped state.
        """
        if spin:
            return flip_array_jax_spin(n, k)
        return flip_array_jax_nspin(n, k)
    
    # Multi-index versions using vectorized operations
    @jax.jit
    def flip_array_jax_spin_multi(n: 'array-like', ks: 'array-like'):
        """
        Flip multiple spins in a JAX array (spin representation).
        Uses advanced indexing for vectorized updates.
        Parameters:
            n (array-like)      : The state to flip.
            ks (array-like)     : The indices of the bits to flip.
        Returns:
            array-like          : The flipped state.
        """
        return n.at[ks].set(-n[ks])

    @jax.jit
    def flip_array_jax_nspin_multi(n: 'array-like', ks: 'array-like'):
        """
        Flip multiple bits in a JAX array (binary representation).
        Uses advanced indexing for vectorized updates.
        Parameters:
            n (array-like)      : The state to flip.
            ks (array-like)     : The indices of the bits to flip.
        Returns:
            array-like          : The flipped state.
        """
        updates = (n[ks] + 1) % 2
        return n.at[ks].set(updates)
    
    def flip_array_jax_multi(n: 'array-like', ks: 'array-like',
                            spin: bool = BACKEND_DEF_SPIN):
        """
        Flip multiple bits in a JAX array.
        Parameters:
            n (array-like)      : The state to flip.
            ks (array-like)     : The indices of the bits to flip.
            spin (bool)         : A flag to indicate whether to use spin values.
        Returns:
            array-like          : The flipped state.
        """
        if spin:
            return flip_array_jax_spin_multi(n, ks)
        return flip_array_jax_nspin_multi(n, ks)
    
    @jax.jit
    def flip_int_traced_jax(n : int, k : int):
        '''
        Internal helper function for flipping a single bit in an integer through JAX tracing.
        Parameters:
            n (array-like)  : The state to flip.
            k (int)         : The index of the bit to flip.
        Returns:
            array-like      : The flipped state.
        '''
        return jnp.where(check_int_traced_jax(n, k), n - lookup_binary_power_jax[k], n + lookup_binary_power_jax[k])
    
    @jax.jit
    def flip_int_traced_jax_multi(n: int, ks: 'array-like'):
        """
        Flip multiple bits in an integer representation using JAX.
        The function uses vectorized operations via vmap.
        """
        ks_arr      = jnp.array(ks)
        # Vectorized check for each index k.
        conds       = jax.vmap(lambda k: check_int_traced_jax(n, k))(ks_arr)
        lookup_vals = lookup_binary_power_jax[ks_arr]
        # Compute the update (delta) for each bit.
        deltas      = jnp.where(conds, -lookup_vals, lookup_vals)
        return n + jnp.sum(deltas)

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