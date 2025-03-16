#!/usr/bin/env python3

"""
binary.py

This module implements binary - manipulation routines that allow you to work with binary representations of integers 
or vectors.

It includes functions to:
- check if a number is a power of 2,
- check a given bit in an integer or an indexable vector,
- convert an integer to a base representation (spin: ±value or binary 0/1),
- convert a base representation back to an integer or a binary string or a vector of values,
- flip bits (all at once or a single bit),
- reverse the bits of a 64 - bit integer,
- extract bits (by ordinal position or via a mask),
- and prepare a bit mask from a list of positions.

Functions are written so that they work with plain Python 
integers as well as with NumPy or JAX arrays.
You can choose the backend (np or jnp) by passing the corresponding module.
"""

import time
from typing import List, Optional
import numpy as np
import numba

from general_python.algebra.utils import _JAX_AVAILABLE, DEFAULT_INT_TYPE, DEFAULT_NP_FLOAT_TYPE
from general_python.algebra.utils import get_backend, maybe_jit, is_traced_jax, DEFAULT_NP_INT_TYPE, JIT
from general_python.common.tests import GeneralAlgebraicTest

BACKEND_REPR       = 0.5
BACKEND_DEF_SPIN   = True

if _JAX_AVAILABLE:
    import jax.numpy as jnp
    import jax
    from general_python.algebra.utils import DEFAULT_JP_INT_TYPE, DEFAULT_JP_FLOAT_TYPE

####################################################################################################
#! Global functions
####################################################################################################

def set_global_defaults(repr_value : float, spin : bool):
    """
    Set the global defaults for the binary module.

    Args:
        repr_value (float): The spin value to use.
        spin (bool): A flag to indicate whether to use spin values.
    """
    global BACKEND_REPR, BACKEND_DEF_SPIN
    BACKEND_REPR       = repr_value
    BACKEND_DEF_SPIN   = spin

####################################################################################################
#! Searching functions
####################################################################################################

__BAD_BINARY_SEARCH_STATE = -1

@maybe_jit
def _binary_search_backend(arr, l_point, r_point, elem, backend: str = 'default') -> int:
    '''
    Perform a binary search for 'elem' in the sorted container 'arr'
    between indices l_point and r_point (inclusive).
    
    If tol is provided (i.e. not None), equality is determined approximately
    within the given tolerance (useful for floating point numbers).
    
    Works with Python lists, NumPy arrays, or JAX arrays (via the appropriate backend).
    
    Returns:
        The index of 'elem' if found; otherwise, returns -1.
    '''
    # Try to use the backend's searchsorted if possible.
    backend_mod = get_backend(backend)
    
    # If the left or right points are not the first or last indices, respectively,
    subarr      = arr[l_point: r_point+1]
        
    # searchsorted returns the insertion index.
    idx         = int(backend_mod.searchsorted(subarr, elem))
    idx         = idx + l_point
    return int(idx)

@numba.njit
def binary_search_numpy(arr, l_point, r_point, elem) -> int:
    '''
    Perform a binary search for 'elem' in the sorted NumPy array 'arr'.
    '''
    if l_point < 0 or r_point >= len(arr):
        return __BAD_BINARY_SEARCH_STATE
    if l_point > r_point:
        return __BAD_BINARY_SEARCH_STATE
    middle = l_point + (r_point - l_point) // 2
    if arr[middle] == elem:
        return middle
    if arr[middle] < elem:
        return binary_search_numpy(arr, middle + 1, r_point, elem)
    return binary_search_numpy(arr, l_point, middle - 1, elem)

def _binary_search_list_notol(arr, l_point, r_point, elem) -> int:
    '''
    Perform a binary search for 'elem' in the sorted list 'arr' (exact equality).
    '''
    if l_point < 0 or r_point >= len(arr):
        return __BAD_BINARY_SEARCH_STATE
    if l_point > r_point:
        return __BAD_BINARY_SEARCH_STATE
    middle = l_point + (r_point - l_point) // 2
    if arr[middle] == elem:
        return middle
    if arr[middle] < elem:
        return _binary_search_list(arr, middle + 1, r_point, elem)
    return _binary_search_list(arr, l_point, middle - 1, elem)

def _binary_search_list(arr, l_point, r_point, elem, tol: Optional[float] = None) -> int:
    '''
    Perform a binary search for 'elem' in the sorted list 'arr'.
    If tol is provided, approximate equality is used.
    '''
    if l_point < 0 or r_point >= len(arr):
        return __BAD_BINARY_SEARCH_STATE
    if l_point > r_point:
        return __BAD_BINARY_SEARCH_STATE
    middle = l_point + (r_point - l_point) // 2
    if tol is not None:
        if abs(arr[middle] - elem) <= tol:
            return middle
        elif arr[middle] < elem:
            return _binary_search_list(arr, middle + 1, r_point, elem, tol)
        return _binary_search_list(arr, l_point, middle - 1, elem, tol)
    return _binary_search_list_notol(arr, l_point, r_point, elem)

def binary_search(arr, l_point, r_point, elem,
                tol: Optional[float] = None, backend: str = 'default') -> int:
    """
    Perform a binary search for 'elem' in the sorted container 'arr'
    between indices l_point and r_point (inclusive).
    
    If tol is provided, approximate equality is used (useful for floats).
    
    Works for Python lists as well as for NumPy/JAX arrays.
    Parameters:
        arr (array-like)        : The sorted array or list to search.
        l_point (int)           : The left index of the search range.
        r_point (int)           : The right index of the search range.
        elem (float)            : The element to search for.
        tol (float, optional)   : The tolerance for approximate equality (default is None).
        backend (str)           : The backend to use ('default', 'numpy', or 'jax').
    Returns:
        The index of 'elem' if found; otherwise, returns -1.
    """
    if hasattr(arr, "shape"):
        idx = _binary_search_backend(arr, l_point, r_point, elem, backend)
        if idx < len(arr) and idx >= 0:
            if tol is not None and abs(arr[idx] - elem) <= tol:
                return idx
            if arr[idx] == elem:
                return idx
        return __BAD_BINARY_SEARCH_STATE
    else:
        if tol is not None:
            return _binary_search_list(arr, l_point, r_point, elem, tol)
        return _binary_search_list_notol(arr, l_point, r_point, elem)

####################################################################################################
#! Integer based functions
####################################################################################################

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

# --------------------------------------------------------------------------------------------------

def binary_power(n : int):
    """
    Computes the power of two by moving a single bit to the left.
    
    Args:
        n (int): The power of two to compute.        
    
    Returns:
        int: The power of two.
    """
    return 1 << n

# --------------------------------------------------------------------------------------------------
#! Mask based functions - for extracting bits from a number
# --------------------------------------------------------------------------------------------------

def extract_ord_left(n: int, size: int, size_l: int) -> int:
    '''
    Extract the leftmost bits of a number.
    
    Args:
        n (int)     : The number to extract the bits from. 
        size (int)  : The number of bits in the number.
        size_l (int): The number of bits to extract.   
    Example:
        n = 0b101101, mask = 0b1011 will extract the bits at positions corresponding to left 4 bits.
    '''
    return n >> (size - size_l)

# --------------------------------------------------------------------------------------------------

def extract_ord_right(n: int, size_r: int) -> int:
    '''
    Extract the rightmost bits of a number.
    
    Args:
        n (int)     : The number to extract the bits from. 
        size (int)  : The number of bits in the number.
        size_r (int): The number of bits to extract.   
    Example:
        n = 0b101101, mask = 0b1101 will extract the bits at positions corresponding to right 4 bits.
    '''
    return n & ((1 << size_r) - 1)

# a global static variable to store the reversed bytes from 0 to 255
lookup_binary_power = np.array([binary_power(i) for i in range(63)], dtype=DEFAULT_NP_INT_TYPE)
lookup_reversal     = np.array([reverse_byte(i) for i in range(256)], dtype = DEFAULT_NP_INT_TYPE)

if _JAX_AVAILABLE:
    lookup_reversal_jax     = jnp.array(lookup_reversal, dtype = DEFAULT_JP_INT_TYPE)
    lookup_binary_power_jax = jnp.array(lookup_binary_power, dtype = DEFAULT_JP_INT_TYPE)

# --------------------------------------------------------------------------------------------------
#! Extract bits using a mask
# --------------------------------------------------------------------------------------------------

def extract(n : int, mask : int):
    """
    Extract bits from a number using a mask.
    Extracts the bits from the number n according to the mask. Then it creates an integer out of it.
	The mask gives the positions of the bits to be extracted. The bits are extracted from the right to the left.
	This looks at the integer state and extracts the bits according to the mask. After that, it creates a new integer
	out of the extracted bits (2^0 * bit_0 + 2^1 * bit_1 + ... + 2^(size - 1) * bit_(size - 1)).

    ! The mask should be a number with 1s at the positions of the bits to be extracted.
    Remember, it does not simply apply the mask to the number. It extracts the bits according to the mask.
    This means that it treats the new bits in a correct, extracted order and then gives the result.
    Example:
        n = 0b101101, mask = 0b001001 will extract 0b11 from the number - exactly the 0th and 3rd bits and
        then create a new integer out of it -> 0b11 = 3.
    
    Args:
        n (int): The number to extract the bits from.
        mask (int): The mask to use for extraction.

    Returns:
        int: The extracted bits.
    """
    res         = 0     # Initialize the result.
    pos_mask    = 0     # Initialize the position in the mask.
    pos_res     = 0     # Initialize the position in the result.
    
    while mask:
        if mask & 1:
            res     |= ((n >> pos_mask) & 1) << pos_res
            pos_res += 1
        mask        >>= 1   # Shift the mask to the right.
        pos_mask    += 1    # Increment the position in the mask.
    return res

# --------------------------------------------------------------------------------------------------
#! Prepare a mask from a list of positions
# --------------------------------------------------------------------------------------------------

def prepare_mask(positions : List[int], inv : bool = False, size : int = 0):
    """
    Prepare a bit mask from a list of positions.

    Args:
        positions (List[int])   : The positions of the bits to set in the mask. Counting 
        from the left as 0, 1, 2, ..., n in binary - therefore the leftmost bit is 0.
        inv (bool)              : A flag to indicate whether to invert the mask - changes
                                the order of the bits in the mask.
        size (int)              : The size of the mask.

    Returns:
        int: The mask with the bits set at the specified positions.
    """
    if not inv:
        mask = 0    # Initialize the mask.
        for pos in positions:
            mask |= 1 << (size - 1 - pos)
        return mask
    # Invert the mask.
    mask = 0
    for pos in positions:
        mask |= 1 << pos
    return mask

# --------------------------------------------------------------------------------------------------
#! Check the power of two
# --------------------------------------------------------------------------------------------------

def is_pow_of_two(n : int):
    """
    Check if a number is a power of two. 

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the number is a power of two, False otherwise.
    """
    return n and not (n & (n - 1))

# --------------------------------------------------------------------------------------------------
#! Vector based functions - for extracting bits from a vector
# --------------------------------------------------------------------------------------------------

if _JAX_AVAILABLE:
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

@numba.njit
def check_int(n, k):
    """
    Check the i-th bit in an integer.
    """
    return n & (1 << k)

@numba.njit
def check_arr_np(n, k : int):
    return n[k] > 0

if _JAX_AVAILABLE:
    @jax.jit
    def check_arr_jax(n, k : int):
        return n[k] > 0

def check(n, k : int):
    """
    Check the i-th bit in an integer.

    Args:
        n (int): The integer to check.
        k (int): The index of the bit to check.

    Returns:
        bool: True if the k-th bit is set, False otherwise.
    """
    if isinstance(n, (int, np.integer)):
        return check_int(n, k)
    
    if is_traced_jax(n):
        return check_int_traced_jax(n, k)
    # no matter if it is a NumPy or JAX array
    return n[k] > 0

# --------------------------------------------------------------------------------------------------

@numba.njit(fastmath=True)
def int2base_np(n,
                size        : int,
                dtype       = DEFAULT_NP_INT_TYPE,
                value_true  = 1,
                value_false = 0):
    """
    Convert an integer to a binary (or spin) representation using NumPy.

    Args:
        n (int)             : The integer to convert.
        size (int)          : The number of bits in the representation.
        dtype               : The desired NumPy dtype for the output array.
        spin (bool)         : If True, outputs spin values (±spin_value) instead of {0,1}.
        spin_value (float)  : The value to use for spin representation.
    Returns:
        np.ndarray          : The binary (or spin) representation of the integer.
    """
    bits = []
    # Iterate from the most significant bit to the least.
    for pos in range(size - 1, -1, -1):
        # For a Python or NumPy integer, we use bit shifting.
        if isinstance(n, (int, np.integer)):
            bit = check_int(n, pos)
        else:
            bit = bool(np.bitwise_and(n, (1 << pos)) != 0)
        bits.append(value_true if bit else value_false)
    return np.array(bits, dtype=dtype)

if _JAX_AVAILABLE:
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
        bits = []
        # Iterate from the most significant bit to the least.
        for pos in range(size - 1, -1, -1):
            if isinstance(n, (int, np.integer)):
                bit = (n >> pos) & 1
            else:
                bit = bool(jnp.bitwise_and(n, (1 << pos)) != 0)
            bits.append(value_true if bit else value_false)
        return jnp.array(bits, dtype=dtype)

def int2base(n          : int,
            size        : int,
            backend             = 'default',
            spin        : bool  = True,
            spin_value  : float = BACKEND_REPR):
    '''
    Convert an integer to a base representation (spin: ±value or binary 0/1).

    Args:
        n (int)             : The integer to convert.
        size (int)          : The number of bits in the binary representation.
        backend (np)        : The backend to use (np or jnp).
        spin (bool)         : A flag to indicate whether to use spin values.
        spin_value (float)  : The spin value to use.
    Returns:
        np.ndarray or jnp.ndarray: The binary representation of the integer.    
    '''
    backend     = get_backend(backend)
    val_true    = spin_value
    val_false   = -spin_value if spin else 0
    if backend == np:
        return int2base_np(n, size, value_true = val_true, value_false = val_false)
    return int2base_jax(n, size, value_true = val_true, value_false = val_false)

# --------------------------------------------------------------------------------------------------

def base2int_spin(vec : 'array-like', spin_value: float = BACKEND_REPR) -> int:
    '''
    Convert a base representation (spin: ±value or binary 0/1) back to an integer.
    Args:
        vec (np.ndarray or jnp.ndarray): The binary representation of the integer.
        spin (bool)                    : A flag to indicate whether to use spin values.
        spin_value (float)             : The spin value to use.
    Returns:
        int                             : The integer representation of the binary vector.
    '''
    size = len(vec)
    if not (0 < size <= 63):
        raise ValueError("The size of the vector must be between 1 and 63 inclusive.")
    val = 0
    # Loop over bits from least-significant (index 0) to most-significant.
    for k in range(size):
        bit_val     =   int((vec[size - 1 - k] / spin_value + 1.0) / 2.0)
        val         +=  bit_val * lookup_binary_power[k]
    return val

def base2int(vec        : 'array-like',
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = BACKEND_REPR) -> int:
    '''
    Convert a base representation back to an integer.
    
    Args:
        vec (np.ndarray or jnp.ndarray): The binary representation of the integer.
        spin (bool)                    : A flag to indicate whether to use spin values.
        spin_value (float)             : The spin value to use.
    
    Returns:
        int                             : The integer representation of the binary vector.
    '''
    
    if spin:
        return base2int_spin(vec, spin_value)
    
    size = len(vec)
    
    if not (0 < size <= 63):
        raise ValueError("The size of the vector must be between 1 and 63 inclusive.")
    val = 0
    # Loop over bits from least-significant (index 0) to most-significant.
    for k in range(size):
        # In the vector the bit order is reversed relative to significance.
        bit_val =   int(vec[size - 1 - k])
        val     +=  bit_val * lookup_binary_power[k]
    return val

# --------------------------------------------------------------------------------------------------
#! Flip bits in an integer or a vector
# --------------------------------------------------------------------------------------------------

if _JAX_AVAILABLE:
    @jax.jit
    def flip_all_array_nspin(n : 'array-like'):
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

@maybe_jit
def flip_all_array_spin(n : 'array-like', spin_value : float = BACKEND_REPR, backend = 'default'):
    """
    Flip all bits in a representation of a state.
    - This is a helper function for flip_all.
    Parameters:
        n (array-like)      : The state to flip.
        spin_value (float)  : The spin value to use.
        backend (str)       : The backend to use (np or jnp or default).
    """
    backend = get_backend(backend)
    return backend.where(n == spin_value, 0, spin_value)

def flip_all(n          : 'array-like',
            size        : int,
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = BACKEND_REPR,
            backend             = 'default'):
    """
    Flip all bits in a representation of a state.

    Args:
        n (int)             : The value to flip.
        size (int)          : The number of bits - the size of the state.
        spin (bool)         : A flag to indicate whether to use spin values.
        spin_value (float)  : The spin value to use.
        backend (np)        : The backend to use (np or jnp or default).    
    Note:
        The function is implemented for both integers and NumPy arrays (np or jnp).
        The function is implemented for both binary and spin representations. 
    Returns:
        A flipped state.
    """
    
    if isinstance(n, (int, np.integer)):
        return lookup_binary_power[size] - 1 - n
    elif isinstance(n, list):
        if spin:
            return [-x for x in n]
        else:
            return [0 if x == spin_value else spin_value for x in n]
    if spin:
        return flip_all_array_spin(n, spin_value, backend)
    return flip_all_array_nspin(n)

# --------------------------------------------------------------------------------------------------

if _JAX_AVAILABLE:

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
    @JIT
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

    @JIT
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
    
    @JIT
    def flip_int_traced_jax(n : int, k : int):
        '''
        Internal helper function for flipping a single bit in an integer through JAX tracing.
        Parameters:
            n (array-like)  : The state to flip.
            k (int)         : The index of the bit to flip.
        Returns:
            array-like      : The flipped state.
        Returns:
            array-like      : The flipped state.
        '''
        return jnp.where(check_int_traced_jax(n, k), n - lookup_binary_power_jax[k], n + lookup_binary_power_jax[k])
    
    @JIT
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

@numba.njit
def flip_array_np_spin(n : 'array-like', k : int):
    """
    Flip a single bit in a representation of a state.
    - This is a helper function for flip.
    """
    n[k] = -n[k]
    return n

@numba.njit
def flip_array_np_nspin(n           : 'array-like',
                        k           : int,
                        spin_value  : float = BACKEND_REPR):
    """
    Flip a single bit in a representation of a state.
    - This is a helper function for flip.
    """
    n[k] = float(not (n[k] == spin_value)) * spin_value
    return n

# Multi-index versions for NumPy arrays.
@numba.njit
def flip_array_np_spin_multi(n: 'array-like', ks: 'array-like'):
    """
    Flip multiple spins in a NumPy array using advanced indexing.
    Parameters:
        n (array-like)      : The state to flip.
        ks (array-like)     : The indices of the bits to flip.
    """
    n[ks] = -n[ks]
    return n

@numba.njit
def flip_array_np_nspin_multi(n: 'array-like', ks: 'array-like', spin_value: float = BACKEND_REPR):
    """
    Flip multiple bits (binary representation) in a NumPy array.
    Since Numba does not support advanced indexing for this use-case, we use a loop.
    Parameters:
        n (array-like)      : The state to flip.
        ks (array-like)     : The indices of the bits to flip.
        spin_value (float)  : The spin value to use.
    """
    for k in ks:
        n[k] = 0 if n[k] == spin_value else spin_value
    return n

def flip_array_np(n         : 'array-like',
                k           : int,
                spin        : bool = BACKEND_DEF_SPIN,
                spin_value  : float = BACKEND_REPR):
    """
    Flip a single bit in a representation of a state.
    - This is a helper function for flip.
    """
    if spin:
        return flip_array_np_spin(n, k)
    return flip_array_np_nspin(n, k, spin_value)

@numba.njit
def flip_array_np_multi(n           : 'array-like',
                        ks          : 'array-like',
                        spin        : bool  = BACKEND_DEF_SPIN,
                        spin_value  : float = BACKEND_REPR):
    """
    Dispatch to the appropriate multi-index NumPy flip function.
    Parameters:
        n (array-like)      : The state to flip.
        ks (array-like)     : The indices of the bits to flip.
        spin (bool)         : A flag to indicate whether to use spin values.
        spin_value (float)  : The spin value to use.
    Returns:
        array-like          : The flipped state.
    """
    if spin:
        return flip_array_np_spin_multi(n, ks)
    return flip_array_np_nspin_multi(n, ks, spin_value)

@numba.njit
def flip_int(n : int, k : int):
    '''
    Flip a single bit in an integer.
    '''
    return n - lookup_binary_power[k] if check_int(n, k) else n + lookup_binary_power[k]

@numba.njit
def flip_int_multi(n: int, ks: 'array-like'):
    """
    Flip multiple bits in an integer representation.
    The update is computed as the sum of the individual deltas.
    Parameters:
        n (int)            : The integer to flip.
        ks (array-like)    : The indices of the bits to flip.
    Returns:
        int                 : The flipped integer.
    """
    delta = 0
    for k in ks:
        if check_int(n, k):
            delta -= lookup_binary_power[k]
        else:
            delta += lookup_binary_power[k]
    return n + delta

def flip(n, k,
        spin        : bool  = BACKEND_DEF_SPIN,
        spin_value  : float = BACKEND_REPR):
    '''
    Flip a single bit in a representation of a state.
    Parameters:
        n (int or array-like):
            The state to flip.
        k (int):
            The index of the bit to flip.
        spin (bool): 
            A flag to indicate whether to use spin values.
        spin_value (float):
            The spin value to use.
        backend (str): 
            The backend to use (np or jnp or default).
    Returns:
        Flipped state (same type as n).    
    '''
    
    single_index = isinstance(k, (int, np.integer))
    
    # handle the case when n is an integer
    if isinstance(n, (int, np.integer)):
        return flip_int(n, k) if single_index else flip_int_multi(n, k)
    # handle the case when n is a list or a NumPy array
    elif isinstance(n, (list, np.ndarray)):
        return flip_array_np(n, k, spin, spin_value) if single_index else flip_array_np_multi(n, k, spin, spin_value)
    # handle the case when n is a traced JAX array
    elif is_traced_jax(n):
        return flip_int_traced_jax(n, k) if single_index else flip_int_traced_jax_multi(n, k)
    # otherwise, n is a NumPy or JAX array
    if spin:
        return flip_array_jax_spin_multi(n, k)
    return flip_array_jax_nspin_multi(n, k)

# --------------------------------------------------------------------------------------------------
#! Reverse the bits in a representation of a binary string
# --------------------------------------------------------------------------------------------------

@maybe_jit
def rev_arr(n : 'array-like', backend = 'default'):
    """
    Reverse the bits of a 64-bit integer.
    - This is a helper function for rev.
    Parameters:
        n (array-like)  : The integer to reverse.
        backend (str)   : The backend to use (np or jnp or default).
    Returns:
        array-like      : The array with the bits reversed.
    Note:
        The function is implemented for both integers and NumPy arrays (np or jnp).
        The function is implemented for both binary and spin representations.
    """
    # arrays are assumed to be NumPy or JAX arrays
    backend = get_backend(backend)
    return backend.flip(n)

def rev(n : 'array-like', size : int, bitsize = 64, backend = 'default'):
    """
    Reverse the bits of a 64-bit integer.

    Args:
        n (int)    : The integer to reverse.
        size (int) : The number of bits in the integer.

    Returns:
        int: The integer with the bits reversed.
    """
    if isinstance(n, int):
        start_chunk = 8
        nout        = lookup_reversal[n & 0xff] << (bitsize - start_chunk)
        start_chunk += 8
        while start_chunk < bitsize:
            nout        |= lookup_reversal[(n >> (start_chunk - 8)) & 0xff] << (bitsize - start_chunk)
            start_chunk += 8
        return nout >> (bitsize - size)
    
    return rev_arr(n, backend)

# --------------------------------------------------------------------------------------------------
#! Bit manipulation functions
# --------------------------------------------------------------------------------------------------

@maybe_jit
def rotate_left_array(n : 'array-like', axis = None, backend = 'default'):
    """
    Rotate the bits of an integer to the left.
    - This is a helper function for rotate_left.
    Args:
        n (array-like)  : The integer to rotate.
        axis (int)      : The axis along which to rotate the bits.
        backend (str)   : The backend to use (np or jnp or default).
    Returns:
        array-like      : The array with the bits rotated to the left.
    """
    # arrays are assumed to be NumPy or JAX arrays
    backend = get_backend(backend)
    return backend.roll(n, -1, axis)

def rotate_left(n : 'array-like', size : int, backend = 'default', axis : Optional[int] = None):
    """
    Rotate the bits of an integer to the left.

    Args:
        n (int)         : The integer to rotate.
        size (int)      : The number of bits in the integer.
        backend (str)   : The backend to use (np or jnp or default).
        axis (int)      : The axis along which to rotate the bits.

    Returns:
        int: The integer with the bits rotated to the left.
    """
    if isinstance(n, int):
        max_power = lookup_binary_power[size - 1]
        return ((n - max_power) * 2 + 1) if (n >= max_power) else (n * 2)
    return rotate_left_array(n, axis, backend)

# --------------------------------------------------------------------------------------------------

@maybe_jit
def rotate_left_by_arr(n: int, shift: int = 1, axis = Optional[int], backend = 'default'):
    """
    Rotate the bits of n to the left by 'shift' positions.
    """
    backend = get_backend(backend)
    return backend.roll(n, shift, axis)

def rotate_left_by_int(n: int, shift: int = 1, size: int = 64):
    """
    Rotate the bits of n to the left by 'shift' positions.
    
    For an integer state, it treats n as having a binary representation of length 'size' 
    and performs a cyclic left shift:
    
        result = ((n << shift) | (n >> (size - shift))) & ((1 << size) - 1)
    
    Parameters:
        n (int): 
            The state to rotate.
        shift (int): 
            The number of positions to shift. Default is 1.
        size (int): 
            The number of bits in the integer representation.
        
    Returns:
        Rotated state (same type as n).
    """
    mask    =   (1 << size) - 1
    return ((n << shift) | (n >> (size - shift))) & mask

def rotate_left_by(n    : 'array-like',
                size    : int,
                shift   : int           = 1,
                backend                 = 'default',
                axis    : Optional[int] = None):
    """
    Rotate the bits of n to the left by 'shift' positions.
    
    For an integer state, it treats n as having a binary representation of length 'size' 
    and performs a cyclic left shift:
    
        result = ((n << shift) | (n >> (size - shift))) & ((1 << size) - 1)
    
    For array-like states (NumPy or JAX arrays), it calls the backend's roll function along 
    the specified axis with a negative shift.
    
    Parameters:
        n (int or array-like): 
            The state to rotate.
        size (int): 
            The number of bits in the integer representation.
        shift (int, optional): 
            The number of positions to shift. Default is 1.
        backend (str, optional): 
            The backend to use (e.g., 'default', 'numpy', or 'jax').
        axis (int, optional): 
            The axis along which to rotate for array-like states.
        
    Returns:
        Rotated state (same type as n).
    """
    if isinstance(n, int):
        return rotate_left_by_int(n, shift, size)
    return rotate_left_by_arr(n, shift, axis, backend)

def rotate_left_ax_int(n            : int,
                        x           : int,
                        row_mask    : Optional[int] = None,
                        y           : Optional[int] = None,
                        z           : Optional[int] = None,
                        shift       : int = 1, 
                        axis        : Optional[int] = None):
    """
    Rotate the bits of n along a specified axis for integer input.
    """
    if axis == 0:
        return rotate_left_by(n, x, shift, 'int')
    if axis == 1:
        if y is None:
            raise ValueError("The y parameter must be provided for axis 1.")
        
        row_mask = (1 << x) - 1 # mask of the row length
        new_stat = 0
        for i in range(y):
            row         = (n >> (i * x)) & row_mask
            row         = rotate_left_by(row, x, shift, 'int')
            new_stat    |= row << (i * x)
        return new_stat

    if y is None or z is None:
        raise ValueError("The y and z parameters must be provided for axis 2.")
    
    s        = x * y
    new_stat = 0
    for j in range(z):
        for i in range(y):
            row         = (n >> (j * s + i * x)) & row_mask
            row         = rotate_left_by(row, x, shift, 'int')
            new_stat    |= row << (j * s + i * x)
    return new_stat

def rotate_left_ax(n    : 'array-like',
                x       : int,
                row_mask: Optional[int] = None,
                y       : Optional[int] = None,
                z       : Optional[int] = None,
                shift : int = 1, axis : Optional[int] = None, backend = 'default'):
    """
    Rotate the bits of an integer along a specified axis.
    
    Args:
        n (array-like): The integer or array-like structure to rotate.
        x (int): The number of bits to rotate.
        row_mask (Optional[int]): The mask for the row length.
        y (Optional[int]): The second dimension for rotation.
        z (Optional[int]): The third dimension for rotation.
        shift (int, optional): The number of positions to shift. Default is 1.
        axis (Optional[int]): The axis along which to rotate. Default is None.
        backend (str, optional): The backend to use (e.g., 'default', 'numpy', or 'jax').
    
    Returns:
        Rotated state (same type as n).
    """
    if isinstance(n, int):
        rotate_left_ax_int(n, x, row_mask, y, z, shift, axis)
    return rotate_left_by(n, shift, axis, backend)

# --------------------------------------------------------------------------------------------------

@maybe_jit
def _rotate_right_array(n : 'array-like', axis = None, backend = 'default'):
    """
    Rotate the bits of an integer to the right.
    - This is a helper function for rotate_right.
    Args:
        n (array-like)  : The integer to rotate.
        axis (int)      : The axis along which to rotate the bits.
        backend (str)   : The backend to use (np or jnp or default).
    Returns:
        array-like      : The array with the bits rotated    
    """
    # arrays are assumed to be NumPy or JAX arrays
    backend = get_backend(backend)
    return backend.roll(n, 1, axis)

def rotate_right(n : 'array-like', size : int, backend = 'default', axis : Optional[int] = None):
    """
    Rotate the bits of an integer to the right.

    Args:
        n (int)         : The integer to rotate.
        size (int)      : The number of bits in the integer.
        backend (str)   : The backend to use (np or jnp or default).
        axis (int)      : The axis along which to rotate the bits (only for arrays - Optional).
    Returns:
        int: The integer with the bits rotated to the right.
    """
    if isinstance(n, int):
        return (n >> 1) | ((n & 1) << (size - 1))
    return _rotate_right_array(n, axis, backend)

# --------------------------------------------------------------------------------------------------

#! Other functions

# --------------------------------------------------------------------------------------------------

def int2binstr(n : int, bits : int):
    """
    Convert an integer to its binary representation with a fixed number of bits - as a string.

    Args:
        n (int): The integer to convert.
        bits (int): The number of bits in the binary representation.

    Returns:
        str: The binary representation of the integer, padded with leading zeros to fit the specified number of bits.
    """
    return f"{n:0{bits}b}"

####################################################################################################

@maybe_jit
def _popcount_spin(n : 'array-like', backend = 'default'):
    """
    Calculate the number of 1-bits in the binary representation of an integer.
    - This is a helper function for popcount.
    """
    # arrays are assumed to be NumPy or JAX arrays
    backend = get_backend(backend)
    return backend.sum(n > 0)
                    

@maybe_jit
def _popcount_nspin(n : 'array-like', backend = 'default'):
    """
    Calculate the number of 1-bits in the binary representation of an integer.
    - This is a helper function for popcount.
    """
    # arrays are assumed to be NumPy or JAX arrays
    backend = get_backend(backend)
    return backend.sum(n)

def popcount(n : int, spin : bool = BACKEND_DEF_SPIN, backend : str = 'default'):
    """
    Calculate the number of 1-bits in the binary representation of an integer.

    Args:
        n (int): The integer whose 1-bits are to be counted.

    Returns:
        int: The number of 1-bits in the binary representation of the input integer.
    """
    if isinstance(n, int):
        return n.bit_count('1')
    return int(_popcount_spin(n, backend) if spin else _popcount_nspin(n, backend))
    

####################################################################################################

class BinaryFunctionTests(GeneralAlgebraicTest):
    """
    A class that implements tests for various binary manipulation functions.
    Each test is implemented as a separate method.
    """
    
    # -------------------------------
    # Individual test methods follow:
    # -------------------------------

    def test_extract(self, n, size, mask, mask_size):
        """Test 1: Extract bits using a mask."""
        test_number         = self.test_count
        self._log(f"Test {test_number}: Extract bits using mask", test_number, color="blue")
        t0                  = time.time()
        extracted           = extract(n, mask)
        t1                  = time.time()
        self._log(f"Input n    : {int2binstr(n, size)}", test_number)
        self._log(f"Mask       : {int2binstr(mask, mask_size)}", test_number)
        self._log(f"Extracted  : {int2binstr(extracted, mask_size)} ({extracted})", test_number)
        self._log(f"Time       : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count     += 1
    # -------------------------------
    def test_extract_left_right(self, n, size):
        """Test 2: Extract leftmost and rightmost 4 bits."""
        test_number         = self.test_count
        self._log(f"Test {test_number}: Extract leftmost and rightmost 4 bits", test_number, color="blue")
        t0                  = time.time()
        leftmost            = extract_ord_left(n, size, 4)
        rightmost           = extract_ord_right(n, 4)
        t1                  = time.time()
        self._log(f"Input n         : {int2binstr(n, size)}", test_number)
        self._log(f"Leftmost 4 bits : {int2binstr(leftmost, 4)}", test_number)
        self._log(f"Rightmost 4 bits: {int2binstr(rightmost, 4)}", test_number)
        self._log(f"Time            : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1
    # -------------------------------
    def test_prepare_mask(self, positions, size):
        """Test 3: Prepare a mask from given positions."""
        test_number     = self.test_count
        self._log(f"Test {test_number}: Prepare mask from positions", test_number, color="blue")
        t0              = time.time()
        mask_prepared   = prepare_mask(positions, size=size)
        t1              = time.time()
        self._log(f"Positions          : {positions}", test_number)
        self._log(f"Prepared Mask ({size} bits) : {int2binstr(mask_prepared, size)}", test_number)
        self._log(f"Time               : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1
    # -------------------------------
    def test_prepare_mask_inverted(self, positions, size):
        """Test 4: Prepare an inverted mask from positions."""
        test_number     = self.test_count
        self._log(f"Test {test_number}: Prepare inverted mask from positions", test_number, color="blue")
        t0              = time.time()
        mask_prepared_inv = prepare_mask(positions, inv=True, size=size)
        t1 = time.time()
        self._log(f"Positions             : {positions}", test_number)
        self._log(f"Inverted Mask ({size} bits): {int2binstr(mask_prepared_inv, size)}", test_number)
        self._log(f"Time                  : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1
    # -------------------------------
    def test_is_power_of_two(self, n, size):
        """Test 5: Check if n is a power of two."""
        test_number         = self.test_count
        self._log(f"Test {test_number}: Check if n is a power of two", test_number, color="blue")
        t0                  = time.time()
        pow_check           = is_pow_of_two(n)
        t1                  = time.time()
        self._log(f"Input n : {int2binstr(n, size)}", test_number)
        self._log(f"Result  : {pow_check}", test_number)
        self._log(f"Time    : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1
    # -------------------------------
    def test_reverse_bits(self, n, size):
        """Test 6: Reverse the bits of n."""
        test_number = self.test_count
        self._log("Test {}: Reverse the bits".format(test_number), test_number, color="blue")
        t0 = time.time()
        reversed_bits = rev(n, size, backend=self.backend)
        t1 = time.time()
        self._log(f"Input n  : {int2binstr(n, size)}", test_number)
        self._log(f"Reversed : {int2binstr(reversed_bits, size)}", test_number)
        self._log(f"Time     : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1
    # -------------------------------
    def test_check_specific_bit(self, n, size, bit_position):
        """Test 7: Check a specific bit (bit_position)."""
        test_number = self.test_count
        self._log("Test {}: Check specific bit".format(test_number), test_number, color="blue")
        t0 = time.time()
        bit_check = check(n, bit_position)
        t1 = time.time()
        self._log(f"Input n          : {int2binstr(n, size)}", test_number)
        self._log(f"Bit position (0-indexed from right): {bit_position}", test_number)
        self._log(f"Bit value        : {bit_check}", test_number)
        self._log(f"Time             : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1
    # -------------------------------
    def test_int_to_base(self, n, size, spin, spin_value):
        """Test 8: Convert integer to base representation."""
        test_number         = self.test_count
        typek               = type(n)
        self._log(f"Test {test_number}: Convert integer to base representation: {typek}", test_number, color="blue")
        t0                  = time.time()
        base_representation = int2base(n, size, backend=self.backend, spin=spin, spin_value=spin_value)
        t1                  = time.time()
        self._log(f"Input n           : {int2binstr(n, size)}", test_number)
        self._log(f"Base representation: {base_representation} (type: {type(base_representation)})", test_number)
        self._log(f"Time              : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count     += 1
        return base_representation

    def test_base_to_int(self, base_representation, spin, spin_value):
        """Test 9: Convert base representation back to integer."""
        test_number         = self.test_count
        typek               = type(base_representation)
        self._log(f"Test {test_number}: Convert base representation back to integer: {typek}", test_number, color="blue")
        t0                  = time.time()
        integer_representation = base2int(base_representation, spin=spin, spin_value=spin_value)
        t1                  = time.time()
        self._log(f"Base representation : {base_representation} (type: {typek})", test_number)
        self._log(f"Recovered integer   : {integer_representation} (type: {type(integer_representation)})", test_number)
        self._log(f"Time                : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count     += 1

    def test_flip_all_bits(self, n, size, spin, spin_value):
        """Test 10: Flip all bits of n."""
        test_number         = self.test_count
        typek               = type(n)
        self._log(f"Test {test_number}: Flip all bits: {typek}", test_number, color="blue")
        t0                  = time.time()
        flipped_all         = flip_all(n, size, spin=spin, spin_value=spin_value, backend=self.backend)
        t1                  = time.time()
        self._log(f"Input n : {int2binstr(n, size)}", test_number)
        self._log(f"Flipped : {int2binstr(flipped_all, size)}", test_number)
        self._log(f"Time    : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count     += 1

    def test_flip_specific_bit_int(self, n, size, bit_position):
        """Test 11: Flip a specific bit in n (integer)."""
        test_number         = self.test_count
        typek               = type(n)
        self._log(f"Test {test_number}: Flip a specific bit in integer: {typek}", test_number, color="blue")
        t0                  = time.time()
        flipped_bit_int     = flip(n, bit_position)
        t1                  = time.time()
        self._log(f"Input n           : {int2binstr(n, size)}", test_number)
        self._log(f"Flipped bit at pos: {bit_position} -> {int2binstr(flipped_bit_int, size)}", test_number)
        self._log(f"Time              : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count     += 1

    def test_flip_specific_bit_base(self, base_representation, bit_position, spin, spin_value):
        """Test 12: Flip a specific bit in the base representation."""
        test_number         = self.test_count
        typek               = type(base_representation)
        self._log(f"Test {test_number}: Flip a specific bit in base representation: {typek}", test_number, color="blue")
        t0                  = time.time()
        flipped_bit_base    = flip(base_representation, bit_position, spin=spin, spin_value=spin_value)
        t1                  = time.time()
        self._log(f"Input base rep    : {base_representation} : {typek}", test_number)
        self._log(f"Flipped bit at pos: {bit_position} -> {flipped_bit_base} : {type(flipped_bit_base)}", test_number)
        self._log(f"Time              : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count     += 1
        return flipped_bit_base

    def test_rotate_left_int(self, n, size):
        """Test 13: Rotate left (integer)."""
        test_number         = self.test_count
        self._log(f"Test {test_number}: Rotate left (integer)", test_number, color="blue")
        t0                  = time.time()
        rotated_left_int    = rotate_left(n, size, backend=self.backend)
        t1                  = time.time()
        self._log(f"Input n       : {int2binstr(n, size)}", test_number)
        self._log(f"Rotated left  : {int2binstr(rotated_left_int, size)}", test_number)
        self._log(f"Time          : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count     += 1

    def test_rotate_left_base(self, base_representation, size):
        """Test 14: Rotate left (base representation)."""
        test_number         = self.test_count
        self._log(f"Test {test_number}: Rotate left (base representation): {type(base_representation)}", test_number, color="blue")
        t0                  = time.time()
        rotated_left_base   = rotate_left(base_representation, size, backend=self.backend)
        t1                  = time.time()
        
        self._log(f"Input base rep: {base_representation} : {type(base_representation)}", test_number)
        self._log(f"Rotated left  : {rotated_left_base} : {type(rotated_left_base)}", test_number)
        self._log(f"Time          : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count     += 1
        return rotated_left_base

    def test_rotate_right_int(self, n, size):
        """Test 15: Rotate right (integer)."""
        test_number         = self.test_count
        self._log(f"Test {test_number}: Rotate right (integer)", test_number, color="blue")
        t0                  = time.time()
        rotated_right_int   = rotate_right(n, size, backend=self.backend)
        t1                  = time.time()
        self._log(f"Input n        : {int2binstr(n, size)}", test_number)
        self._log(f"Rotated right  : {int2binstr(rotated_right_int, size)}", test_number)
        self._log(f"Time           : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1

    def test_rotate_right_base(self, base_representation, size):
        """Test 16: Rotate right (base representation)."""
        test_number         = self.test_count
        typek               = type(base_representation)
        self._log(f"Test {test_number}: Rotate right (base representation): {typek}", test_number, color="blue")
        t0 = time.time()
        rotated_right_base = rotate_right(base_representation, size, backend=self.backend)
        t1 = time.time()
        self._log(f"Input base rep : {base_representation} : {typek}", test_number)
        self._log(f"Rotated right  : {rotated_right_base} : {type(rotated_right_base)}", test_number)
        self._log(f"Time           : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1
        return rotated_right_base
    # -------------------------------
    def test_popcount(self, n, size, spin):
        """Test: popcount (number of up bits/spins)."""
        test_number = self.test_count
        self._log(f"Test {test_number}: popcount", test_number, color="blue")
        t0 = time.time()
        count = popcount(n, spin=spin, backend=self.backend)
        t1 = time.time()
        if isinstance(n, int):
            expected = n.bit_count()  # or bin(n).count('1') if bit_count unavailable
            self._log(f"Input n         : {int2binstr(n, size)}", test_number)
            self._log(f"Expected count  : {expected}", test_number)
        else:
            self._log(f"Input state     : {n}", test_number)
            expected = "N/A"
        self._log(f"popcount result : {count}", test_number)
        self._log(f"Time            : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count += 1
    # -------------------------------
    def test_rotate_left_ax(self, n, x, row_mask, y, z, shift, axis):
        '''
        Test: rotate_left_ax
        '''
        test_number = self.test_count
        self._log(f"Test {test_number}: rotate_left_ax", test_number, color="blue")
        t0 = time.time()
        rotated = rotate_left_ax(n, x, row_mask, y, z, shift, axis)
        t1 = time.time()
        self._log(f"Input state     : {n}", test_number)
        self._log(f"Rotated state   : {rotated}", test_number)
        self._log(f"Time            : {t1 - t0:.6f} seconds", test_number)
        self._log("-" * 50, test_number)
        self.test_count             += 1
        
    # -------------------------------
    #! Master test runner:
    # -------------------------------
    def run_tests(self,
                  size          = 6,
                  mask          = 0b1011,
                  mask_size     = None,
                  positions     = None,
                  bit_position  = 2,
                  spin          = True,
                  spin_value    = 1.0,
                  backend       = 'default'):
        """
        Runs a series of tests on the binary manipulation functions by calling individual test methods.
        
        Parameters:
            size         (int): Number of bits for the binary representations (default: 6).
            mask         (int): A mask value used for bit extraction (default: 0b1011).
            mask_size    (int): Number of bits to represent the mask. If None, defaults to size.
            positions    (list): Bit positions to prepare a mask from (default: [0, 1, 3]).
            bit_position (int): Bit position to test for flip and check operations (default: 2).
            spin         (bool): Whether to use spin representation (±spin_value) for conversion.
            spin_value   (float): The value representing spin-up in spin representation.
        """
        
        self.change_backend(backend)
        
        if mask_size is None:
            mask_size = size
        if positions is None:
            positions = [0, 1, 3]
        # Reset test counter
        self.test_count = 1

        separator = "=" * 50
        self._log(separator, 0)
        self._log("TESTING BINARY FUNCTIONS".center(50), 0)
        self._log(separator, 0)
        total_start = time.time()

        # Generate a random integer for testing.
        n = np.random.randint(0, 2 ** size)
        self._log(f"Random integer n = {n} (binary: {int2binstr(n, size)})", 0)
        self._log("-" * 50, 0)  # Added missing separator log
        self._log("-" * 50, 0)

        # Run each individual test by passing all required arguments including backend.
        self.test_extract(n, size, mask, mask_size)
        self.test_extract_left_right(n, size)
        self.test_prepare_mask(positions, size)
        self.test_prepare_mask_inverted(positions, size)
        self.test_is_power_of_two(n, size)
        self.test_reverse_bits(n, size)
        self.test_check_specific_bit(n, size, bit_position)
        base_representation = self.test_int_to_base(n, size, spin, spin_value)
        self.test_base_to_int(base_representation, spin, spin_value)
        self.test_flip_all_bits(n, size, spin, spin_value)
        self.test_flip_specific_bit_int(n, size, bit_position)
        base_representation = self.test_flip_specific_bit_base(base_representation, bit_position, spin, spin_value)
        self.test_rotate_left_int(n, size)
        base_representation = self.test_rotate_left_base(base_representation, size)
        self.test_rotate_right_int(n, size)
        base_representation = self.test_rotate_right_base(base_representation, size)

        total_time = time.time() - total_start
        self._log(separator, 0)
        self._log(f"Total testing time: {total_time:.6f} seconds", 0, color="green")
        self._log(separator, 0)
        self.test_count = 0
        self._log("Testing completed.", 0, color = "green")

# --------------------------------------------------------------------------------------------------