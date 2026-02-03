"""
binary.py

This module implements binary - manipulation routines that allow you to work with binary representations of integers 
or vectors.

It includes functions to:
- check if a number is a power of 2,
- check a given bit in an integer or an indexable vector,
- convert an integer to a base representation (spin: +/- value or binary 0/1),
- convert a base representation back to an integer or a binary string or a vector of values,
- flip bits (all at once or a single bit),
- reverse the bits of a 64 - bit integer,
- extract bits (by ordinal position or via a mask),
- and prepare a bit mask from a list of positions.

Functions are written so that they work with plain Python 
integers as well as with NumPy or JAX arrays.
You can choose the backend (np or jnp) by passing the corresponding module.

-----------------------------------------------------------------------
Author          : Maks Kliczkowski
Date            : December 2025
Description     : This module implements binary - manipulation routines that allow you to work with binary representations of integers
                or vectors.
-----------------------------------------------------------------------
"""

import  time
import  numba
import  numpy as np
from    typing import Optional

try:
    from ..algebra.utils    import DEFAULT_NP_FLOAT_TYPE, Array
    from ..algebra.utils    import get_backend, maybe_jit, is_traced_jax, DEFAULT_NP_INT_TYPE, JAX_AVAILABLE, BACKEND_REPR, BACKEND_DEF_SPIN
    from ..common.testing   import GeneralAlgebraicTest
except ImportError:
    raise ImportError("general_python.common.binary module requires general_python.common.algebra.utils and general_python.common.testing modules.")

####################################################################################################

if JAX_AVAILABLE:
    from ..common.embedded  import binary_jax as jaxpy
else:
    jaxpy = None

#! extraction
try:
    from ..common.embedded  import bit_extract      as extract
    from ..common.embedded  import binary_search    as bin_search
except ImportError:
    raise ImportError("general_python.common.binary module requires general_python.common.embedded.bit_extract and binary_search modules.")

__all__                     = [
                                # Core Numba-safe bit ops
                                'ctz64', 'popcount64',
                                'mask_from_indices', 'indices_from_mask',
                                'complement_mask', 'complement_indices',
                                # Integer checks
                                'check_int', 'popcount',
                                # Conversion
                                'int2base', 'base2int', 'int2binstr'
                                # Bit manipulation
                                'flip', 'flip_all', 'rev', 'rotate_left', 'rotate_right', 'rotate_left_by',
                            ]

####################################################################################################
#! Numba-safe bit operations for tight loops (uint64)
####################################################################################################

@numba.njit(cache=True, inline='always')
def ctz64(x: np.uint64) -> np.int64:
    """
    Count trailing zeros in a 64-bit unsigned integer (Numba-safe).
    
    Returns 64 if x == 0 (no bits set). Uses binary search - O(log bits).
    
    Args:
        x: 64-bit unsigned integer
        
    Returns:
        Number of trailing zero bits (0-64)
        
    Example:
        >>> ctz64(np.uint64(8))  # 0b1000 -> 3 trailing zeros
        3
    """
    if x == 0:
        return np.int64(64)
    n = np.int64(0)
    if (x & np.uint64(0xFFFFFFFF)) == 0:
        n += 32
        x >>= 32
    if (x & np.uint64(0xFFFF)) == 0:
        n += 16
        x >>= 16
    if (x & np.uint64(0xFF)) == 0:
        n += 8
        x >>= 8
    if (x & np.uint64(0xF)) == 0:
        n += 4
        x >>= 4
    if (x & np.uint64(0x3)) == 0:
        n += 2
        x >>= 2
    if (x & np.uint64(0x1)) == 0:
        n += 1
    return n

@numba.njit(cache=True, inline='always')
def popcount64(x: np.uint64) -> np.int64:
    """
    Count number of set bits in a 64-bit integer (Numba-safe).
    
    Uses parallel bit-counting algorithm - O(1).
    
    Args:
        x: 64-bit unsigned integer
        
    Returns:
        Number of set bits (0-64)
        
    Example:
        >>> popcount64(np.uint64(0b1011))
        3
    """
    x = x - ((x >> 1) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x * np.uint64(0x0101010101010101)) >> 56
    return np.int64(x)

@numba.njit(cache=True)
def mask_from_indices(idxs: np.ndarray) -> np.uint64:
    """
    Convert array of bit indices to a bitmask (Numba-safe).
    
    Args:
        idxs: Array of indices (int64) indicating which bits to set
        
    Returns:
        64-bit mask with bits set at given indices
        
    Example:
        >>> mask_from_indices(np.array([0, 2, 3], dtype=np.int64))
        np.uint64(13)  # 0b1101
    """
    m = np.uint64(0)
    for i in range(idxs.shape[0]):
        m |= np.uint64(1) << np.uint64(idxs[i])
    return m

@numba.njit(cache=True)
def indices_from_mask(mask: np.uint64) -> np.ndarray:
    """
    Convert bitmask to array of set bit indices (Numba-safe).
    
    Returns indices in ascending order. Uses ctz64 for efficiency.
    
    Args:
        mask: 64-bit mask
        
    Returns:
        Array of indices (int64) where bits are set
        
    Example:
        >>> indices_from_mask(np.uint64(13))  # 0b1101
        array([0, 2, 3], dtype=int64)
    """
    count   = popcount64(mask)
    out     = np.empty(count, dtype=np.int64)
    m       = mask
    i       = 0
    while m != 0:
        pos     = ctz64(m)
        out[i]  = pos
        i      += 1
        m      &= m - np.uint64(1) # Clear LSB
    return out

@numba.njit(cache=True)
def complement_mask(mask: np.uint64, ns: int) -> np.uint64:
    """
    Return the complement of a mask within ns bits.
    
    Args:
        mask: Original bitmask
        ns: Number of bits in the system (1-64)
        
    Returns:
        Complement mask (bits flipped within range [0, ns))
    """
    full = np.uint64((1 << ns) - 1)
    return full ^ mask

def complement_indices(n: int, indices: np.ndarray) -> np.ndarray:
    """
    Return indices in [0..n) not in `indices`.
    
    O(n) boolean scratch, minimal allocations.

    Args:
        n: Upper bound of the range (exclusive)
        indices: Input indices to exclude
        
    Returns:
        Array of complementary indices (sorted)
        
    Example:
        >>> complement_indices(5, np.array([1, 3]))
        array([0, 2, 4], dtype=int64)
    """
    mark            = np.zeros(n, dtype=np.bool_)
    mark[indices]   = True
    return np.nonzero(~mark)[0].astype(np.int64, copy=False)

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

def binary_power(n : int):
    """
    Computes the power of two by moving a single bit to the left.
    
    Args:
        n (int): The power of two to compute.        
    
    Returns:
        int: The power of two.
    """
    return 1 << n

# a global static variable to store the reversed bytes from 0 to 255
lookup_binary_power = np.array([binary_power(i) for i in range(63)], dtype = DEFAULT_NP_INT_TYPE)
lookup_reversal     = np.array([reverse_byte(i) for i in range(256)], dtype = DEFAULT_NP_INT_TYPE)

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

@numba.njit(inline = 'always')
def check_int(n, k):
    """
    Checks if the k-th bit in the binary representation of an integer n is set (1).

    Args:
        n (int): The integer to check.
        k (int): The position of the bit to check (0-indexed, from the right).

    Returns:
        int: A non-zero value if the k-th bit is set, otherwise 0.
    """
    out = bool(n & (1 << k))
    return int(out)

@numba.njit(inline = 'always')
def check_int_l(n, k, ns):
    """
    Check the k-th bit in the binary representation of an integer n.

    Args:
        n (int): The integer to check.
        k (int): The position of the bit to check (0-indexed, from the left).

    Returns:
        int: A non-zero value if the k-th bit is set, otherwise 0.
    """
    out = bool(n & (1 << (ns - k - 1)))
    return int(out)

@numba.njit(inline = 'always')
def check_arr_np(n, k : int):
    '''
    Check the value at numpy array
    '''
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
        return jaxpy.check_int_traced_jax(n, k)
    
    return n[k] > 0

# --------------------------------------------------------------------------------------------------

def int2base_np(n,
                size        : int,
                dtype       = DEFAULT_NP_FLOAT_TYPE,
                value_true  = BACKEND_REPR,
                value_false = -BACKEND_REPR):
    """
    Convert an integer to a binary (or spin) representation using NumPy.

    Args:
        n (int)             : The integer to convert.
        size (int)          : The number of bits in the representation.
        dtype               : The desired NumPy dtype for the output array.
        spin (bool)         : If True, outputs spin values (+/- spin_value) instead of {0,1}.
        spin_value (float)  : The value to use for spin representation.
    Returns:
        np.ndarray          : The binary (or spin) representation of the integer.
    """
    if not (0 < size <= 63):
        raise ValueError("size must be in [1, 63].")

    nn          = np.uint64(n)
    shifts      = np.arange(size - 1, -1, -1, dtype=np.uint64)      # MSB->LSB
    bits        = ((nn >> shifts) & np.uint64(1)).astype(np.bool_)  # True for 1s

    out         = np.empty(size, dtype=dtype)
    out[:]      = value_false
    out[bits]   = value_true
    return out


@numba.njit(inline="always")
def int2base_numba(n            : int,
                   size         : int,
                   value_true   : float,
                   value_false  : float,
                   out          : np.ndarray):
    # out: preallocated 1D array length=size
    nn = np.uint64(n)
    for i in range(size):
        pos     = size - 1 - i                          # shift from MSB->LSB
        bit     = (nn >> np.uint64(pos)) & 1            # extract bit
        out[i]  = value_true if bit else value_false    # set value

def int2base(n          : int,
            size        : int,
            backend                         = 'default',
            spin        : bool              = True,
            spin_value  : float             = BACKEND_REPR, 
            out         : Optional[Array]   = None
            ):
    '''
    Convert an integer to a base representation (spin: +/- value or binary 0/1).

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
    val_false   = (-spin_value if spin else 0)

    if backend == np:
        
        if out is not None:
            int2base_numba(n, size, val_true, val_false, out)
            return out
        
        return int2base_np(n, size,
                           dtype       = DEFAULT_NP_FLOAT_TYPE,
                           value_true  = val_true,
                           value_false = val_false)

    return jaxpy.int2base_jax(n, size, value_true=val_true, value_false=val_false)

# --------------------------------------------------------------------------------------------------

@numba.njit(inline="always")
def base2int_binary(vec) -> np.int64:
    
    size = vec.shape[0]
    if size <= 0 or size > 63:
        raise ValueError("The size of the vector must be between 1 and 63 inclusive.")
    
    val = np.int64(0)
    # Interpret vec[0] as MSB, vec[size-1] as LSB
    for i in range(size):
        bit = np.int64(vec[i] != 0)
        val = (val << 1) | bit
    return val

@numba.njit(inline = 'always')
def base2int_spin(vec : Array, spin_value: float = BACKEND_REPR) -> np.int64:
    '''
    Convert a base representation (spin: +/- value or binary 0/1) back to an integer.
    Args:
        vec (np.ndarray or jnp.ndarray): The binary representation of the integer.
        spin (bool)                    : A flag to indicate whether to use spin values.
        spin_value (float)             : The spin value to use.
    Returns:
        int                             : The integer representation of the binary vector.
    '''
    size = vec.shape[0]
    if size <= 0 or size > 63:
        raise ValueError("The size of the vector must be between 1 and 63 inclusive.")
    # If spin_value > 0 and encoding is +/- spin_value, then sign test is enough.
    val = np.int64(0)
    for i in range(size):
        bit = np.int64(vec[i] > 0.0)
        val = (val << 1) | bit
    return val

@numba.njit(inline = 'always')
def base2int(vec        : Array,
            spin        : bool  = BACKEND_DEF_SPIN,
            spin_value  : float = BACKEND_REPR) -> np.int64:
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
    return base2int_binary(vec)

# --------------------------------------------------------------------------------------------------
#! Flip bits in an integer or a vector
# --------------------------------------------------------------------------------------------------

@maybe_jit
def flip_all_array_nspin(n : Array, spin_value : float = BACKEND_REPR, backend = 'default'):
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

def flip_all(n          : Array,
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
        return jaxpy.flip_all_array_spin(n)
    return flip_all_array_nspin(n, spin_value, backend)

# --------------------------------------------------------------------------------------------------
#! SINGLE BIT FLIP
# --------------------------------------------------------------------------------------------------

@numba.njit(inline = 'always')
def flip_array_np_spin(n : Array, k : int):
    """
    Flip a single bit in a representation of a state.
    - This is a helper function for flip.
    """
    n[k] = -n[k]
    return n

@numba.njit(inline = 'always')
def flip_array_np_nspin(n           : Array,
                        k           : int,
                        spin_value  : float = 1.0):
    """
    Flip a single bit in a representation of a state.
    - This is a helper function for flip.
    """
    n[k] = (0 if n[k] > 0 else 1) * spin_value
    return n

# Multi-index versions for NumPy arrays.
@numba.njit(inline = 'always')
def flip_array_np_spin_multi(n: Array, ks: Array):
    """
    Flip multiple spins in a NumPy array using advanced indexing.
    Parameters:
        n (array-like)      : The state to flip.
        ks (array-like)     : The indices of the bits to flip.
    """
    n[ks] = -n[ks]
    return n

@numba.njit(inline = 'always')
def flip_array_np_nspin_multi(n: Array, ks: Array, spin_value: float = BACKEND_REPR):
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

@numba.njit(inline = 'always')
def flip_array_np(n         : Array,
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

@numba.njit(inline = 'always')
def flip_array_np_multi(n           : Array,
                        ks          : Array,
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

@numba.njit(inline = 'always')
def flip_int(n : int, k : int):
    '''
    Flip a single bit in an integer.
    '''
    return n - lookup_binary_power[k] if check_int(n, k) else n + lookup_binary_power[k]

@numba.njit(inline = 'always')
def flip_int_multi(n: int, ks: Array):
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

def flip(n, 
        k,
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
        return jaxpy.flip_int_traced_jax(n, k) if single_index else jaxpy.flip_int_traced_jax_multi(n, k)
    # otherwise, n is a NumPy or JAX array
    if spin:
        return jaxpy.flip_array_jax_spin_multi(n, k)
    return jaxpy.flip_array_jax_nspin_multi(n, k)

# --------------------------------------------------------------------------------------------------
#! Reverse the bits in a representation of a binary string
# --------------------------------------------------------------------------------------------------

@maybe_jit
def rev_arr(n : Array, backend = 'default'):
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

def rev(n : Array, size : int, bitsize = 64, backend = 'default'):
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
def rotate_left_array(n : Array, axis = None, backend = 'default'):
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

def rotate_left(n : Array, size : int, backend = 'default', axis : Optional[int] = None):
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

def rotate_left_by(n    : Array,
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

def rotate_left_ax(n    : Array,
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
def _rotate_right_array(n : Array, axis = None, backend = 'default'):
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

def rotate_right(n : Array, size : int, backend = 'default', axis : Optional[int] = None):
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
    if isinstance(n, (int, np.integer)):
        return f"{n:0{bits}b}"
    return '0' * (len(n) - bits) + str(n)

####################################################################################################

@maybe_jit
def _popcount_spin(n : Array, backend = 'default'):
    """
    Calculate the number of 1-bits in the binary representation of an integer.
    - This is a helper function for popcount.
    """
    # arrays are assumed to be NumPy or JAX arrays
    backend = get_backend(backend)
    return backend.sum(n > 0)

@maybe_jit
def _popcount_nspin(n : Array, backend = 'default'):
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
    if isinstance(n, (int, np.integer)):
        # Python's int.bit_count() returns the number of set bits; no argument required
        return n.bit_count()
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
        leftmost            = jaxpy.extract_ord_left(n, size, 4)
        rightmost           = jaxpy.extract_ord_right(n, 4)
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
        mask_prepared   = jaxpy.prepare_mask(positions, size=size)
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
        mask_prepared_inv = jaxpy.prepare_mask(positions, inv=True, size=size)
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
            spin         (bool): Whether to use spin representation (+/- spin_value) for conversion.
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