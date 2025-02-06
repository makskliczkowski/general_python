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

import numpy as np
from typing import Union, List
# Try to import JAX’s numpy if available.
try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

# define a type alias for the backend
Backend         = Union[None, np.ndarray, jnp.ndarray]              # The backend type alias.
Backend_Out     = jnp if jnp else np                                # The backend output type alias.
Backend_Repr    = 0.5                                               # The backend representation - for instance 0.5 for spins.                                  

####################################################################################################

# Integer based functions

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

# Mask based functions - for extracting bits from a number

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
lookup_reversal     = [reverse_byte(i) for i in range(256)]
lookup_binary_power = [binary_power(i) for i in range(63)]

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

# Check the power of two

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

# Vector based functions - for extracting bits from a vector

# --------------------------------------------------------------------------------------------------

def check(n : Backend, k : int):
    """
    Check the i-th bit in an integer.

    Args:
        n (int): The integer to check.
        i (int): The index of the bit to check.

    Returns:
        bool: True if the i-th bit is set, False otherwise.
    """
    if isinstance(n, (int, np.integer)):
        return bool(n & (1 << k))
    elif hasattr(n, "shape"):
        return n[k] > 0     # Assume n is a NumPy or JAX array; index directly.
    else:
        return n[k] > 0

# --------------------------------------------------------------------------------------------------

def int2base(n          : int,
            size        : int,
            backend             = Backend_Out,
            spin        : bool  = True,
            spin_value  : float = Backend_Repr):
    '''
    Convert an integer to a base representation (spin: ±value or binary 0/1).

    Args:
        n (int)        : The integer to convert.
        size (int)     : The number of bits in the binary representation.
        backend (np)   : The backend to use (np or jnp).
        spin (bool)    : A flag to indicate whether to use spin values.
        spin_value (float): The spin value to use.
    Returns:
        np.ndarray or jnp.ndarray: The binary representation of the integer.    
    '''
    bits = []
    
    # Loop over the bits in the integer.
    for k in range(size):
        pos = size - 1 - k

        if isinstance(n, (int, np.integer)):
            bit = check(n, pos)
        else:
            bit = bool(backend.bitwise_and(n, (1 << pos)) != 0)
        
        # Append the bit to the list.
        if spin:
            bits.append(spin_value if bit else -spin_value)
        else:
            bits.append(1 if bit else 0)
            
    # Return the list as a NumPy or JAX array.
    return backend.array(bits)

# --------------------------------------------------------------------------------------------------

def base2int(vec, spin: bool = True, spin_value: float = Backend_Repr) -> int:
    '''
    Convert a base representation back to an integer.
    
    Args:
        vec (np.ndarray or jnp.ndarray): The binary representation of the integer.
        spin (bool)                    : A flag to indicate whether to use spin values.
        spin_value (float)             : The spin value to use.
    
    Returns:
        int: The integer representation of the binary vector.
    '''
    size = len(vec)
    
    if not (0 < size <= 63):
        raise ValueError("The size of the vector must be between 1 and 63 inclusive.")
    
    val = 0
    # Loop over bits from least-significant (index 0) to most-significant.
    for k in range(size):
        # In the vector the bit order is reversed relative to significance.
        if spin:
            # Convert from ±spin_value to 0/1: ((v/spin_value + 1) / 2)
            bit_val = int((vec[size - 1 - k] / spin_value + 1.0) / 2.0)
        else:
            bit_val = int(vec[size - 1 - k])
        val += bit_val * lookup_binary_power[k]
    return val

# --------------------------------------------------------------------------------------------------

def flip_all(n : Backend, size : int, spin : bool = True, spin_value : float = Backend_Repr):
    """
    Flip all bits in a representation of a state.

    Args:
        n (int)     : The value to flip.
        size (int)  : The number of bits - the size of the state.
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
    # arrays are assumed to be NumPy or JAX arrays
    if spin:
        return n * (-1)

    # does not work for spin values
    module = np
    if jnp is not None and isinstance(n, jnp.ndarray):
        module = jnp
    return module.where(n == spin_value, 0, spin_value)

# --------------------------------------------------------------------------------------------------

def flip(n, k, spin : bool = True, spin_value : float = Backend_Repr):
    '''
    Flip a single bit in a representation of a state.
    '''
    
    if isinstance(n, (int, np.integer)):
        return n - lookup_binary_power[k] if check(n, k) else n + lookup_binary_power[k]
    elif isinstance(n, list):
        if spin:
            n[k] = -n[k]
        else:
            n[k] = 0 if n[k] == spin_value else spin_value
        return n

    # arrays are assumed to be NumPy or JAX arrays
    if spin:
        n = n.at[k].set(-n[k])
    else:
        module = np
        if jnp is not None and isinstance(n, jnp.ndarray):
            module = jnp
        n = module.where(n == spin_value, 0, spin_value)
    return n

# --------------------------------------------------------------------------------------------------

def rev(n : Backend, size : int, bitsize = 64):
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
    
    # arrays are assumed to be NumPy or JAX arrays
    module = np
    if jnp is not None and isinstance(n, jnp.ndarray):
        module = jnp
    # reverse the bits
    return module.flip(n)

# --------------------------------------------------------------------------------------------------

# Bit manipulation functions

# --------------------------------------------------------------------------------------------------

def rotate_left(n : Backend, size : int):
    """
    Rotate the bits of an integer to the left.

    Args:
        n (int): The integer to rotate.
        size (int): The number of bits in the integer.

    Returns:
        int: The integer with the bits rotated to the left.
    """
    if isinstance(n, int):
        max_power = lookup_binary_power[size - 1]
        return ((n - max_power) * 2 + 1) if (n >= max_power) else (n * 2)
    
    # arrays are assumed to be NumPy or JAX arrays
    module = np
    if jnp is not None and isinstance(n, jnp.ndarray):
        module = jnp
    # rotate the bits
    return module.roll(n, -1)

# --------------------------------------------------------------------------------------------------

def rotate_right(n : Backend, size : int):
    """
    Rotate the bits of an integer to the right.

    Args:
        n (int): The integer to rotate.
        size (int): The number of bits in the integer.

    Returns:
        int: The integer with the bits rotated to the right.
    """
    if isinstance(n, int):
        max_power = lookup_binary_power[size - 1]
        return (n >> 1) | ((n & 1) << (size - 1))
    
    # arrays are assumed to be NumPy or JAX arrays
    module = np
    if jnp is not None and isinstance(n, jnp.ndarray):
        module = jnp
    # rotate the bits
    return module.roll(n, 1)

# --------------------------------------------------------------------------------------------------

# Other functions

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

def popcount(n : int):
    """
    Calculate the number of 1-bits in the binary representation of an integer.

    Args:
        n (int): The integer whose 1-bits are to be counted.

    Returns:
        int: The number of 1-bits in the binary representation of the input integer.
    """
    return n.bit_count()

####################################################################################################

# Test the binary functions

####################################################################################################

import time

def test_binary_functions(
    size        =   6,
    mask        =   0b1011,
    mask_size   =   None,
    positions   =   None,
    bit_position=   2,
    spin        =   True,
    spin_value  =   Backend_Repr):
    """
    Runs a series of tests on the binary manipulation functions.
    
    Parameters:
        size         : (int) Number of bits for the binary representations (default: 6).
        mask         : (int) A mask value used for bit extraction (default: 0b1011).
        mask_size    : (int) Number of bits to represent the mask. If None, defaults to size.
        positions    : (list) Bit positions to prepare a mask from (default: [0, 1, 3]).
        bit_position : (int) Bit position to test for flip and check operations (default: 2).
        spin         : (bool) Whether to use spin representation (±spin_value) for conversion (default: True).
        spin_value   : (float) The value representing spin-up in spin representation (default: 1.0).
    
    Each test is timed, and the function prints a detailed report of the input, output,
    and execution time.
    """
    if mask_size is None:
        mask_size = size
    if positions is None:
        positions = [0, 1, 3]

    # Header for the test run
    print("=" * 50)
    print("TESTING BINARY FUNCTIONS".center(50))
    print("=" * 50)

    total_start = time.time()

    # Generate a random integer within the specified bit-width.
    n = np.random.randint(0, 2**size)
    print(f"Random integer n = {n} (binary: {int2binstr(n, size)})")
    print("-" * 50)

    # Test 1: Extract bits using a mask
    t0 = time.time()
    extracted = extract(n, mask)
    t1 = time.time()
    print("Test 1: Extract bits using mask")
    print(f"  Input n    : {int2binstr(n, size)}")
    print(f"  Mask       : {int2binstr(mask, mask_size)}")
    print(f"  Extracted  : {int2binstr(extracted, mask_size)} ({extracted})")
    print(f"  Time       : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 2: Extract leftmost and rightmost bits
    t0 = time.time()
    leftmost = extract_ord_left(n, size, 4)
    rightmost = extract_ord_right(n, 4)
    t1 = time.time()
    print("Test 2: Extract leftmost and rightmost 4 bits")
    print(f"  Input n         : {int2binstr(n, size)}")
    print(f"  Leftmost 4 bits : {int2binstr(leftmost, 4)}")
    print(f"  Rightmost 4 bits: {int2binstr(rightmost, 4)}")
    print(f"  Time            : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 3: Prepare a mask from given positions
    t0 = time.time()
    mask_prepared = prepare_mask(positions, size=4)
    t1 = time.time()
    print("Test 3: Prepare mask from positions")
    print(f"  Positions          : {positions}")
    print(f"  Prepared Mask (4b) : {int2binstr(mask_prepared, 4)}")
    print(f"  Time               : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 4: Prepare an inverted (reversed) mask from positions
    t0 = time.time()
    mask_prepared_inv = prepare_mask(positions, inv=True, size=4)
    t1 = time.time()
    print("Test 4: Prepare inverted mask from positions")
    print(f"  Positions             : {positions}")
    print(f"  Inverted Mask (4 bits): {int2binstr(mask_prepared_inv, 4)}")
    print(f"  Time                  : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 5: Check if n is a power of two
    t0 = time.time()
    pow_check = is_pow_of_two(n)
    t1 = time.time()
    print("Test 5: Check if n is a power of two")
    print(f"  Input n : {int2binstr(n, size)}")
    print(f"  Result  : {pow_check}")
    print(f"  Time    : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 6: Reverse the bits of n
    t0 = time.time()
    reversed_bits = rev(n, size)
    t1 = time.time()
    print("Test 6: Reverse the bits")
    print(f"  Input n  : {int2binstr(n, size)}")
    print(f"  Reversed : {int2binstr(reversed_bits, size)}")
    print(f"  Time     : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 7: Check a specific bit (bit_position)
    t0 = time.time()
    bit_check = check(n, bit_position)
    t1 = time.time()
    print("Test 7: Check specific bit")
    print(f"  Input n          : {int2binstr(n, size)}")
    print(f"  Bit position (0-indexed from right): {bit_position}")
    print(f"  Bit value        : {bit_check}")
    print(f"  Time             : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 8: Convert integer to base representation
    t0 = time.time()
    base_representation = int2base(n, size, spin=spin, spin_value=spin_value)
    t1 = time.time()
    print("Test 8: Convert integer to base representation")
    print(f"  Input n           : {int2binstr(n, size)}")
    print(f"  Base representation: {base_representation} (type: {type(base_representation)})")
    print(f"  Time              : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 9: Convert the base representation back to an integer
    t0 = time.time()
    integer_representation = base2int(base_representation, spin=spin, spin_value=spin_value)
    t1 = time.time()
    print("Test 9: Convert base representation back to integer")
    print(f"  Base representation : {base_representation}")
    print(f"  Recovered integer : {integer_representation}")
    print(f"  Time              : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 10: Flip all bits of n
    t0 = time.time()
    flipped_all = flip_all(n, size)
    t1 = time.time()
    print("Test 10: Flip all bits")
    print(f"  Input n : {int2binstr(n, size)}")
    print(f"  Flipped : {int2binstr(flipped_all, size)}")
    print(f"  Time    : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 11: Flip a specific bit in n (integer)
    t0 = time.time()
    flipped_bit_int = flip(n, bit_position)
    t1 = time.time()
    print("Test 11: Flip a specific bit in integer")
    print(f"  Input n           : {int2binstr(n, size)}")
    print(f"  Flipped bit at pos: {bit_position} -> {int2binstr(flipped_bit_int, size)}")
    print(f"  Time              : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 12: Flip a specific bit in the base representation
    t0 = time.time()
    flipped_bit_base = flip(base_representation, bit_position)
    t1 = time.time()
    print("Test 12: Flip a specific bit in base representation")
    print(f"  Input base rep    : {base_representation}")
    print(f"  Flipped bit at pos: {bit_position} -> {flipped_bit_base}")
    print(f"  Time              : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 13: Rotate left (integer)
    t0 = time.time()
    rotated_left_int = rotate_left(n, size)
    t1 = time.time()
    print("Test 13: Rotate left (integer)")
    print(f"  Input n       : {int2binstr(n, size)}")
    print(f"  Rotated left  : {int2binstr(rotated_left_int, size)}")
    print(f"  Time          : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 14: Rotate left (base representation)
    t0 = time.time()
    rotated_left_base = rotate_left(base_representation, size)
    t1 = time.time()
    print("Test 14: Rotate left (base representation)")
    print(f"  Input base rep: {base_representation}")
    print(f"  Rotated left  : {rotated_left_base}")
    print(f"  Time          : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 15: Rotate right (integer)
    t0 = time.time()
    rotated_right_int = rotate_right(n, size)
    t1 = time.time()
    print("Test 15: Rotate right (integer)")
    print(f"  Input n        : {int2binstr(n, size)}")
    print(f"  Rotated right  : {int2binstr(rotated_right_int, size)}")
    print(f"  Time           : {t1 - t0:.6f} seconds")
    print("-" * 50)

    # Test 16: Rotate right (base representation)
    t0 = time.time()
    rotated_right_base = rotate_right(base_representation, size)
    t1 = time.time()
    print("Test 16: Rotate right (base representation)")
    print(f"  Input base rep : {base_representation}")
    print(f"  Rotated right  : {rotated_right_base}")
    print(f"  Time           : {t1 - t0:.6f} seconds")
    print("-" * 50)

    total_time = time.time() - total_start
    print("=" * 50)
    print(f"Total testing time: {total_time:.6f} seconds")
    print("=" * 50)
    
    
####################################################################################################