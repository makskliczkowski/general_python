'''
file    : general_python/common/embedded/bit_extract.py

High-performance helpers for
    * packing selected bits of an integer into a new integer
    * building masks / shift tables
Back-ends: pure Python, Numba, JAX.

----------------
File            : general_python/common/embedded/bit_extract.py
Author          : Maksymilian Kliczkowski
----------------
'''

from    typing import Iterable, Literal, Callable, Union, Optional
import  math
import  numpy as np

# numba
try:
    import  numba
    from    numba import njit as numba_njit
except ImportError as e:
    def numba_njit(*args, **kwargs):
        # Dummy decorator that returns the function unchanged
        def decorator(func):
            return func
        return decorator

# JAX
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE   = True
except ImportError:
    jax             = None
    JAX_AVAILABLE   = False

#################################
#! Pure Python
#################################

@numba_njit(inline='always')
def extract(n: int, mask: int) -> int:
    """
    Pure Python, *O(popcount(mask))*.
    
    Parameters
    ---
        n (int) :
            integer number to mask
        mask (int) : 
            mask for the integer
    
    Returns
    ---
        The extracted integer.
        
    Example
    >>> extract(0b101101, 0b001001)
    >>> 3
    """
    res, out_pos = 0, 0
    while mask:
        lsb       = mask & -mask              # isolate lowest set bit
        in_pos    = lsb.bit_length() - 1
        res      |= ((n >>  in_pos) & 1) << out_pos
        out_pos  += 1
        mask     &= mask - 1                  # clear processed bit
    return res

@numba_njit(inline='always')
def extract_with_shifts(n: int, shifts: np.ndarray) -> int:
    res = 0
    # shifts[i] is the input bit position to read for output bit i
    for out_pos, in_pos in enumerate(shifts.tolist()):
        res |= ((n >> in_pos) & 1) << out_pos
    return res

#################################
#! Helpers for mask / shift-table creation
#################################

def prepare_mask(positions  : Iterable[int],
                size        : int,
                *,
                msb0        : bool = True) -> int:
    mask = 0
    if msb0:
        for p in positions:
            mask |= 1 << (size - 1 - p)
    else:   # positions from the right
        for p in positions:
            mask |= 1 << p
    return mask

def to_one_hot(positions    : Iterable[int],
                size        : int,
                *,
                asbool      : bool = True) -> np.ndarray:
    ''' Convert list of bit positions to one-hot array. '''
    y       = np.zeros(shape=size, dtype=np.int32)
    idx     = np.array(list(positions), dtype=int)
    y[idx]  = 1
    if asbool:
        return y.astype(bool)
    return y

def shift_table_from_mask(mask: int, *, msb0: bool = True, size: int | None = None) -> np.ndarray:
    if size is None:
        size = mask.bit_length()

    shifts  : list[int] = []
    m = mask
    while m:
        lsb    = m & -m
        pos    = lsb.bit_length() - 1              # position from LSB=0
        shifts.append((size - 1 - pos) if msb0 else pos)
        m     &= m - 1
    return np.asarray(shifts, dtype=np.int64)

#################################
#! Numba
#################################

if numba:
    @numba_njit(cache=True)
    def extract_nb(n: int, mask: int) -> int:
        """
        Extracts bits from an integer `n` at positions specified by the bitmask `mask` and packs them into a contiguous integer.

        Args:
            n (int):
                The input integer from which bits are to be extracted.
            mask (int):
                A bitmask indicating which bits to extract from `n`.
                Each set bit in `mask` specifies a position in `n` to extract.

        Returns:
            int: An integer composed of the extracted bits from `n`, packed contiguously starting from the least significant bit.

        Example:
            >>> extract_nb(0b110101, 0b10110)  # Extracts bits at positions 1, 2, and 4 (from LSB), returns 0b101
        """
        res, out_pos = 0, 0
        while mask:
            lsb       = mask & -mask
            in_pos    = int(math.log2(lsb))   # faster than bit_length in Numba
            res      |= ((n >>  in_pos) & 1) << out_pos
            out_pos  += 1
            mask     &= mask - 1
        return res

    @numba_njit(cache=True, inline='always')
    def extract_vnb(state: int, shifts: np.ndarray) -> int:
        """
        Extracts bits from a given integer state at specified positions and packs them into a new integer.

        Args:
            state (int):
                The integer from which bits will be extracted.
            shifts (np.ndarray):
                An array of integer positions indicating which bits to extract from `state`.

        Returns:
            int: An integer composed of the extracted bits, packed into the lower bits in the order specified by `shifts`.
        """
        res = 0
        for k in range(shifts.size):
            res |= ((state >> shifts[k]) & 1) << k
        return res
    
    @numba_njit(cache=True, inline='always')
    def extract_vnb_v(state, mask_idx, ns):
        """
        Extract bits from `state` at positions in `mask_idx` (leftmost = highest bit index)
        and pack them into a new integer, filling from the right (LSB) to left (MSB).

        Args:
            state (int): The integer to extract bits from.
            mask_idx (np.ndarray): Indices (from left, MSB=0) of bits to extract.
            ns (int): Total number of bits in the state.

        Returns:
            int: The packed integer, with extracted bits placed from right to left.
        """
        res     = 0
        mask_s  = mask_idx.shape[0]
        for k in range(mask_s):
            # Extract the bit at position (ns - 1 - mask_idx[k]) from state
            bit = (state >> (ns - 1 - mask_idx[k])) & 1
            # Place it at position k in the result (right to left, LSB first)
            res |= bit << (mask_s - 1 - k)
        return res
    
else:
    extract_nb      = None
    extract_vnb     = None
    extract_vnb_v   = None

#################################
#! JAX
#################################
    
if JAX_AVAILABLE:
    def extract_jax(state: int, mask: int) -> int:
        """JIT-compiled after first call; works on scalars or arrays."""
        
        bit_range = jnp.arange(mask.bit_length(), dtype=jnp.uint64)
        pos       = bit_range[(mask >> bit_range) & 1]            # positions
        weights   = 1 << jnp.arange(pos.size, dtype=jnp.uint64)
        def body(x):
            bits = ((x >> pos) & 1) * weights
            return bits.sum(dtype=jnp.uint64)
        return jax.jit(jnp.vectorize(body, signature="()->()"), static_argnums=1)(state) 
    
    def extract_vjax(state  : Union[int, jnp.array],
                    masks   : np.ndarray) -> jnp.ndarray:
        masks       = jnp.asarray(masks, dtype=jnp.uint64)
        state       = jnp.asarray(state, dtype=jnp.uint64)

        max_bits    = int(jnp.max(masks)).bit_length()
        bit_axis    = jnp.arange(max_bits, dtype=jnp.uint64)

        mask_bits   = (masks[:, None] >> bit_axis) & 1          # (m, b)
        state_bits  = (state[..., None] >> bit_axis) & 1       # (..., b)

        prefix      = jnp.cumsum(mask_bits, axis=1, dtype=jnp.uint64) - mask_bits
        weights     = 1 << prefix

        packed      = jnp.sum((mask_bits & state_bits) * weights,
                            axis=1, dtype=jnp.uint64)
        return packed
else:
    jnp             = None
    jax             = None
    extract_jax     = None
    extract_vjax    = None

#################################
#! Other
#################################

def extract_ord_left(n: int, size: int, size_l: int) -> int:
    '''
    Extract the leftmost bits of a number.
    
    Args:
        n (int) :
            The number to extract the bits from. 
        size (int) :
            The number of bits in the number.
        size_l (int) :
            The number of bits to extract.   
    Example:
        n = 0b101101, mask = 0b1011 will extract the bits at positions corresponding to left 4 bits.
    '''
    return n >> (size - size_l)

def extract_ord_right(n: int, size_r: int) -> int:
    '''
    Extract the rightmost bits of a number.
    
    Args:
        n (int) :
            The number to extract the bits from. 
        size (int) :
            The number of bits in the number.
        size_r (int) :
            The number of bits to extract.   
    Example:
        n = 0b101101, mask = 0b1101 will extract the bits at positions corresponding to right 4 bits.
    '''
    return n & ((1 << size_r) - 1)

#################################
#! Factory: choose backend once, call many times
#################################

Backend = Literal["python", "numba", "numba_shifts", "jax_scalar", "jax_vector", "numba_vnb"]

def make_extractor( mask    : Union[int, np.ndarray],
                    size    : Optional[int] = None,
                    *,
                    backend : Backend = "python") -> Callable[[int], int] | Callable:

    if backend == "python":
        if not isinstance(mask, int):
            raise TypeError("python backend expects a scalar mask.")
        def f(state: int) -> int:
            return extract(state, mask)
        return f

    elif backend == "numba":
        if extract_nb is None:
            raise RuntimeError("Numba not installed.")
        if not isinstance(mask, int):
            raise TypeError("numba backend expects a scalar mask.")

        @numba_njit(cache=True, inline='always')
        def f(state: int) -> int:
            return extract_nb(state, mask)
        return f        

    elif backend == "numba_shifts":
        if extract_vnb_v is None:
            raise RuntimeError("Numba not installed.")
        if isinstance(mask, int):
            mask_   = mask
            shifts  = shift_table_from_mask(mask_, size)
        else:
            shifts  = np.asarray(mask, dtype=np.int64)
            shifts  = np.array([size - 1 - p for p in shifts], dtype=np.int64)
            
        @numba_njit(cache=True, inline='always')
        def wrapped(state: int) -> int:
            return extract_vnb(state, shifts)
        return wrapped
    
    elif backend == "numba_vnb":
        if extract_vnb is None:
            raise RuntimeError("Numba not installed.")
        
        if not isinstance(mask, np.ndarray):
            raise TypeError("numba_vnb backend expects mask array "
                            "(one packed integer per row).")
        # compile once, capture masks
        
        @numba_njit(cache=True, inline='always')
        def jit_fn(state: int) -> int:
            return extract_vnb_v(state, mask, size)
        return jit_fn

    elif backend == "jax_scalar":
        if extract_jax is None:
            raise RuntimeError("JAX not installed.")
        if not isinstance(mask, int):
            raise TypeError("jax_scalar backend expects scalar mask.")
        return lambda state: extract_jax(state, mask)

    elif backend == "jax_vector":
        if extract_vjax is None:
            raise RuntimeError("JAX not installed.")
        if not isinstance(mask, np.ndarray):
            raise TypeError("jax_vector backend expects mask array "
                            "(one packed integer per row).")
        # compile once, capture masks
        jit_fn = jax.jit(lambda state: extract_vjax(state, mask))
        return jit_fn

    else:
        raise ValueError(f"Unknown backend {backend!r}")

####################################
#! Colorize extractors
####################################

def _join(chars) -> str:
    return ''.join(chars)

def colorize_extractor(state_bin_str: str, mask_a_positions : np.ndarray, mask_b_positions : np.ndarray, logger : 'Logger') -> str:
    """
    Highlights specific bits in a binary string using colorization based on provided mask positions.
    Args:
        state_bin_str (str):
            The binary string representing the state to be colorized.
        mask_a_positions (np.ndarray):
            Array of integer indices specifying positions to colorize in green.
        mask_b_positions (np.ndarray):
            Array of integer indices specifying positions to colorize in cyan.
        logger (Logger):
            Logger instance with a `colorize` method for applying color to string segments.
    Returns:
        str: The binary string with specified bits colorized according to the masks.
    """
    
    colored_bits = []
    for i, bit in enumerate(state_bin_str):
        if i in mask_a_positions:
            colored_bits.append(logger.colorize(bit, "green"))
        elif i in mask_b_positions:
            colored_bits.append(logger.colorize(bit, "cyan"))
        else:
            colored_bits.append(bit)
    return _join(colored_bits)

####################################
#! Test
####################################

def test(size           : int   = 12,
        size_a          : int   = 5,
        state_a_mask    : int   = 0x66 + 1,
        nsamples        : int   = 1) -> None:
    
    from ..binary import int2binstr
    from ..flog import Logger
    import time 
    logger = Logger(append_ts=True)



    rng             = np.random.default_rng()
    size_b          = size - size_a
    La, Lb, L       = size_a, size_b, size

    all_bits_mask   = (1 << L) - 1
    state_b_mask    = all_bits_mask ^ state_a_mask

    logger.info(f"State A mask = ({state_a_mask}) {int2binstr(state_a_mask, L)}")
    logger.info(f"State B mask = ({state_b_mask}) {int2binstr(state_b_mask, L)}")

    # ---------------------------------------------------------------------
    # Pre-compute the *vector* form (list of bit positions, high->low order)
    # that the vector extractor needs.  We cache them once per test-run.
    # ---------------------------------------------------------------------
    pos_a = [i for i in range(L) if (state_a_mask >> (L - 1 - i)) & 1]
    pos_b = [i for i in range(L) if (state_b_mask >> (L - 1 - i)) & 1]

    pos_a.sort(reverse=True) # high -> low  (requirement of extract_vnb)
    pos_b.sort(reverse=True)

    shifts_a = np.array([L - 1 - p for p in pos_a], dtype=np.uint64)
    shifts_b = np.array([L - 1 - p for p in pos_b], dtype=np.uint64)

    # ---------------------------------------------------------------------
    # Actual test loop
    # ---------------------------------------------------------------------
    for _ in range(nsamples):
        state      = int(rng.integers(0, 1 << L, dtype=np.uint64))
        state_bin  = int2binstr(state, L)

        # Colorize bits for subsystem A (e.g., green), B (e.g., cyan), rest (default)
        colored_state_bin = colorize_extractor(state_bin, pos_a, pos_b, logger)

        state_A_bin = _join(state_bin[i] for i in pos_a)
        state_B_bin = _join(state_bin[i] for i in pos_b)

        logger.info(f"\nFull ({state})    = {colored_state_bin}")
        logger.info(f"  State A bits      = {logger.colorize(state_A_bin, 'green')}")
        logger.info(f"  State B bits      = {logger.colorize(state_B_bin, 'cyan')}")

        # ---- scalar-mask extractor --------------------------------------
        t0              = time.perf_counter()
        packed_A        = extract(state, state_a_mask)
        dt              = (time.perf_counter() - t0) * 1e6
        logger.info(rf"  A scalar-mask : {int2binstr(packed_A, La)}   [{dt:.1f} \mus]")

        t0              = time.perf_counter()
        packed_B        = extract(state, state_b_mask)
        dt              = (time.perf_counter() - t0) * 1e6
        logger.info(rf" B scalar-mask : {int2binstr(packed_B, Lb)}   [{dt:.1f} \mus]")

        # ---- vector-mask extractor --------------------------------------
        t0              = time.perf_counter()
        packed_A_vec    = extract_vnb(state, shifts_a)
        dt              = (time.perf_counter() - t0) * 1e6
        logger.info(rf"  A vector-mask : {int2binstr(packed_A_vec, La)}   [{dt:.1f} \mus]")

        t0              = time.perf_counter()
        packed_B_vec    = extract_vnb(state, shifts_b)
        dt              = (time.perf_counter() - t0) * 1e6
        logger.info(rf"  B vector-mask : {int2binstr(packed_B_vec, Lb)}   [{dt:.1f} \mus]")

        # ---- sanity check -----------------------------------------------
        assert packed_A == packed_A_vec, "Mismatch for subsystem A!"
        assert packed_B == packed_B_vec, "Mismatch for subsystem B!"

        logger.title('', desired_size=50, fill = '+')
        logger.endl(1)
        
#! End file