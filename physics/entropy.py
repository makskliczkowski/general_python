'''
This module provides optimized functions for computing entanglement entropy, 
mutual information, and topological entanglement entropy. It also includes 
static methods for predicting entropy in various physical scenarios.

--------------------------------
Author      : Maksymilian Kliczkowski
email       : maksymilian.kliczkowski@pwr.edu.pl
--------------------------------
'''

from    enum            import Enum, unique
import  numpy           as np
import  numba
import  math
import  scipy.linalg    as la
from    typing          import List, Dict, Tuple, Union, Optional, Any

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    
# ----------------------------------

if JAX_AVAILABLE:
    from ..physics import entropy_jax as entropy_jax
    from .entropies.entropy_constants import *
else:
    entropy_jax = None
    
###################################

###################################
#! JIT-optimized entropy kernels
###################################

@numba.njit(cache=True)
def _clean_probs(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    r"""Clip negatives & zeros from round-off, renormalise."""
    q           = np.where(p < eps, 0.0, p)
    s           = q.sum()
    if s <= 0:  return q
    return q / s if abs(s - 1.0) > 1e-14 else q

@numba.njit(cache=True, fastmath=True)
def vn_entropy(lam: np.ndarray, base: float = np.e) -> float:
    r"""
    Calculates the von Neumann entropy for a given probability distribution.c

    Parameters
    ----------
    lam : np.ndarray
        Array of eigenvalues or probabilities. Should sum to 1 and contain non-negative values.
    base : float, optional
        The logarithm base to use for entropy calculation. Default is the natural logarithm (np.e).

    Returns
    -------
    float
        The von Neumann entropy of the probability distribution.

    Notes
    -----
    The function internally cleans the probability array to ensure numerical stability.
    A small constant (1e-30) is added to probabilities before taking the logarithm to avoid log(0).
    """
    lam     = _clean_probs(lam)
    # Avoid log(0) for exactly zero elements
    ent     = 0.0
    for p in lam:
        if p > 0:
            ent -= p * math.log(p)
    if base != np.e:
        ent /= math.log(base)
    return ent

@numba.njit(cache=True, fastmath=True)
def renyi_entropy(lam: np.ndarray, q: float, base: float = np.e) -> float:
    r"""
    Calculates the Rényi entropy of a probability distribution.

    The Rényi entropy is a generalization of the Shannon (von Neumann) entropy, parameterized by q.
    For q = 1, it reduces to the von Neumann entropy.

    Args:
        lam (np.ndarray):
            Probability distribution (array of probabilities or eigenvalues).
        q (float):
            Order of the Rényi entropy. Must be a positive real number.
        base (float, optional):
            Logarithm base to use. Default is the natural logarithm (np.e).

    Returns:
        float:
            The Rényi entropy of the distribution.

    Raises:
        ValueError: If q is not a positive real number.

    Notes:
        - If q == 1, the function returns the von Neumann entropy.
        - The input probabilities are cleaned to remove zeros and ensure normalization.
    """
    if abs(q - 1.0) < 1e-9:
        return vn_entropy(lam, base)
    lam         = _clean_probs(lam)
    s           = (lam ** q).sum()
    if s <= 0:  return 0.0
    log_s       = math.log(s)
    if base != np.e:
        log_s /= math.log(base)
    return log_s / (1.0 - q)

@numba.njit(cache=True, fastmath=True)
def tsallis_entropy(lam: np.ndarray, q: float) -> float:
    r"""
    Compute the Tsallis entropy for a given probability distribution.

    The Tsallis entropy is a generalization of the standard von Neumann entropy,
    parameterized by a real number q. For q=1, it reduces to the von Neumann entropy.

    Args:
        lam (np.ndarray):
            Array of eigenvalues or probabilities (should sum to 1).
        q (float):
            Entropy parameter. For q=1, returns the von Neumann entropy.

    Returns:
        float: The Tsallis entropy of the input distribution.

    References:
        - C. Tsallis, "Possible generalization of Boltzmann-Gibbs statistics," J. Stat. Phys. 52, 479-487 (1988).
    """
    if abs(q - 1.0) < 1e-9:
        return vn_entropy(lam)
    lam = _clean_probs(lam)
    return (1.0 - (lam ** q).sum()) / (q - 1.0)

@numba.njit(cache=True)
def sp_correlation_entropy(lam: np.ndarray, q: float, base: float = np.e):    
    r"""
    Compute the single-particle correlation entropy for a set of eigenvalues.
    This function calculates either the von Neumann entropy (for q=1) or the Rényi entropy (for generic q)
    of a set of eigenvalues `lam` (typically from a correlation matrix), after mapping each eigenvalue
    from the interval [-1, 1] to a probability in [0, 1].
    Parameters
    ----------
    lam : np.ndarray
        Array of eigenvalues (\lambda), each in the interval [-1, 1].
    q : float
        Entropy order parameter. If q == 1, computes the von Neumann entropy; otherwise, computes the Rényi entropy.
    base : float, optional
        The logarithm base to use (default is the natural logarithm, base e).
    Returns
    -------
    float
        The computed entropy value.
    Notes
    -----
    - For q == 1, the function computes the von Neumann entropy:
          S = -\Sigma  [p * log(p) + (1-p) * log(1-p)]
      where p = 0.5 * (1 + \lambda).
    - For q \neq  1, the function computes the Rényi entropy:
          S_q = (1 / (1-q)) * \Sigma  log(p^q + (1-p)^q)
    - The entropy is normalized by the logarithm of the specified base.
    """
    
    log_base = np.log(base)
    
    #! von-Neumann entropy (q == 1)
    if np.abs(q - 1.0) < 1e-12:
        s = 0.0
        LOG_TWO = np.log(2.0)
        for l in lam:
            if l > -1.0:
                s += (1.0 + l) * (np.log1p(l) - LOG_TWO)
            if l < 1.0:
                s += (1.0 - l) * (np.log1p(-l) - LOG_TWO)
        return -0.5 * s  # Negative sign to match von Neumann entropy definition

    #! Rényi entropy (generic q)
    inv_1mq = 1.0 / (1.0 - q)
    s       = 0.0
    for l in lam:
        p  = 0.5 * (1.0 + l)
        pm = 1.0 - p
        s += np.log(p ** q + pm ** q)
    return inv_1mq * s / log_base

#! Participation entropies

@numba.njit(cache=True)
def information_entropy(states: np.ndarray, threshold: float = 1e-12):
    """
    Compute S_j = -∑_i p_{i,j} ln p_{i,j},  p_{i,j}=|states[i,j]|^2,
    dropping p<=threshold.  Works for 1D (n,) or 2D (n,m) input.

    Parameters
    ----------
    states : np.ndarray
        Complex array, shape (n,) or (n, m).
    threshold : float
        Values of p<=threshold are treated as zero.

    Returns
    -------
    out : float or np.ndarray
        Scalar entropy if input was 1D, else 1D array of length m.
    """
    # reshape 1D -> 2D(n,1)
    if states.ndim == 1:
        S = states.reshape(states.shape[0], 1)
        single = True
    else:
        S = states
        single = False

    n, m    = S.shape
    ent     = np.zeros(m, dtype=np.float64)

    # parallel over columns, minimal memory
    for j in numba.prange(m):
        acc = 0.0
        for i in range(n):
            c = S[i, j]
            p = c.real*c.real + c.imag*c.imag
            if p > threshold:
                acc += p * math.log(p)
        ent[j] = -acc

    return ent[0] if single else ent

@numba.njit(cache=True)
def participation_entropy(states: np.ndarray, q: float = 1.0, threshold: float = 1e-12, square = False) -> float:
    """
    Compute the participation entropy for a given probability distribution.

    The participation entropy is a measure of how evenly the probability is distributed among the eigenvalues.
    It is defined as the negative logarithm of the sum of the squares of the probabilities.

    Args:
        lam (np.ndarray):
            Array of eigenvalues or probabilities (should sum to 1).
        q (float):
            Entropy parameter. For q=1, returns the von Neumann entropy.

    Returns:
        float: The participation entropy of the input distribution.
    """
    single = False
    if states.ndim == 1:
        states = states.reshape(states.shape[0], 1)
        single = True

    n, m    = states.shape
    out     = np.empty(m, dtype=np.float64)
    two_q   = 2.0 * q if square else q
    is_q1   = math.fabs(q - 1.0) < 1e-12
    
    # parallel over columns
    for j in numba.prange(m):
        acc = 0.0
        if is_q1:
            # Shannon‐type
            for i in range(n):
                c   = states[i, j]
                p   = abs(c) ** two_q
                if p > threshold:
                    acc += p * math.log(p)
            out[j] = -acc
        else:
            # Rényi‐type: sum p^q, then (1/(1-q))\cdot ln(...)
            for i in range(n):
                c   = states[i, j]
                p   = abs(c) ** q
                if p > threshold:
                    acc += p
            # protect against acc==0 (all below threshold)
            out[j] = math.log(acc) / (1.0 - q) if acc > 0.0 else 0.0
    return out

# ----------------------------------

@unique
class Entanglement(Enum):
    VN          = 1
    RENYI       = 2
    TSALLIS     = 3
    SINGLE      = 4
    PARTIC      = 5

def entropy(lam: np.ndarray, q: float = 1.0, base: float = np.e, *,
        typek: Entanglement = Entanglement.RENYI, backend: str = "numpy", **kwargs) -> float:
    """
    Calculates the entropy of a probability distribution using the specified entanglement entropy type.

    Parameters:
        lam (np.ndarray):
            The probability distribution (eigenvalues) for which to compute the entropy.
        q (float, optional):
            The order parameter for Rényi and Tsallis entropies. Default is 1.0.
        base (float, optional):
            The logarithm base to use in entropy calculations. Default is the natural logarithm (np.e).
        typek (Entanglement, optional):
            The type of entanglement entropy to compute. Must be one of:
                - Entanglement.VN:
                    Von Neumann entropy
                - Entanglement.RENYI:
                    Rényi entropy
                - Entanglement.TSALLIS:
                    Tsallis entropy
                - Entanglement.SINGLE:
                    Single-particle correlation entropy

    Returns:
        float: The computed entropy value.

    Raises:
        ValueError: If an unsupported entanglement type is provided.
    """
    if backend.lower() == 'numpy':
        lam = np.asarray(lam)
        if typek == Entanglement.VN:
            return vn_entropy(lam, base)
        elif typek == Entanglement.RENYI:
            return renyi_entropy(lam, q, base)
        elif typek == Entanglement.TSALLIS:
            return tsallis_entropy(lam, q, base)
        elif typek == Entanglement.SINGLE:
            return sp_correlation_entropy(lam, q, base)
        elif typek == Entanglement.PARTIC:
            return participation_entropy(lam, q, kwargs.get('threshold', 1e-12))
    elif backend.lower() == 'jax' and JAX_AVAILABLE:
        if typek == Entanglement.VN:
            return entropy_jax.vn_entropy_jax(lam, base)
        elif typek == Entanglement.RENYI:
            return entropy_jax.renyi_entropy_jax(lam, q, base)
        elif typek == Entanglement.TSALLIS:
            return entropy_jax.tsallis_entropy_jax(lam, q, base)
        elif typek == Entanglement.SINGLE:
            return entropy_jax.sp_correlation_entropy_jax(lam, q, base)
        elif typek == Entanglement.PARTIC:
            return entropy_jax.participation_entropy_jax(lam, q, kwargs.get('threshold', 1e-12))
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'jax'.")

def mutual_information(psi, i, j, ns, q: float = 1.0, base: float = np.e, **kwargs) -> Tuple[float, Tuple[float, float, float]]:
    """
    Compute mutual information I(i:j) = S_i + S_j - S_ij.
    
    Parameters:
    -----------
    psi : np.ndarray
        The state vector of the quantum system.
    i : int
        Index of the first subsystem.
    j : int
        Index of the second subsystem.
    ns : int
        Total number of subsystems.
    q : float, optional
        The order parameter for Rényi and Tsallis entropies. Default is 1.0.
    base : float, optional
    """
    from .density_matrix import rho, rho_spectrum
    
    # We use our optimized rho which handles masking correctly
    rho_i   = rho(psi, va=[i], ns=ns)
    rho_j   = rho(psi, va=[j], ns=ns)
    rho_ij  = rho(psi, va=[i, j], ns=ns)

    spec_i  = rho_spectrum(rho_i)
    spec_j  = rho_spectrum(rho_j)
    spec_ij = rho_spectrum(rho_ij)

    Si      = entropy(spec_i, q, base, **kwargs)
    Sj      = entropy(spec_j, q, base, **kwargs)
    Sij     = entropy(spec_ij, q, base, **kwargs)
    
    return Si + Sj - Sij, (Si, Sj, Sij)

def topological_entropy(
    psi         : np.ndarray, 
    regions     : Dict[str, List[int]], 
    ns          : int, 
    q           : float = 1.0, 
    base        : float = np.e, 
    **kwargs
) -> Dict[str, Any]:
    r"""
    Calculate topological entanglement entropy (TEE) \gamma.
    Optimized for speed and memory.
    """
    from .density_matrix import schmidt
    
    topo_kind   = kwargs.get('topological', 'kitaev_preskill')
    entropies   = {}
    
    for name, indices in regions.items():
        indices         = np.asarray(indices, dtype=np.int64)
        size_a          = len(indices)
        
        if size_a == 0 or size_a == ns:
            entropies[name] = 0.0
            continue
            
        probs           = schmidt(psi, va=indices, ns=ns, eig=False, contiguous=False, square=True, return_vecs=False)
        entropies[name] = entropy(probs, q, base, **kwargs)
        
    # Calculate Gamma
    gamma = 0.0
    if topo_kind.startswith('kitaev') or topo_kind.startswith('levin'):
        # Both KP and LW can use the 7-region formula if regions are defined that way
        keys = ['A', 'B', 'C', 'AB', 'BC', 'AC', 'ABC']
        if all(k in entropies for k in keys):
            gamma = (entropies['A'] + entropies['B'] + entropies['C'] 
                   - entropies['AB'] - entropies['BC'] - entropies['AC'] 
                   + entropies['ABC'])
        elif topo_kind.startswith('levin'):
            # Fallback for old concentric annuli formula
            keys = ['inner', 'outer', 'inner_outer']
            if all(k in entropies for k in keys):
                gamma = (entropies['inner'] + entropies['outer'] - entropies['inner_outer'])
         
    return {
        'gamma'     : gamma,
        'entropies' : entropies,
        'regions'   : regions
    }

####################################

class Fractal:
    
    @staticmethod
    def fractal_dim_s_info(S_lp1, S_l, lp1, l):
        '''
        Calculate the fractal dimension out of the information entropy of the system.
        - S_lp1 : Entropy of the system with L+1 sites.
        - S_l   : Entropy of the system with L sites.
        '''
        return (S_lp1 - S_l) / (np.log(2**lp1) - np.log(2**l))
    
    @staticmethod
    def fractal_dim_s_info_mean(dq_lp1, dq_l):
        '''
        Calculate the fractal dimension out of the information entropy of the system.
        Average over system sizes to get the mean value.
        - dq_lp1    : fractal of the system with L+1 sites.
        - dq_l      : fractal of the system with L sites.
        '''
        return (dq_lp1 - dq_l) / 2.0

    ################################
    
    @staticmethod
    def fractal_dim_pr(pr_lp1, pr_l, q):
        '''
        Calculate the fractal dimension out of the information entropy of the system.
        - pr_lp1: Entropy of the system with L+1 sites.
        - pr_l: Entropy of the system with L sites.
        '''
        if q != 1.0:    
            return (np.log2(pr_lp1) - np.log2(pr_l)) / (1 - q)
        else:
            return (np.log2(pr_lp1) - np.log2(pr_l))
        
# ---------------------------------
#! EOF
# ---------------------------------
