'''
file    : QES/general_python/physics/entropy.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
date    : 2025-05-01

This module provides a collection of static methods for predicting entanglement entropy in quantum systems under various physical scenarios.
It includes analytical and semi-analytical formulas for entanglement entropy, including volume law, chaotic, random Gaussian, and Page values, with and without U(1) conservation.

'''

# Adds higher directory to python modules path.
from enum import Enum, unique
from general_python.common.hdf5_lib import *
from scipy.special import digamma, polygamma, binom, psi
import numpy as np
import numba

from general_python.algebra.utils import JAX_AVAILABLE, Array
if JAX_AVAILABLE:
    from general_python.physics import entropy_jax as jnp
else:
    jnp = None
    
###################################

class EntropyPredictions:
    """
    A collection of static methods for predicting entanglement entropy in quantum systems under various physical scenarios.

    This class provides analytical and semi-analytical formulas for entanglement entropy, including volume law, chaotic, random Gaussian, and Page values, with and without U(1) conservation. The methods are based on results from quantum statistical mechanics and random matrix theory.

    Attributes:
        entro_ff (float): Entropy value from PRL 119, 020601 (2017).

    Methods:
        volume_law(L, f=0.5):
            Computes the volume law entropy for a system of size L and filling f.

        quadratic_chaotic(L: int, f: float):
            Computes the entropy for quadratic chaotic systems as described in PRL 125, 180604 (2020).

        entro_random_gaussian(L: int, f: float):
            Computes the entropy for random Gaussian states in the thermodynamic limit.

        entro_random_gaussian_u1(L: int, f: float, n=0.5):
            Computes the entropy for random Gaussian states with U(1) conservation in the thermodynamic limit.

        entro_page_th(L: int, f: float):
            Computes the Page value (average entropy of a subsystem) in the thermodynamic limit.

        entro_page(f: float, L: int):
            Computes the Page value for a given subsystem size.

        entro_page_u1(f: float, L: int, n=0.5):
            Placeholder for Page value computation with U(1) correction.

        entro_page_u1_th(f: float, L: int, n=0.5):
            Computes the Page value with U(1) correction in the thermodynamic limit.
    """

    # ---------------------------------
    
    class Mean:
        
        @staticmethod
        def volume_law(L, f = 0.5):
            """
            Calculates the entanglement entropy according to the volume law for a system of size L.

            Parameters:
                L (int or float):
                    The size of the system (e.g., number of sites or particles).
                f (float, optional):
                    Fraction of the system considered. Default is 0.5.

            Returns:
                float: The entanglement entropy given by (f * L) * log(2).

            Notes:
                The volume law states that the entanglement entropy scales linearly with the size of the subsystem.
            """
            return (f * L) * np.log(2)

        @staticmethod
        def quadratic_chaotic(L : int, f : float):
            '''
            PRL 125, 180604 (2020)
            - La: subsystem size
            - f : filling
            '''
            La = int(L * f)
            return (1. - (1. + (1. - f) * np.log(1. - f) / f) / np.log(2.)) * La * np.log(2.)

        @staticmethod
        def free_fermions_half(L : int, f : float):
            '''
            PRL 119, 020601 (2017)
            - La: subsystem size
            - f : filling
            '''
            return 0.5378
    
        @staticmethod
        def random_gaussian_th(L: int, f: float):
            """
            Random Gaussian states in the thermodynamic limit.

            Parameters:
            L (int): System size.
            f (float): Filling fraction.

            Returns:
            float: Entanglement entropy for random Gaussian states.
            """
            return L * (f * (np.log(2.) - 1.) + (f - 1.) * np.log(1. - f)) + 0.5 * f + 0.25 * np.log(1. - f)

        @staticmethod
        def random_gaussian_u1_th(L: int, f: float, n: float = 0.5):
            """
            Random Gaussian states with U(1) conservation in the thermodynamic limit.

            Parameters:
            L (int): System size.
            f (float): Filling fraction.
            n (float, optional): Fermionic filling. Default is 0.5.

            Returns:
            float: Entanglement entropy for random Gaussian states with U(1) conservation.
            """
            # Note: The variable V is not defined in the original code. Please define V if needed.
            return L * ((f - 1.0) * np.log(1.0 - f) + f * ((n - 1.0) * np.log(1.0 - n) - n * np.log(n) - 1))  # + (f * (1.0 - f + n * (1.0 - n))) / (12 * (1.0 - f) * (1.0 - n) * n * V)

        @staticmethod
        def page_th(L: int, f: float):
            """
            Page value in the thermodynamic limit.

            Parameters:
            L (int): System size.
            f (float): Filling fraction.

            Returns:
            float: Page value.
            """
            return f * L * np.log(2) - binom(2 * f * L, f * L) / binom(L, L / 2) / 2

        @staticmethod
        def page(La: int, Lb: int):
            """
            Page value for given subsystem sizes.

            Parameters:
            La (int): Subsystem A size.
            Lb (int): Subsystem B size.

            Returns:
            float: Page value.
            """
            da = 2 ** La
            db = 2 ** Lb
            return digamma(da * db + 1) - digamma(db + 1) - (da - 1) / (2 * db)

        @staticmethod
        def page_u1(La: int, Lb: int, n: float = 0.5):
            """
            Page result with the correction for U(1).

            Parameters:
            La (int): Subsystem A size.
            Lb (int): Subsystem B size.
            n (float, optional): Fermionic filling. Default is 0.5.

            Returns:
            float: Page value with U(1) correction.
            """
            Sval = 0
            L_tot = int(La + Lb)
            N = int(L_tot * n)
            for na in range(0, min(N, La) + 1):
                d_a = binom(La, na)
                d_b = binom(Lb, N - na)
                d_N = binom(L_tot, N)
            # page_result2 is not defined in the original code. Replace with appropriate function if needed.
            Sval += d_a * d_b / d_N * (digamma(d_a * d_b + 1) - digamma(d_b + 1) - (d_a - 1) / (2 * d_b) + digamma(d_N + 1) - digamma(d_a * d_b + 1))
            return Sval

        @staticmethod
        def page_u1_th(f: float, L: int, n: float = 0.5):
            """
            Page results with U(1) correction in the thermodynamic limit.

            Parameters:
            f (float): Filling fraction.
            L (int): System size.
            n (float, optional): Fermionic filling. Default is 0.5.

            Returns:
            float: Page value with U(1) correction in the thermodynamic limit.
            """
            return ((n - 1.0) * np.log(1.0 - n) - n * np.log(n)) * f * L \
            - np.sqrt(n * (1.0 - n) / (2.0 * np.pi)) * np.abs(np.log((1.0 - n) / n)) * (1.0 if f == 0.5 else 0.0) * np.sqrt(L) \
            + (f + np.log(1 - f)) / 2.0 \
            - 0.5 * (1.0 if f == 0.5 else 0.0) * (1.0 if n == 0.5 else 0.0)

    # ---------------------------------
    
    class Var:
        """
        A collection of static methods for calculating the variance of entanglement entropy in quantum systems.

        This class provides methods to compute the variance of entanglement entropy based on the Page value and other parameters.

        Methods:
            page_var(LA, LB):
                Computes the variance of entanglement entropy based on subsystem sizes LA and LB.
        """

        # ---------------------------------

        @staticmethod
        def page_var(LA, LB):
            """
            Computes the variance of the entanglement entropy (Page variance) for subsystems of sizes LA and LB.

            Parameters:
            LA (int): Size of subsystem A.
            LB (int): Size of subsystem B.

            Returns:
            float: Variance of the entanglement entropy.
            """
            d_a     = 2 ** LA
            d_b     = 2 ** LB
            term1   = ((d_a + d_b) / (d_a * d_b + 1.0)) * polygamma(1, d_b + 1)
            term2   = polygamma(1, d_a * d_b + 1)
            term3   = ((d_a - 1) * (d_a + 2.0 * d_b - 1.0)) / (4.0 * d_b ** 2 * (d_a * d_b + 1.0))
            return term1 - term2 - term3
    
        # ---------------------------------
        
    mean = Mean()
    var  = Var()
    
    # ---------------------------------

    ################################# TYPICAL ##############################

    @staticmethod
    def entro_random_gaussian(L : int, f : float):
        '''
        Random Gaussian states in thermodynamic limit
        - L : system size
        - f : filling
        '''
        La = int(L * f)
        return  (L - 0.5) * psi(2 * L) + (0.5 + La - L) * psi(2 * L - 2 * La) + (0.25 - La) * psi(L) - 0.25 * psi(L - La) - La
        # return L * (f * (np.log(2.) - 1.) + (f - 1.) * np.log(1. - f)) + 0.5 * f + 0.25 * np.log(1. - f)

    @staticmethod
    def entro_random_gaussian_u1(L : int, f : float, n = 0.5):
        '''
        Random Gaussian states with U(1) conservation in thermodynamic limit
        - L : system size
        - f : filling
        '''
        return L * ((f - 1.0) * np.log(1.0 - f) + f * ((n - 1.0) * np.log(1.0 - n) - n * np.log(n) - 1)) + (f * (1.0 - f + n * (1.0 - n))) / (12 * (1.0 - f) * (1.0 - n) * n * V)

    ################################# MB CHAOS ##############################

    @staticmethod
    def entro_page_th(L : int, f : float):
        '''
        Page value in thermodynamic limit.
        - L : system size
        - f : filling
        '''
        return f * L * np.log(2) - binom(2 * f * L, f * L) / binom(L, L/2) / 2

    @staticmethod
    def entro_page(f    : float, 
                   L    : int):
        '''
        Page value for a given subsystem sizes.
        - La : subsystem size
        - Lb : subsystem size
        '''
        La = int(L * f)
        Lb = L - La
        da = 2**La
        db = 2**Lb
        return digamma(da * db + 1) - digamma(db + 1) - (da - 1) / (2*db)

    @staticmethod
    def entro_page_u1(f : float, 
                      L : int, 
                      n = 0.5):

        '''
        Page result with the correction for U1.
        - La : subsystem size
        - Lb : subsystem size
        - n  : fermionic filling
        '''
        pass
        # Sval = 0
        # L_tot = int(La + Lb)
        # N = int(L_tot * n)
        # for na in range(0, min(N, L_a) + 1):
        #     d_a = binom(La, na)
        #     d_b = binom(Lb, N - na)
        #     d_N = binom(L_tot, N)
        #     Sval += d_a * d_b / d_N * ( page_result2(d_a, d_b) + digamma(d_N + 1) - digamma(d_a * d_b + 1) )
        # return Sval

    @staticmethod
    def entro_page_u1_th(f  : float,
                         L  : int,
                         n  = 0.5):
        '''
        Page results with U1 correction in thermodynamic limit
        - f : filling
        - L : system size
        - n : fermionic filling
        '''
        return ((n-1.0) * np.log(1.0-n) - n*np.log(n))*f*L - np.sqrt(n*(1.0-n)/2.0/np.pi) * np.abs(np.log((1.0-n)/n)) * (1.0 if f == 0.5 else 0.) * np.sqrt(L) + (f+np.log(1-f))/2. - 0.5 * (1. if f == 0.5 else 0) * (1. if n == 0.5 else 0.0)

###################################
#! helpers: eigenvalues from ρ
###################################

@numba.njit(cache=True)
def _eigvals_numba(rho: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(rho) # LAPACK inside

@numba.njit(cache=True)
def _clean_probs(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Clip negatives & zeros from round-off, renormalise."""
    q   = np.where(p < eps, 0.0, p)
    s   = q.sum()
    return q / s if s != 1.0 else q

# ----------------------------------

@numba.njit(cache=True)
def vn_entropy(lam: np.ndarray, base: float = np.e) -> float:
    """
    Calculates the von Neumann entropy for a given probability distribution.

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
    log     = np.log(lam + 1e-30)
    if base != np.e:
        log /= np.log(base)
    return -np.dot(lam, log)

@numba.njit(cache=True)
def renyi_entropy(lam: np.ndarray, q: float, base: float = np.e) -> float:
    """
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
    if q == 1.0:
        return vn_entropy(lam, base)
    lam     = _clean_probs(lam)
    s       = (lam ** q).sum()
    log     = np.log(s)
    
    if base != np.e:
        log /= np.log(base)
    return log / (1.0 - q)

@numba.njit(cache=True)
def tsallis_entropy(lam: np.ndarray, q: float) -> float:
    """
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
        - C. Tsallis, "Possible generalization of Boltzmann-Gibbs statistics," J. Stat. Phys. 52, 479–487 (1988).
    """
    if q == 1.0:
        return vn_entropy(lam)
    lam = _clean_probs(lam)
    return (1.0 - (lam ** q).sum()) / (q - 1.0)

@numba.njit(cache=True)
def sp_correlation_entropy(lam: np.ndarray, q: float, base: float = np.e):    
    """
    Compute the single-particle correlation entropy for a set of eigenvalues.
    This function calculates either the von Neumann entropy (for q=1) or the Rényi entropy (for generic q)
    of a set of eigenvalues `lam` (typically from a correlation matrix), after mapping each eigenvalue
    from the interval [-1, 1] to a probability in [0, 1].
    Parameters
    ----------
    lam : np.ndarray
        Array of eigenvalues (λ), each in the interval [-1, 1].
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
          S = -Σ [p * log(p) + (1-p) * log(1-p)]
      where p = 0.5 * (1 + λ).
    - For q ≠ 1, the function computes the Rényi entropy:
          S_q = (1 / (1-q)) * Σ log(p^q + (1-p)^q)
    - The entropy is normalized by the logarithm of the specified base.
    """
    
    log_base = np.log(base)
    
    #! von‑Neumann entropy  (q == 1)
    if np.abs(q - 1.0) < 1e-12:
        s = 0.0
        LOG_TWO = np.log(2.0)
        for l in lam:
            if l > -1.0:
                s += (1.0 + l) * (np.log1p(l) - LOG_TWO)
            if l < 1.0:
                s += (1.0 - l) * (np.log1p(-l) - LOG_TWO)
        return -0.5 * s

    #! Rényi entropy  (generic q)
    inv_1mq = 1.0 / (1.0 - q)
    s = 0.0
    for l in lam:
        p  = 0.5 * (1.0 + l)
        pm = 1.0 - p
        s += np.log(p ** q + pm ** q)
    return inv_1mq * s / log_base

@unique
class Entanglement(Enum):
    VN      = 1
    RENYI   = 2
    TSALIS  = 3
    SINGLE  = 4

def entropy(lam: np.ndarray, q: float = 1.0, base: float = np.e, *, typek: Entanglement = Entanglement.VN, backend: str = "numpy") -> float:
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
                - Entanglement.TSALIS:
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
        elif typek == Entanglement.TSALIS:
            return tsallis_entropy(lam, q, base)
        elif typek == Entanglement.SINGLE:
            return sp_correlation_entropy(lam, q, base)
    elif backend.lower() == 'jax' and JAX_AVAILABLE:
        if typek == Entanglement.VN:
            return jnp.vn_entropy(lam, base)
        elif typek == Entanglement.RENYI:
            return jnp.renyi_entropy(lam, q, base)
        elif typek == Entanglement.TSALIS:
            return jnp.tsallis_entropy(lam, q, base)
        elif typek == Entanglement.SINGLE:
            return jnp.sp_correlation_entropy(lam, q, base)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'jax'.")

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

####################################