'''
Math utilities for various mathematical operations, fittings, and distributions.
It includes functions for finding maxima, nearest values, and fitting data to various models.

Imports:
- scipy.optimize.curve_fit
- scipy.interpolate.splrep
- scipy.interpolate.BSpline

Functions:
- Fitter
- FitterParams

File    : general_python/maths/math_utils.py
Version : 0.1.0
Author  : Maksymilian Kliczkowski
License : MIT
'''

# fitting and distributions
from scipy.optimize import curve_fit as fit
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
from scipy.stats import poisson, norm, expon, gaussian_kde
from scipy.special import gamma

# fit the functions
try:
    import pandas as pd
except ImportError:
    pd = None

import math
import numpy as np

# Optional JAX support without import-time side effects
try:
    import jax.numpy as jnp     # type: ignore
    _JAX_AVAILABLE = True
except Exception:               # pragma: no cover - optional dependency
    jnp = None                  # type: ignore
    _JAX_AVAILABLE = False

TWOPI       = math.pi * 2
PI          = math.pi
PIHALF      = math.pi / 2

#################################### FINDERS ####################################

def find_maximum_idx(x):
    """
    Find maximum index in a DataFrame, numpy array, or JAX array.

    Parameters
    ----------
    x : Union[pd.DataFrame, np.ndarray, jnp.ndarray]
        Input data.

    Returns
    -------
    int or index
        Index of the maximum value.

    Raises
    ------
    TypeError
        If input is not a DataFrame, numpy array, or JAX array.
    """
    if pd is not None and isinstance(x, pd.DataFrame):
        return x.idxmax(axis=1)
    elif isinstance(x, np.ndarray):
        return np.argmax(x, axis=1)
    elif _JAX_AVAILABLE:
        try:
            return jnp.argmax(x, axis=1)
        except Exception:
            raise TypeError("Unsupported JAX array type or shape for argmax")
    else:
        raise TypeError("Input must be a DataFrame, numpy array, or JAX array")
    
def find_nearest_val(x, val, col):
    """
    Find the nearest value to the given value.

    Parameters
    ----------
    x : Union[pd.DataFrame, np.ndarray, jnp.ndarray]
        Input data container.
    val : scalar
        The value to search for.
    col : str
        Column name (only for DataFrame inputs).

    Returns
    -------
    scalar
        The value in `x` closest to `val`.
    """
    if pd is not None and isinstance(x, pd.DataFrame):
        return x.loc[(x[col]-val).abs().idxmin()]
    elif isinstance(x, np.ndarray):
        return np.array((np.abs(x - val)).argmin())
    elif _JAX_AVAILABLE:
        try:
            return jnp.array((jnp.abs(x - val)).argmin())
        except Exception:
            raise TypeError("Unsupported JAX array type for nearest value computation")
    else:
        raise TypeError("Input must be a DataFrame, numpy array, or JAX array")

def find_nearest_idx(x, val : float, **kwargs):
    """
    Find the index of the nearest value to the given value.

    Parameters
    ----------
    x : Union[pd.DataFrame, np.ndarray, jnp.ndarray]
        Input data container.
    val : float
        The target value.
    **kwargs
        Additional arguments, e.g., `col` for DataFrame column name.

    Returns
    -------
    int or index
        Index of the nearest value.
    """
    if pd is not None and isinstance(x, pd.DataFrame):
        col = kwargs.get('col', None)
        if col is None:
            raise ValueError("Column name must be provided for DataFrame.")
        return (x[col]-val).abs().idxmin()
    elif isinstance(x, np.ndarray):
        return (np.abs(x - val)).argmin()
    elif _JAX_AVAILABLE:
        try:
            return jnp.array((jnp.abs(x - val)).argmin())
        except Exception:
            raise TypeError("Unsupported JAX array type for nearest index computation")
    else:
        raise TypeError("Input must be a DataFrame, numpy array, or JAX array")
    
###################################### FITS ######################################

class FitterParams(object):
    """
    Class that stores the parameters of a fit function.

    Attributes
    ----------
    popt : list
        Parameters of the fit.
    pcov : list
        Covariance matrix of the fit.
    funct : callable
        The fitted function.
    """
    
    def __init__(self, funct, popt, pcov):
        """
        Initialize the FitterParams.

        Parameters
        ----------
        funct : callable
            Function of the fit.
        popt : list or array
            Parameters of the fit.
        pcov : list or array
            Covariance matrix of the fit.
        """
        self._popt   = popt
        self._pcov   = pcov
        self._funct  = funct
    
    def get_popt(self):
        return self._popt
    
    def get_pcov(self):
        return self._pcov
    
    def get_fun(self):
        return self._funct
    
    @property
    def popt(self):
        return self._popt
    
    @property
    def pcov(self):
        return self._pcov

    @property
    def funct(self):
        return self._funct
    
    def __call__(self, x):
        return self._funct(x)
    
    def __str__(self):
        return 'FitterParams: ' + str(self._popt) + ' ' + str(self._pcov)
    
class Fitter:
    """
    Class providing curve fitting functionalities.

    Contains methods for fitting various mathematical models (linear, exponential, power law, etc.)
    to data provided in the constructor or passed directly to static methods.
    """
    
    ###################################################
    
    def __init__(self, x : np.ndarray, y : np.ndarray):
        """
        Initialize the Fitter.

        Parameters
        ----------
        x : np.ndarray
            Arguments (independent variable).
        y : np.ndarray
            Values (dependent variable).
        """
        self._x      = x
        self._y      = y
        self._fitter = FitterParams(None, None, None)

    ###################################################
    
    def apply(self, x : np.ndarray):
        return self._fitter.funct(x) 
        
    ###################################################
    
    @staticmethod
    def skip(x, y, skipF = 0, skipL = 0):
        """
        Skips a certain part of the values for the fit.

        Parameters
        ----------
        x : array-like
            Arguments to trim.
        y : array-like
            Values to trim.
        skipF : int, optional
            Number of first elements to skip.
        skipL : int, optional
            Number of last elements to skip.

        Returns
        -------
        tuple
            (trimmed_x, trimmed_y)
        """
        xfit            =       x[skipF if skipF!= 0 else None : -skipL if skipL!= 0 else None]
        yfit            =       y[skipF if skipF!= 0 else None : -skipL if skipL!= 0 else None]
        return xfit, yfit
    
    #################### F I T S ! #################### 
    
    def fit_linear(self, skipF = 0, skipL = 0):
        """
        Fits a linear function (y = ax + b).

        Parameters
        ----------
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.
        """
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        a, b            =       np.polyfit(xfit, yfit, 1)
        self._fitter    =       FitterParams(lambda x: a * x + b, [a, b], [])

    @staticmethod
    def fitLinear(  x, y,
                    skipF = 0,
                    skipL = 0):
        """
        Fits a linear function (y = ax + b) statically.

        Parameters
        ----------
        x : array-like
            Arguments.
        y : array-like
            Values.
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.

        Returns
        -------
        FitterParams
            Fitted parameters and function.
        """
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        a, b            =       np.polyfit(xfit, yfit, 1)
        return FitterParams(lambda x: a * x + b, [a, b], [])

    #############
    
    def fit_exp(self,
                skipF   = 0,
                skipL   = 0):
        """
        Fits an exponential function [a * exp(-b * x)].

        Parameters
        ----------
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.
        """
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        funct           =       lambda x, a, b: a * np.exp(-b * x)
        popt, pcov      =       fit(funct, xfit, yfit)
        self._fitter     =       FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
        
    @staticmethod
    def fitExp( x,
                y,
                skipF   = 0,
                skipL   = 0):
        """
        Fits an exponential function [a * exp(-b * x)] statically.

        Parameters
        ----------
        x : array-like
            Arguments.
        y : array-like
            Values.
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.

        Returns
        -------
        FitterParams
            Fitted parameters and function.
        """
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        funct           =       lambda x, a, b: a * np.exp(-b * x)
        popt, pcov      =       fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
    
    #############

    def fit_x_plus_x2(  self,
                        skipF   = 0,
                        skipL   = 0):
        """
        Fits [a * x + b * x^2].

        Parameters
        ----------
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.
        """
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        funct           =       lambda x, a, b: (a * x) + (b * x ** 2)
        popt, pcov      =       fit(funct, xfit, yfit)
        self._fitter    =       FitterParams(lambda x: popt[0] * x + popt[1] * x ** 2, popt, pcov)
        
    @staticmethod
    def fitXPlusX2( x,
                    y,
                    skipF   = 0,
                    skipL   = 0):
        """
        Fits [a * x + b * x^2] statically.

        Parameters
        ----------
        x : array-like
            Arguments.
        y : array-like
            Values.
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.

        Returns
        -------
        FitterParams
            Fitted parameters and function.
        """
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        funct           =       lambda x, a, b: (a * x) + (b * x ** 2)
        popt, pcov      =       fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * x + popt[1] * x ** 2, popt, pcov)
    
    #############
    
    def fit_power(self, 
                  skipF = 0,
                  skipL = 0):
        """
        Fits a power law function [a * x^b].

        Parameters
        ----------
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.
        """
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        funct           =       lambda x, a, b: a * x ** b
        popt, pcov      =       fit(funct, xfit, yfit)
        self._fitter    =       FitterParams(funct, popt, pcov)
        
    @staticmethod
    def fitPower( x,
                  y,
                  skipF = 0,
                  skipL = 0):
        """
        Fits a power law function [a * x^b] statically.

        Parameters
        ----------
        x : array-like
            Arguments.
        y : array-like
            Values.
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.

        Returns
        -------
        FitterParams
            Fitted parameters and function.
        """
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        funct           =       lambda x, a, b: a * x ** b
        popt, pcov      =       fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * (x ** popt[1]), popt, pcov)
        
    #################### G E N R L #################### 

    def fit_any(self,
                funct,
                skipF   = 0,
                skipL   = 0):
        """
        Fits an arbitrary user-provided function.

        Parameters
        ----------
        funct : callable
            Function to fit.
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.
        """
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        popt, pcov      =       fit(funct, xfit, yfit)
        self._fitter     =       FitterParams(funct, popt, pcov)
    
    @staticmethod
    def fitAny( x,
                y,  
                funct,
                skipF   = 0,
                skipL   = 0, 
                bounds  = []):
        """
        Fits an arbitrary user-provided function statically.

        Parameters
        ----------
        x : array-like
            Arguments.
        y : array-like
            Values.
        funct : callable
            Function to fit.
        skipF : int, optional
            Skip first N elements.
        skipL : int, optional
            Skip last N elements.
        bounds : list, optional
            Bounds for parameters.

        Returns
        -------
        FitterParams
            Fitted parameters and function.
        """
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        if bounds == []:
            popt, pcov  =       fit(funct, xfit, yfit)
            return FitterParams(lambda x: funct(x, *popt), popt, pcov)
        else:
            popt, pcov  =       fit(funct, xfit, yfit, bounds = bounds)
            return FitterParams(lambda x: funct(x, *popt), popt, pcov)
    
    #################### H I S T O #################### 

    @staticmethod
    def gen_cauchy(x, v = 1.0, gamma = 1.0, alpha = 1.0, beta = 1.0):
        """
        Generalized Cauchy distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        v : float
            Normalization factor.
        gamma : float
            Scale parameter related to width.
        alpha : float
            Stability/shape parameter.
        beta : float
            Scale parameter.

        Returns
        -------
        array-like
            Distribution values.
        """
        return v * gamma * (1 + (x * gamma / beta)**2) ** (-(alpha + 1) / 2)
    
    @staticmethod
    def cauchy(x, x0 = 0., gamma = 1.0, v = 1.0):
        """
        Cauchy distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        x0 : float
            Location parameter.
        gamma : float
            Scale parameter.
        v : float
            Normalization/Amplitude.

        Returns
        -------
        array-like
            Distribution values.
        """
        y = v / ((x - x0)**2 + gamma**2)
        return y

    @staticmethod
    def pareto(x, v = 1.0, alpha = 1.0, xm = 1.0, mu = 0.0):
        """
        Pareto distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        v : float
            Normalization factor.
        alpha : float
            Shape parameter.
        xm : float
            Scale parameter.
        mu : float
            Location parameter.

        Returns
        -------
        array-like
            Distribution values.
        """
        return v * np.power(1.0 + alpha * (np.abs(x - mu)), -1.0 / xm - 1.0)

    @staticmethod
    def poisson(x, lambd = 1.0, v = 1.0):
        """
        Poisson distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        lambd : float
            Lambda parameter.
        v : float
            Normalization factor.

        Returns
        -------
        array-like
            Distribution values.
        """
        return v * np.exp(-lambd * (np.abs(x)))
    
    @staticmethod
    def chi2(x, k = 1.0, v = 1.0, z = 1.0):
        """
        Chi-squared distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        k : float
            Degrees of freedom.
        v : float
            Normalization factor.
        z : float
            Scale factor.

        Returns
        -------
        array-like
            Distribution values.
        """
        return v * np.exp(-np.abs(z * x) / 2) * (np.abs(x) ** (k / 2 - 1)) / (2 ** (k/2) * gamma(k/2))
    
    @staticmethod
    def gaussian(x, mu = 0.0, sigma = 1.0):
        """
        Gaussian distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        mu : float
            Mean.
        sigma : float
            Standard deviation.

        Returns
        -------
        array-like
            Distribution values.
        """
        return norm.pdf(x, mu, sigma)
    
    @staticmethod
    def laplace(x, lambd = 1.0, v = 1.0, mu = 0.0):
        """
        Laplace distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        lambd : float
            Scale parameter (decay).
        v : float
            Normalization factor.
        mu : float
            Mean.

        Returns
        -------
        array-like
            Distribution values.
        """
        return (0.5 + 0.5 * np.sign(x-mu) * (1.0 - np.exp(-np.abs(x - mu) / lambd))) / (v)
    
    @staticmethod
    def exponential(x, lambd, sigma):
        """
        Exponential distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        lambd : float
            Lambda parameter.
        sigma : float
            Normalization/Scale factor.

        Returns
        -------
        array-like
            Distribution values.
        """
        return sigma * np.exp(-lambd * np.abs(x))
    
    @staticmethod
    def lorentzian(x, v = 1.0, g = 1.0):
        """
        Lorentzian distribution.

        Parameters
        ----------
        x : array-like
            Arguments.
        v : float
            Amplitude.
        g : float
            Gamma (width) parameter.

        Returns
        -------
        array-like
            Distribution values.
        """
        return np.abs(v) * g / ((x)**2 + g**2)
    
    @staticmethod
    def two_lorentzian(x, v = 1.0, g1 = 1.0, g2 = 1.0, v2 = 1.0):
        """
        Sum of two Lorentzian distributions.

        Parameters
        ----------
        x : array-like
            Arguments.
        v : float
            Amplitude of first Lorentzian.
        g1 : float
            Gamma of first Lorentzian.
        g2 : float
            Gamma of second Lorentzian.
        v2 : float
            Amplitude of second Lorentzian.

        Returns
        -------
        array-like
            Distribution values.
        """
        if g1 < 0 or g2 < 0:
            return 1.0E10
        if v < 0 or v2 < 0:
            return 1.0E10
        
        # return v * (g1 / ((x)**2 + g1**2) + g2 / ((x)**2 + g2**2))
        return Fitter.lorentzian(x, v, g1) + Fitter.lorentzian(x, v2, g2)
    
    # parametrized
    
    @staticmethod
    def lorentzian_system_size(param):
        """
        Returns a Lorentzian function parametrized by system size.
        """
        def lorentzian(x, g = 1.0, v = 1.0):
            return v * (g / param[0]) / ( (x)**2 + (g / param[0])**2 )
        return lorentzian
    
    #################### H I S T O ####################
    
    @staticmethod
    def fit_histogram(edges,
                      counts, 
                      typek     = 'gaussian',
                      skipF     = 0,
                      skipL     = 0,
                      centers   = [],
                      params    = [],
                      bounds    = None):
        """
        Fit a histogram with a specified distribution.

        Parameters
        ----------
        edges : array-like
            Histogram bin edges.
        counts : array-like
            Histogram counts.
        typek : str, optional
            Type of distribution ('gaussian', 'poisson', 'laplace', 'exponential', 'pareto', 'cauchy', 'gen_cauchy', 'chi2', 'lorentzian', 'two_lorentzian', 'lorentzian_system_size').
        skipF : int, optional
            Skip first N bins.
        skipL : int, optional
            Skip last N bins.
        centers : array-like, optional
            Bin centers. If empty, computed from edges.
        params : list, optional
            Extra parameters (e.g., for lorentzian_system_size).
        bounds : tuple, optional
            Bounds for optimization.

        Returns
        -------
        FitterParams
            Fitted parameters and function.
        """
        
        if len(centers) == 0:
            if len(edges) <= 1:
                raise ValueError('Edges must have at least 2 elements')
            centers = ((edges[:-1] + edges[1:]) / 2)
            
        match typek:
            case 'gaussian':
                funct = Fitter.gaussian
            case 'poisson':
                funct = Fitter.poisson
            case 'laplace':
                funct = Fitter.laplace
            case 'exponential':
                funct = Fitter.exponential
            case 'pareto':
                funct = Fitter.pareto
            case 'cauchy':
                funct = Fitter.cauchy
            case 'gen_cauchy':
                funct = Fitter.gen_cauchy
            case 'chi2':
                funct = Fitter.chi2
            case 'lorentzian':
                funct = Fitter.lorentzian
            case 'two_lorentzian':
                funct = Fitter.two_lorentzian
            case 'lorentzian_system_size':
                funct = Fitter.lorentzian_system_size(params)
            case _:
                raise ValueError('Type not recognized: ' + typek)
        
        centers, counts = Fitter.skip(centers, counts, skipF, skipL)
        popt, pcov      = fit(funct, centers, counts)
        return FitterParams(lambda x: funct(x, *popt), popt, pcov)
    
    @staticmethod
    def get_histogram(typek = 'gaussian'):
        """
        Get the histogram function based on the type of distribution.
        
        Parameters
        ----------
        typek : str
            Distribution type name.

        Returns
        -------
        callable
            The distribution function.
        """
        if typek == 'gaussian':
            return Fitter.gaussian
        elif typek == 'poisson':
            return Fitter.poisson
        elif typek == 'laplace':
            return Fitter.laplace
        elif typek == 'exponential':
            return Fitter.exponential
        elif typek == 'pareto':
            return Fitter.pareto
        elif typek == 'cauchy':
            return Fitter.cauchy
        else:
            raise ValueError('Type not recognized: ' + typek)

##############

def next_power(x : float, base : int = 2):
    """
    Get the next power of a number (base) that is greater than x.

    Parameters
    ----------
    x : float
        Number to get the next power of.
    base : int, optional
        Base of the power (default 2 for binary, can be 10 for decimal).

    Returns
    -------
    int
        Next power.
    """
    return base ** math.ceil(math.log(x) / math.log(base))

def prev_power(x : float, base : int = 2):
    """
    Get the previous power of a number (base) that is smaller than x.

    Parameters
    ----------
    x : float
        Number to get the previous power of.
    base : int, optional
        Base of the power (default 2 for binary, can be 10 for decimal).

    Returns
    -------
    int
        Previous power.
    """
    return base ** math.floor(math.log(x) / math.log(base))

#################################################################################

def mod_euc(a: int, b: int) -> int:
    """
    Compute the modified Euclidean remainder of a divided by b.
    
    This function ensures that the result has the same sign as b.
    
    Parameters
    ----------
    a : int
        Dividend.
    b : int
        Divisor.

    Returns
    -------
    int
        Euclidean remainder.

    Raises
    ------
    ValueError
        If b is zero.
    """
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    m = a % b
    if m < 0:
        m = m - b if b < 0 else m + b
    return m

def mod_floor(a: int, b: int) -> int:
    """
    Compute the modified floor division of a divided by b.
    
    This function ensures that the result has the same sign as b.
    
    Parameters
    ----------
    a : int
        Dividend.
    b : int
        Divisor.

    Returns
    -------
    int
        Floor division result.

    Raises
    ------
    ValueError
        If b is zero.
    """
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    m = a // b
    if (a < 0) != (b < 0) and a % b != 0:
        m -= 1
    return m

def mod_ceil(a: int, b: int) -> int:
    """
    Compute the modified ceiling division of a divided by b.
    
    This function ensures that the result has the same sign as b.
    
    Parameters
    ----------
    a : int
        Dividend.
    b : int
        Divisor.

    Returns
    -------
    int
        Ceiling division result.

    Raises
    ------
    ValueError
        If b is zero.
    """
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    m = a // b
    if (a < 0) == (b < 0) and a % b != 0:
        m += 1
    return m

def mod_trunc(a: int, b: int) -> int:
    """
    Compute the modified truncation division of a divided by b.
    
    This function ensures that the result has the same sign as a.
    
    Parameters
    ----------
    a : int
        Dividend.
    b : int
        Divisor.

    Returns
    -------
    int
        Truncation result.

    Raises
    ------
    ValueError
        If b is zero.
    """
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    return a // b

def mod_round(a: int, b: int) -> int:
    """
    Compute the modified rounding division of a divided by b.
    
    This function ensures that the result has the same sign as a.
    
    Parameters
    ----------
    a : int
        Dividend.
    b : int
        Divisor.

    Returns
    -------
    int
        Rounded result.

    Raises
    ------
    ValueError
        If b is zero.
    """
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    m = a / b
    if m < 0:
        m = m - 1 if a % b < 0 else m + 1
    return int(m)

#################################################################################
