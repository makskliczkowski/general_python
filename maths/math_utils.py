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
    ''' 
    Find maximum index in a DataFrame, numpy array, or JAX array
    - x : DataFrame, numpy array, or JAX array
    '''
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
    ''' 
    Find the nearest value to the value given 
    - x     : a DataFrame or numpy array
    - val   : a scalar
    - col   : a string on which column to find the nearest
    '''
    if pd is not None and isinstance(x, pd.DataFrame):
        idx = (x[col]-val).abs().idxmin()
        return x.at[idx, col]
    elif isinstance(x, np.ndarray):
        idx = (np.abs(x - val)).argmin()
        return x.flat[idx]
    elif _JAX_AVAILABLE:
        try:
            idx = (jnp.abs(x - val)).argmin()
            return x.ravel()[idx]
        except Exception:
            raise TypeError("Unsupported JAX array type for nearest value computation")
    else:
        raise TypeError("Input must be a DataFrame, numpy array, or JAX array")

def find_nearest_idx(x, val : float, **kwargs):
    ''' 
    Find the nearest idx to the value given 
    - x     : a DataFrame or numpy array
    - val   : a scalar
    - col   : a string on which column to find the nearest
    Returns the index of the nearest value to the given value
    '''
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
    '''
    Class that stores only the parameters of the fit function
    - popt  :   parameters of the fit
    - pcov  :   covariance matrix of the fit
    - funct :   function of the fit
    '''
    
    def __init__(self, funct, popt, pcov):
        '''
        Initialize the class
        - funct :   function of the fit
        - popt  :   parameters of the fit
        - pcov  :   covariance matrix of the fit
        '''
        self._popt   = popt
        self._pcov   = pcov
        self._funct  = funct
    
    def get_popt(self):
        """Optimized fit parameters."""
        return self._popt
    
    def get_pcov(self):
        """Estimated covariance matrix of the fitted parameters."""
        return self._pcov
    
    def get_fun(self):
        """Fitted callable."""
        return self._funct
    
    @property
    def popt(self):
        """Optimized fit parameters."""
        return self._popt
    
    @property
    def pcov(self):
        """Estimated covariance matrix of the fitted parameters."""
        return self._pcov

    @property
    def funct(self):
        """Fitted callable."""
        return self._funct
    
    def __call__(self, x):
        return self._funct(x)
    
    def __str__(self):
        return 'FitterParams: ' + str(self._popt) + ' ' + str(self._pcov)
    
# -----------------------------------------------------------------------------    
    
class Fitter:
    '''
    Class that contains the fit functions and their general usage.
    - x     :   arguments
    - y     :   values
    - fitter:   FitterParams object
    '''
    
    ###################################################
    
    def __init__(self, x : np.ndarray, y : np.ndarray):
        '''
        Initialize the class
        - x     : arguments
        - y     : values
        '''
        self._x      = x
        self._y      = y
        self._fitter = FitterParams(None, None, None)

    ###################################################
    
    def apply(self, x : np.ndarray):
        """Evaluate the current fitted function at ``x``."""
        return self._fitter.funct(x) 
        
    ###################################################
    
    @staticmethod
    def skip(x, y, skipF = 0, skipL = 0):
        '''
        Skips a certain part of the values for the fit
        - x     :   arguments to trim
        - y     :   values to trim
        - skipF :   number of first elements to skip
        - skipL :   number of last elements to skip
        '''
        xfit            =       x[skipF if skipF!= 0 else None : -skipL if skipL!= 0 else None]
        yfit            =       y[skipF if skipF!= 0 else None : -skipL if skipL!= 0 else None]
        return xfit, yfit

    # ------------------------------------------------------------------
    # Robust helpers for scaling analysis
    # ------------------------------------------------------------------

    _MEAN_ALIASES   = {
        'arithmetic'    : {'arithmetic', 'arith', 'avg', 'average', 'mean'},
        'typical'       : {'typical', 'typ', 'geometric', 'geo', 'log-mean', 'logmean'},
        'median'        : {'median', 'med'},
        'harmonic'      : {'harmonic', 'harm'},
    }

    @staticmethod
    def _canonical_mean(mean_type: str) -> str:
        key = str(mean_type).strip().lower()
        
        for canon, aliases in Fitter._MEAN_ALIASES.items():
            if key in aliases:
                return canon
        raise ValueError(f"Unknown mean_type '{mean_type}'. Use one of: arithmetic/avg/mean, typical/typ/geometric, median, harmonic.")

    @staticmethod
    def _as_1d(a, dtype=float):
        return np.asarray(a, dtype=dtype).ravel()

    @staticmethod
    def _prepare_xy(x, y, *, require_positive_x: bool = False, require_positive_y: bool = False, eps: float = 1e-300, dtype=float):
        """Prepare x and y arrays for fitting by filtering out non-finite values and optionally enforcing positivity. Returns the filtered x and y arrays."""
        
        x_arr = Fitter._as_1d(x, dtype=dtype)
        y_arr = Fitter._as_1d(y, dtype=dtype)
        if x_arr.size != y_arr.size:
            raise ValueError(f"x and y must have same length, got {x_arr.size} and {y_arr.size}.")

        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if require_positive_x and x_arr.dtype in [np.float32, np.float64]:
            mask &= x_arr > 0.0
        if require_positive_y and y_arr.dtype in [np.float32, np.float64]:
            mask &= y_arr > eps

        x_ok = x_arr[mask]
        y_ok = y_arr[mask]
        if x_ok.size < 2:
            raise ValueError("Need at least 2 valid points after filtering.")
        return x_ok, y_ok

    @staticmethod
    def aggregate(values, *, mean_type: str = 'arithmetic', weights=None, trim_fraction: float = 0.0, eps: float = 1e-300):
        """
        Aggregate samples with configurable averaging convention.

        Parameters
        ----------
        values : array-like
            Input samples.
        mean_type : str
            One of arithmetic/avg/mean, typical/typ/geometric, median, harmonic.
        weights : array-like, optional
            Optional weights for arithmetic averaging only.
        trim_fraction : float, default=0.0
            Fraction trimmed from each tail for arithmetic mean (0 <= trim < 0.5).
        eps : float, default=1e-300
            Positivity cutoff for logarithmic/harmonic aggregations.
        """
        arr     = Fitter._as_1d(values, dtype=float)
        arr     = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan

        mode    = Fitter._canonical_mean(mean_type)

        if mode == 'median':
            return float(np.median(arr))

        if mode == 'typical':
            pos = arr[arr > eps]
            if pos.size == 0:
                return np.nan
            return float(np.exp(np.mean(np.log(pos))))

        if mode == 'harmonic':
            pos = arr[arr > eps]
            if pos.size == 0:
                return np.nan
            return float(pos.size / np.sum(1.0 / pos))

        # arithmetic
        trim = float(trim_fraction)
        if trim < 0.0 or trim >= 0.5:
            raise ValueError("trim_fraction must satisfy 0 <= trim_fraction < 0.5.")

        if trim > 0.0 and arr.size >= 3:
            arr = np.sort(arr)
            k = int(trim * arr.size)
            if 2 * k < arr.size:
                arr = arr[k: arr.size - k]

        if weights is None:
            return float(np.mean(arr))

        w = Fitter._as_1d(weights, dtype=float)
        if w.size != Fitter._as_1d(values, dtype=float).size:
            raise ValueError("weights must have the same length as values.")
        
        # apply mask to both values and weights
        mask    = np.isfinite(Fitter._as_1d(values, dtype=float)) & np.isfinite(w)
        v       = Fitter._as_1d(values, dtype=float)[mask]
        wv      = w[mask]
        s       = np.sum(wv)
        if s == 0.0:
            return np.nan
        return float(np.dot(v, wv) / s)

    @staticmethod
    def fit_loglog_linear(x, y):
        """Fit ``log(y) = a + b log(x)`` and return ``(a, b, r2)``."""
        x_ok, y_ok  = Fitter._prepare_xy(x, y, require_positive_x=True, require_positive_y=True)
        lx          = np.log(x_ok)
        ly          = np.log(y_ok)
        b, a        = np.polyfit(lx, ly, 1)
        yhat        = a + b * lx
        ss_res      = float(np.sum((ly - yhat) ** 2))
        ss_tot      = float(np.sum((ly - np.mean(ly)) ** 2))
        r2          = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
        return float(a), float(b), float(r2)

    @staticmethod
    def fit_power_scaling(x, y):
        """Fit ``y = A x^beta`` and return ``(A, beta, r2_log)``."""
        a, b, r2 = Fitter.fit_loglog_linear(x, y)
        return float(np.exp(a)), float(b), float(r2)

    @staticmethod
    def fit_inverse_power_scaling(x, y):
        """Fit ``y = A x^{-alpha}`` and return ``(A, alpha, r2_log)``."""
        A, beta, r2 = Fitter.fit_power_scaling(x, y)
        return float(A), float(-beta), float(r2)

    @staticmethod
    def fit_ipr_scaling(dimensions, iprs, q: float = 2.0):
        r"""
        Fit ``IPR(N) = A N^{-alpha}`` and infer ``D_q = alpha/(q-1)``.

        Returns
        -------
        tuple
            ``(A, alpha, D_q, r2_log)``.
        """
        A, alpha, r2    = Fitter.fit_inverse_power_scaling(dimensions, iprs)
        Dq              = alpha / (q - 1.0) if q != 1.0 else np.inf
        return float(A), float(alpha), float(Dq), float(r2)
    
    #################### F I T S ! #################### 
    
    def fit_linear(self, skipF = 0, skipL = 0):
        '''
        Fits a linear function.
        - skipF : skip first arguments
        - skipL : skip last arguments
        '''
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        a, b            =       np.polyfit(xfit, yfit, 1)
        self._fitter    =       FitterParams(lambda x: a * x + b, [a, b], [])

    @staticmethod
    def fitLinear(  x, y,
                    skipF = 0,
                    skipL = 0):
        '''
        Fits a linear function.
        - x     : arguments
        - y     : values
        - skipF : skip first arguments
        - skipL : skip last arguments
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        a, b            =       np.polyfit(xfit, yfit, 1)
        return FitterParams(lambda x: a * x + b, [a, b], [])

    #############
    
    def fit_exp(self,
                skipF   = 0,
                skipL   = 0):
        '''
        Fits [a * exp(-b * x)]
        - skipF : skip first arguments
        - skipL : skip last arguments
        '''
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        funct           =       lambda x, a, b: a * np.exp(-b * x)
        popt, pcov      =       fit(funct, xfit, yfit)
        self._fitter     =       FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
        
    @staticmethod
    def fitExp( x,
                y,
                skipF   = 0,
                skipL   = 0):
        '''
        Fits [a * exp(-b * x)]
        - x     : arguments
        - y     : values
        - skipF : skip first arguments
        - skipL : skip last arguments
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        funct           =       lambda x, a, b: a * np.exp(-b * x)
        popt, pcov      =       fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
    
    #############

    def fit_x_plus_x2(  self,
                        skipF   = 0,
                        skipL   = 0):
        '''
        Fits [a * x + b * x ** 2]
        - skipF :   number of elements to skip on the left
        - skipR :   number of elements to skip on the right
        '''
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        funct           =       lambda x, a, b: (a * x) + (b * x ** 2)
        popt, pcov      =       fit(funct, xfit, yfit)
        self._fitter    =       FitterParams(lambda x: popt[0] * x + popt[1] * x ** 2, popt, pcov)
        
    @staticmethod
    def fitXPlusX2( x,
                    y,
                    skipF   = 0,
                    skipL   = 0):
        '''
        Fits [a * x + b * x ** 2]
        - x     :   arguments to the fit
        - y     :   values to fit
        - skipF :   number of elements to skip on the left
        - skipR :   number of elements to skip on the right
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        funct           =       lambda x, a, b: (a * x) + (b * x ** 2)
        popt, pcov      =       fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * x + popt[1] * x ** 2, popt, pcov)
    
    #############
    
    def fit_power(self, 
                  skipF = 0,
                  skipL = 0):
        '''
        Fits function [a*x**b]
        - x     :   arguments to the fit
        - y     :   values to the fit
        - skipF :   number of elements to skip on the left
        - skipR :   number of elements to skip on the right
        '''
        xfit, yfit      =       Fitter.skip(self._x, self._y, skipF, skipL)
        funct           =       lambda x, a, b: a * x ** b
        popt, pcov      =       fit(funct, xfit, yfit)
        self._fitter    =       FitterParams(funct, popt, pcov)
        
    @staticmethod
    def fitPower( x,
                  y,
                  skipF = 0,
                  skipL = 0):
        '''
        Fits function [a*x**b]
        - x     :   arguments to the fit
        - y     :   values to the fit
        - skipF :   number of elements to skip on the left
        - skipR :   number of elements to skip on the right
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        funct           =       lambda x, a, b: a * x ** b
        popt, pcov      =       fit(funct, xfit, yfit)
        return FitterParams(lambda x: popt[0] * (x ** popt[1]), popt, pcov)
        
    #################### G E N R L #################### 

    def fit_any(self,
                funct,
                skipF   = 0,
                skipL   = 0):
        '''
        Fits function [any]
        - funct :   function to fit to
        - skipF :   number of elements to skip on the left
        - skipR :   number of elements to skip on the right
        '''
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
        '''
        Fits function [any]
        - x     :   arguments to fit
        - y     :   values to fit
        - funct :   function to fit to
        - skipF :   number of elements to skip on the left
        - skipR :   number of elements to skip on the right
        '''
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
        r'''
        Generalized Cauchy distribution
        - v is the normalization factor
        - \alpha is the stability parameter, often referred to as the shape parameter,
        - \beta is the scale parameter,
        - \gamma is a scale parameter related to the width of the distribution.
        '''
        return v * gamma * (1 + (x * gamma / beta)**2) ** (-(alpha + 1) / 2)
    
    @staticmethod
    def cauchy(x, x0 = 0., gamma = 1.0, v = 1.0):
        '''
        Cauchy distribution
        - x     :   arguments
        - x0    :   x0 parameter
        - gamma :   gamma parameter
        '''
        y = v / ((x - x0)**2 + gamma**2)
        return y

    @staticmethod
    def pareto(x, v = 1.0, alpha = 1.0, xm = 1.0, mu = 0.0):
        '''
        Pareto distribution
        - x     :   arguments
        - alpha :   alpha parameter
        - xm    :   xm parameter
        '''
        return v * np.power(1.0 + alpha * (np.abs(x - mu)), -1.0 / xm - 1.0)

    @staticmethod
    def poisson(x, lambd = 1.0, v = 1.0):
        '''
        Poisson distribution
        - k     :   arguments
        - lamb  :   lambda parameter
        '''
        return v * np.exp(-lambd * (np.abs(x)))
    
    @staticmethod
    def chi2(x, k = 1.0, v = 1.0, z = 1.0):
        '''
        Chi2 distribution
        - x     :   arguments
        - k     :   k parameter
        '''
        return v * np.exp(-np.abs(z * x) / 2) * (np.abs(x) ** (k / 2 - 1)) / (2 ** (k/2) * gamma(k/2))
    
    @staticmethod
    def gaussian(x, mu = 0.0, sigma = 1.0):
        '''
        Gaussian distribution
        - x     :   arguments
        - mu    :   mean
        - sigma :   standard deviation
        '''
        return norm.pdf(x, mu, sigma)
    
    @staticmethod
    def laplace(x, lambd = 1.0, v = 1.0, mu = 0.0):
        '''
        Laplace distribution
        - x     :   arguments
        - mu    :   mean
        - b     :   scale parameter
        '''
        return (0.5 + 0.5 * np.sign(x-mu) * (1.0 - np.exp(-np.abs(x - mu) / lambd))) / (v)
    
    @staticmethod
    def exponential(x, lambd, sigma):
        '''
        Exponential distribution
        - x     :   arguments
        - lambd :   lambda parameter
        '''
        return sigma * np.exp(-lambd * np.abs(x))
    
    @staticmethod
    def lorentzian(x, v = 1.0, g = 1.0):
        '''
        Lorentzian distribution
        - x     :   arguments
        - v     :   multiplication constant
        - g     :   gamma parameter
        '''
        return np.abs(v) * g / ((x)**2 + g**2)
    
    @staticmethod
    def two_lorentzian(x, v = 1.0, g1 = 1.0, g2 = 1.0, v2 = 1.0):
        '''
        Two Lorentzian distribution
        - x     :   arguments
        - x0    :   x0 parameter
        - gamma :   gamma parameter
        '''
        if g1 < 0 or g2 < 0:
            return 1.0E10
        if v < 0 or v2 < 0:
            return 1.0E10
        
        # return v * (g1 / ((x)**2 + g1**2) + g2 / ((x)**2 + g2**2))
        return Fitter.lorentzian(x, v, g1) + Fitter.lorentzian(x, v2, g2)
    
    # parametrized
    
    @staticmethod
    def lorentzian_system_size(param):
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
        Get the histogram function based on the type of distribution
        
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
    '''
    Get the next power of a number (base) that is greater than x\
    - x     : number to get the next power
    - base  : base of the power (default 2 for binary, can be 10 for decimal)
    '''
    return base ** math.ceil(math.log(x) / math.log(base))

def prev_power(x : float, base : int = 2):
    '''
    Get the previous power of a number (base) that is smaller than x\
    - x     : number to get the next power
    - base  : base of the power (default 2 for binary, can be 10 for decimal)
    '''
    return base ** math.floor(math.log(x) / math.log(base))

#################################################################################

def mod_euc(a: int, b: int) -> int:
    '''
    Compute the modified Euclidean remainder of a divided by b.
    
    This function ensures that the result has the same sign as b.
    
    - a : integer dividend
    - b : integer divisor
    '''
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    m = a % b
    if m < 0:
        m = m - b if b < 0 else m + b
    return m

def mod_floor(a: int, b: int) -> int:
    '''
    Compute the modified floor division of a divided by b.
    
    This function ensures that the result has the same sign as b.
    
    - a : integer dividend
    - b : integer divisor
    '''
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    m = a // b
    if (a < 0) != (b < 0) and a % b != 0:
        m -= 1
    return m

def mod_ceil(a: int, b: int) -> int:
    '''
    Compute the modified ceiling division of a divided by b.
    
    This function ensures that the result has the same sign as b.
    
    - a : integer dividend
    - b : integer divisor
    '''
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    m = a // b
    if (a < 0) == (b < 0) and a % b != 0:
        m += 1
    return m

def mod_trunc(a: int, b: int) -> int:
    '''
    Compute the modified truncation division of a divided by b.
    
    This function ensures that the result has the same sign as a.
    
    - a : integer dividend
    - b : integer divisor
    '''
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    return a // b

def mod_round(a: int, b: int) -> int:
    '''
    Compute the modified rounding division of a divided by b.
    
    This function ensures that the result has the same sign as a.
    
    - a : integer dividend
    - b : integer divisor
    '''
    if b == 0:
        raise ValueError("Divisor 'b' cannot be zero.")
        
    m = a / b
    if m < 0:
        m = m - 1 if a % b < 0 else m + 1
    return int(m)

#################################################################################
