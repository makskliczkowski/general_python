from .__random__ import *
from scipy.optimize import curve_fit as fit
from scipy.interpolate import splrep, BSpline
from scipy import interpolate

from scipy.stats import poisson, norm, expon, gaussian_kde
from scipy.special import gamma

# fit the functions
import pandas as pd

import math

TWOPI       = math.pi * 2
PI          = math.pi
PIHALF      = math.pi / 2

#################################### FINDERS ####################################

def findMaximumIdx(x):
    ''' 
    Find maximum index in a Dataframe
    - x : Dataframe or numpy array
    '''
    if isinstance(x, pd.DataFrame):
        return x.idxmax(axis=1)
    else:
        return np.argmax(x, axis = 1)
    
def findNearestVal(x, val, col):
    ''' 
    Find the nearest value to the value given 
    - x     : a DataFrame or numpy array
    - val   : a scalar
    - col   : a string on which column to find the nearest
    '''
    if isinstance(x, pd.DataFrame):
        return x.loc[(x[col]-val).abs().idxmin()]
    else:
        return np.array((np.abs(x - val)).argmin())

def findNearestIdx(x, val, col = ''):
    ''' 
    Find the nearest idx to the value given 
    - x     : a DataFrame or numpy array
    - val   : a scalar
    - col   : a string on which column to find the nearest
    '''
    if isinstance(x, pd.DataFrame):
        return (x[col]-val).abs().idxmin()
    else:
        return (np.abs(x - val)).argmin()
    
###################################### FITS ######################################

class FitterParams(object):
    '''
    Class that stores only the parameters of the fit function
    '''
    
    def __init__(self, funct, popt, pcov) -> None:
        self.popt   = popt
        self.pcov   = pcov
        self.funct  = funct
    
    def get_popt(self):
        return self.popt
    
    def get_pcov(self):
        return self.pcov
    
    def get_fun(self):
        return self.funct
    
class Fitter:
    '''
    Class that contains the fit functions and their general usage.
    '''
    ###################################################
    
    def __init__(self, x : np.ndarray, y : np.ndarray):
        self.x      = x
        self.y      = y
        self.fitter = FitterParams()

    ###################################################
    
    def apply(self, x : np.ndarray):
        return self.fitter.funct(x) 
        
    ###################################################
    
    @staticmethod
    def skip(x, 
             y,
             skipF = 0,
             skipL = 0):
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
    
    #################### F I T S ! #################### 
    
    def fit_linear( self,
                    skipF = 0,
                    skipL = 0):
        '''
        Fits a linear function.
        '''
        xfit, yfit      =       Fitter.skip(self.x, self.y, skipF, skipL)
        a, b            =       np.polyfit(xfit, yfit, 1)
        self.fitter     =       FitterParams(lambda x: a * x + b, [a, b], [])

    @staticmethod
    def fitLinear(  x,
                    y,
                    skipF = 0,
                    skipL = 0):
        '''
        Fits linear function (a * x + b)
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
        xfit, yfit      =       Fitter.skip(self.x, self.y, skipF, skipL)
        funct           =       lambda x, a, b: a * np.exp(-b * x)
        popt, pcov      =       fit(funct, xfit, yfit)
        self.fitter     =       FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
        
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
        xfit, yfit      =       Fitter.skip(self.x, self.y, skipF, skipL)
        funct           =       lambda x, a, b: (a * x) + (b * x ** 2)
        popt, pcov      =       fit(funct, xfit, yfit)
        self.fitter     =       FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
        
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
        return FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
    
    #############
    
    def fit_power(self, 
                  skipF = 0,
                  skipL = 0):
        '''
        Fits function [a*x**b]
        - skipF :   number of elements to skip on the left
        - skipR :   number of elements to skip on the right
        '''
        xfit, yfit      =       Fitter.skip(self.x, self.y, skipF, skipL)
        funct           =       lambda x, a, b: a * x ** b
        popt, pcov      =       fit(funct, xfit, yfit)
        self.fitter     =       FitterParams(funct, popt, pcov)
        
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
        xfit, yfit      =       Fitter.skip(self.x, self.y, skipF, skipL)
        popt, pcov      =       fit(funct, xfit, yfit)
        self.fitter     =       FitterParams(funct, popt, pcov)
    
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
            popt, pcov      =       fit(funct, xfit, yfit, bounds = bounds)
            return FitterParams(lambda x: funct(x, *popt), popt, pcov)
    
    #################### H I S T O #################### 

    @staticmethod
    def gen_cauchy(x, v = 1.0, gamma = 1.0, alpha = 1.0, beta = 1.0):
        '''
        Generalized Cauchy distribution
        - v is the normalization factor
        - α is the stability parameter, often referred to as the shape parameter,
        - β is the scale parameter,
        - γ is a scale parameter related to the width of the distribution.
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
        return (alpha * xm ** alpha) / (np.abs(x) ** (alpha + 1))
    
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
                      typ       = 'gaussian',
                      skipF     = 0,
                      skipL     = 0,
                      centers   = [],
                      params    = [],
                      bounds    = None):
        
        if len(centers) == 0:
            if len(edges) <= 1:
                raise ValueError('Edges must have at least 2 elements')
            centers = ((edges[:-1] + edges[1:]) / 2)
            
        funct   = Fitter.gaussian
        if typ == 'gaussian':
            funct = Fitter.gaussian
        elif typ == 'poisson':
            funct = Fitter.poisson
        elif typ == 'laplace':
            funct = Fitter.laplace
        elif typ == 'exponential':
            funct = Fitter.exponential
        elif typ == 'pareto':
            funct = Fitter.pareto
        elif typ == 'cauchy':
            funct = Fitter.cauchy
        elif typ == 'gen_cauchy':
            funct = Fitter.gen_cauchy
        elif typ == 'chi2':
            funct = Fitter.chi2
        elif typ == 'lorenzian':
            funct = Fitter.lorentzian
        elif typ == 'two_lorenzian':
            funct = Fitter.two_lorentzian
        elif typ == 'lorenzian_system_size':
            funct = Fitter.lorentzian_system_size(params)
        else:
            raise ValueError('Type not recognized: ' + typ)
        
        centers, counts = Fitter.skip(centers, counts, skipF, skipL)
        popt, pcov      = fit(funct, centers, counts)
        return FitterParams(lambda x: funct(x, *popt), popt, pcov)
    
    @staticmethod
    def get_histogram(typ = 'gaussian'):
        funct = Fitter.gaussian
        if typ == 'gaussian':
            funct = Fitter.gaussian
        elif typ == 'poisson':
            funct = Fitter.poisson
        elif typ == 'laplace':
            funct = Fitter.laplace
        elif typ == 'exponential':
            funct = Fitter.exponential
        elif typ == 'pareto':
            funct = Fitter.pareto
        elif typ == 'cauchy':
            funct = Fitter.cauchy
        else:
            raise ValueError('Type not recognized: ' + typ)
        return funct
