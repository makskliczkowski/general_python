from __random__ import *
from scipy.optimize import curve_fit as fit
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

class Fitter:
    '''
    Class that contains the fit functions and their general usage.
    '''
    
    class FitterParams(object):
        '''
        Class that stores only the parameters of the fit function
        '''
        
        def __init__(self, funct = None, popt = [], pcov = []) -> None:
            self.popt   = popt
            self.pcov   = pcov
            self.funct  = funct
        
        def popt(self):
            return self.popt
        
        def pcov(self):
            return self.pcov
        
        def fun(self):
            return self.fun
        
    ###################################################
    
    def __init__(self, x : np.ndarray, y : np.ndarray):
        self.x      = x
        self.y      = y
        self.fitter = Fitter.FitterParams()

    
    ###################################################
    
    def apply(self, x : np.ndarray):
        return self.funct(x) 
        
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
        xfit            =       x[skipF if skipF!= 0 else None : skipL if skipL!= 0 else None]
        yfit            =       y[skipF if skipF!= 0 else None : skipL if skipL!= 0 else None]
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
        self.fitter     =       Fitter.FitterParams(lambda x: a * x + b, [a, b], [])

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
        return Fitter.FitterParams(lambda x: a * x + b, [a, b], [])

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
        self.fitter     =       Fitter.FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
        
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
        return Fitter.FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
    
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
        self.fitter     =       Fitter.FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
        
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
        return Fitter.FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
     
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
        self.fitter     =       Fitter.FitterParams(lambda x: popt[0] * np.exp(-popt[1] * x), popt, pcov)
    
    @staticmethod
    def fitAny( x,
                y,  
                funct,
                skipF   = 0,
                skipL   = 0):
        '''
        Fits function [any]
        - x     :   arguments to fit
        - y     :   values to fit
        - funct :   function to fit to
        - skipF :   number of elements to skip on the left
        - skipR :   number of elements to skip on the right
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        popt, pcov      =       fit(funct, xfit, yfit)
        return Fitter.FitterParams(lambda x: funct(x, *popt), popt, pcov)