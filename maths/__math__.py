import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as fit

#################################### FINDERS ####################################

''' Find maximum index in a Dataframe'''
def findMaximumIdx(x):
    if isinstance(x, pd.DataFrame):
        return x.idxmax(axis=1)
    else:
        return np.argmax(x, axis = 1)
    
''' Find the nearest value to the value given '''
def findNearestVal(x, val, col):
    if isinstance(x, pd.DataFrame):
        return x.loc[(x[col]-val).abs().idxmin()]
    else:
        return np.array((np.abs(x - val)).argmin())

''' Find the nearest idx to the value given '''
def findNearestIdx(x, val, col = ''):
    if isinstance(x, pd.DataFrame):
        return (x[col]-val).abs().idxmin()
    else:
        return (np.abs(x - val)).argmin()
    
##################################### RANDOM #####################################

'''
Create the CUE matrix using QR decomposition
- n     : size of the matrix (n X n)
- simple: use the straightforward method
'''
def CUE_QR( n        :   int,
            simple   =   True):
    x       =   np.random.normal(size = (n, n)) + 1j * np.random.normal(size = (n, n))
    x       /=  np.sqrt(2)
    Q, R    =   np.linalg.qr(x)
    if not simple:
        d       =   np.diagonal(R)
        ph      =   d / np.abs(d)
        Q       =   np.matmul(Q, ph) * Q
    return Q

class Fitter:
    ########## F I T S ! ########## 
    
    @staticmethod
    def skip(x, 
             y,
             skipF = 0,
             skipL = 0):
        '''
        Skips a certain part of the values for the fit
        '''
        xfit            =       x[skipF if skipF!= 0 else None : skipL if skipL!= 0 else None]
        yfit            =       y[skipF if skipF!= 0 else None : skipL if skipL!= 0 else None]
        return xfit, yfit
    
    @staticmethod
    def fit_linear( x,
                    y,
                    skipF = 0,
                    skipL = 0):
        '''
        Fits linear function
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        a, b            =       np.polyfit(xfit, yfit, 1)
        # return fun and coefficients
        return (lambda x: a * x + b), a, b

    @staticmethod
    def fit_exp_x(  x,
                    y,
                    skipF   = 0,
                    skipL   = 0):
        '''
        Fits a * exp(-b * x)
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        funct           =       lambda x, a, b: a * np.exp(-b * x)
        popt, pcov      =       fit(funct, xfit, yfit)
        # return fun and coefficients
        return funct, popt[0], popt[1]
    
    @staticmethod
    def fit_x_plus_x_sq(    x,
                            y,
                            skipF   = 0,
                            skipL   = 0):
        '''
        Fits a * x + b * x ** 2
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        funct           =       lambda x, a, b: a * x + b * x ** 2
        popt, pcov      =       fit(funct, xfit, yfit)
        # return fun and coefficients
        return lambda x: funct(x, popt[0], popt[1]), popt[0], popt[1]
    
    ########## G E N R L ########## 

    @staticmethod
    def fit_general_fun(    x,
                            y,
                            funct,
                            skipF   = 0,
                            skipL   = 0):
        '''
        Fits any function
        '''
        xfit, yfit      =       Fitter.skip(x, y, skipF, skipL)
        popt, pcov      =       fit(funct, xfit, yfit)
        # return fun and coefficients
        return funct, popt