import numpy as np
import pandas as pd
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
        return array[(np.abs(x - val)).argmin()]

''' Find the nearest idx to the value given '''
def findNearestIdx(x, val, col = ''):
    if isinstance(x, pd.DataFrame):
        return (x[col]-val).abs().idxmin()
    else:
        return (np.abs(x - val)).argmin()