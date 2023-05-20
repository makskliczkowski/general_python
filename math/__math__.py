import numpy as np

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
        return df.loc[(df[col]-val).abs().idxmin()]
    else:
        return array[(np.abs(array - value)).argmin()]

''' Find the nearest idx to the value given '''
def findNearestIdx(x, val, col):
    if isinstance(x, pd.DataFrame):
        return (df[col]-val).abs().idxmin()
    else:
        return (np.abs(array - value)).argmin()