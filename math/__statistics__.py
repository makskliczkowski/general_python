import numpy as np
import math

TWOPI       = math.pi * 2
PI          = math.pi
PIHALF      = math.pi / 2

############################################### STATISTICAL AVERAGING ###############################################

'''
Calculates a bin average of an array. Does so by computing the new shape of the array
'''
def rebin(arr, av_num, d : int, verbose = False):
    
    # check if the array ain't too small
    if len(arr)/av_num < 1 or av_num == 1:
        printV("Too small number of averages", verbose)
        return arr
    
    # cut the last part of an array to calculate the mean
    printV(f'\t->To rebin must take out {len(arr)%av_num} states', verbose)
    arr = arr[0:len(arr) - (len(arr)%av_num)]
    
    if d == 3:
        return arr.reshape(av_num, arr.shape[0]//av_num, arr.shape[1], arr.shape[2]).mean(0)
    elif d == 2: 
        return arr.reshape(av_num, arr.shape[0]//av_num, arr.shape[1]).mean(0)
    else:
        return arr.reshape(av_num, arr.shape[0]//av_num).mean(0)

#################################################### PERMUTATION ####################################################


'''
Apply a random permutation to arrays - any number really
'''
def permute(*args):
    p = np.random.permutation(len(args[0]))
    t = tuple([i[p] for i in args])
    return t

###################################################### AVERAGES #####################################################

""" 
Calculate the bin average of an array
"""
def avgBin(myArray, N=2):
    cum = np.cumsum(myArray,0)
    result = cum[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = myArray.shape[0] % N
    if remainder != 0:
        if remainder < myArray.shape[0]:
            lastAvg = (cum[-1]-cum[-1-remainder])/float(remainder)
        else:
            lastAvg = cum[-1]/float(remainder)
        result = np.vstack([result, lastAvg])

    return result

################################################### MOVING AVERAGES

""" 
Moving average with cumsum
"""
def moveAverage(a, n : int) :
    if isinstance(a, np.ndarray):
        ret         =       np.cumsum(a, dtype=float)
        ret[n:]     =       ret[n:] - ret[:-n]
        return      ret[n - 1:] / n
    elif isinstance(a, pd.DataFrame):
        df_tmp = a.copy()
        for i, row in df_tmp.iterrows():
            df_tmp.loc[i] = savgol_filter(row, window, 3)
        return df_tmp 
        # return df.rolling(window, axis=1).mean().dropna(axis = 1), en[:-window+1]
        # return np.array(pd.Series(x).rolling(window=n).mean().iloc[n-1:].values)
    else:
        beg = int(n/2)
        end = len(x) - int(n/2)
        zeros = [0 for i in range(beg)]
        for k in range(beg, end):
            begin = int(k-n/2)
            av = np.sum(x[begin:begin + n])
            x[k] = x[k] - (av/float(n))
        return x[beg:-beg]

################################################# CONSIDER FLUCTUATIONS 

""" 
Neglect average in data and leave fluctuations only
"""
def removeMean(a, n : int):
    return a[n-1:] - moveAverage(a,n)

##################################################### DISTRIBUTIONS ##################################################

'''
Gaussian PDF
'''
def gauss(x : np.ndarray, mu, sig, *args):
    if len(args) == 0:
        return 1/np.sqrt(2 * PI * sig**2) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    elif len(args) == 1:
        return args[0] * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    elif len(args) == 2:
        return args[1] + args[0] * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    else:
        return 1/np.sqrt(2 * PI * sig**2) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
######################################################## FINDERS #####################################################
