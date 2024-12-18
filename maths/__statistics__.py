import sys
from scipy.signal import savgol_filter
from scipy.stats import gmean
from scipy.stats import kurtosis, binned_statistic
from numpy.lib.stride_tricks import sliding_window_view

from .__math__ import *


class Statistics:
    '''
    Class for statistical operations - mean, median, etc. 
    '''
    
    @staticmethod 
    def bin_avg(data, x, centers, delta = 0.05, typical = False, cutoffNum = 10, func = np.mean):
        '''
        Bin average of the data
        - data      : data to average
        - x         : values to bin
        - centers   : centers of the bins
        - delta     : width of the bin
        '''
        averages = []
        # go through the centers
        for c in centers:
            # go through realizations
            averagesin  = []
            for ir, data_r in enumerate(data):
                xin     = x[ir]
                # Find values within |x_n - c| < delta    
                bin_values = data_r[np.abs(xin - c) < delta]

                if len(bin_values) < cutoffNum:
                    c_arg       = np.argmin(np.abs(xin - c))
                    bin_values  = data_r[max(c_arg - cutoffNum // 2, 0) : min(c_arg + cutoffNum // 2, len(data_r))]
                # Compute the mean of the values in this bin
                if len(bin_values) > 0:
                    if typical:
                        averagesin.append(func(np.log(bin_values)))
                    else:
                        averagesin.append(func(bin_values))
            if typical:
                averages.append(np.mean(np.exp(averagesin)))
            else:
                averages.append(np.mean(averagesin))
        return np.nan_to_num(averages, nan = 0.0)

def take_fraction(frac : float, data):
    '''
    Take a fraction of the data
    '''
    sizeData = len(data)
    if frac < 1.0:
        toTake = int(frac * sizeData)
        if toTake <= 1 or toTake == sizeData:
            return data
        return data[sizeData // 2 - toTake // 2: sizeData // 2 + toTake // 2]
    elif frac > 1.0:
        toTake = int(frac)
        if toTake >= sizeData:
            return data
        return data[sizeData // 2 - toTake // 2: sizeData // 2 + toTake // 2]
    return data

############################################### STATISTICAL AVERAGING ###############################################

def rebin(arr, av_num : int, d : int, rng = None):
    '''
    Calculates a bin average of an array. Does so by computing the new shape of the array.
    - arr : array to rebin
    - av_num : average number
    - d : dimensionality of the array
    ''' 
    
    if rng is None:
        rng = np.random.default_rng()
    
    # check if the array ain't too small
    if (len(arr) / av_num < 1) or av_num == 1:
        # logger.info("Too small number of averages", 3)
        return arr
    
    # cut the last part of an array to calculate the mean
    shuffled = arr[0:len(arr) - (len(arr)%av_num)]
    rng.random.Generator.shuffle(shuffled)

    if d == 3:
        return shuffled.reshape(av_num, shuffled.shape[0]//av_num, shuffled.shape[1], shuffled.shape[2]).mean(0)
    elif d == 2: 
        return shuffled.reshape(av_num, shuffled.shape[0]//av_num, shuffled.shape[1]).mean(0)
    else:
        return shuffled.reshape(av_num, shuffled.shape[0]//av_num).mean(0) 
    
#################################################### PERMUTATION ####################################################

def permute(*args, rng = None):
    '''
    Apply a random permutation to arrays - any number really
    '''
    if sys.version_info[1] >= 10:
        if rng is None:
            rng = np.random.default_rng()
        p = rng.random.Generator.permutation(len(args[0]))
        t = tuple([i[p] for i in args])
        return t
    else:
        p = np.random.permutation(len(args[0]))
        t = tuple([i[p] for i in args])
        return t

###################################################### AVERAGES #####################################################

def avgBin(myArray, N=2):
    """ 
    Calculate the bin average of an array
    - myArray   : array to average into bins
    - N         : number of bins
    """
    cum         = np.cumsum(myArray,0)
    result      = cum[N-1::N]/float(N)
    result[1:]  = result[1:] - result[:-1]

    remainder   = myArray.shape[0] % N
    if remainder != 0:
        if remainder < myArray.shape[0]:
            lastAvg = (cum[-1]-cum[-1-remainder]) / float(remainder)
        else:
            lastAvg = cum[-1]/float(remainder)
        result = np.vstack([result, lastAvg])

    return result

################################################### MOVING AVERAGES

def moveAverage(a, n : int) :
    """ 
    Moving average with cumsum or sliding window. This is applied along the first axis of the array.
    
    Args:
        a: Input data, can be a numpy array, pandas DataFrame, or list.
        n: Window size for the moving average.

    Returns:
        A numpy array, pandas DataFrame, or list with the moving averages.
    """
    if isinstance(a, np.ndarray):
        if n > a.shape[0]:
            raise ValueError("Window size n must be less than or equal to the size of the input array.")
        # Use sliding window for a moving average
        return sliding_window_view(a, n, axis=0).mean(axis=-1)
    
    elif isinstance(a, pd.DataFrame):
        df_tmp      =       []
        for idx, row in a.iterrows():
            df_tmp.append(moveAverage(np.array(row), n))
        return pd.DataFrame(np.array(df_tmp))
    
    elif isinstance(a, list):
        a = np.array(a)  # Convert list to numpy array for consistency
        if n > len(a):
            raise ValueError("Window size n must be less than or equal to the size of the input list.")
        # Use sliding window for a moving average
        return list(sliding_window_view(a, n).mean(axis=-1))

    else:
        raise TypeError("Input must be a numpy array, pandas DataFrame, or list.")
    
def fluctAboveAverage(a, n : int):
    """ 
    Calculate fluctuations above the moving average.
    
    Args:
        a: Input data, can be a numpy array, pandas DataFrame, or list.
        n: Window size for the moving average.

    Returns:
        Fluctuations above the moving average.
    """
    
    # Calculate moving average
    moving_avg = moveAverage(a, n)
    
    if isinstance(a, np.ndarray):
        # Calculate fluctuations above the moving average
        fluctuations = a[n-1:] - moving_avg
        return fluctuations

    elif isinstance(a, pd.DataFrame):
        # Calculate fluctuations for each row in the DataFrame
        df_fluctuations = a.copy()
        for idx, row in a.iterrows():
            df_fluctuations.loc[idx] = row.to_numpy()[n-1:] - moveAverage(row.to_numpy(), n)
        return df_fluctuations

    elif isinstance(a, list):
        # Convert list to numpy array and calculate fluctuations
        a = np.array(a)
        fluctuations = a[n-1:] - np.array(moving_avg)
        return list(fluctuations)

    else:
        raise TypeError("Input must be a numpy array, pandas DataFrame, or list.")

################################################# CONSIDER FLUCTUATIONS 

def removeMean(a, 
               n : int, 
               moving_average = []):
    """ 
    Neglect average in data and leave fluctuations only
    """
    N = min(n, len(a))
    if moving_average is not None:
        return a[N-1:] - moveAverage(a, N)
    else:
        return a[min(len(a), len(moving_average)):] - moving_average

##################################################### DISTRIBUTIONS ##################################################

def gauss(x : np.ndarray, mu, sig, *args):
    '''
    Gaussian PDF
    '''
    if len(args) == 0:
        return 1.0 / np.sqrt(2 * PI * sig**2) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    elif len(args) == 1:
        return args[0] * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    elif len(args) == 2:
        return args[1] + args[0] * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    else:
        return 1/np.sqrt(2 * PI * sig**2) * np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    
######################################################## FINDERS #####################################################

class Histogram:
    
    @staticmethod
    def iqr(data : np.ndarray) -> float:
        '''
        Interquartile range of the data
        - data : data to calculate the IQR
        '''
        return float(np.percentile(data, 75)) - float(np.percentile(data, 25))

    @staticmethod
    def freedman_diaconis_rule(nobs : int, iqr : float, mx : float, mn = 0.0):
        '''
        Freedman-Diaconis rule for the number of bins. 
        - nobs : number of observations
        - iqr : interquartile range
        - mx : maximum value
        - mn : minimum value
        '''
        h = (2.0 * iqr / np.power(nobs, 1.0 / 3.0))
        return int(np.ceil((mx - mn) / h))
    
    #################################################
    
    @staticmethod
    def limit_logspace_val(value, base = 10):
        if base == 10:
            return np.ceil(np.log10(value))
        elif base == 2:
            return np.ceil(np.log2(value))
        else:
            return np.ceil(np.log(value))
