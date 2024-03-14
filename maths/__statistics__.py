from scipy.signal import savgol_filter
from scipy.stats import gmean
from scipy.stats import kurtosis

from .__math__ import *

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
    if rng is None:
        rng = np.random.default_rng()
    p = rng.random.Generator.permutation(len(args[0]))
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
    Moving average with cumsum. This is applied along the first axis of the array.
    
    """
    if isinstance(a, np.ndarray):
        ret         =       np.cumsum(a, axis = 0, dtype=float)
        ret[n:]     =       ret[n:] - ret[:-n]
        return      ret[n - 1:] / n
    elif isinstance(a, pd.DataFrame):
        df_tmp      =       []
        for idx, row in a.iterrows():
            df_tmp.append(moveAverage(np.array(row), n))
        return pd.DataFrame(np.array(df_tmp))
    else:
        beg     = int(n/2)
        end     = len(a) - int(n/2)
        for k in range(beg, end):
            begin   = int(k-n/2)
            av      = np.sum(a[begin:begin + n])
            a[k]    = a[k] - (av/float(n))
        return a[beg:-beg]

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
