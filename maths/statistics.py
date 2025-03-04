import sys
from numpy.lib.stride_tricks import sliding_window_view
from .math_utils import *

from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.stats import gmean
from scipy.stats import kurtosis, binned_statistic

##############################################################

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

##############################################################

from typing import List, Tuple, Union, Optional, Callable, Sequence
import numpy as np
import pandas as pd

##############################################################

class Statistics:
    '''
    Class for statistical operations - mean, median, etc. 
    '''
    
    ##########################################################
    @staticmethod 
    def bin_avg(data, x, centers, delta = 0.05, typical = False, cutoffNum = 10, func = lambda x: np.mean(x, axis = 0), verbose = False):
        '''
        Bin average of the data
        - data      : data to average - first axis is realizations!
        - x         : values to bin - first axis is realizations!
        - centers   : centers of the bins
        - delta     : width of the bin
        - typical   : if True, apply log transformation to data
        - cutoffNum : minimum number of values in a bin
        - func      : function to apply to bin values (default: np.mean)
        - verbose   : if True, print additional information
        '''
        averages = []
        if typical:
            data = np.log(data)
        
        for c in centers:
            averagesin  = []
            for ir, data_r in enumerate(data):
                xin         = x[ir]
                mask        = np.abs(xin - c) < delta
                bin_values  = data_r[mask]

                if len(bin_values) < cutoffNum:
                    c_arg       = np.argmin(np.abs(xin - c))
                    start_idx   = max(c_arg - cutoffNum // 2, 0)
                    end_idx     = min(c_arg + cutoffNum // 2, len(data_r))
                    bin_values  = data_r[start_idx : end_idx]
                
                if len(bin_values) > 0:
                    averagesin.append(np.mean(func(bin_values)))
                    
            if len(averagesin) > 0:
                if typical:
                    averages.append(np.mean(np.exp(averagesin)))
                else:
                    averages.append(np.mean(averagesin))
        return np.nan_to_num(averages, nan = 0.0)


    ##########################################################
    @staticmethod
    def rebin(arr, av_num : int, d : int, rng = None):
        '''
        Calculates a bin average of an array. Does so by computing the new shape of the array.
        - arr    : array to rebin
        - av_num : average number
        - d      : dimensionality of the array
        - rng    : random number generator (default: None)
        ''' 
        
        if rng is None:
            rng = np.random.default_rng()
        
        if (len(arr) / av_num < 1) or av_num == 1:
            return arr
        
        shuffled = arr[0:len(arr) - (len(arr) % av_num)]
        rng.shuffle(shuffled)

        if d == 3:
            return shuffled.reshape(av_num, shuffled.shape[0] // av_num, shuffled.shape[1], shuffled.shape[2]).mean(0)
        elif d == 2: 
            return shuffled.reshape(av_num, shuffled.shape[0] // av_num, shuffled.shape[1]).mean(0)
        else:
            return shuffled.reshape(av_num, shuffled.shape[0] // av_num).mean(0) 
        
    ##########################################################
    @staticmethod
    def permute(*args, rng = None):
        '''
        Apply a random permutation to arrays - any number really
        - args : arrays to permute
        - rng  : random number generator (default: None)
        '''
        if sys.version_info[1] >= 10:
            if rng is None:
                rng = np.random.default_rng()
            p = rng.permutation(len(args[0]))
            return tuple([i[p] for i in args])
        else:
            p = np.random.permutation(len(args[0]))
            return tuple([i[p] for i in args])
    
    ##########################################################
    
    @staticmethod
    def calculate_fluctuations(signals, bin_size, axis=1):
        """
        Calculate fluctuations around each signal within a bin, handling NaN values correctly, and keep the original dimensions.

        Args:
            signals (numpy.ndarray): Input signals array of any shape.
            bin_size (int): Size of the bin for computing fluctuations.
            axis (int): Axis along which to calculate fluctuations.

        Returns:
            numpy.ndarray: Fluctuations for each signal, same shape as input.
        """
        if bin_size < 1:
            raise ValueError("Bin size must be at least 1")
        
        # Create a mask for NaN values
        nan_mask = np.isnan(signals)
        temp_signals = np.where(nan_mask, 0, signals)  # Replace NaNs with 0 temporarily for mean calculations

        # Count the number of valid (non-NaN) values in each bin
        valid_counts = uniform_filter1d(
            (~nan_mask).astype(int), 
            size=bin_size, 
            axis=axis, 
            mode='nearest'
        )
        valid_counts = np.where(valid_counts > 0, valid_counts, np.nan)  # Replace zeros with NaN to avoid division by zero
        
        if np.all(np.isnan(valid_counts)):
            return np.full(signals.shape, np.nan)
        
        # Compute the mean, considering only valid values
        sum_values = uniform_filter1d(temp_signals, size=bin_size, axis=axis, mode='nearest')
        means = np.where(valid_counts > 0, sum_values / valid_counts, np.nan)
        
        # Compute squared deviations from the mean
        squared_deviations = (signals - means) ** 2
        squared_deviations = np.where(nan_mask, 0, squared_deviations)  # Replace NaNs in squared deviations with 0 for variance calculation

        # Compute variance, considering only valid values
        sum_squared_deviations = uniform_filter1d(
            squared_deviations, 
            size=bin_size, 
            axis=axis, 
            mode='nearest'
        )
        variance = np.where(valid_counts > 0, sum_squared_deviations / valid_counts, np.nan)

        # Return the standard deviation as fluctuations
        return np.sqrt(variance)                                                  # Return the standard deviation as fluctuations         
    
    ##########################################################
    
    @staticmethod
    def get_cdf(x, y, gammaval = 0.5, BINVAL = 21):
        """
        Calculate the cumulative distribution function (CDF) and find the gamma value.

        Parameters:
        x (array-like): The independent variable values.
        y (array-like): The dependent variable values, which may contain NaNs.
        gammaval (float, optional): The target CDF value to find the corresponding gamma value. Default is 0.5.

        Returns:
        tuple: A tuple containing:
            - x (array-like): The input independent variable values.
            - y (array-like): The input dependent variable values with NaNs removed.
            - cdf (array-like): The cumulative distribution function values.
            - gammaf (float): The value of the independent variable corresponding to the target CDF value.
        """
        # Apply the moving average to smooth y
        y_smoothed  = np.convolve(y, np.ones(BINVAL)/BINVAL, mode='same')
        cdf         = np.cumsum(y_smoothed * np.diff(np.insert(x, 0, 0)))
        cdf         /= cdf[-1]
        y_smoothed  /= cdf[-1]
        gammaf      = x[np.argmin(np.abs(cdf - gammaval))]
        return x, y_smoothed, cdf, gammaf
    
    @staticmethod    
    def find_peak_and_interpolate(alphas, values):
        """
        Find the peak value in the given data and interpolate to improve peak precision.
        This function removes NaN values from the input arrays, performs spline interpolation
        to improve the precision of the peak detection, and then finds the maximum value and
        its corresponding alpha. A fine-grained search around the peak is performed to find
        a more precise maximum.
        Parameters:
        - alphas (array-like): The array of alpha values.
        - values (array-like): The array of corresponding values.
        Returns:
        - tuple: A tuple containing the refined alpha value at the peak and the refined peak value.
        """
        # Remove NaN values and filter the alphas and values
        valid_alphas = alphas[~np.isnan(values)]
        valid_values = values[~np.isnan(values)]
        
        # Interpolate to improve peak precision (using spline interpolation)
        poly_coeffs = np.polyfit(valid_alphas, valid_values, deg=9)
        spline_func = np.poly1d(poly_coeffs)
        
        # Find the approximate maximum value and its corresponding alpha
        max_value = np.nanmax(valid_values)
        max_index = np.argmax(valid_values)
        max_alpha = valid_alphas[max_index]
        
        # Perform a fine-grained search around the peak to find a more precise maximum
        fine_alphas = np.linspace(valid_alphas[max_index - 1], valid_alphas[max_index + 1], 100)
        fine_values = spline_func(fine_alphas)
        refined_max_alpha = fine_alphas[np.argmax(fine_values)]
        refined_max_value = np.max(fine_values)

        return refined_max_alpha, refined_max_value

############################################### STATISTICAL AVERAGING ###############################################

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
    """
    A histogram class that stores the bin edges and the bin counts.
    
    Convention:
        - The bins are defined by an array `bin_edges` of length (n_bins+1).
        - The first bin (index 0) collects all values below bin_edges[0] (underflow).
        - For 1 <= i < n_bins, bin i collects values in [bin_edges[i], bin_edges[i+1]).
        - The last bin (index n_bins) collects values greater than or equal to bin_edges[-1] (overflow).
    """
    
    def __init__(self, n_bins: Optional[int] = None, edges: Optional[Sequence[float]] = None, dtype = None):
        """
        Initialize the histogram with either a specified number of bins or specific edges.
        Parameters:
            n_bins  : Number of bins (if edges is None).
            edges   : Specific bin edges (if n_bins is None).
            dtype   : Data type for the bin edges.
        Raises:
            ValueError: If both n_bins and edges are None, or if edges is not a one-dimensional array with at least two elements.    
        Notes:
            - If both n_bins and edges are None, a histogram with one bin (0 to 0) is created.
            - If edges are provided, the number of bins is determined from the length of edges.
            - The bin counts are initialized to zero.    
        """
        self.dtype = np.float64 if dtype is None else dtype
        
        if edges is not None:
            # If edges are provided, assume they form the full set of boundaries.
            self.bin_edges = np.array(edges, dtype=self.dtype)
            if self.bin_edges.ndim != 1 or self.bin_edges.size < 2:
                raise ValueError("edges must be a one-dimensional array with at least two elements.")
            self.n_bins     = self.bin_edges.size - 1
            self.bin_counts = np.zeros(self.n_bins + 1, dtype=np.uint64)
        elif n_bins is not None:
            self.n_bins     = n_bins
            # In the default (number-of-bins) constructor, we allocate n_bins+1 counts.
            # The bin_edges array will have length n_bins+1.
            self.bin_edges  = np.zeros(self.n_bins + 1, dtype=self.dtype)
            self.bin_counts = np.zeros(self.n_bins + 1, dtype=np.uint64)
        else:
            self.n_bins     = 1
            self.bin_edges  = np.zeros(2, dtype=self.dtype)
            self.bin_counts = np.zeros(2, dtype=np.uint64)

    #######################################################
    #! Setters
    #######################################################
    
    def set_histogram_counts(self,
                            values      : Union[np.ndarray, Sequence[Union[float, complex]]],
                            set_bins    : bool  = True) -> None:
        """
        For the specified values, set the histogram counts.
        If set_bins is True, the bin edges will be determined from the minimum and maximum of the data.
        For complex-valued inputs, only the real part is used.
        """
        # Convert input to a NumPy array; if complex, take the real part.
        values = np.asarray(values)

        if set_bins:
            v_min = values.min() if not np.iscomplexobj(values) else values.real.min()
            v_max = values.max() if not np.iscomplexobj(values) else values.real.max()
            # Set the bin edges based on the min and max of the data.
            self.bin_edges = np.linspace(v_min, v_max, self.n_bins + 1)
        
        # Reset counts to zero before computing new counts.
        self.bin_counts.fill(0)
        
        # Compute underflow, proper bins, and overflow.
        # 1) Underflow: values below the first edge.
        self.bin_counts[0] = np.sum(values < self.bin_edges[0])
        # 2) Overflow: values greater than or equal to the last edge.
        self.bin_counts[-1] = np.sum(values >= self.bin_edges[-1])
        # 3) For the bins in between.
        for i in range(self.n_bins - 1):
            self.bin_counts[i+1] = np.sum((values >= self.bin_edges[i]) & (values < self.bin_edges[i+1]))
    
    def set_edges(self, edges: Union[np.ndarray, Sequence[float]]) -> None:
        """
        Set the bin edges from an array or list.
        The number of bins is set to len(edges)-1.
        Arguments:
            edges: A one-dimensional array or list of bin edges.
        Raises:
            ValueError: If edges is not a one-dimensional array or list with at least two elements.
        """
        edges = np.asarray(edges, dtype=self.dtype)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a one-dimensional array with at least two elements.")
        self.bin_edges  = edges
        self.n_bins     = edges.size - 1
        # Reinitialize counts to match new edges.
        self.bin_counts = np.zeros(self.n_bins + 1, dtype=np.uint64)
    
    #######################################################
    #! Getters
    #######################################################
    
    @property
    def edges(self) -> np.ndarray:
        """Return the bin edges."""
        return self.bin_edges
    
    @property
    def counts(self, i: Optional[int] = None) -> Union[np.uint64, np.ndarray]:
        """
        If i is provided, return the count for that bin index.
        Otherwise, return the full counts array.
        """
        if i is not None:
            return self.bin_counts[i]
        return self.bin_counts

    def counts_col(self) -> np.ndarray:
        """
        Return the counts as a column vector.
        """
        return self.bin_counts.reshape(-1, 1)

    #######################################################
    #! Statistics
    #######################################################
    
    @staticmethod
    def iqr(data: Union[np.ndarray, Sequence[float]]) -> float:
        """
        Calculate the interquartile range (IQR) of the data.
        Splits the sorted data into two halves and computes the difference between the medians.
        """
        data    = np.asarray(data, dtype=float)
        n       = data.size
        mid     = n // 2
        q1      = np.median(data[:mid])
        q3      = np.median(data[mid:])
        return q3 - q1

    @staticmethod
    def freedman_diaconis_rule(n_obs: int, iqr_val: float, max_val: float, min_val: float = 0) -> int:
        """
        Calculate the number of bins using the Freedman-Diaconis rule.
        """
        h = (2.0 * iqr_val) / (n_obs ** (1.0 / 3.0))
        if h == 0:
            return 1
        return int(math.ceil((max_val - min_val) / h))
    
    #######################################################
    #! Reset
    #######################################################

    def _reset_with_n_bins(self, n_bins: int) -> None:
        """
        Reset the histogram with a new number of bins.
        """
        self.n_bins     = n_bins
        self.bin_edges  = np.zeros(n_bins + 1, dtype=self.dtype)
        self.bin_counts = np.zeros(n_bins + 1, dtype=np.uint64)
    
    def reset(self, nbins: int = None) -> None:
        """
        Reset the histogram counts and (optionally) the bin edges to zero.
        Parameters:
        - nbins: If provided, reset the histogram with this number of bins.
        """
        if nbins is not None:
            return self._reset_with_n_bins(nbins)
        self.bin_counts.fill(0)
        self.bin_edges.fill(0)

    #######################################################
    #! Setters
    #######################################################
    
    def uniform(self, v_max: float, v_min: float = 0) -> None:
        """
        Create a uniform distribution of bins between v_min and v_max.
        Parameters:
        - v_max: Maximum value for the histogram.
        - v_min: Minimum value for the histogram.
        """
        self.bin_edges = np.linspace(v_min, v_max, self.n_bins + 1)
    
    def uniform_log(self, v_max: float, v_min: float = 1e-5, base: int = 10) -> None:
        """
        Create a logarithmic distribution of bins between v_min and v_max.
        Parameters:
        - v_max: Maximum value for the histogram.
        - v_min: Minimum value for the histogram.
        """
        if base == 10:
            start           = math.log10(v_min)
            stop            = math.log10(v_max)
            self.bin_edges  = np.logspace(start, stop, self.n_bins + 1, base=10)
        else:
            # General case: use logarithms with the specified base.
            start           = math.log(v_min, base)
            stop            = math.log(v_max, base)
            exponents       = np.linspace(start, stop, self.n_bins + 1)
            self.bin_edges  = np.power(base, exponents)
    
    #######################################################
    #! Append
    #######################################################
    
    def append(self, values) -> int:
        """
        Append a value to the histogram by determining its bin and incrementing the corresponding count.
        Returns the bin index.
        Parameters:
        - value: The value to append to the histogram.
        Returns:
        - bin indices: The indices of the bin where the value was appended.        
        """
        if hasattr(values, 'ndim') and values.ndim == 0 or np.isscalar(values):
            values = np.array([values])
            
        indices = np.searchsorted(self.bin_edges, values, side='right')
        # Correct for underflow and overflow
        indices = np.where(values < self.bin_edges[0], 0, indices)
        indices = np.where(values >= self.bin_edges[-1], self.n_bins, indices)
        # Update bin counts in-place. np.add.at handles repeated indices correctly.
        np.add.at(self.bin_counts, indices, 1)
        return indices

    def merge(self, other: "Histogram") -> None:
        """
        Merge another histogram into this one.
        The histograms must have the same number of bins and matching bin edges.
        """
        if self.n_bins != other.n_bins or not np.allclose(self.bin_edges, other.bin_edges):
            raise ValueError("Cannot merge histograms of different bin sizes or edges.")
        self.bin_counts += other.bin_counts

class HistogramAverage(Histogram):
    """
    Additional properties for the histogram class, adding bin averages.
    This class allows one to have a function f(x) averaged over the bins.
    The binAverages are the sum of the function evaluated in each bin,
    and they can be normalized by the bin counts.
    """
    
    def __init__(self, n_bins: Optional[int] = None, edges: Optional[Sequence[float]] = None, dtype = None):
        """
        Initialize the histogram with either a specified number of bins or specific edges.
        Parameters:
            n_bins  : Number of bins (if edges is None).
            edges   : Specific bin edges (if n_bins is None).
            dtype   : Data type for the bin edges.
        Raises:
            ValueError: If both n_bins and edges are None, or if edges is not a one-dimensional array with at least two elements. 
        Notes:
            - If both n_bins and edges are None, a histogram with one bin (0 to 0) is created.
            - If edges are provided, the number of bins is determined from the length of edges.
            - The bin counts and averages are initialized to zero.
        """       
        super().__init__(n_bins=n_bins, edges=edges, dtype=dtype)
        self.bin_averages = np.zeros(self.bin_counts.shape, dtype=self.dtype)
    
    ###############################################
    #! Getters
    ###############################################
    
    def averages(self, i: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Return the bin averages. If i is provided, return the average for that bin.
        Otherwise, return the full averages array.
        """
        if i is not None:
            return self.bin_averages[i]
        return self.bin_averages

    def averages_av(self, is_typical: bool = False) -> np.ndarray:
        """
        Get the average of the function over the bins normalized by the counts.
        If is_typical is True, exponentiate the normalized averages (useful if the averages
        represent logarithms).
        """
        out             = self.bin_averages.copy()
        # Normalize each bin where the count is nonzero.
        nonzero         = self.bin_counts != 0
        out[nonzero]    = out[nonzero] / self.bin_counts[nonzero]
        if is_typical:
            return np.exp(out)
        return out
    
    ###############################################
    #! Reset
    ###############################################
    
    def _reset_with_n_bins(self, n_bins: int) -> None:
        """
        Reset the histogram and bin averages with a new number of bins.
        """
        super()._reset_with_n_bins(n_bins)
        self.bin_averages = np.zeros(self.bin_counts.shape, dtype=float)
        
    def reset(self, nbins = None) -> None:
        """
        Reset both the histogram counts and the bin averages.
        """
        if nbins is not None:
            return self._reset_with_n_bins(nbins)
        super().reset()
        self.bin_averages.fill(0)

    ################################################
    #! Append
    ################################################
    
    def append(self, values, elements) -> int:
        """
        Append a value to the histogram and add the corresponding element to the bin average.

        Parameters:
            - values: The value to append to the histogram.
            - elements: The element to add to the bin average.
        Returns:
            - bin_idx: The indices of the bin where the value was appended.
        """
        bin_idx                     = super().append(values)
        self.bin_averages[bin_idx]  += elements
        return bin_idx
    
    # ------------------ Merge ------------------
    def merge(self, other: "HistogramAverage") -> None:
        """
        Merge another HistogramAverage into this one.
        Warning: The histograms must have the same number of bins and matching bin edges.
        """
        super().merge(other)
        self.bin_averages += other.bin_averages
    
################################################################################

class Fraction:
    """
    Class to handle fractions.
    """
    
    @staticmethod
    def diag_cut(fraction: float, size: int) -> int:
        """
        Calculate the number of states to take based on a fraction of the total size.
        Parameters:
        fraction : fraction of the total size to take.
        size     : total size of the Hilbert space.
        Returns:
        The number of states to take.
        """
        
        if fraction == 1.0:
            return size
        if fraction >= 1.0:
            states = min(int(fraction), size)
        else:
            states = max(1, int(fraction * size))
        return size if states >= size else states
    
    @staticmethod
    def around_idx(l: int, r: int, idx: int, size: int) -> Tuple[int, int]:
        """
        Get the specific indices in a range around a given index in the Hilbert space.
        Checks for boundaries.
        
        Parameters:
        l           : number of elements to the left of idx.
        r           : number of elements to the right of idx.
        idx         : center index.
        size        : total size of the Hilbert space.
        
        Returns:
        A tuple (min_index, max_index) with the allowed index range.
        """
        min_index = max(0, idx - l)
        max_index = min(size - 1, idx + r)
        return (min_index, max_index)
    
    @staticmethod
    def take_fraction(frac  : float, 
                    data    : np.ndarray, 
                    around          = None,
                    fraction_left   = 0.5,
                    fraction_right  = 0.5,
                    around_idx      = None
                ) -> list:
        """
        Take a fraction of the data.

        Parameters:
        frac (float)                    : The fraction of the data to take. If frac is less than 1.0, it is treated as a fraction of the total data size.
                                        If frac is greater than 1.0, it is treated as the number of elements to take.
        data (list)                     : The list of data from which to take the fraction.
        around (float, optional)        : The index around which to take the fraction. If None, it defaults to half the size of the data.
        fraction_left (float, optional) : The fraction of the left side to take. Default is 0.5.
        fraction_right (float, optional): The fraction of the right side to take. Default is 0.5.
        around_idx (int, optional)      : The index around which to take the fraction. If None, it defaults to half the size of the data.

        Returns:
        list: A list containing the central portion of the original data, based on the specified fraction.
            If the calculated number of elements to take is less than or equal to 1, or equal to the size of the data, 
            the original data is returned.
        """
        size_data = len(data)
        if (around_idx is not None and not 0 < around_idx < size_data) or around_idx is None:
            if around is not None:
                around_idx = Fraction.find_nearest_idx(data, around)
            else:
                around_idx = size_data // 2
                        
        numstates = Fraction.diag_cut(frac, size_data)
        l         = int(numstates * fraction_left)
        r         = int(numstates * fraction_right)
        l, r      = Fraction.around_idx(l, r, around_idx, size_data)
        return data[l:r]
    
    @staticmethod
    def is_close_target(l: float, r: float, target: float = 0.0, tol: float = 0.0015) -> bool:
        """
        Check if the average of two energies (l and r) is within tol of a target energy.
        Parameters:
            l       : first energy.
            r       : second energy.
        Returns:
            True if the average is close to the target, False otherwise.
        """
        return np.abs((l + r) / 2.0 - target) < tol

    @staticmethod
    def is_difference_close_target(l: float, r: float, target: float = 0.0, tol: float = 0.0015) -> bool:
        """
        Check if the absolute energy difference between l and r is within tol of a target difference.
        """
        return np.abs(np.abs(l - r) - target) < tol
    
    @staticmethod
    def is_fraction_difference_between(l: float, r: float, min_val: float, max_val: float) -> bool:
        """
        Check if the absolute energy difference between l and r lies between min_val and max_val.
        """
        diff = np.abs(l - r)
        return min_val <= diff <= max_val

    #!TODO: add a function that checks if the difference is between two values
    @staticmethod
    def hs_fraction_offdiag(mn: int,
                            max_val: int, 
                            hilbert_size: int,
                            energies: np.ndarray,
                            target_en: float = 0.0,
                            tol: float = 0.0015,
                            sort: bool = True) -> List[Tuple[float, int, int]]:
        pass
        """
        Get the off-diagonal Hilbert-space fraction information.
        
        Iterates over the energy spectrum (from index mn to max_val) and for each pair (i, j) with j > i,
        if the average energy is within tol of target_en then store a tuple of
        (energy difference, j, i)
        in the output list. Finally, sort the list by the energy difference (first element).
        
        Parameters:
        mn        : starting index (inclusive).
        max_val   : ending index (exclusive).
        hilbert_size: size of the Hilbert space (not used in computation here, but kept for consistency).
        energies  : 1D NumPy array of energies.
        target_en : target energy for the mean.
        tol       : tolerance for closeness.
        sort      : whether to sort the output list by the energy difference.
        
        Returns:
        A list of tuples (omega, j, i) sorted by omega if sort is True.
        """
        out = []
        for i in range(mn, max_val):
            en_l = energies[i]
            for j in range(i + 1, max_val):
                en_r = energies[j]
                if hs_fraction_close_mean(en_l, en_r, target=target_en, tol=tol):
                    out.append((abs(en_r - en_l), j, i))
        if sort:
            out.sort(key=lambda tup: tup[0])
        return out
    
    @staticmethod
    def spectral_function_fraction(x, target, tolerance, tolerance_function = is_close_target):
        '''
        Calculate the spectral function fraction based on the given target and tolerance.
        '''
        n            = x.shape[0]
        # get the upper triangle indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        # calculate the mask for the target
        mask         = tolerance_function(x[i_idx], x[j_idx], target, tolerance)
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        differences  = x[j_idx] - x[i_idx]
        return differences, i_idx, j_idx

    @staticmethod
    def find_nearest_idx(array, value):
        """
        Find the index of the nearest value in an array.
        """
        idx = (np.abs(array - value)).argmin()
        return idx
    
####################################################################################
