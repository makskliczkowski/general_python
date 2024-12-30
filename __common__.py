##################### G E N E R A L   I M P O R T S #####################
import sys
import typing
# Adds higher directory to python modules path.
sys.path.append("./common/")

""" Standard numerical """
import numpy as np
import itertools
import os

"""       Flogger      """
from common.__flog__ import *
loggingLvl  =   logging.INFO
logger      =   Logger("", loggingLvl)
logger.info("Including common files to the project.", 0)

############################################### CHECK ALL EQUAL ELEMENTS ###############################################

def allEqual(iterator):
    '''
    In iterable, check if all items are equal.
    - iterator : iterator to be checked
    '''
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

############################################ CHECK THE INDICES OF DUPLICATES ###########################################

def listDups(seq, item):
    '''
    In a given sequence find the list of duplicate indices
    - seq   : sequence to find in
    - item  : item that is repeated
    '''
    start_at    = -1
    locs        = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

########################################################################################################################

def appDict(d, k, v):
    '''
    Append a value to a key in a dictionary
    - d : dictionary
    - k : key
    - v : value
    '''
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]
        
########################################################################################################################

def meanSlice(arr, idx, left, right):
    """
    Compute the mean of a slice of an array around a specified index.

    Parameters:
    - arr   : numpy.ndarray
        The array from which to compute the mean.
    - idx   : int
        The central index for the slice.
    - left  : int
        The number of elements to include to the left of idx.
    - right : int
        The number of elements to include to the right of idx.

    Returns:
    - float
        The mean of the slice.
    """
    # Ensure idx is within the bounds of the array
    if idx < 0 or idx >= len(arr):
        raise ValueError(f"Index 'idx' ({idx}) is out of bounds for array of length {len(arr)}.")

    # Define left and right bounds with clamping
    idx_l = max(0, idx - left)
    idx_r = min(len(arr), idx + right + 1)  # +1 to include the element at idx + right

    # Compute and return the mean of the slice
    return np.mean(arr[idx_l:idx_r])