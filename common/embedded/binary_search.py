'''


'''

import numpy as np
import numba
from typing import Optional, Union, List
from general_python.algebra.utils import DEFAULT_NP_FLOAT_TYPE, Array
from general_python.algebra.utils import get_backend, maybe_jit, is_traced_jax, DEFAULT_NP_INT_TYPE

_BAD_BINARY_SEARCH_STATE = -1

####################################################################################################
#! Searching functions
####################################################################################################

@maybe_jit
def _binary_search_backend(arr, l_point, r_point, elem, backend: str = 'default') -> int:
    '''
    Perform a binary search for 'elem' in the sorted container 'arr'
    between indices l_point and r_point (inclusive).
    
    If tol is provided (i.e. not None), equality is determined approximately
    within the given tolerance (useful for floating point numbers).
    
    Works with Python lists, NumPy arrays, or JAX arrays (via the appropriate backend).
    
    Returns:
        The index of 'elem' if found; otherwise, returns -1.
    '''
    # Try to use the backend's searchsorted if possible.
    backend_mod = get_backend(backend)
    
    # If the left or right points are not the first or last indices, respectively,
    subarr      = arr[l_point: r_point+1]
        
    # searchsorted returns the insertion index.
    idx         = int(backend_mod.searchsorted(subarr, elem))
    idx         = idx + l_point
    return int(idx)

@numba.njit
def binary_search_numpy(arr, l_point, r_point, elem) -> int:
    '''
    Perform a binary search for 'elem' in the sorted NumPy array 'arr'.
    '''
    if l_point < 0 or r_point >= len(arr):
        return _BAD_BINARY_SEARCH_STATE
    if l_point > r_point:
        return _BAD_BINARY_SEARCH_STATE
    middle = l_point + (r_point - l_point) // 2
    if arr[middle] == elem:
        return middle
    if arr[middle] < elem:
        return binary_search_numpy(arr, middle + 1, r_point, elem)
    return binary_search_numpy(arr, l_point, middle - 1, elem)

def _binary_search_list_notol(arr, l_point, r_point, elem) -> int:
    '''
    Perform a binary search for 'elem' in the sorted list 'arr' (exact equality).
    '''
    if l_point < 0 or r_point >= len(arr):
        return _BAD_BINARY_SEARCH_STATE
    if l_point > r_point:
        return _BAD_BINARY_SEARCH_STATE
    middle = l_point + (r_point - l_point) // 2
    if arr[middle] == elem:
        return middle
    if arr[middle] < elem:
        return _binary_search_list(arr, middle + 1, r_point, elem)
    return _binary_search_list(arr, l_point, middle - 1, elem)

def _binary_search_list(arr, l_point, r_point, elem, tol: Optional[float] = None) -> int:
    '''
    Perform a binary search for 'elem' in the sorted list 'arr'.
    If tol is provided, approximate equality is used.
    '''
    if l_point < 0 or r_point >= len(arr):
        return _BAD_BINARY_SEARCH_STATE
    if l_point > r_point:
        return _BAD_BINARY_SEARCH_STATE
    middle = l_point + (r_point - l_point) // 2
    if tol is not None:
        if abs(arr[middle] - elem) <= tol:
            return middle
        elif arr[middle] < elem:
            return _binary_search_list(arr, middle + 1, r_point, elem, tol)
        return _binary_search_list(arr, l_point, middle - 1, elem, tol)
    return _binary_search_list_notol(arr, l_point, r_point, elem)

def binary_search(arr, l_point, r_point, elem,
                tol: Optional[float] = None, backend: str = 'default') -> int:
    """
    Perform a binary search for 'elem' in the sorted container 'arr'
    between indices l_point and r_point (inclusive).
    
    If tol is provided, approximate equality is used (useful for floats).
    
    Works for Python lists as well as for NumPy/JAX arrays.
    Parameters:
        arr (array-like)        : The sorted array or list to search.
        l_point (int)           : The left index of the search range.
        r_point (int)           : The right index of the search range.
        elem (float)            : The element to search for.
        tol (float, optional)   : The tolerance for approximate equality (default is None).
        backend (str)           : The backend to use ('default', 'numpy', or 'jax').
    Returns:
        The index of 'elem' if found; otherwise, returns -1.
    """
    if hasattr(arr, "shape"):
        idx = _binary_search_backend(arr, l_point, r_point, elem, backend)
        if idx < len(arr) and idx >= 0:
            if tol is not None and abs(arr[idx] - elem) <= tol:
                return idx
            if arr[idx] == elem:
                return idx
        return _BAD_BINARY_SEARCH_STATE
    else:
        if tol is not None:
            return _binary_search_list(arr, l_point, r_point, elem, tol)
        return _binary_search_list_notol(arr, l_point, r_point, elem)

####################################################################################################