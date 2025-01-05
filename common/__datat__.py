"""
This module provides a DataHandler class for handling and processing data arrays. 
It includes methods for filtering, initializing, interpolating, aggregating, 
concatenating, and averaging data arrays.
Classes:
    DataHandler: A class containing static methods for data handling and processing.
Methods:
    _filter_typical_values(current_x, current_y, typical, threshold=1.0) -> tuple:
    _initialize_combined_arrays(y_list, x_list, typical, threshold=1.0) -> tuple:
    _interpolate_and_update(x_combined, y_combined, current_x, current_y, divider) -> tuple:
    _aggregate_and_update(x_combined, y_combined, current_x, current_y, divider) -> tuple:
        Aggregates and updates combined x and y data arrays with current x and y data arrays.
    concat_and_average(y_list, x_list, typical=False, use_interpolation=True, threshold=1.0) -> tuple:
    concat_and_fill(y_list, x_list, lengths, missing_val=np.nan) -> tuple:
"""
import numpy as np

class DataHandler:
    """
    """

    @staticmethod
    def _filter_typical_values(current_x, current_y, typical, threshold = 1.0) -> tuple:
        """
        Filters y values less than the threshold.
        """
        
        if typical:
            indices     = current_y < threshold
            current_x   = current_x[indices]
            current_y   = current_y[indices]
        return current_x, current_y
    
    @staticmethod
    def _initialize_combined_arrays(y_list, x_list, typical, threshold = 1.0) -> tuple:
        """
        Initializes and combines arrays from given lists.

        This method takes two lists of arrays, `y_list` and `x_list`, and combines their first elements. 
        If the `typical` flag is set to True, it filters the combined arrays to include only elements 
        where the values in `y_combined` are less than 1.0.

        Args:
            y_list (list of numpy.ndarray): List of arrays to be combined for the y-axis.
            x_list (list of numpy.ndarray): List of arrays to be combined for the x-axis.
            typical (bool): Flag to determine if filtering should be applied.

        Returns:
            tuple: A tuple containing the combined y-axis array and x-axis array.
        """
        y_combined      = y_list[0]
        x_combined      = x_list[0]
        return DataHandler._filter_typical_values(x_combined, y_combined, typical, threshold)

    @staticmethod
    def _interpolate_and_update(x_combined, y_combined, current_x, current_y, divider):
        """
        Interpolates and updates combined x and y data arrays with current x and y data arrays.

        This function takes in combined x and y data arrays, and current x and y data arrays,
        interpolates them to a common set of x values, and updates the combined y data array
        by adding the interpolated current y data array. It also updates the divider array
        by interpolating it to the new x values and incrementing it by 1.

        Parameters:
        x_combined (np.ndarray): The combined x data array.
        y_combined (np.ndarray): The combined y data array.
        current_x (np.ndarray): The current x data array.
        current_y (np.ndarray): The current y data array.
        divider (np.ndarray): The divider array.

        Returns:
        tuple: A tuple containing:
            - new_x_combined (np.ndarray): The new combined x data array after interpolation.
            - y_combined (np.ndarray): The updated combined y data array.
            - divider (np.ndarray): The updated divider array.
        """
        new_x_combined          = np.sort(np.unique(np.concatenate([x_combined, current_x])))           # Create a new combined x grid
        y_interpolated_combined = np.interp(new_x_combined, x_combined, y_combined, left=0, right=0)    # Interpolate previous y values onto the new grid
        y_interpolated_current  = np.interp(new_x_combined, current_x, current_y, left=0, right=0)      # Interpolate current y onto the new grid
        y_combined              = y_interpolated_combined + y_interpolated_current                      # Update combined y values        
        divider                 = np.interp(new_x_combined, x_combined, divider, left=0, right=0) + 1   # Update divider - increment by 1
        return new_x_combined, y_combined, divider

    @staticmethod
    def _aggregate_and_update(x_combined, y_combined, current_x, current_y, divider):
        # Find common bins and separate unique bins
        common_bins         = np.intersect1d(x_combined, current_x, assume_unique=True) # Common bins in combined and current
        unique_x_combined   = np.setdiff1d(x_combined, common_bins, assume_unique=True) # Unique bins in combined - previous x's
        unique_x_current    = np.setdiff1d(current_x, common_bins, assume_unique=True)  # Unique bins in current - new x's 

        if common_bins.size > 0:
            common_indices_combined = np.isin(x_combined, common_bins)                  # Indices of common bins in combined
            common_indices_current  = np.isin(current_x, common_bins)                   # Indices of common bins in current
            y_combined[common_indices_combined] += current_y[common_indices_current]    # Sum y values
            divider[common_indices_combined]    += 1                                    # Update divider - increment by 1

        if unique_x_current.size > 0:
            x_combined  = np.concatenate([x_combined, unique_x_current])                # Append unique bins
            y_combined  = np.concatenate([y_combined, current_y[np.isin(current_x, unique_x_current)]])
            divider     = np.concatenate([divider, np.ones_like(unique_x_current)])     # Update divider - join the list of ones (first occurrence)

        sort_indices    = np.argsort(x_combined)                                        # Sort combined arrays
        x_combined      = x_combined[sort_indices]                                      # Sort x_combined
        y_combined      = y_combined[sort_indices]                                      # Sort y_combined
        divider         = divider[sort_indices]                                         # Sort divider

        return x_combined, y_combined, divider
    
    ################################################################################################
    
    @staticmethod
    def concat_and_average(y_list, x_list, typical = False, use_interpolation = True, threshold = 1.0):
        """
        Concatenates and averages y values across multiple histograms.

        :param y_list           : List of y matrices (each one corresponding to a realization).
        :param x_list           : List of x vectors (each one corresponding to a realization).
        :param typical          : If True, filter y values less than 1.0.
        :param use_interpolation: If True, interpolate y values for non-matching bins.
                                  If False, aggregate only exact matches and append unique bins.
        :param threshold        : The threshold value for filtering y values (default: 1.0).
        :returns                : Combined y values and x bins after averaging.
        """
        # chack the instances
        if not isinstance(y_list, list) or not isinstance(x_list, list) or not isinstance(typical, np.ndarray) or not isinstance(use_interpolation, np.ndarray):
            raise ValueError("Input lists must be of type list or numpy.ndarray.")
            
        # check if the arrays are already one dimensional and return them
        if len(y_list[0].shape) == 1:
            return y_list, x_list
        # check if the arrays are empty or have different lengths
        if len(y_list) == 0 or len(x_list) == 0:
            raise ValueError("Input lists cannot be empty.")
        # check if the arrays have the same length - when they are multidimensional
        if len(y_list) != len(x_list):
            raise ValueError("Input lists must have the same length.")
        # check if the arrays have only one element and return them
        if len(x_list) == 1:
            return y_list[0], x_list[0]
        
        # first initialization
        y_combined, x_combined  = DataHandler._initialize_combined_arrays(y_list, x_list, typical)
        divider                 = np.ones_like(y_combined)

        # loop over the rest of the arrays
        for i in range(1, len(x_list)):
            current_x, current_y                = DataHandler._filter_typical_values(x_list[i], y_list[i], typical, threshold)
            if use_interpolation:
                x_combined, y_combined, divider = DataHandler._interpolate_and_update(x_combined, y_combined, current_x, current_y, divider)
            else:
                x_combined, y_combined, divider = DataHandler._aggregate_and_update(x_combined, y_combined, current_x, current_y, divider)
        return y_combined / divider, x_combined # Final averaging - it shall divide each element by the number of realizations [each element in the divider]

    @staticmethod
    def concat_and_fill(y_list, x_list, lengths, missing_val = np.nan):
        """
        Concatenates y values across multiple histograms, combines x vectors into a single sorted array,
        and fills missing values.

        :param y_list: List of y arrays (each one corresponding to a realization).
        :param x_list: List of x arrays (each one corresponding to a realization group).
        :param lengths: List indicating how many y arrays correspond to each x array.
        :param missing_val: Value to fill for missing data points after interpolation (default: np.nan).
        :returns: A 2D NumPy array of y values interpolated to a common x grid and the combined x bins.
        """
        # check the instances
        if not isinstance(y_list, list) or not isinstance(x_list, list) or not isinstance(lengths, list):
            raise ValueError("Input lists must be of type list.")
        # check if the arrays are already one dimensional and return them
        if len(y_list[0].shape) == 1:
            return y_list, x_list
        if len(y_list) == 0 or len(x_list) == 0:
            raise ValueError("Input lists cannot be empty.")
        if len(lengths) != len(x_list):
            raise ValueError("Lengths list must match the size of x_list.")

        # Combine all x vectors into a single sorted, unique array
        x_combined  = np.sort(np.unique(np.concatenate(x_list)))

        # Interpolate each realization onto the combined x grid
        y_all       = []
        for il, length in enumerate(lengths):
            num_realizations    = length[0] if (isinstance(length, list) or isinstance(length, tuple)) else length
            
            for ii in range(num_realizations):
                y               = y_list[il][ii]
                # Interpolate current y onto x_combined grid
                y_all.append(np.interp(x_combined, x_list[il], y, left = missing_val, right = missing_val))
        
        # Return as 2D array and combined x bins
        return y_all, x_combined

    ################################################################################################
    
    @staticmethod
    def cut_matrix_bad_vals_zero(M, 
                                axis           = 0, 
                                tol            = 1e-9, 
                                check_limit    : float | None = 10):
        """
        Cut off the slices (along any specified axis) in matrix M where all elements are close to zero.
        If a 1D vector is provided, it returns the vector unless all elements are close to zero, 
        in which case it returns an empty array.

        Parameters:
        - M (numpy.ndarray) : The input matrix or vector.
        - axis (int)        : The axis along which to check for zero elements. 
                            For example, 0 for rows, 1 for columns, etc.
                            Ignored if M is a 1D vector.
        - tol (float)       : The tolerance for considering elements as zero.
        - check_limit (int) : The maximum number of elements along the axis to check for zeros.

        Returns:
        - numpy.ndarray: The resulting matrix after removing slices (along the specified axis) 
                        that are close to zero, or the vector after removing if all elements are close to zero.
        """
        
        # handle vector shape!
        if M.ndim == 1:
            if check_limit is not None:
                check_limit     = min(check_limit, M.shape[0])
                mask            = np.isclose(M[:check_limit], 0.0, atol=tol)
            else:
                mask            = np.isclose(M, 0.0, atol=tol)
            return M[~mask]
        
        # handle matrix shape!
        M_moved                 = np.moveaxis(M, axis, 0)
        if check_limit is not None:
            check_limit         = min(check_limit, M_moved.shape[1])
            mask                = ~np.all(np.isclose(M_moved[:, :check_limit], 0.0, atol=tol), axis=1)
        else:
            mask                = ~np.all(np.isclose(M_moved, 0.0, atol=tol), axis=1)
        
        M_filtered = M_moved[mask]              # Use the mask to filter the slices along the moved axis
        return np.moveaxis(M_filtered, 0, axis) # Move the axis back to its original position

    @staticmethod
    def cut_matrix_bad_vals(M, 
                            axis        = 0, 
                            threshold   = -1e4, 
                            check_limit = None):
        """
        Cut off the rows or columns in matrix M where the first `check_limit` elements are all below a threshold.

        Parameters:
        - M (numpy.ndarray): The input matrix.
        - axis (int): The axis along which to check for elements below the threshold (0 for rows, 1 for columns).
        - threshold (float): The threshold value.
        - check_limit (int, optional): The number of elements to check from each row or column.

        Returns:
        - numpy.ndarray: The resulting matrix after removing rows or columns where the first `check_limit` elements are below the threshold.
        """
        if axis == 0:
            # Check rows
            if check_limit is not None:
                check_limit = min(check_limit, M.shape[1])
                mask        = ~np.all(M[:, :check_limit] < threshold, axis=1)
            else:
                mask        = ~np.all(M < threshold, axis=1)
        elif axis == 1:
            # Check columns
            if check_limit is not None:
                check_limit = min(check_limit, M.shape[0])
                mask        = ~np.all(M[:check_limit, :] < threshold, axis=0)
            else:
                mask        = ~np.all(M < threshold, axis=0)
        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")   # Invalid axis
        return M[mask] if axis == 0 else M[:, mask]                     # Return the matrix with the rows or columns removed

    ################################################################################################
    
####################################################################################################
