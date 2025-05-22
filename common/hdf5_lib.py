import sys
# Adds higher directory to python modules path.

from .directories import *
import numpy as np
import h5py
import os

####################################################### READ HDF5 FILE #######################################################

def allbottomkeys(group):
    """
    Recursively collect all dataset keys in an HDF5 group.
    """
    datasetJAX_RND_DEFAULT_KEYs = []

    def collectJAX_RND_DEFAULT_KEYs(obj):
        if isinstance(obj, h5py.Group):
            for key, value in obj.items():
                collectJAX_RND_DEFAULT_KEYs(value)
        else:
            datasetJAX_RND_DEFAULT_KEYs.append(obj.name)
    
    # Start the recursive key collection
    collectJAX_RND_DEFAULT_KEYs(group)
    
    return datasetJAX_RND_DEFAULT_KEYs

####################################################### READ HDF5 FILE #######################################################

def read_hdf5(file_path, 
              keys      = [], 
              verbose   = False, 
              removeBad = False):
    """
    Read the hdf5 saved file 
    - keys : if we input keys, they will be used for reading. Otherwise use the available ones.
    Parameters:
    - file_path (str): Path to the HDF5 file.
    - keys (list, optional): Specific keys to read. If None, read all dataset keys.
    - verbose (bool, optional): If True, log detailed information.
    - remove_bad (bool, optional): If True, remove the file if it's corrupted or unreadable.
    
    Returns:
    - dict: A dictionary with keys as dataset paths and values as numpy arrays.
    """
    
    data = {}
    
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist")
        return data
    try:
        # check the file
        if not file_path.endswith(('.h5', '.hdf5', '.hdf')):
            logging.error(f"File {file_path} is not an HDF5 file")
            return data
        
        with h5py.File(file_path, "r") as f:
            # all root level object names (aka keys) 
                
            # Determine keys to read
            if keys is None or len(keys) == 0:
                keys = allbottomkeys(f)
                if verbose:
                    logging.info(f"Available keys: {keys}")
                    
            # Read datasets into the dictionary
            for key in keys:
                try:
                    data[key] = f[key][()]
                except KeyError:
                    if verbose:
                        logging.warning(f"Key {key} not found in file {file_path}")
                except Exception as e:
                    if verbose:
                        logging.error(f"Error reading {key} from {file_path}: {str(e)}")
        data["filename"] = file_path
        return data
        
    except Exception as e:
        logging.error(f"Error opening file {file_path}: {str(e)}")
        if "truncated" in str(e) or "doesn't exist" in str(e) and removeBad:
            logging.info(f"Removing corrupted file {file_path}")
            os.remove(file_path)
        return {}

####################################################### READ HDF5 FILE #######################################################

def read_hdf5_extract(hdf5files,
                      key       : "str",
                      ):
    '''
    Yield the data from the hdf5 files for the given key.
    ''' 
    for f in hdf5files:
        yield f[key]

def read_hdf5_extract_concat(hdf5files,
                            key      : str,
                            repeatax : int  = 0,
                            verbose  : bool = False, 
                            isvector : bool = False,
                            cut_0_ax = None,
                            cut_v_ax = None,
                            padding  : bool = False,
                            check_limit     = None                       
                            ):
    '''
    For a specific set of hdf5 files, try to create a large array of the same form for the given key.
    One of the shapes must be the same for all the files.
    - hdf5files : list of hdf5 files
    - key       : key to read
    - repeatax  : axis to repeat
    - verbose   : if True, print the shape of the data
    '''
    shapeRepeat = None
    data        = []
    sizes       = []
    for ii, f in enumerate(hdf5files):
        try:
            d   = f[key]
            
            if verbose:
                logging.warning(f"{f['filename']}: Reading {key} with shape {d.shape}")
            
            if shapeRepeat is None:
                shapeRepeat = d.shape[repeatax]
            
            # incorrect shape
            if d.shape[repeatax] != shapeRepeat:
                if padding:
                    # Pad or truncate to match the repeat shape
                    padding_shape           = list(d.shape)
                    padding_shape[repeatax] = shapeRepeat
                    padded_d                = np.zeros(tuple(padding_shape), dtype=d.dtype)
                    slices                  = [slice(None)] * len(d.shape)
                    slices[repeatax]        = slice(0, min(d.shape[repeatax], shapeRepeat))
                    padded_d[tuple(slices)] = d[tuple(slices)]
                    d                       = padded_d
                else:
                    if verbose:
                        logging.error(f"Shape mismatch for {key} in {f['filename']}, skipping this file.")
                    continue
            
            # check the shape on the repeat axis 
            if d.shape[repeatax] == shapeRepeat:
                # check if it should be a vector
                if d.shape[0] == 1 and len(d.shape) == 2 and isvector:
                    d = d[0]
                data.append(d)
                sizes.append(d.shape)
            else:
                if verbose:
                    logging.error(f"Shape mismatch for {key} in {f['filename']}")
        except Exception as e:
            if verbose:
                logging.error(f"Error reading {key} from {f['filename']}: {str(e)}")
            continue
        
    # concatenate the data
    if len(data) > 0:
        # if verbose:
            # for i, d in enumerate(data):
                # logging.info(f"Data {i}: {d.shape}")
        data = np.concatenate(data, axis = 0)
    else:
        if verbose:
            logging.error(f"No data found for {key}")
        return np.array([])
    
    # cut the data if necessary
    if cut_0_ax is not None:
        data = cut_matrix_bad_vals_zero(data, axis = cut_0_ax, check_limit = check_limit)
    if cut_v_ax is not None:
        data = cut_matrix_bad_vals(data, axis = cut_v_ax, check_limit = check_limit)
    
    return data   

####################################################### READ HDF5 FILE #######################################################

def read_multiple_hdf5(directories  : list, 
                       conditions   = [],
                       keys         = [],
                       verbose      = False,
                       sortme       = True,
                       listme       = True,
                       logger       = None
                       ):
    '''
    Parse multiple hdf5 files. 
    - directories : list of directories to parse
    - conditions  : conditions to be met
    - keys        : keys to be read
    @return       : generator - this is lazy evaluated loading!
    '''
    files = Directories.listDirs(directories, conditions = conditions, appendDir = True, sortCondition = (lambda x: x) if sortme else None)
    
    if len(files) == 0:
        # logging.error("No files found")
        return None 
    
    for f in files:
        if verbose:
            logging.info(f"\t\tReading {f}")
        yield read_hdf5(f, keys, verbose = verbose)

def read_multiple_hdf5l(directories  : list,
                        conditions   = [],
                        keys         = [],
                        verbose      = False,
                        sortme       = True,
                        listme       = True
                        ):
    '''
    Parse multiple hdf5 files. 
    - directories : list of directories to parse
    - conditions  : conditions to be met
    - keys        : keys to be read
    @return       : list - this is eager evaluated loading!
    '''
    return list(read_multiple_hdf5(directories, conditions, keys, verbose, sortme, listme))

####################################################### READ HDF5 FILE #######################################################

def read_hdf5_extract_and_concat(directories  : list, 
                        conditions  = [],
                        key         = None,
                        verbose     = False,
                        is_vector   = False,
                        repeatax    : int  = 0,
                        cut_0_ax    = None,
                        cut_v_ax    = None,
                        padding     : bool = False,
                        check_limit = None      
                       ):
    ''' 
    Do the same as read_hdf5_extract but skip reading multiple files.
    '''
    files = read_multiple_hdf5(directories, conditions, [key] if key is not None else [], verbose)
    if files is None:
        return np.array([])
    
    return read_hdf5_extract_concat(files, key if isinstance(key, str) else '',
                                    repeatax, verbose, is_vector, cut_0_ax, cut_v_ax, padding, check_limit)

def read_hdf5_extract_and_concat_list(directories  : list,
                                    conditions  = [],
                                    key         = None,
                                    verbose     = False,
                                    is_vector   = False,
                                    repeatax    : int  = 0,
                                    cut_0_ax    = None,
                                    cut_v_ax    = None,
                                    padding     : bool = False,
                                    check_limit = None      
                                    ):
    '''
    Do the same as read_hdf5_extract but concat single file and return a list
    '''
    return [read_hdf5_extract_and_concat([d], conditions, key, verbose, is_vector, repeatax, cut_0_ax, cut_v_ax, padding, check_limit) for d in directories]

####################################################### SAVE HDF5 FILE #######################################################

def create_labels_hdf5(data, keys : list | dict, shape : tuple = ()) -> list:
    if len(keys) == len(data) or shape != ():
        return keys
    elif len(keys) == 1:
        return keys
    elif len(keys) > 1:
        return [keys[0] + "_" + str(i) for i in range(len(data))]
    else:
        return ['data_' + str(i) for i in range(len(data))]

def save_hdf5(directory, filename, data, shape: tuple = (), keys: list = []):
    '''
    Creates and saves an ndarray as hdf5 file.
    - filename  : name of the file to be saved
    - data      : data to be saved (can be a list of data or a dictionary)
    - shape     : shape of which the data shall be (if not the same as the data)
    - keys      : if len(keys) == len(data) we sample that and save each iteration
    '''
    # create a file first
    filename = filename if (filename.endswith(".h5") or filename.endswith(".hdf5")) else filename + ".h5"
    hf = h5py.File(os.path.join(directory, filename), 'w')

    # create the labels
    labels = []
    if not isinstance(data, dict):
        labels = create_labels_hdf5(data, keys, shape)

    # save the file
    if isinstance(data, (np.ndarray, list)):
        dtype = np.complex128 if np.iscomplexobj(data) else np.float64
        if len(labels) == 1:
            hf.create_dataset(labels[0], data=np.array(data, dtype = dtype).reshape(shape) if shape != () else data)
        else:
            for i, lbl in enumerate(labels):
                hf.create_dataset(lbl, data=data[i].reshape(shape) if shape != () else np.array(data[i], dtype = dtype))
    elif isinstance(data, dict):
        for key in data.keys():
            dtype = np.complex128 if np.iscomplexobj(data[key]) else np.float64
            hf.create_dataset(key, data=np.array(data[key], dtype=dtype).reshape(shape) if shape != () else np.array(data[key], dtype=dtype))
    # close
    hf.close()

def append_hdf5(directory, filename, new_data, keys=[], override = True):
    """
    Append new data to an existing HDF5 file.
    - directory: Directory where the file is located.
    - filename: Name of the HDF5 file.
    - new_data: Data to be appended (can be a list of data or a dictionary).
    - keys: Keys for the new data. If empty, keys will be generated.
    """
    file_path = os.path.join(directory, filename)
    if not (file_path.endswith(".h5") or file_path.endswith(".hdf5")):
        file_path += ".h5"
    
    if not os.path.exists(file_path if (file_path.endswith(".h5") or file_path.endswith(".hdf5")) else file_path + ".h5"):
        logging.debug(f"File {file_path} does not exist")
        return save_hdf5(directory, filename, new_data, shape = None, keys = keys)
            
    with h5py.File(file_path, 'a') as hf:
        labels = []
        if not isinstance(new_data, dict):
            labels = create_labels_hdf5(new_data, keys)
        
        if isinstance(new_data, (np.ndarray, list)):
            # go through the data
            for i, lbl in enumerate(labels):
                if lbl in hf:
                    if override:
                        del hf[lbl]
                        hf.create_dataset(lbl, data=new_data[i], maxshape=(None,) + new_data[i].shape[1:])
                    else:
                        hf[lbl].resize((hf[lbl].shape[0] + new_data[i].shape[0]), axis=0)
                        hf[lbl][-new_data[i].shape[0]:] = new_data[i]
                else:
                    hf.create_dataset(lbl, data=new_data[i], maxshape=(None,) + new_data[i].shape[1:])
        elif isinstance(new_data, dict):
            for lbl in new_data.keys():
                dtype = np.complex128 if np.iscomplexobj(new_data[lbl]) else np.float64
                if lbl in hf:
                    if override:
                        del hf[lbl]
                        hf.create_dataset(lbl, data=np.array(new_data[lbl], dtype=dtype))
                else:
                    hf.create_dataset(lbl, data=np.array(new_data[lbl], dtype=dtype))

########################################################### CUTTER ##########################################################

def concat_and_average(y_list, x_list, typical=False, use_interpolation=True):
    """
    Concatenates and averages y values across multiple histograms.

    :param y_list: List of y matrices (each one corresponding to a realization).
    :param x_list: List of x vectors (each one corresponding to a realization).
    :param typical: If True, filter y values less than 1.0.
    :param use_interpolation: If True, interpolate y values for non-matching bins.
                              If False, aggregate only exact matches and append unique bins.
    :returns: Combined y values and x bins after averaging.
    """
    if len(y_list) == 0 or len(x_list) == 0:
        raise ValueError("Input lists cannot be empty.")
    if len(y_list) != len(x_list):
        raise ValueError("Input lists must have the same length.")
    if len(x_list) == 1:
        return y_list[0], x_list[0]

    # Initialize combined arrays
    y_combined = y_list[0]
    x_combined = x_list[0]

    if typical:
        indices     = y_combined < 1.0
        x_combined  = x_combined[indices]
        y_combined  = y_combined[indices]

    divider         = np.ones_like(y_combined)

    for i in range(1, len(x_list)):
        current_x   = x_list[i]
        current_y   = y_list[i]

        if typical:
            indices     = current_y < 1.0
            current_x   = current_x[indices]
            current_y   = current_y[indices]

        if use_interpolation:
            # Create a new combined x grid
            new_x_combined          = np.sort(np.unique(np.concatenate([x_combined, current_x])))

            # Interpolate y values onto the new grid
            y_interpolated_combined = np.interp(new_x_combined, x_combined, y_combined, left=0, right=0)
            y_interpolated_current  = np.interp(new_x_combined, current_x, current_y, left=0, right=0)

            # Update combined y values and divider
            y_combined              = y_interpolated_combined + y_interpolated_current
            divider                 = np.interp(new_x_combined, x_combined, divider, left=0, right=0) + 1

            x_combined = new_x_combined
        else:
            # Find common bins and separate unique bins
            common_bins         = np.intersect1d(x_combined, current_x, assume_unique = True)   # Common bins in combined and current
            unique_x_combined   = np.setdiff1d(x_combined, common_bins, assume_unique = True)   # Unique bins in combined - previous x's
            unique_x_current    = np.setdiff1d(current_x, common_bins, assume_unique = True)    # Unique bins in current - new x's

            # Handle common bins: sum y values and update divider
            if common_bins.size > 0:
                common_indices_combined     = np.isin(x_combined, common_bins)                  # Indices of common bins in combined
                common_indices_current      = np.isin(current_x, common_bins)                   # Indices of common bins in current
                y_combined[common_indices_combined] += current_y[common_indices_current]        # Sum y values
                divider[common_indices_combined]    += 1                                        # Update divider

            # Append unique bins
            if unique_x_current.size > 0:
                x_combined                  = np.concatenate([x_combined, unique_x_current])
                y_combined                  = np.concatenate([y_combined, current_y[np.isin(current_x, unique_x_current)]])
                divider                     = np.concatenate([divider, np.ones_like(unique_x_current)])

            # if unique_x_combined.size > 0:
            #     x_combined                  = np.concatenate([x_combined, unique_x_combined])
            #     y_combined                  = np.concatenate([y_combined, np.zeros_like(unique_x_combined)])
            #     divider                     = np.concatenate([divider, np.ones_like(unique_x_combined)])

            # Sort combined arrays
            sort_indices                    = np.argsort(x_combined)
            x_combined                      = x_combined[sort_indices]
            y_combined                      = y_combined[sort_indices]
            divider                         = divider[sort_indices]

    # Final averaging - it shall divide each element by the number of realizations [each element in the divider]
    y_combined /= divider

    return y_combined, x_combined

########################################################### CUTTER ##########################################################

def concat_and_fill(y_list, x_list, lengths, missing_val=np.nan):
    """
    Concatenates y values across multiple histograms, combines x vectors into a single sorted array,
    and fills missing values.

    :param y_list: List of y arrays (each one corresponding to a realization).
    :param x_list: List of x arrays (each one corresponding to a realization group).
    :param lengths: List indicating how many y arrays correspond to each x array.
    :param missing_val: Value to fill for missing data points after interpolation (default: np.nan).
    :returns: A 2D NumPy array of y values interpolated to a common x grid and the combined x bins.
    """

    if len(y_list) == 0 or len(x_list) == 0:
        raise ValueError("Input lists cannot be empty.")
    if len(lengths) != len(x_list):
        raise ValueError("Lengths list must match the size of x_list.")

    # Combine all x vectors into a single sorted, unique array
    x_combined  = np.sort(np.unique(np.concatenate(x_list)))

    # Interpolate each realization onto the combined x grid
    y_all       = []
    y_index     = 0  # Tracks the position in y_list
    for il, length in enumerate(lengths):
        num_realizations    = length[0] if (isinstance(length, list) or isinstance(length, tuple)) else length
        
        for ii in range(num_realizations):
            y               = y_list[il][ii]
            y_index         += 1
            # Interpolate current y onto x_combined grid
            y_all.append(np.interp(x_combined, x_list[il], y, left = missing_val, right = missing_val))
    
    # Return as 2D array and combined x bins
    return y_all, x_combined

########################################################### CUTTER ##########################################################

def cut_matrix_bad_vals_zero(M, 
                             axis           = 0, 
                             tol            = 1e-9, 
                             check_limit    = 10):
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
    M_moved = np.moveaxis(M, axis, 0)
    if check_limit is not None:
        check_limit = min(check_limit, M_moved.shape[1])
        mask        = ~np.all(np.isclose(M_moved[:, :check_limit], 0.0, atol=tol), axis=1)
    else:
        mask        = ~np.all(np.isclose(M_moved, 0.0, atol=tol), axis=1)
    
    # Use the mask to filter the slices along the moved axis
    M_filtered = M_moved[mask]

    # Move the axis back to its original position
    return np.moveaxis(M_filtered, 0, axis)

########################################################### CUTTER ##########################################################

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
            mask = ~np.all(M[:, :check_limit] < threshold, axis=1)
        else:
            mask = ~np.all(M < threshold, axis=1)
    elif axis == 1:
        # Check columns
        if check_limit is not None:
            check_limit = min(check_limit, M.shape[0])
            mask = ~np.all(M[:check_limit, :] < threshold, axis=0)
        else:
            mask = ~np.all(M < threshold, axis=0)
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")

    # Return the matrix with the rows or columns removed
    return M[mask] if axis == 0 else M[:, mask]

######################################################### CHANGE H5 #########################################################

def change_h5_bad(directory : Directories,
                  filename  : str,
                  keys2new  = {},
                  c_zero_ax = None,
                  c_val_ax  = None,
                  verbose   : bool  = False,
                  directoryS: str   = "",
                  checkLim  : int   = 10):
    '''
    Take a given hdf5 file and change the keys (if they exist).
    Otherwise, leave them as they are.
    Additionally, add check whether the file contains bad elements in the shape.
    - directory : directory where the file is located
    - filename  : filename of the hdf5 file
    - keys2new  : dictionary with keys to be changed
    - c_zero_ax : axis to check for zeros, if -1, no check is performed
    - c_val_ax  : axis to check for values, if -1, no check is performed
    - verbose   : if True, print the information
    - directoryS: if not "", save the file in a different directory
    - checkLim  : number of elements to check for zeros or values. Default = 10
    '''
    
    h5data      = read_hdf5(directory + kPS + filename, verbose = verbose)
    # check the true keys, not the filename
    truekeys    = [x for x in list(h5data.keys()) if (x != 'filename')]
   
    if verbose:
        logging.info(f"Reading {filename}")
        logging.info(f"Keys: {list(h5data.keys())}")
    
    if len(h5data) == 0:
        logging.error(f"Could not read {filename}")
        return
    
    # check for zeros
    if c_zero_ax is not None:
        for key in truekeys:
            h5data[key] = cut_matrix_bad_vals_zero(h5data[key], axis = c_zero_ax, check_limit = checkLim)
            
    # check for values (default = -1e4)
    if c_val_ax is not None:
        for key in truekeys:
            h5data[key] = cut_matrix_bad_vals(h5data[key], axis = c_val_ax, check_limit = checkLim)
            
    # change the keys so that they are the new ones
    if len(keys2new) > 0:
        for key in truekeys:
            if key in keys2new.keys():
                h5data[keys2new[key]] = h5data.pop(key)
    
    # remove the filename
    if 'filename' in h5data:
        del h5data['filename']
    
    # save the file
    if directoryS == "":
        printV("Saving the file in the same directory", verbose, 1)
        save_hdf5(directory, filename, h5data, (), keys = list(h5data.keys()))
    else:
        printV("Saving the file in a different directory: " + directoryS, verbose, 1)
        save_hdf5(directoryS, filename, h5data, (), keys = list(h5data.keys()))

def change_h5_bad_dirs(
        directories : list,
        conditions  = [],
        keys2new    = {},
        testrun     : bool  = False,
        c_zero_ax   = None,
        c_val_ax    = None,
        verbose     : bool  = False,
        directoryS  : str   = "",
        checkLim    = None,
        exceptionHandler = None
    ):
    '''
    '''
    for directory in directories:
        try:
            if verbose:
                print(f"Reading files in {directory}")
                
            # get the files
            files = Directories.listDirs([directory], conditions = conditions, appendDir = False)
            
            
            # prepare the test run
            if testrun:
                directoryS = (directory + kPS + "testrun") if directoryS == "" else directoryS
                       
            # change the files
            for f in files:
                if verbose:
                    print(f"\tChanging {directory + kPS + f}", '->', directoryS + kPS + f)
                    
                change_h5_bad(directory, f, keys2new, 
                              c_zero_ax, 
                              c_val_ax, 
                              verbose, 
                              directoryS, 
                              checkLim)
                
        except Exception as e:
            # if not isinstance(e, FileExistsError) and not isinstance(e, FileNotFoundError):
            if exceptionHandler is not None:
                exceptionHandler.handle(e, "Uknown error", FileExistsError, FileNotFoundError)
            else:
                print("Exception:", e)
        # ExceptionHandler().handle(e, "Uknown error", FileExistsError, FileNotFoundError)

######################################################### CHANGE H5 #########################################################

from .datah import DataHandler

class HDF5Handler:
    """
    A class for reading and writing HDF5 files. 
    """
    
    ############### PRIVATE METHODS ###############
    
    @staticmethod
    def _allbottomkeys(group):
        """
        This method traverses an HDF5 group and collects the names of all datasets 
        contained within it, including those in nested groups.

        Args:
            group (h5py.Group): The HDF5 group to traverse.

        Returns:
            list: A list of dataset keys (names) found within the group.
        """
        datasetJAX_RND_DEFAULT_KEYs = []

        def collectJAX_RND_DEFAULT_KEYs(obj):
            if isinstance(obj, h5py.Group):
                for key, value in obj.items():
                    collectJAX_RND_DEFAULT_KEYs(value)
            else:
                datasetJAX_RND_DEFAULT_KEYs.append(obj.name)

        collectJAX_RND_DEFAULT_KEYs(group)
        return datasetJAX_RND_DEFAULT_KEYs

    ############### PUBLIC METHODS ################
    
    @staticmethod
    def read_hdf5(file_path, keys=None, verbose=False, remove_bad=False, logger=None):
        """
        Read an HDF5 file and return a dictionary of datasets.
        """
        data = {}
        if not os.path.exists(file_path):
            if logger is not None:
                logger.error(f"File {file_path} does not exist")
            return data

        try:
            if not file_path.endswith(('.h5', '.hdf5', '.hdf')):
                if logger is not None:
                    logger.error(f"File {file_path} is not an HDF5 file")
                return data

            with h5py.File(file_path, "r") as f:
                if keys is None or len(keys) == 0:
                    keys = HDF5Handler._allbottomkeys(f)
                    if verbose and logger is not None:
                        logger.info(f"Available keys: {keys}")

                for key in keys:
                    try:
                        data[key] = f[key][()]
                    except KeyError:
                        if verbose and logger is not None:
                            logger.warning(f"Key {key} not found in file {file_path}")
                    except Exception as e:
                        if verbose and logger is not None:
                            logger.error(f"Error reading {key} from {file_path}: {str(e)}")

            data["filename"] = file_path
            return data

        except Exception as e:
            if logger is not None:
                logger.error(f"Error opening file {file_path}: {str(e)}")
            if "truncated" in str(e) or "doesn't exist" in str(e) and remove_bad:
                if logger is not None:
                    logger.info(f"Removing corrupted file {file_path}")
                os.remove(file_path)
            return {}

    @staticmethod
    def read_hdf5_extract(hdf5files, key):
        """
        Yield data for the given key from a list of HDF5 files.
        :param hdf5files: List of HDF5 files.
        :param key      : Key to read.
        :return         : Generator of data.
        """
        for f in hdf5files:
            yield f[key]

    @staticmethod
    def read_hdf5_extract_concat(hdf5files, key, repeatax=0, verbose=False, isvector=False, cut_0_ax=None, cut_v_ax=None, padding=False, check_limit=None, logger=None):
        """
        Concatenate data from multiple HDF5 files along a specified axis for a given key.

        Parameters:
        -----------
        hdf5files : list
            List of HDF5 file objects to read the data from.
        key : str
            Key to access the dataset within each HDF5 file.
        repeatax : int, optional
            Axis along which to concatenate the data. Default is 0.
        verbose : bool, optional
            If True, enables verbose logging. Default is False.
        isvector : bool, optional
            If True, treats 2D datasets with shape (1, N) as vectors and flattens them. Default is False.
        cut_0_ax : int, optional
            Axis along which to cut bad values to zero. Default is None.
        cut_v_ax : int, optional
            Axis along which to cut bad values. Default is None.
        padding : bool, optional
            If True, pads or truncates datasets to match the shape of the first dataset along the repeat axis. Default is False.
        check_limit : float, optional
            Limit to check for bad values when cutting matrices. Default is None.

        Returns:
        --------
        np.ndarray
            Concatenated data from the HDF5 files. If no valid data is found, returns an empty array.

        Raises:
        -------
        Exception
            If an error occurs while reading a dataset from an HDF5 file, it logs the error if verbose is True and continues with the next file.
        """
        shape_repeat    = None
        data            = []
        for f in hdf5files:
            try:
                d       = f[key]                                                                # Read the dataset from the file
                if verbose and logger is not None:
                    logger.warning(f"{f['filename']}: Reading {key} with shape {d.shape}")

                if shape_repeat is None:
                    shape_repeat = d.shape[repeatax]                                            # Set the shape of the repeat axis for the first dataset

                if d.shape[repeatax] != shape_repeat:                                           # Check if the shape of the repeat axis matches (first dataset will set the shape)  
                    if padding:                                                                 # Pad or truncate to match the repeat shape if padding is enabled
                        padding_shape           = list(d.shape)                                 # Create a new shape with the repeat axis padded or truncated
                        padding_shape[repeatax] = shape_repeat                                  # Set the repeat axis to the desired shape
                        padded_d                = np.zeros(tuple(padding_shape), dtype=d.dtype) # Create a new array with the padded shape
                        slices                  = [slice(None)] * len(d.shape)                  # Create a list of slices to copy the data
                        slices[repeatax]        = slice(0, min(d.shape[repeatax], shape_repeat))       
                        padded_d[tuple(slices)] = d[tuple(slices)]
                        d                       = padded_d                                      # Assign the padded array to the dataset
                    else:
                        continue

                if d.shape[repeatax] == shape_repeat:                                           # Check if the shape of the repeat axis matches the expected shape
                    if d.shape[0] == 1 and len(d.shape) == 2 and isvector:                      # Check if the dataset should be treated as a vector
                        d = d[0]
                    data.append(d)
            except Exception as e:
                if verbose:
                    logging.error(f"Error reading {key}: {str(e)}")
                continue

        if data:
            data = np.concatenate(data, axis=0)
            if cut_0_ax is not None:
                data = DataHandler.cut_matrix_bad_vals_zero(data, axis = cut_0_ax, check_limit = check_limit)
            if cut_v_ax is not None:
                data = DataHandler.cut_matrix_bad_vals(data, axis = cut_v_ax, check_limit = check_limit)
            return data
        else:
            return np.array([])

    ###############################################
    
    @staticmethod
    def save_hdf5(directory, filename, data, shape=(), keys=[]):
        '''
        Creates and saves an ndarray as hdf5 file.
        - filename  : name of the file to be saved
        - data      : data to be saved (can be a list of data or a dictionary)
        - shape     : shape of which the data shall be (if not the same as the data)
        - keys      : if len(keys) == len(data) we sample that and save each iteration
        '''
        # create a file first
        filename    = filename if (filename.endswith(".h5") or filename.endswith(".hdf5")) else filename + ".h5"
        
        # create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # create the file
        hf          = h5py.File(os.path.join(directory, filename), 'w')

        # create the labels
        labels      = []
        if not isinstance(data, dict):
            labels = create_labels_hdf5(data, keys, shape)

        # save the file
        if isinstance(data, (np.ndarray, list)):
            dtype = np.complex128 if np.iscomplexobj(data) else np.float64
            if len(labels) == 1:
                hf.create_dataset(labels[0], data=np.array(data, dtype = dtype).reshape(shape) if shape != () else data)
            else:
                for i, lbl in enumerate(labels):
                    hf.create_dataset(lbl, data=data[i].reshape(shape) if shape != () else np.array(data[i], dtype = dtype))
        elif isinstance(data, dict):
            for key in data.keys():
                dtype = np.complex128 if np.iscomplexobj(data[key]) else np.float64
                hf.create_dataset(key, data=np.array(data[key], dtype=dtype).reshape(shape) if shape != () else np.array(data[key], dtype=dtype))
        hf.close()

    @staticmethod
    def append_hdf5(directory, filename, new_data, 
                    keys        : list | dict | None    = [], 
                    override    : bool                  = True):
        """
        Append new data to an existing HDF5 file.
        - directory: Directory where the file is located.
        - filename: Name of the HDF5 file.
        - new_data: Data to be appended (can be a list of data or a dictionary).
        - keys: Keys for the new data. If empty, keys will be generated.
        """
        file_path = os.path.join(directory, filename)
        if not (file_path.endswith(".h5") or file_path.endswith(".hdf5")):
            file_path += ".h5"
        
        if not os.path.exists(file_path if (file_path.endswith(".h5") or file_path.endswith(".hdf5")) else file_path + ".h5"):
            logging.debug(f"File {file_path} does not exist")
            return HDF5Handler.save_hdf5(directory, filename, new_data, shape = None, keys = keys)
                
        with h5py.File(file_path, 'a') as hf:
            labels = []
            if not isinstance(new_data, dict):
                labels = create_labels_hdf5(new_data, keys)
            
            if isinstance(new_data, (np.ndarray, list)):
                for i, lbl in enumerate(labels):
                    if lbl in hf:
                        if override:
                            del hf[lbl]
                            hf.create_dataset(lbl, data=new_data[i], maxshape=(None,) + new_data[i].shape[1:])
                        else:
                            hf[lbl].resize((hf[lbl].shape[0] + new_data[i].shape[0]), axis=0)
                            hf[lbl][-new_data[i].shape[0]:] = new_data[i]
                    else:
                        hf.create_dataset(lbl, data=new_data[i], maxshape=(None,) + new_data[i].shape[1:])
            elif isinstance(new_data, dict):
                for lbl in new_data.keys():
                    dtype = np.complex128 if np.iscomplexobj(new_data[lbl]) else np.float64
                    if lbl in hf:
                        if override:
                            del hf[lbl]
                            hf.create_dataset(lbl, data=np.array(new_data[lbl], dtype=dtype))
                    else:
                        hf.create_dataset(lbl, data=np.array(new_data[lbl], dtype=dtype))
                        
    ########################################################### CUTTER ##########################################################


    ########################################################### CUTTER ##########################################################

    
    ######################################################### CHANGE H5 #########################################################

    @staticmethod
    def change_h5_bad(directory : Directories,
                    filename  : str,
                    keys2new  = {},
                    c_zero_ax = None,
                    c_val_ax  = None,
                    verbose   : bool  = False,
                    directoryS: str   = "",
                    checkLim  : int   = 10):
        '''
        Take a given hdf5 file and change the keys (if they exist).
        Otherwise, leave them as they are.
        Additionally, add check whether the file contains bad elements in the shape.
        - directory : directory where the file is located
        - filename  : filename of the hdf5 file
        - keys2new  : dictionary with keys to be changed
        - c_zero_ax : axis to check for zeros, if -1, no check is performed
        - c_val_ax  : axis to check for values, if -1, no check is performed
        - verbose   : if True, print the information
        - directoryS: if not "", save the file in a different directory
        - checkLim  : number of elements to check for zeros or values. Default = 10
        '''
        
        h5data      = read_hdf5(directory + kPS + filename, verbose = verbose)
        # check the true keys, not the filename
        truekeys    = [x for x in list(h5data.keys()) if (x != 'filename')]
    
        if verbose:
            logging.info(f"Reading {filename}")
            logging.info(f"Keys: {list(h5data.keys())}")
        
        if len(h5data) == 0:
            logging.error(f"Could not read {filename}")
            return
        
        # check for zeros
        if c_zero_ax is not None:
            for key in truekeys:
                h5data[key] = cut_matrix_bad_vals_zero(h5data[key], axis = c_zero_ax, check_limit = checkLim)
                
        # check for values (default = -1e4)
        if c_val_ax is not None:
            for key in truekeys:
                h5data[key] = cut_matrix_bad_vals(h5data[key], axis = c_val_ax, check_limit = checkLim)
                
        # change the keys so that they are the new ones
        if len(keys2new) > 0:
            for key in truekeys:
                if key in keys2new.keys():
                    h5data[keys2new[key]] = h5data.pop(key)
        
        # remove the filename
        if 'filename' in h5data:
            del h5data['filename']
        
        # save the file
        if directoryS == "":
            printV("Saving the file in the same directory", verbose, 1)
            save_hdf5(directory, filename, h5data, (), keys = list(h5data.keys()))
        else:
            printV("Saving the file in a different directory: " + directoryS, verbose, 1)
            save_hdf5(directoryS, filename, h5data, (), keys = list(h5data.keys()))
    
    @staticmethod
    def change_h5_bad_dirs(
            directories : list,
            conditions  = [],
            keys2new    = {},
            testrun     : bool  = False,
            c_zero_ax   = None,
            c_val_ax    = None,
            verbose     : bool  = False,
            directoryS  : str   = "",
            checkLim    = None,
            exceptionHandler = None
        ):
        '''
        '''
        for directory in directories:
            try:
                if verbose:
                    print(f"Reading files in {directory}")
                    
                # get the files
                files = Directories.listDirs([directory], conditions = conditions, appendDir = False)
                
                
                # prepare the test run
                if testrun:
                    directoryS = (directory + kPS + "testrun") if directoryS == "" else directoryS
                        
                # change the files
                for f in files:
                    if verbose:
                        print(f"\tChanging {directory + kPS + f}", '->', directoryS + kPS + f)
                        
                    change_h5_bad(directory, f, keys2new, 
                                c_zero_ax, 
                                c_val_ax, 
                                verbose, 
                                directoryS, 
                                checkLim)
                    
            except Exception as e:
                # if not isinstance(e, FileExistsError) and not isinstance(e, FileNotFoundError):
                if exceptionHandler is not None:
                    exceptionHandler.handle(e, "Uknown error", FileExistsError, FileNotFoundError)
                else:
                    print("Exception:", e)
            # ExceptionHandler().handle(e, "Uknown error", FileExistsError, FileNotFoundError)

    ######################################################### CHANGE H5 #########################################################
    
#############################################################
#! END OF FILE