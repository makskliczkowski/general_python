import sys
# Adds higher directory to python modules path.

from .__directories__ import *
import numpy as np
import h5py
import os

####################################################### READ HDF5 FILE #######################################################

def allbottomkeys(group):
    """
    Recursively collect all dataset keys in an HDF5 group.
    """
    dataset_keys = []

    def collect_keys(obj):
        if isinstance(obj, h5py.Group):
            for key, value in obj.items():
                collect_keys(value)
        else:
            dataset_keys.append(obj.name)
    
    # Start the recursive key collection
    collect_keys(group)
    
    return dataset_keys

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
            if keys is None:
                keys = allbottomkeys(f)
                if verbose:
                    logging.info(f"Available keys: {keys}")
                    
            # Read datasets into the dictionary
            for key in keys:
                try:
                    data[key] = f[key][()]
                except KeyError:
                    logging.warning(f"Key {key} not found in file {file_path}")
                except Exception as e:
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
    for ii, f in enumerate(hdf5files):
        try:
            d   = f[key]
            if verbose:
                logging.warning(f"Reading {key} with shape {d.shape}")
            
            if shapeRepeat is None:
                shapeRepeat = d.shape[repeatax]
            
            # check the shape on the repeat axis 
            if d.shape[repeatax] == shapeRepeat:
                data.append(d)
            else:
                logging.error(f"Shape mismatch for {key} in {f['filename']}")
        except Exception as e:
            logging.error(f"Error reading {key} from {f['filename']}: {str(e)}")
    
    # concatenate the data
    if len(data) > 0:
        return np.concatenate(data, axis = 0)
    else:
        logging.error(f"No data found for {key}")
        return np.array()
        
####################################################### READ HDF5 FILE #######################################################

def read_multiple_hdf5(directories  : list, 
                       conditions   = [],
                       keys         = []
                       ):
    '''
    Parse multiple hdf5 files. 
    - directories : list of directories to parse
    - conditions  : conditions to be met
    - keys        : keys to be read
    @return       : generator - this is lazy evaluated loading!
    '''
    files = Directories.listDirs(directories, conditions = conditions, appendDir = True)
    for f in files:
        yield read_hdf5(f, keys)

####################################################### SAVE HDF5 FILE #######################################################
    
def save_hdf5(directory, filename, data : np.ndarray, shape : tuple, keys = []):
    '''
    Creates and saves an ndarray as hdf5 file.
    - filename : name of the file to be saved
    - data : data to be saved
    - shape : shape of which the data shall be
    - keys : if len(keys) == len(data) we sample that and save each iteration
    '''
    # create a file first
    hf = h5py.File(directory + filename + '.h5', 'w')
    
    # create the labels
    labels = keys if len(keys) == len(data) else ([keys[0]] if len(keys) != 0 else 'green')
    
    # save the file
    if len(labels) == 1:
        hf.create_dataset(labels[0], data=data)
    else:
        for i, lbl in enumerate(labels):
            hf.create_dataset(lbl, data=data[i].reshape(shape))
    # close
    hf.close()

def app_hdf5(directory, filename, data : np.ndarray, key : str):
    '''
    Appends hdf5 file
    - key : given key to append the data with
    '''
    # create a file first
    hf = h5py.File(directory + filename + '.h5', 'a')
    # save
    hf.create_dataset(key, data=data)
    # close
    hf.close()
    
########################################################### CUTTER ##########################################################

def cut_matrix_bad_vals_zero(M, 
                             axis           = 0, 
                             tol            = 1e-9, 
                             check_limit    = 10):
    """
    Cut off the rows or columns in matrix M where all elements are close to zero.

    Parameters:
    - M (numpy.ndarray) : The input matrix.
    - axis (int)        : The axis along which to check for zero elements (0 for rows, 1 for columns).
    - tol (float)       : The tolerance for considering elements as zero.

    Returns:
    - numpy.ndarray: The resulting matrix after removing zero rows or columns.
    """
    if axis == 0:
        # Check rows
        if check_limit is not None:
            check_limit = min(check_limit, M.shape[1])
            mask = ~np.all(np.isclose(M[:, :check_limit], 0.0, atol=tol), axis=1)
        else:
            mask = ~np.all(np.isclose(M, 0.0, atol=tol), axis=1)
    elif axis == 1:
        # Check columns
        if check_limit is not None:
            check_limit = min(check_limit, M.shape[0])
            mask = ~np.all(np.isclose(M[:check_limit, :], 0.0, atol=tol), axis=0)
        else:
            mask = ~np.all(np.isclose(M, 0.0, atol=tol), axis=0)
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")

    # Return the matrix with the zero rows or columns removed
    return M[mask] if axis == 0 else M[:, mask]

########################################################### CUTTER ##########################################################

def cut_matrix_bad_vals(M, 
                        axis        = 0, 
                        threshold   = 1e-4, 
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

########################################################### CUTTER ##########################################################