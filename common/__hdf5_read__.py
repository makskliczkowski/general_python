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

        return data
        
    except Exception as e:
        logging.error(f"Error opening file {file_path}: {str(e)}")
        if "truncated" in str(e) or "doesn't exist" in str(e) and removeBad:
            logging.info(f"Removing corrupted file {file_path}")
            os.remove(file_path)
        return {}

####################################################### READ HDF5 FILE #######################################################

def read_multiple_hdf5(directories  : list[Directories], 
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
    files = Directories.listDirs(directories, conditions = conditions)
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