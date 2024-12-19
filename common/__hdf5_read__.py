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
                       listme       = True
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

####################################################### SAVE HDF5 FILE #######################################################
    
def create_labels_hdf5(data, keys, shape : tuple = ()):
    if len(keys) == len(data) or shape != ():
        return keys
    elif len(keys) == 1:
        return keys
    elif len(keys) > 1:
        return [keys[0] + "_" + str(i) for i in range(len(data))]
    else:
        return ['data_' + str(i) for i in range(len(data))]

def save_hdf5(directory, filename, data, shape : tuple = (), keys : list = []):
    '''
    Creates and saves an ndarray as hdf5 file.
    - filename  : name of the file to be saved
    - data      : data to be saved (can be a list of data or a dictionary)
    - shape     : shape of which the data shall be (if not the same as the data)
    - keys      : if len(keys) == len(data) we sample that and save each iteration
    '''
    # create a file first
    filename    = filename if (filename.endswith(".h5") or filename.endswith(".hdf5")) else filename + ".h5"
    hf          = h5py.File(directory + kPS + filename, 'w')
    
    # create the labels
    labels      = []
    if not isinstance(data, dict):
        labels = create_labels_hdf5(data, keys, shape)
    
    # save the file
    if type(data) == np.ndarray or isinstance(data, list):
        if len(labels) == 1:
            hf.create_dataset(labels[0], data = np.array(data).reshape(shape) if shape != () else data)
        else:
            for i, lbl in enumerate(labels):
                hf.create_dataset(lbl, data = data[i].reshape(shape) if shape != () else data[i])
    elif isinstance(data, dict):
        for key in data.keys():
            hf.create_dataset(key, data = np.array(data[key]).reshape(shape) if shape != () else data[key])
    # close
    hf.close()
    
def append_hdf5(directory, filename, new_data, keys=[]):
    """
    Append new data to an existing HDF5 file.
    - directory: Directory where the file is located.
    - filename: Name of the HDF5 file.
    - new_data: Data to be appended (can be a list of data or a dictionary).
    - keys: Keys for the new data. If empty, keys will be generated.
    """
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist")
        return
    
    with h5py.File(file_path, 'a') as hf:
        labels = []
        if not isinstance(new_data, dict):
            labels = create_labels_hdf5(new_data, keys)
        
        if isinstance(new_data, (np.ndarray, list)):
            for i, lbl in enumerate(labels):
                if lbl in hf:
                    hf[lbl].resize((hf[lbl].shape[0] + new_data[i].shape[0]), axis=0)
                    hf[lbl][-new_data[i].shape[0]:] = new_data[i]
                else:
                    hf.create_dataset(lbl, data=new_data[i], maxshape=(None,) + new_data[i].shape[1:])
        elif isinstance(new_data, dict):
            for lbl in new_data.keys():
                if lbl in hf:
                    hf[lbl].resize((hf[lbl].shape[0] + new_data[lbl].shape[0]), axis=0)
                    hf[lbl][-new_data[lbl].shape[0]:] = new_data[lbl]
                else:
                    hf.create_dataset(lbl, data=np.array(new_data[lbl], dtype=np.float64))

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

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

# FOR THE FUTURE - encapsulate this in a class

class HDF5Handler:
    def __init__(self, file_path=None, verbose=False, remove_bad=False):
        self.file_path = file_path
        self.verbose = verbose
        self.remove_bad = remove_bad
        self.data = {}

    def _allbottomkeys(self, group):
        """
        Recursively collect all dataset keys in an HDF5 group.
        - group: The HDF5 group to start the search from.
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

    ###################################################################
    
    def read_hdf5(self, keys=None):
        """
        Read the HDF5 file.
        - keys: Specific keys to read. If None, read all dataset keys.
        Returns:
        - dict: A dictionary with keys as dataset paths and values as numpy arrays.
        """
        if not self.file_path or not os.path.exists(self.file_path):
            logging.error(f"File {self.file_path} does not exist")
            return {}

        try:
            # Check if the file is indeed an HDF5 file
            if not self.file_path.endswith(('.h5', '.hdf5', '.hdf')):
                logging.error(f"File {self.file_path} is not an HDF5 file")
                return {}

            with h5py.File(self.file_path, "r") as f:
                # Determine keys to read
                if not keys:
                    keys = self._allbottomkeys(f)
                    if self.verbose:
                        logging.info(f"Available keys: {keys}")

                # Read datasets into the dictionary
                for key in keys:
                    try:
                        self.data[key] = f[key][()]
                    except KeyError:
                        if self.verbose:
                            logging.warning(f"Key {key} not found in file {self.file_path}")
                    except Exception as e:
                        if self.verbose:
                            logging.error(f"Error reading {key} from {self.file_path}: {str(e)}")

                self.data["filename"] = self.file_path
                return self.data

        except Exception as e:
            logging.error(f"Error opening file {self.file_path}: {str(e)}")
            if "truncated" in str(e) or "doesn't exist" in str(e) and self.remove_bad:
                logging.info(f"Removing corrupted file {self.file_path}")
                os.remove(self.file_path)
            return {}

    ###################################################################
    
    def extract_key_data(self, hdf5files, key):
        """
        Yield the data from the HDF5 files for the given key. 
        - hdf5files: List of HDF5 files
        """
        for f in hdf5files:
            yield f[key]
    
    def extract_and_concat_data(self, hdf5files, 
                                key, repeatax=0, isvector=False, cut_0_ax=None, cut_v_ax=None, padding=False, check_limit=None):
        """
        Concatenate data from multiple HDF5 files along the specified axis.
        """
        shape_repeat = None
        data = []

        for f in hdf5files:
            try:
                d = f[key]
                if self.verbose:
                    logging.warning(f"{f['filename']}: Reading {key} with shape {d.shape}")

                if shape_repeat is None:
                    shape_repeat = d.shape[repeatax]

                # Handle shape mismatch
                if d.shape[repeatax] != shape_repeat:
                    if padding:
                        padding_shape = list(d.shape)
                        padding_shape[repeatax] = shape_repeat
                        padded_d = np.zeros(tuple(padding_shape), dtype=d.dtype)
                        slices = [slice(None)] * len(d.shape)
                        slices[repeatax] = slice(0, min(d.shape[repeatax], shape_repeat))
                        padded_d[tuple(slices)] = d[tuple(slices)]
                        d = padded_d
                    else:
                        if self.verbose:
                            logging.error(f"Shape mismatch for {key} in {f['filename']}, skipping this file.")
                        continue

                # Concatenate data
                if d.shape[repeatax] == shape_repeat:
                    if d.shape[0] == 1 and len(d.shape) == 2 and isvector:
                        d = d[0]
                    data.append(d)
                else:
                    if self.verbose:
                        logging.error(f"Shape mismatch for {key} in {f['filename']}")

            except Exception as e:
                if self.verbose:
                    logging.error(f"Error reading {key} from {f['filename']}: {str(e)}")
                continue

        # Concatenate collected data
        if data:
            data = np.concatenate(data, axis=0)
        else:
            if self.verbose:
                logging.error(f"No data found for {key}")
            return np.array([])

        # Apply cutting functions if necessary
        if cut_0_ax is not None:
            data = self.cut_matrix_bad_vals_zero(data, axis=cut_0_ax, check_limit=check_limit)
        if cut_v_ax is not None:
            data = self.cut_matrix_bad_vals(data, axis=cut_v_ax, check_limit=check_limit)

        return data

    ###################################################################
    
    @staticmethod
    def cut_matrix_bad_vals_zero(M, axis=0, tol=1e-9, check_limit=10):
        """
        Remove slices where elements are close to zero along the specified axis.
        """
        if M.ndim == 1:
            check_limit = min(check_limit, M.shape[0])
            mask = np.isclose(M[:check_limit], 0.0, atol=tol)
            return M[~mask]

        M_moved = np.moveaxis(M, axis, 0)
        check_limit = min(check_limit, M_moved.shape[1])
        mask = ~np.all(np.isclose(M_moved[:, :check_limit], 0.0, atol=tol), axis=1)
        M_filtered = M_moved[mask]

        return np.moveaxis(M_filtered, 0, axis)

    @staticmethod
    def cut_matrix_bad_vals(M, axis=0, threshold=-1e4, check_limit=None):
        """
        Remove rows or columns where elements are below a threshold along the specified axis.
        """
        if axis == 0:
            check_limit = min(check_limit, M.shape[1])
            mask = ~np.all(M[:, :check_limit] < threshold, axis=1) if check_limit else ~np.all(M < threshold, axis=1)
        elif axis == 1:
            check_limit = min(check_limit, M.shape[0])
            mask = ~np.all(M[:check_limit, :] < threshold, axis=0) if check_limit else ~np.all(M < threshold, axis=0)
        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")

        return M[mask] if axis == 0 else M[:, mask]

    ###################################################################

    def save_hdf5(self, directory, filename, data, shape=(), keys=[]):
        """
        Save data as an HDF5 file.
        """
        filename = filename if filename.endswith(".h5") or filename.endswith(".hdf5") else filename + ".h5"
        hf = h5py.File(os.path.join(directory, filename), 'w')

        labels = self._create_labels_hdf5(data, keys)

        if isinstance(data, (np.ndarray, list)):
            for i, lbl in enumerate(labels):
                hf.create_dataset(lbl, data=data[i].reshape(shape) if shape else data[i])
        elif isinstance(data, dict):
            for lbl in labels:
                hf.create_dataset(lbl, data=data[lbl].reshape(shape) if shape else data[lbl])
        hf.close()

    def _create_labels_hdf5(self, data, keys):
        """
        Create labels for HDF5 datasets.
        """
        if len(keys) == len(data):
            return keys
        elif keys:
            return [keys[0] + "_" + str(i) for i in range(len(data))]
        else:
            return ['data_' + str(i) for i in range(len(data))]

    ###################################################################
    
    def change_h5_bad(self, directory, filename, keys2new={}, c_zero_ax=None, c_val_ax=None, directoryS="", checkLim=10):
        """
        Modify HDF5 file keys and remove bad elements.
        """
        self.file_path = os.path.join(directory, filename)
        h5data = self.read_hdf5()

        if not h5data:
            logging.error(f"Could not read {filename}")
            return

        truekeys = [x for x in list(h5data.keys()) if x != 'filename']

        if c_zero_ax is not None:
            for key in truekeys:
                h5data[key] = self.cut_matrix_bad_vals_zero(h5data[key], axis=c_zero_ax, check_limit=checkLim)

        if c_val_ax is not None:
            for key in truekeys:
                h5data[key] = self.cut_matrix_bad_vals(h5data[key], axis=c_val_ax, check_limit=checkLim)

        if keys2new:
            for key in truekeys:
                if key in keys2new:
                    h5data[keys2new[key]] = h5data.pop(key)

        if 'filename' in h5data:
            del h5data['filename']

        if directoryS:
            save_dir = directoryS
        else:
            save_dir = directory

        self.save_hdf5(save_dir, filename, h5data, shape=())
