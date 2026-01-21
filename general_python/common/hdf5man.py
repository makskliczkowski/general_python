'''
A collection of functions to read, write, and process HDF5 files. 
'''
from __future__ import annotations

from pathlib import Path
import os
import h5py
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Generator, Tuple, Callable

try:
    from ..common.directories import Directories
    
except ImportError:
    raise ImportError("Required modules from 'common' package are missing.")

# --------------------------------

class LazyHDF5Entry:
    """A proxy that holds metadata but loads data only on demand."""
    def __init__(self, filepath, params):
        self.filepath   = filepath
        self.filename   = Path(filepath).name
        self.params     = params
        self._cache     = {}

    def __getitem__(self, key):
        """Behaves like a dict, but loads from HDF5 on the fly."""
        if key in self._cache:
            return self._cache[key]
        
        data = HDF5Manager.read_hdf5(self.filepath, keys=[key], verbose=False)
        if key in data:
            self._cache[key] = data[key]
            return data[key]
        raise KeyError(f"Key '{key}' not found in {self.filename}")

    def get(self, key, default=None):
        """Get method with default value."""
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
        
    def __contains__(self, key):
        """Check if key exists without loading data."""
        if key in self._cache:
            return True
        data = HDF5Manager.read_hdf5(self.filepath, keys=[key], verbose=False)
        return key in data
    
    def __len__(self):
        """Number of keys available (may require loading)."""
        if self._cache:
            return len(self._cache)
        data = HDF5Manager.read_hdf5(self.filepath, verbose=False)
        self._cache.update(data)
        return len(self._cache)
    
    def keys(self):
        """List of keys available (may require loading)."""
        if self._cache:
            return self._cache.keys()
        data = HDF5Manager.read_hdf5(self.filepath, verbose=False)
        self._cache.update(data)
        return self._cache.keys()

    def load_all(self):
        """Force load everything if needed."""
        self._cache = HDF5Manager.read_hdf5(self.filepath, verbose=False)
        return self

# --------------------------------

#! HDF5Manager
class HDF5Manager:
    '''
    A class encapsulating methods for reading, writing, and processing HDF5 files.
    Methods include:
        - load_file_data: Read data from a single HDF5 file.
        - stream_key_from_loaded_files: Generator to yield specific dataset from loaded data.
    '''
    
    # ---------------------------------
    #! Data Processing Methods
    
    @staticmethod
    def _get_all_dataset_paths(h5_group: h5py.Group) -> List[str]:
        """
        Recursively collects all dataset paths within an HDF5 group.
        A dataset path is its full path starting from the root.
        """
        dataset_paths = []
        def _collect_paths(obj, current_path=""):
            if isinstance(obj, h5py.Dataset):
                dataset_paths.append(obj.name)
            elif isinstance(obj, h5py.Group):
                for name, item in obj.items():
                    _collect_paths(item, os.path.join(current_path, name))
        
        _collect_paths(h5_group)
        return dataset_paths

    @staticmethod
    def _validate_file(file_path, verbose: bool = False):
        """
        Validates if the file exists and is an HDF5 file.
        """
        if not os.path.exists(file_path):
            if verbose:
                logging.error(f"File does not exist: {file_path}")
            return False
        
        if not str(file_path).lower().endswith(('.h5', '.hdf5', '.hdf')):
            if verbose:
                logging.error(f"File is not an HDF5 file (based on extension): {file_path}")
            return False
        return True

    @staticmethod
    def _read_data_key(hf: h5py.File, key: str) -> Optional[np.ndarray]:
        """
        Reads a specific dataset from an HDF5 file.
        Returns None if the dataset is not found or cannot be read.
        """
        try:
            return hf[key][()]
        except KeyError:
            logging.error(f"Dataset key '{key}' not found in HDF5 file.")
            return None
        except Exception as e:
            logging.error(f"Error reading dataset '{key}': {e}")
            return None

    # ---------------------------------
    #! Loading and Concatenation Methods
    # ---------------------------------
    
    @staticmethod
    def load_file_data(
        file_path               : str,
        dataset_keys            : Optional[List[str]]   = None,
        verbose                 : bool                  = False,
        remove_corrupted_file   : bool                  = False,
        strict_keys             : bool                  = True) -> Dict[str, Any]:
        """
        Reads data from an HDF5 file.

        - If `dataset_keys` is provided:
            * strict_keys=True  -> skip file entirely if any key is missing
            * strict_keys=False -> load only available keys
        """
        data: Dict[str, Any] = {}
        if not HDF5Manager._validate_file(file_path, verbose):
            return data

        try:
            with h5py.File(file_path, "r") as hf:

                if dataset_keys:
                    keys_to_read = dataset_keys
                else:
                    # Only scan tree if we want EVERYTHING
                    keys_to_read = HDF5Manager._get_all_dataset_paths(hf)

                # strict key check
                if dataset_keys and strict_keys:
                    missing = [k for k in dataset_keys if k not in keys_to_read]
                    if missing:
                        if verbose:
                            _logger.warning(f"Skipping file {file_path}: missing keys {missing}")
                        return {}

                if verbose:
                    _logger.info(f"Reading {len(keys_to_read)} datasets from {file_path}")

                for key in keys_to_read:
                    if key in hf:
                        data_in = HDF5Manager._read_data_key(hf, key)
                        if data_in is not None:
                            data[key] = data_in
                    elif strict_keys:
                        if verbose: 
                            _logger.warning(f"Missing strict key: {key}")
                        return {} # Fail fast

            if data:  # only attach filename if some data loaded
                data["filename"] = file_path
            return data

        except Exception as e:
            logging.error(f"Error opening or reading HDF5 file {file_path}: {e}")
            if remove_corrupted_file and ("truncated" in str(e).lower() or "doesn't exist" in str(e).lower()):
                _logger.warning(f"Attempting to remove corrupted file: {file_path}")
                try:
                    os.remove(file_path)
                    _logger.info(f"Successfully removed corrupted file: {file_path}")
                except OSError as oe:
                    logging.error(f"Failed to remove corrupted file {file_path}: {oe}")
            return {}

    @staticmethod
    def read_hdf5(file_path, keys=None, verbose=False, remove_bad=False):
        return HDF5Manager.load_file_data(file_path, dataset_keys=keys, verbose=verbose, remove_corrupted_file=remove_bad)

    # ---------------------------------

    @staticmethod
    def stream_key_from_loaded_files(loaded_hdf5_data_list : List[Dict[str, Any]], key : str) -> Generator[np.ndarray, None, None]:
        """
        Yields data for a specific key from a list of already loaded HDF5 data dictionaries.
        Each dictionary in loaded_hdf5_data_list is expected to be the output of 'load_file_data'.

        Args:
            loaded_hdf5_data_list:
                List of dictionaries, where each dict contains data from an HDF5 file.
            key:
                The dataset key to extract from each dictionary.

        Yields:
            numpy.ndarray: The dataset corresponding to the key.
        """
        for data_dict in loaded_hdf5_data_list:
            if key in data_dict:
                yield data_dict[key]
            else:
                _logger.warning(f"Key '{key}' not found in loaded data from file: {data_dict.get('filename', 'Unknown file')}")

    @staticmethod
    def concatenate_key_from_loaded_files(
        loaded_hdf5_data_list   : List[Dict[str, Any]],
        key                     : str,
        concat_axis             : int  = 0,
        target_shape_axis       : Optional[int] = None,             # The axis whose dimension should be consistent
        allow_padding           : bool = False,
        is_vector               : bool = False,
        clean_zeros_params      : Optional[Dict[str, Any]] = None,  # e.g., {'axis': 0, 'check_limit': 10}
        clean_threshold_params  : Optional[Dict[str, Any]] = None,  # e.g., {'axis': 0, 'threshold': -1e4}
        verbose                 : bool = False) -> np.ndarray:
        """
        Concatenates a specific dataset key from a list of loaded HDF5 data.
        Handles shape mismatches by padding (if enabled) or skipping.

        Args:
            loaded_hdf5_data_list:
                List of dictionaries (from 'load_file_data').
            key:
                Dataset key to extract and concatenate.
            concat_axis:
                Axis along which to concatenate arrays.
            target_shape_axis:
                If specified, datasets must have the same size along this axis
                as the first valid dataset found. Usually this is 'concat_axis'.
            allow_padding:
                If True, pads/truncates datasets to match the target shape on 'target_shape_axis'.
            is_vector:
                If True and data is 2D (1, N), flatten to 1D (N,).
            clean_zeros_params:
                Optional dict of parameters for 'clean_data_remove_zeros'.
            clean_threshold_params:
                Optional dict of parameters for 'clean_data_remove_thresholded'.
            verbose:    
                If True, log detailed information.

        Returns:
            A concatenated numpy array. Returns an empty array if no data is found or processed.
        """
        datasets_to_concat = []
        reference_dim_size = None

        if target_shape_axis is None:
            target_shape_axis = concat_axis # Default to concat_axis if not specified

        for data_dict in loaded_hdf5_data_list:
            filename = data_dict.get('filename', 'Unknown file')
            if key not in data_dict:
                if verbose:
                    _logger.warning(f"Key '{key}' not in {filename}, skipping.")
                continue

            d = data_dict[key]
            if not isinstance(d, np.ndarray):
                if verbose:
                    _logger.warning(f"Data for key '{key}' in {filename} is not a numpy array, skipping.")
                continue
            
            if verbose:
                _logger.info(f"{filename}: Reading key '{key}' with shape {d.shape}")

            if d.ndim <= target_shape_axis:
                if verbose:
                    logging.error(f"Dataset for key '{key}' in {filename} has insufficient dimensions ({d.ndim}) for target_shape_axis {target_shape_axis}, skipping.")
                continue

            current_dim_size = d.shape[target_shape_axis]

            if reference_dim_size is None:
                reference_dim_size = current_dim_size
            
            if current_dim_size != reference_dim_size:
                if allow_padding:
                    padding_shape                       = list(d.shape)
                    padding_shape[target_shape_axis]    = reference_dim_size
                    padded_d                            = np.zeros(tuple(padding_shape), dtype=d.dtype)
                    slices                              = [slice(None)] * d.ndim
                    slices[target_shape_axis]           = slice(0, min(current_dim_size, reference_dim_size))
                    
                    # Ensure compatible slicing for assignment
                    source_slices                       = [slice(None)] * d.ndim
                    source_slices[target_shape_axis]    = slice(0, min(current_dim_size, reference_dim_size))

                    padded_d[tuple(slices)]             = d[tuple(source_slices)]
                    d                                   = padded_d
                    if verbose:
                        _logger.info(f"Padded/truncated data for key '{key}' in {filename} to shape {d.shape} on axis {target_shape_axis}.")
                else:
                    if verbose:
                        _logger.warning(f"Shape mismatch for key '{key}' in {filename} on axis {target_shape_axis} "
                                        f"(expected {reference_dim_size}, got {current_dim_size}). Skipping file.")
                    continue
            
            if is_vector and d.ndim == 2 and d.shape[0] == 1:
                d = d.ravel()
            
            datasets_to_concat.append(d)

        if not datasets_to_concat:
            if verbose:
                _logger.warning(f"No data found or processed for key '{key}'.")
            return np.array([])

        try:
            concatenated_data = np.concatenate(datasets_to_concat, axis=concat_axis)
        except ValueError as e:
            logging.error(f"Error concatenating data for key '{key}': {e}. Check shapes: {[arr.shape for arr in datasets_to_concat]}")
            return np.array([])

        if clean_zeros_params:
            concatenated_data = HDF5Manager.clean_data_remove_zeros(concatenated_data, **clean_zeros_params)
        if clean_threshold_params:
            concatenated_data = HDF5Manager.clean_data_remove_thresholded(concatenated_data, **clean_threshold_params)
            
        return concatenated_data

    # ---------------------------------
    #! File Management Methods
    # ---------------------------------    

    @staticmethod
    def stream_data_from_multiple_files(
            file_paths      : List[str],
            dataset_keys    : Optional[List[str]] = None,
            sort_files      : bool = True,
            verbose         : bool = False,
            strict_keys     : bool = True,     # pass-through
        ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams data dictionary (from 'load_file_data') for each HDF5 file found in specified paths.

        Args:
            file_paths:
                List of HDF5 file paths.
            dataset_keys:
                Specific dataset keys to load from each file.
            sort_files: Whether to sort the found files by name.
            verbose: If True, log detailed information.

        Yields:
            dict: Data dictionary from each processed HDF5 file.
        """
        if sort_files:
            file_paths.sort()

        for file_path in file_paths:
            if verbose:
                _logger.info(f"Processing file: {file_path}")
            data = HDF5Manager.load_file_data(file_path, dataset_keys, verbose=verbose, strict_keys=strict_keys)
            if data: # Only yield if data was successfully loaded
                yield data

    @staticmethod
    def load_data_from_multiple_files(
        file_paths            : List[str] | list[str],
        dataset_keys          : Optional[List[str]] = None,
        sort_files            : bool = True,
        verbose               : bool = False) -> List[Dict[str, Any]]:
        """
        Loads data from multiple HDF5 files into a list of dictionaries. Eager evaluation.
        (This was 'read_multiple_hdf5' before)
        """
        return list(HDF5Manager.stream_data_from_multiple_files(file_paths, dataset_keys, sort_files, verbose))

    # ---------------------------------
    #! Saving Methods
    # ---------------------------------
    
    @staticmethod
    def _get_standard_dtype(arr: Any):
        if np.iscomplexobj(arr): return np.complex128
        if arr.dtype.kind in np.typecodes['AllInteger']: return np.int64
        return np.float64
    
    @staticmethod
    def _create_labels_dataset(data, keys : list | dict, shape : tuple = ()) -> list:
        if len(keys) == len(data) or shape != ():
            return keys
        elif len(keys) == 1:
            return keys
        elif len(keys) > 1:
            return [keys[0] + "_" + str(i) for i in range(len(data))]
        else:
            return ['data_' + str(i) for i in range(len(data))]

    @staticmethod
    def save_data_to_file(directory: str, filename: str, data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]], shape: Tuple = (), keys: list = [], override: bool = True):
        '''
        Creates and saves data to an HDF5 file. (Refactored to use robust logic).

        Parameters:
        -----------
        directory : str
            Directory where the file will be saved.
        filename : str
            Name of the HDF5 file.
        data : Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
            Data to be saved. Can be a single array, a list of arrays, or a dictionary of arrays.
        shape : Tuple, optional
            Desired shape to reshape the data before saving (default is ()).
        keys : list, optional
            Keys/labels for the datasets if data is a list or array (default is []).
        override : bool, optional
            If True, overwrite existing datasets (default is True).
        '''
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        if override and os.path.exists(os.path.join(directory, filename)):
            os.remove(os.path.join(directory, filename))
            
        # Ensure filename has .h5 extension
        filename = str(filename)
        filename = filename if (filename.endswith(".h5") or filename.endswith(".hdf5")) else filename + ".h5"
        path     = os.path.join(directory, filename)

        # Use 'w' mode (overwrite/truncate is implied by the original function's behavior)
        # The original function did not check for existence, so 'w' is the correct mode.
        # If the original function relied on explicit os.remove, this is simplified here.
        try:
            with h5py.File(path, 'w') as hf:
                if isinstance(data, dict):
                    # Process Dictionary Input
                    for key, arr in data.items():
                        dtype   = HDF5Manager._get_standard_dtype(np.array(arr))
                        arr     = np.array(arr, dtype=dtype)
                        if shape:
                            arr = arr.reshape(shape)
                        hf.create_dataset(key, data=arr)
                else:
                    # Process List/Array Input
                    datasets = [data] if isinstance(data, np.ndarray) else list(data)
                    
                    # Determine labels: Use keys if provided, otherwise rely on the external
                    if keys and len(keys) == len(datasets):
                        labels = keys
                    else:
                        labels = HDF5Manager._create_labels_dataset(data, keys, shape) 

                    # Save each dataset
                    for lbl, arr in zip(labels, datasets):
                        dtype = HDF5Manager._get_standard_dtype(np.array(arr))
                        arr   = np.array(arr, dtype=dtype)
                        if shape:
                            arr = arr.reshape(shape)
                        hf.create_dataset(lbl, data=arr)
        except Exception as e:
            logging.error(f"Error saving HDF5 file {path}: {e}")
            raise
    
    @staticmethod
    def append_data_to_file(directory: str, filename: str, new_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]], 
                    keys: list = [], override: bool = True):
        """
        Append new data to an existing HDF5 file. (Refactored to use robust logic).
        - directory: Directory where the file is located.
        - filename: Name of the HDF5 file.
        - new_data: Data to be appended (can be a list of data or a dictionary).
        - keys: Keys for the new data.
        - override: If True, overwrite datasets instead of appending (original function's 'override').
        """
        
        # Ensure filename has .h5 extension
        filename    = filename if (filename.endswith(".h5") or filename.endswith(".hdf5")) else filename + ".h5"
        path        = os.path.join(directory, filename)

        # Check existence and fallback to save_hdf5 (matching original logic)
        if not os.path.exists(path):
            return HDF5Manager.save_data_to_file(directory, filename, new_data, shape=(), keys=keys)
                
        # Append or overwrite datasets
        try:
            with h5py.File(path, 'a') as hf:
                
                # Determine labels/items for iteration
                if isinstance(new_data, dict):
                    items = new_data.items()
                else:
                    datasets = [new_data] if isinstance(new_data, np.ndarray) else list(new_data)
                    
                    # Determine labels
                    if keys and len(keys) == len(datasets):
                        labels = keys
                    else:
                        # NOTE: This call relies on an external definition.
                        labels = HDF5Manager._create_labels_dataset(new_data, keys) 
                        
                    items = zip(labels, datasets)

                # Process each dataset
                for k, arr in items:
                    arr = np.array(arr)
                    
                    if k in hf:
                        if override:
                            del hf[k]
                            hf.create_dataset(k, data=arr)
                        else:
                            if arr.ndim == 0:
                                raise ValueError(f"Cannot append scalar data to existing dataset '{k}'. Use override=True.")

                            hf[k].resize(hf[k].shape[0] + arr.shape[0], axis=0)
                            hf[k][-arr.shape[0]:] = arr
                    else:
                        # Create new dataset
                        # Use maxshape=(None,) for append compatibility in future calls
                        maxshape_val = (None,) + arr.shape[1:] if arr.ndim > 0 else (None,)
                        hf.create_dataset(k, data=arr, maxshape=maxshape_val)
                        
        except Exception as e:
            logging.error(f"Error appending HDF5 file {path}: {e}")
            raise

    save_hdf5   = save_data_to_file 
    append_hdf5 = append_data_to_file

    # ---------------------------------
    #! Folders
    # ---------------------------------

    @staticmethod
    def file_list_matching(directories          : Union[List, Directories, str],
                        *args,                  # additional arguments to create the directories
                        conditions              : List[Callable]    = [],
                        check_hdf5_condition    : bool              = True,
                        as_string               : bool              = True):
        """
        Returns a list of HDF5 files in the specified directories matching given conditions.
        Args:
            directories:
                A list of directory paths (str) or Directories objects, or a single one.
            *args:
                Additional arguments passed to Directories constructor if directories are str.
            conditions:
                A list of callables that take a filename and return True if it matches the condition.
            check_hdf5_condition:
                If True (default), adds a condition to only include files ending with .h5 or .hdf5.
            as_string:
                If True (default), returns file paths as strings. If False, returns as Path objects.
        Returns:
            A sorted list of file paths matching the conditions.
        """

        if isinstance(directories, str) or isinstance(directories, Directories):
            directories = [directories]

        if not isinstance(conditions, list):
            if callable(conditions):
                conditions = [conditions]
            else:
                conditions = []

        if check_hdf5_condition:
            conditions = conditions + [lambda x: str(x).endswith('.h5') or str(x).endswith('.hdf5')]

        # get all directories
        directories_in  = [Directories(d, *args) for d in directories]
        filelist        = [x for d in directories_in for x in d.list_files(filters = conditions)]
        filelist        = sorted(filelist)
        if as_string:
            filelist = [str(x) for x in filelist]
        return filelist

    @staticmethod
    def stream_data_from_multiple_folders(
        directory_paths     : List[Directories],
        file_conditions     : Optional[List[Any]] = None, # Conditions for Directories
        dataset_keys        : Optional[List[str]] = None,
        sort_files          : bool = True,
        verbose             : bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Streams data dictionary (from 'load_file_data') for each HDF5 file found in specified directories.

        Args:
            directory_paths:
                List of directories to search for HDF5 files.
            file_conditions:
                Conditions passed to 'Directories.listDirs' for filtering files.
            dataset_keys:
                Specific dataset keys to load from each file.
            sort_files: Whether to sort the found files by name.
            verbose: If True, log detailed information.

        Yields:
            dict: Data dictionary from each processed HDF5 file.
        """
        # Use a placeholder for sortCondition if not specifically different
        sort_lambda = (lambda x: x) if sort_files else None
        
        try:
            # Assuming Directories.listDirs returns full paths if appendDir=True
            # and uses conditions to filter filenames.
            # For safety, always use appendDir=True here or handle path construction.
            file_paths  =   [d.list_files(filters=file_conditions, sort_key=sort_lambda)
                                for d in directory_paths]
            
        except NameError: # Directories class might not be defined if import failed
            logging.error("'Directories' class is not available. Cannot list files.")
            return # yield nothing
        except Exception as e:
            logging.error(f"Error listing directories: {e}")
            return

        if not file_paths:
            _logger.warning(f"No files found in {directory_paths} matching conditions.")
            return

        for file_path in file_paths:
            if verbose:
                _logger.info(f"Processing file: {file_path}")
            data = HDF5Manager.load_file_data(file_path, dataset_keys, verbose=verbose)
            if data: # Only yield if data was successfully loaded
                yield data
    
    @staticmethod
    def load_data_from_multiple_folders(
        directory_paths     : List[str],
        file_conditions     : Optional[List[Any]] = None,
        dataset_keys        : Optional[List[str]] = None,
        sort_files          : bool = True,
        verbose             : bool = False) -> List[Dict[str, Any]]:
        """
        Loads data from multiple HDF5 files into a list of dictionaries. Eager evaluation.
        (This was 'read_multiple_hdf5l' before)
        """
        return list(HDF5Manager.stream_data_from_multiple_files(
            directory_paths, file_conditions, dataset_keys, sort_files, verbose))

    @staticmethod
    def load_and_concatenate_key_from_folders(
        directory_paths         : List[str],
        key_to_extract          : str,
        file_conditions         : Optional[List[Any]] = None,
        concat_axis             : int = 0,
        target_shape_axis       : Optional[int] = None,
        allow_padding           : bool = False,
        is_vector               : bool = False,
        clean_zeros_params      : Optional[Dict[str, Any]] = None,
        clean_threshold_params  : Optional[Dict[str, Any]] = None,
        sort_files              : bool = True,
        verbose                 : bool = False) -> np.ndarray:
        """
        Reads a specific dataset key from multiple HDF5 files found in directories and concatenates them.
        (This was 'read_hdf5_extract_and_concat' before)
        """
        if target_shape_axis is None:
            target_shape_axis = concat_axis

        loaded_data_stream  = HDF5Manager.stream_data_from_multiple_folders(
            directory_paths,
            file_conditions,
            dataset_keys    =   [key_to_extract] if key_to_extract else None, # Load only the required key
            sort_files      =   sort_files,
            verbose         =   verbose)
        
        # Collect data first, then pass to concatenate_key_from_loaded_data
        # This is because concatenate_key_from_loaded_data needs the first dataset to establish reference shape
        # A more memory-efficient version would require more complex logic if files are huge and padding is involved.
        # For now, this matches the implied logic of the original code.
        data_dicts_list     = [data for data in loaded_data_stream if data]

        if not data_dicts_list:
            if verbose:
                _logger.warning(f"No files processed or data loaded for key '{key_to_extract}'.")
            return np.array([])

        return HDF5Manager.concatenate_key_from_loaded_files(
            data_dicts_list,
            key_to_extract,
            concat_axis             =   concat_axis,
            target_shape_axis       =   target_shape_axis,
            allow_padding           =   allow_padding,
            is_vector               =   is_vector,
            clean_zeros_param       =   clean_zeros_params,
            clean_threshold_params  =   clean_threshold_params,
            verbose                 =   verbose)

    @staticmethod
    def load_and_concatenate_key_per_directory(
        list_of_directory_paths : List[str], # Each item is a directory to process independently
        key_to_extract          : str,
        file_conditions         : Optional[List[Any]] = None,
        concat_axis             : int = 0,
        target_shape_axis       : Optional[int] = None,
        allow_padding           : bool = False,
        is_vector               : bool = False,
        clean_zeros_params      : Optional[Dict[str, Any]] = None,
        clean_threshold_params  : Optional[Dict[str, Any]] = None,
        verbose                 : bool = False) -> List[np.ndarray]:
        """
        For each directory in 'list_of_directory_paths', loads and concatenates data for 'key_to_extract'.
        Returns a list of concatenated numpy arrays, one for each input directory.
        (This was 'read_hdf5_extract_and_concat_list' before)
        """
        results = []
        for dir_path in list_of_directory_paths:
            concatenated_data = HDF5Manager.load_and_concatenate_key_from_folders(
                directory_paths         =   [dir_path], # Process one directory at a time
                key_to_extract          =   key_to_extract,
                file_conditions         =   file_conditions,
                concat_axis             =   concat_axis,
                target_shape_axis       =   target_shape_axis,
                allow_padding           =   allow_padding,
                is_vector               =   is_vector,
                clean_zeros_params      =   clean_zeros_params,
                clean_threshold_params  =   clean_threshold_params,
                sort_files              =   True, # Default sort within directory
                verbose                 =   verbose
            )
            results.append(concatenated_data)
        return results

    # ---------------------------------

    @staticmethod
    def _generate_dataset_names(
        num_datasets    : int,
        proposed_names  : Optional[Union[List[str], str]] = None) -> List[str]:
        """
        Generates dataset names for saving multiple datasets.
        """
        if isinstance(proposed_names, str):
            return [f"{proposed_names}_{i}" for i in range(num_datasets)]

        if isinstance(proposed_names, list):
            if len(proposed_names) == num_datasets:
                if any(name == "" for name in proposed_names):
                    raise ValueError("Dataset names must not be empty strings.")
                return proposed_names
            if len(proposed_names) == 1:
                prefix = proposed_names[0]
                if not prefix:
                    raise ValueError("Dataset name prefix must not be an empty string.")
                return [f"{prefix}_{i}" for i in range(num_datasets)]

        # Default naming
        return [f"dataset_{i}" for i in range(num_datasets)]

    @staticmethod
    def save_data_to_file(
        directory           : Union[str, Directories],
        filename            : str,
        data_to_save        : Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        target_shape        : Optional[Tuple[int, ...]] = None,
        dataset_names_config: Optional[Union[List[str], str]] = None, # Used if data_to_save is list/ndarray
        overwrite           : bool = True):
        """
        Saves data to an HDF5 file.

        Args:
            directory: Directory to save the file.
            filename: Name of the HDF5 file (extension .h5 or .hdf5 will be ensured).
            data_to_save: 
                Data to save. Can be a single np.ndarray, a list of np.ndarrays,
                or a dictionary {name: np.ndarray}.
            target_shape: 
                If specified, datasets will be reshaped to this shape before saving.
                dataset_names_config: Names for datasets if 'data_to_save' is a list/ndarray.
                If a string, used as a prefix. If a list, used as names.
            overwrite:
                If True (default), overwrites the file if it exists.
        """
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                logging.error(f"Could not create directory {directory}: {e}")
                return

        base, ext = os.path.splitext(filename)
        if ext.lower() not in ['.h5', '.hdf5']:
            filename = base + '.h5'
        
        file_path = os.path.join(directory, filename)
        
        mode = 'w' if overwrite else 'w-' # 'w-' fails if file exists

        try:
            with h5py.File(file_path, mode) as hf:
                if isinstance(data_to_save, dict):
                    for name, dataset_array in data_to_save.items():
                        dtype = np.complex128 if np.iscomplexobj(dataset_array) else np.float64 # Or infer from array
                        array_to_write = np.array(dataset_array, dtype=dtype)
                        if target_shape:
                            array_to_write = array_to_write.reshape(target_shape)
                        hf.create_dataset(name, data=array_to_write)
                
                elif isinstance(data_to_save, (list, np.ndarray)):
                    datasets = data_to_save
                    if isinstance(data_to_save, np.ndarray) and data_to_save.ndim == 0: # scalar
                        datasets = [data_to_save] # treat as list of one
                    elif isinstance(data_to_save, np.ndarray) and data_to_save.ndim > 0:
                        # If it's a single multidim array, and no dataset_names_config is given,
                        # or dataset_names_config is a single string, save as one dataset.
                        # If dataset_names_config is a list, it implies data_to_save should be a list of arrays.
                        # This behavior needs to be clear. Assuming if ndarray, it's one dataset unless names imply multiple.
                        if not dataset_names_config or isinstance(dataset_names_config, str) or \
                                            (isinstance(dataset_names_config, list) and len(dataset_names_config) == 1):
                            datasets = [data_to_save] # Treat as a list containing one dataset
                        # else: if names_config is list of N > 1, but data is single ndarray, that's ambiguous.


                    names = HDF5Manager._generate_dataset_names(len(datasets), dataset_names_config)
                    
                    for i, name in enumerate(names):
                        dataset_array = datasets[i]
                        dtype = np.complex128 if np.iscomplexobj(dataset_array) else np.float64
                        array_to_write = np.array(dataset_array, dtype=dtype)
                        if target_shape:
                            array_to_write = array_to_write.reshape(target_shape)
                        hf.create_dataset(name, data=array_to_write)
                else:
                    logging.error(f"Unsupported data type for saving: {type(data_to_save)}. Must be dict, list, or ndarray.")

        except Exception as e:
            logging.error(f"Error saving HDF5 file {file_path}: {e}")

    @staticmethod
    def append_data_to_file(
        directory                   : str,
        filename                    : str,
        new_data                    : Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        dataset_names_config        : Optional[Union[List[str], str]] = None, # Used if new_data is list/ndarray
        overwrite_existing_datasets : bool = True, # If dataset exists, overwrite or append rows
        allow_dataset_creation      : bool = True # If dataset does not exist, create it
    ):
        """
        Appends data to an existing HDF5 file or creates it if it doesn't exist.

        Args:
            directory:
                Directory of the HDF5 file.
            filename:
                Name of the HDF5 file.
            new_data:
                Data to append.
            dataset_names_config:   
                Names for datasets if 'new_data' is list/ndarray.
            overwrite_existing_datasets:
                - If True and dataset exists, it's deleted and recreated.
                - If False and dataset exists, data is appended (row-wise).
                Requires dataset to be resizable.
            allow_dataset_creation:
                If True, new datasets are created if they don't exist.
        """
        base, ext = os.path.splitext(filename)
        if ext.lower() not in ['.h5', '.hdf5']:
            filename = base + '.h5'
        file_path = os.path.join(directory, filename)

        if not os.path.exists(file_path):
            if allow_dataset_creation:
                _logger.info(f"File {file_path} does not exist. Creating and saving new data.")
                HDF5Manager.save_data_to_file(directory, filename, new_data, dataset_names_config=dataset_names_config, overwrite=True)
                return
            else:
                logging.error(f"File {file_path} does not exist and dataset creation is not allowed.")
                return

        try:
            with h5py.File(file_path, 'a') as hf:
                data_items_to_process: Dict[str, np.ndarray] = {}
                if isinstance(new_data, dict):
                    data_items_to_process = new_data
                elif isinstance(new_data, (list, np.ndarray)):
                    datasets = new_data
                    if isinstance(new_data, np.ndarray) and new_data.ndim > 0: # Single ndarray
                        datasets = [new_data]
                    
                    names = HDF5Manager._generate_dataset_names(len(datasets), dataset_names_config)
                    data_items_to_process = {name: arr for name, arr in zip(names, datasets)}
                else:
                    logging.error("Invalid data type for appending.")
                    return

                for name, data_array in data_items_to_process.items():
                    data_array_np = np.array(data_array) # Ensure it's a numpy array
                    
                    if name in hf:
                        if overwrite_existing_datasets:
                            del hf[name]
                            # Create with maxshape for potential future non-overwrite appends
                            hf.create_dataset(name, data=data_array_np, maxshape=(None,) + data_array_np.shape[1:])
                        else: # Append rows
                            if not hf[name].maxshape or hf[name].maxshape[0] is None : # Check if resizable
                                original_shape = hf[name].shape
                                hf[name].resize((original_shape[0] + data_array_np.shape[0]), axis=0)
                                hf[name][-data_array_np.shape[0]:] = data_array_np
                            else:
                                logging.error(f"Dataset '{name}' in {file_path} is not resizable for appending. Maxshape: {hf[name].maxshape}")
                    elif allow_dataset_creation:
                        # Create with maxshape for future appends
                        hf.create_dataset(name, data=data_array_np, maxshape=(None,) + data_array_np.shape[1:] if data_array_np.ndim > 0 else (None,))
                    else:
                        _logger.warning(f"Dataset '{name}' not found in {file_path} and creation is not allowed.")
        except Exception as e:
            logging.error(f"Error appending to HDF5 file {file_path}: {e}")

    # ---------------------------------
    #! Data Cleaning Methods
    # ---------------------------------
    
    @staticmethod
    def _coerce_array(a: Any, axis_realization: int = 0) -> np.ndarray:
        """
        Coerce HDF5 dataset/list/scalar to a numeric ndarray.
        
        """
        if a is None:
            return np.array([], dtype=float)
        
        arr = np.asarray(a)
        if arr.dtype == object:  # ragged: list of arrays/scalars
            parts = []
            for e in arr:
                if e is None:
                    continue
                e = np.asarray(e)
                if e.size == 0:
                    continue
                parts.append(e)
            if not parts:
                return np.array([], dtype=float)
            
            # all 1D or all 2D?
            ndims = {p.ndim for p in parts}
            if len(ndims) != 1:
                raise ValueError("Mixed ranks in object array; cannot concatenate cleanly.")
            
            if parts[0].ndim == 1:
                return np.concatenate([p.reshape(-1) for p in parts], axis=axis_realization)
            elif parts[0].ndim == 2:
                dset = {p.shape[1] for p in parts}
                if len(dset) != 1:
                    raise ValueError("Inconsistent second dimension in ragged 2D parts.")
                return np.concatenate(parts, axis=axis_realization)
            else:
                raise ValueError("Only 1D or 2D arrays are supported.")
            
        # scalar -> (1,)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    @staticmethod
    def process_data(
        data,
        keys                : str | list[str] | tuple[str, ...],
        throw_if_bad        : bool = False,
        unpack              : bool = True,
        expected_ndim       : int | None = None,
        expected_dim0       : int | None = None,
        expected_dim1       : int | None = None,
        expected_first_val  : Any = None,
        return_skipped      : bool = False) -> np.ndarray | tuple[np.ndarray, list[str]]:
        """
        Collects arrays from iterable of mappings and concatenates them robustly.

        Parameters:
            - data
                iterable of dict-like objects
            - keys: 
                key or list of possible keys to try (first available is used)
            - throw_if_bad: 
                whether to throw if no valid arrays are found
            - unpack: 
                whether to flatten nested arrays along first axis
            - expected_ndim: 
                enforce specific ndim (1 or 2). If None, auto-infer from first valid array.
                This is stricter than just checking consistency.
            - expected_dim0: 
                enforce first dimension length (skip mismatches)
            - expected_dim1: 
                enforce second dimension length (skip mismatches)
            - return_skipped: 
                if True, return (array, skipped_filenames)

        Returns:
            - ndarray (default)
            - (ndarray, skipped_filenames) if return_skipped=True
        
        Example:
        >>> energies = HDF5Manager.process_data(data, "energies")
        >>> energies = HDF5Manager.process_data(data, ["energies", "E"], expected_ndim=1)
        >>> obs = HDF5Manager.process_data(data, "observables", expected_ndim=2, expected_dim1=4)
        """
        if isinstance(keys, str):
            keys = [keys]

        arrays: list[np.ndarray]    = []
        target_dtype                = None
        target_ndim                 = expected_ndim
        target_dim1                 = expected_dim1
        target_first_val            = expected_first_val
        skipped: list[str]          = []

        for idx, x in enumerate(data):
            fname = x.get("filename", f"<item {idx}>")
            try:
                # find a matching key
                found_key = next((k for k in keys if k in x), None)
                if found_key is None:
                    skipped.append(fname)
                    continue

                arr = np.asarray(x[found_key])
                if arr.size == 0:
                    skipped.append(fname)
                    continue
                
                if target_first_val is not None:
                    if arr.ndim == 0:
                        first_val = arr.item()
                    elif arr.ndim >= 1 and arr.shape[0] > 0:
                        first_val = arr.flat[0]
                    else:
                        skipped.append(fname)
                        continue
                    if not np.isclose(first_val, target_first_val):
                        skipped.append(fname)
                        continue

                # normalize rank
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)

                # enforce ndim
                if expected_ndim is not None and arr.ndim != expected_ndim:
                    skipped.append(fname)
                    continue

                if target_ndim is None:
                    target_ndim = arr.ndim
                    if target_ndim == 2:
                        target_dim1 = arr.shape[1]

                # consistency checks
                if target_ndim == 1:
                    if arr.ndim == 2:
                        if arr.shape[1] != 1:
                            skipped.append(fname)
                            continue
                        arr = arr.reshape(-1)
                elif target_ndim == 2:
                    if arr.ndim != 2:
                        skipped.append(fname)
                        continue
                    if target_dim1 is not None and arr.shape[1] != target_dim1:
                        skipped.append(fname)
                        continue

                if expected_dim0 is not None and arr.shape[0] != expected_dim0:
                    skipped.append(fname)
                    continue
                
                if expected_dim1 is not None and arr.ndim > 1 and arr.shape[1] != expected_dim1:
                    skipped.append(fname)
                    continue

                if target_dtype is None:
                    target_dtype = arr.dtype
                    
                arrays.append(arr)

            except Exception as e:
                logging.error(f"Error processing {fname}: {e}")
                skipped.append(fname)

        if not arrays:
            if throw_if_bad:
                raise ValueError(f"No valid data found for keys {keys}")
            result = np.array([], dtype=float)
        else:
            if target_ndim == 1:
                if unpack:
                    result = np.array([x for arr in arrays for x in arr], dtype=target_dtype)
                else:
                    result = np.array(arrays, dtype=target_dtype)
            else:  # target_ndim == 2
                if unpack:
                    result = np.concatenate(arrays, axis=0)
                else:
                    result = np.array(arrays, dtype=target_dtype)

        return (result, skipped) if return_skipped else result

    # ---------------------------------
    #! Data Processing Methods
    # ---------------------------------

    @staticmethod
    def clean_data_remove_zeros(
        matrix      : np.ndarray,
        axis        : int           = 0,
        tolerance   : float         = 1e-9,
        check_limit : Optional[int] = 10) -> np.ndarray:
        """
        Removes slices (e.g., rows or columns) from a matrix where initial elements are all close to zero.
        For 1D vector, removes elements close to zero from the beginning up to check_limit.

        Args:
            matrix:
                Input numpy array.
            axis:
                Axis along which to check for zero elements and remove slices.
            tolerance:
                Tolerance for considering an element as zero.
            check_limit:
                Max number of elements along the slice (or vector) to check.
                If None, checks all elements in the slice.

        Returns:
            Cleaned numpy array.
        """
        if matrix.ndim == 0: # Scalar
            return matrix 
        if matrix.size == 0: # Empty
            return matrix

        if matrix.ndim == 1:
            limit               = matrix.shape[0] if check_limit is None else min(check_limit, matrix.shape[0])
            elements_to_check   = matrix[:limit]
            if np.all(np.isclose(elements_to_check, 0.0, atol=tolerance)):
                if np.all(np.isclose(matrix, 0.0, atol=tolerance)): # Check all elements if initial ones are zero
                    return np.array([]) # Return empty if all are zero
            return matrix # Or return as is if not all (checked) are zero

        if axis < 0 or axis >= matrix.ndim:
            logging.error(f"Invalid axis {axis} for matrix with {matrix.ndim} dimensions.")
            return matrix

        # Move the target axis to the first position for easier processing
        matrix_moved            = np.moveaxis(matrix, axis, 0)
        num_slices              = matrix_moved.shape[0]
        elements_per_slice_dim  = tuple(range(1, matrix_moved.ndim)) # Dims other than the first (moved) one

        limit                   = None
        if check_limit is not None:
            # For multi-dim slices, check_limit applies to the first dimension of the slice.
            # This interpretation might need refinement based on exact intent.
            # Assuming check_limit applies to the first dimension of the slice itself.
            # Example: if axis=0 (rows), check_limit applies to columns of each row.
            # If matrix_moved.shape = (num_rows, C, D), slice shape is (C, D).
            # Check M_moved[:, :check_limit, ...] if C is the dimension to check.
            # The original code: M_moved[:, :check_limit] (implicitly checking columns up to check_limit)
            if matrix_moved.ndim > 1: # Slices are at least 1D
                limit = min(check_limit, matrix_moved.shape[1]) if matrix_moved.shape[1] > 0 else None


        valid_slice_indices = []
        for i in range(num_slices):
            current_slice = matrix_moved[i]
            slice_to_check = current_slice
            if limit is not None and current_slice.ndim > 0: # if slice is not scalar
                # Take the first 'limit' elements along the first dimension of the slice
                # Example: if slice is (C, D, E), slice_to_check is current_slice[:limit, :, :]
                idx             = [slice(None)] * current_slice.ndim
                idx[0]          = slice(0, limit)
                slice_to_check  = current_slice[tuple(idx)]


            if not np.all(np.isclose(slice_to_check, 0.0, atol=tolerance)):
                valid_slice_indices.append(i)
        
        if not valid_slice_indices: # All slices were "bad"
            final_shape = list(matrix.shape)
            final_shape[axis] = 0
            return np.empty(tuple(final_shape), dtype=matrix.dtype)

        matrix_filtered = matrix_moved[valid_slice_indices]
        return np.moveaxis(matrix_filtered, 0, axis)

    @staticmethod
    def clean_data_remove_thresholded(
        matrix          : np.ndarray,
        axis            : int = 0,
        threshold       : float = -1e4,
        check_limit     : Optional[int] = None) -> np.ndarray:
        """
        Removes slices from a matrix where initial elements are all below a threshold.
        Improved to handle any axis using np.moveaxis.

        Args:
            matrix:
                Input numpy array.
            axis:
                Axis along which to check and remove slices.
            threshold:
                Threshold value. Slices are removed if all checked elements are < threshold.
            check_limit:
                Max number of elements along the slice to check. If None, checks all.

        Returns:
            Cleaned numpy array.
        """
        if matrix.ndim == 0: return matrix
        if matrix.size == 0: return matrix
        if axis < 0 or axis >= matrix.ndim:
            logging.error(f"Invalid axis {axis} for matrix with {matrix.ndim} dimensions.")
            return matrix

        matrix_moved = np.moveaxis(matrix, axis, 0)
        
        num_slices = matrix_moved.shape[0]
        limit = None

        if check_limit is not None and matrix_moved.ndim > 1 and matrix_moved.shape[1] > 0:
            limit = min(check_limit, matrix_moved.shape[1])

        valid_slice_indices = []
        for i in range(num_slices):
            current_slice = matrix_moved[i]
            slice_to_check = current_slice

            if limit is not None and current_slice.ndim > 0:
                idx = [slice(None)] * current_slice.ndim
                idx[0] = slice(0, limit) # Check along the first dimension of the slice
                slice_to_check = current_slice[tuple(idx)]
            
            if not np.all(slice_to_check < threshold):
                valid_slice_indices.append(i)

        if not valid_slice_indices:
            final_shape = list(matrix.shape)
            final_shape[axis] = 0
            return np.empty(tuple(final_shape), dtype=matrix.dtype)
            
        matrix_filtered = matrix_moved[valid_slice_indices]
        return np.moveaxis(matrix_filtered, 0, axis)

    # ---------------------------------
    #! Batch Processing
    # ---------------------------------
    
    @staticmethod
    def process_file_content(
        source_directory        : str,
        source_filename         : str,
        key_map                 : Optional[Dict[str, str]] = None, # Maps old keys to new keys
        clean_zeros_axis        : Optional[int] = None,
        clean_values_axis       : Optional[int] = None,
        clean_check_limit       : int = 10,
        output_directory        : Optional[str] = None, # If None, overwrites source
        verbose                 : bool = False):
        """
        Loads an HDF5 file, optionally renames keys, cleans data, and saves it.
        (This was 'change_h5_bad' before)

        Args:
            source_directory:
                Directory of the source HDF5 file.
            source_filename:
                Filename of the source HDF5 file.
            key_map:
                Dictionary to rename dataset keys {old_key: new_key}.
            clean_zeros_axis:
                Axis for 'clean_data_remove_zeros'.
            clean_values_axis:
                Axis for 'clean_data_remove_thresholded'.
            clean_check_limit:  
                'check_limit' for cleaning functions.
            output_directory:
                Directory to save the processed file. If None, overwrites original.
            verbose:
                If True, log detailed information.
        """
        full_source_path = os.path.join(source_directory, source_filename)
        
        data = HDF5Manager.load_file_data(full_source_path, verbose=verbose)
        if not data or 'filename' not in data: # load_file_data returns empty dict on failure
            logging.error(f"Could not read or data is empty for {full_source_path}")
            return

        #! Remove 'filename' key before processing datasets
        original_filepath_in_data   = data.pop('filename', None)
        processed_data              = {}
        
        for current_key, dataset_array in data.items():
            # Apply cleaning
            if clean_zeros_axis is not None:
                dataset_array = HDF5Manager.clean_data_remove_zeros(dataset_array, axis=clean_zeros_axis, check_limit=clean_check_limit)
            if clean_values_axis is not None:
                dataset_array = HDF5Manager.clean_data_remove_thresholded(dataset_array, axis=clean_values_axis, check_limit=clean_check_limit)
            
            # Apply key mapping
            new_key                 = key_map.get(current_key, current_key) if key_map else current_key
            processed_data[new_key] = dataset_array

        if not processed_data:
            if verbose: _logger.info(f"No data left after processing {full_source_path}")
            # Decide if an empty HDF5 file should be saved or not.
            # Current: does not save if processed_data is empty.
            return

        target_dir      = output_directory if output_directory else source_directory
        target_filename = source_filename # Assumes filename remains the same

        if verbose:
            action = "Overwriting" if target_dir == source_directory else "Saving to"
            _logger.info(f"{action} {os.path.join(target_dir, target_filename)}")

        HDF5Manager.save_data_to_file(target_dir, target_filename, processed_data, overwrite=True)

    @staticmethod
    def batch_process_files_in_dirs(
        source_directories      : List[str],
        file_conditions         : Optional[List[Any]] = None,
        key_map                 : Optional[Dict[str, str]] = None,
        clean_zeros_axis        : Optional[int] = None,
        clean_values_axis       : Optional[int] = None,
        clean_check_limit       : int = 10,
        output_directory_base   : Optional[str] = None, # If set, processed files go to output_directory_base/original_subdir_structure
        is_test_run             : bool = False, # If true, appends "testrun" to output dir names
        verbose                 : bool = False,
        exception_handler       : Optional[Callable[[Exception, str], None]] = None
    ):
        """
        Processes multiple HDF5 files across directories.
        (This was 'change_h5_bad_dirs' before)
        """
        for current_source_dir in source_directories:
            try:
                if verbose:
                    _logger.info(f"Processing files in directory: {current_source_dir}")

                # Directories.listDirs with appendDir=False returns relative filenames
                relative_filenames = Directories.listDirs(
                    [current_source_dir], conditions=file_conditions or [], appendDir=False
                )

                if not relative_filenames:
                    if verbose:
                        _logger.info(f"No files matching conditions found in {current_source_dir}")
                    continue

                for rel_filename in relative_filenames:
                    target_output_dir = None
                    if output_directory_base:
                        # Recreate subdirectory structure if current_source_dir is nested
                        relative_dir_path = os.path.relpath(current_source_dir, start=min(source_directories, key=len)) #Simplistic base
                        if relative_dir_path == '.': relative_dir_path = ""
                        target_output_dir = os.path.join(output_directory_base, relative_dir_path)
                    elif is_test_run: # original behavior for testrun
                        target_output_dir = os.path.join(current_source_dir, "testrun")
                    # else: target_output_dir remains None, so process_file_content overwrites source.

                    if verbose:
                        _logger.info(f"Processing {os.path.join(current_source_dir, rel_filename)} -> "
                                    f"Output dir: {target_output_dir if target_output_dir else current_source_dir}")
                    
                    HDF5Manager.process_file_content(
                        current_source_dir,
                        rel_filename,
                        key_map=key_map,
                        clean_zeros_axis=clean_zeros_axis,
                        clean_values_axis=clean_values_axis,
                        clean_check_limit=clean_check_limit,
                        output_directory=target_output_dir,
                        verbose=verbose
                    )

            except Exception as e:
                logging.error(f"Error processing directory {current_source_dir}: {e}")
                if exception_handler:
                    exception_handler(e, f"Error processing directory {current_source_dir}")
                # else: print("Exception:", e) # Original behavior

    # --------------------------------
    #! Histogram / Data Series Combination Utilities
    # --------------------------------
    
    @staticmethod
    def average_histograms(
        y_arrays_list       : List[np.ndarray],
        x_arrays_list       : List[np.ndarray],
        filter_y_lt_one     : bool = False,
        use_interpolation   : bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combines and averages y-values (e.g., histogram counts) across multiple series,
        aligning them by their x-values (e.g., bin centers).

        Args:
            y_arrays_list: List of Y-value arrays.
            x_arrays_list: List of corresponding X-value arrays (bins).
            filter_y_lt_one: 
                If True, y-values < 1.0 (and corresponding x) are filtered out
                before averaging. (Original 'typical' parameter).
            use_interpolation: 
                - If True, interpolates Y-values onto a common X-grid.
                - If False, aggregates only at exact X-matches and appends unique X-bins.

        Returns:
            Tuple (y_combined_averaged, x_common_grid).
        """
        if not y_arrays_list or not x_arrays_list:
            raise ValueError("Input y_arrays_list and x_arrays_list cannot be empty.")
        if len(y_arrays_list) != len(x_arrays_list):
            raise ValueError("y_arrays_list and x_arrays_list must have the same length.")

        if len(x_arrays_list) == 1:
            y_curr, x_curr = y_arrays_list[0], x_arrays_list[0]
            if filter_y_lt_one:
                mask = y_curr >= 1.0
                return y_curr[mask], x_curr[mask]
            return y_curr, x_curr

        processed_series = []
        for y, x in zip(y_arrays_list, x_arrays_list):
            if filter_y_lt_one:
                mask = y >= 1.0
                processed_series.append((y[mask], x[mask]))
            else:
                processed_series.append((y, x))

        if use_interpolation:
            # Create a common, sorted, unique x-grid from all series
            all_x_values = np.concatenate([s[1] for s in processed_series])
            x_common_grid = np.sort(np.unique(all_x_values))
            
            sum_y_on_common_grid = np.zeros_like(x_common_grid, dtype=float)
            counts_on_common_grid = np.zeros_like(x_common_grid, dtype=int)

            for y_s, x_s in processed_series:
                if x_s.size == 0: continue # Skip empty series
                # Interpolate y_s onto x_common_grid
                y_interp = np.interp(x_common_grid, x_s, y_s, left=0, right=0) # Or np.nan and handle later
                sum_y_on_common_grid += y_interp
                # Count contributions: where x_common_grid values fall within the range of x_s
                # A simpler way is to count non-zero interpolated values if left/right are 0
                # Or, more accurately, for each point in x_common_grid, count how many original series could contribute
                min_x_s, max_x_s = np.min(x_s), np.max(x_s)
                counts_on_common_grid += ((x_common_grid >= min_x_s) & (x_common_grid <= max_x_s) & (y_interp != 0)) # Approximation
            
            # Avoid division by zero
            valid_counts = counts_on_common_grid > 0
            y_combined_averaged = np.zeros_like(x_common_grid, dtype=float)
            y_combined_averaged[valid_counts] = sum_y_on_common_grid[valid_counts] / counts_on_common_grid[valid_counts]
            
            return y_combined_averaged, x_common_grid

        else: # Non-interpolation method (original logic)
            # This part is complex to replicate exactly without deeper understanding of "divider".
            # The original non-interpolation part sums y-values for common bins and appends unique bins.
            # Averaging then divides by a 'divider' array that tracks contributions.
            # Let's simplify for now or stick to interpolation if possible, which is often more robust.
            # For a direct port of original non-interpolation:
            combined_dict = {} # x_value -> (sum_y, count)
            for y_s, x_s in processed_series:
                for i, x_val in enumerate(x_s):
                    if x_val not in combined_dict:
                        combined_dict[x_val] = [0.0, 0]
                    combined_dict[x_val][0] += y_s[i]
                    combined_dict[x_val][1] += 1
            
            sorted_x = sorted(combined_dict.keys())
            x_final = np.array(sorted_x)
            y_final = np.array([combined_dict[x][0] / combined_dict[x][1] for x in sorted_x])
            return y_final, x_final

    @staticmethod
    def align_and_fill_histograms(
        y_arrays_list   : List[np.ndarray], # List of lists of y-arrays if grouped by x_arrays
        x_arrays_list   : List[np.ndarray], # List of x-arrays
        group_lengths   : List[int],        # Number of y_arrays corresponding to each x_array
        fill_value      : float = np.nan) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Aligns multiple y-value series (histograms) to a common x-grid by interpolation,
        filling values for x-points not present in an original series.

        Args:
            y_arrays_list: 
                A list where each element can be a list of y-arrays (if multiple realizations share an x_array)
                or a single y-array. The structure should align with group_lengths.
                Example: [[y1_real1, y1_real2], [y2_real1]]
            x_arrays_list:
                List of x-value arrays (bins), one for each group of y_arrays.
                Example: [x1_bins, x2_bins]
            group_lengths:
                List indicating how many y-arrays in 'y_arrays_list' correspond to each x-array
                in 'x_arrays_list'. Example: [2, 1] means y_arrays_list[0] (a list of 2 y-arrays)
                uses x_arrays_list[0], and y_arrays_list[1] (a list of 1 y-array) uses x_arrays_list[1].
                If y_arrays_list elements are single y-arrays, then group_lengths would be [1, 1, ...].
            fill_value:
                Value used for points in the common x-grid that are outside an original series' x-range.

        Returns:
            Tuple (y_aligned_all, x_common_grid).
            y_aligned_all: List of 1D numpy arrays, each y-series interpolated to x_common_grid.
        """
        if not y_arrays_list or not x_arrays_list:
            raise ValueError("Input lists cannot be empty.")
        if len(group_lengths) != len(x_arrays_list): # Each x_array defines a group
            raise ValueError("group_lengths must match the number of x_arrays.")

        # Create a common, sorted, unique x-grid from all x_arrays_list
        all_x_values    = np.concatenate(x_arrays_list)
        x_common_grid   = np.sort(np.unique(all_x_values))

        y_aligned_all   = []
        y_list_flat_idx = 0 # To iterate through y_arrays_list elements correctly
        
        for group_idx, num_realizations_in_group in enumerate(group_lengths):
            current_x_array = x_arrays_list[group_idx]
            
            for _ in range(num_realizations_in_group):
                # y_arrays_list could be a list of lists or a flat list of y-arrays
                # Assuming y_arrays_list is structured such that each element corresponds to an x_array group,
                # and if num_realizations_in_group > 1, then y_arrays_list[group_idx] is a list of y-arrays.
                # If y_arrays_list is flat: y_current_realization = y_arrays_list[y_list_flat_idx]
                # If y_arrays_list is nested: y_current_realization = y_arrays_list[group_idx][real_idx_in_group]
                # The original had 'y = y_list[il][ii]', suggesting y_list was list of lists.
                # Let's assume y_arrays_list[group_idx] gives us the y-array or list of y-arrays for this group.

                y_data_for_group = y_arrays_list[group_idx] # This might be a single array or a list of arrays
                
                if num_realizations_in_group == 1 and isinstance(y_data_for_group, np.ndarray) and y_data_for_group.ndim == 1:
                    y_current_realization = y_data_for_group
                elif isinstance(y_data_for_group, list) and len(y_data_for_group) == num_realizations_in_group:
                    y_current_realization = y_data_for_group[_] # _ is realization index within group
                else: # Fallback or error for mismatched structure
                    # This part depends on exact structure of y_arrays_list.
                    # Original 'y = y_list[il][ii]' means y_list[il] is indexable (a list/array of y_arrays).
                    # For simplicity, let's adjust input expectations for y_arrays_list.
                    # Assume y_arrays_list is a FLAT list of all y_arrays.
                    if y_list_flat_idx >= len(y_arrays_list):
                        raise ValueError("Mismatch between group_lengths and total number of y_arrays.")
                    y_current_realization   = y_arrays_list[y_list_flat_idx]
                    y_list_flat_idx        += 1

                if current_x_array.size == 0: # Skip if x array is empty
                    y_aligned_all.append(np.full_like(x_common_grid, fill_value, dtype=float))
                    continue

                y_interp = np.interp(
                    x_common_grid, current_x_array, y_current_realization,
                    left=fill_value, right=fill_value
                )
                y_aligned_all.append(y_interp)
        
        return y_aligned_all, x_common_grid

# ----------------------------------------
#! END OF HDF5Manager CLASS