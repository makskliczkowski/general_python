''' 
Generic data loader for HDF5 results.

This module provides functions to load and filter HDF5 result files
from a specified directory. It supports parsing parameters from filenames,
applying manual and generic filters, and returning a list of LazyHDF5Entry objects.

Important: We assume that HDF5 files are used to store results.
TODO: Add support for other file formats if needed.

--------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2025-12-10
--------------------------------------------------------------
'''

from    pathlib import Path
from    typing  import List, Callable, Union, TYPE_CHECKING
import  numpy   as np
import  re
import  json
import  pickle

if TYPE_CHECKING:
    from general_python.common.flog     import Logger
    from general_python.common.hdf5man  import LazyHDF5Entry

# --------------------------------------------------------------
# Lazy Entry Classes for other formats
# --------------------------------------------------------------

class LazyDataEntry:
    """Base class for lazy data entries."""
    def __init__(self, filepath, params):
        self.filepath   = filepath
        self.filename   = Path(filepath).name
        self.params     = params
        self._cache     = {}

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        self._load_item(key)
        return self._cache[key]

    def _load_item(self, key):
        # Default implementation loads everything
        self.load_all()

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key):
        if key in self._cache:
            return True
        return key in self.keys()

    def keys(self):
        if not self._cache:
            self.load_all()
        return self._cache.keys()

    def __len__(self):
        return len(self.keys())

    def load_all(self):
        raise NotImplementedError

class LazyNpzEntry(LazyDataEntry):
    """Lazy loader for .npz files."""
    def _load_item(self, key):
        try:
            with np.load(self.filepath) as data:
                if key in data:
                    self._cache[key] = data[key]
                else:
                    raise KeyError(f"Key '{key}' not found in {self.filename}")
        except Exception as e:
            raise KeyError(f"Error loading {key} from {self.filename}: {e}")

    def keys(self):
        # If cache is populated, return keys from there
        if self._cache:
            return self._cache.keys()
        # Otherwise peek into file
        try:
            with np.load(self.filepath) as data:
                return list(data.files)
        except Exception:
            return []

    def load_all(self):
        with np.load(self.filepath) as data:
            for k in data.files:
                self._cache[k] = data[k]
        return self

class LazyPickleEntry(LazyDataEntry):
    """Lazy loader for .pkl/.pickle files."""
    def load_all(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self._cache.update(data)
                else:
                    self._cache['default'] = data
        except Exception as e:
            # You might want to log this error
            pass
        return self

class LazyJsonEntry(LazyDataEntry):
    """Lazy loader for .json files."""
    def load_all(self):
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        # Try to convert lists to numpy arrays
                        try:
                            self._cache[k] = np.array(v)
                        except:
                            self._cache[k] = v
                else:
                    try:
                        self._cache['default'] = np.array(data)
                    except:
                        self._cache['default'] = data
        except Exception as e:
            pass
        return self

# --------------------------------------------------------------
# Data loading functions
# --------------------------------------------------------------

def parse_filename(filename):
    """Parse parameters from HDF5 filename."""
    params      = {}
    pattern     = r'(\w+)=([\d\.\-]+)'
    matches     = re.findall(pattern, filename)
    
    # Convert values to float if possible
    for key, value in matches:
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    return params

def load_hdf5_data(filepath, keys=None, verbose=False):
    """Load all data from an HDF5 file."""
    
    try:
        from general_python.common.hdf5man import HDF5Manager
    except ImportError as e:
        raise ImportError("Required general_python modules not found. Please ensure general_python is installed and accessible.") from e
    
    data = HDF5Manager.read_hdf5(filepath, keys=keys, verbose=verbose)
    return data

# --------------------------------------------------------------
# Main data loader
# --------------------------------------------------------------

def load_results(data_dir                   : str, 
                *, 
                filters                     =   None, 
                lx                          =   None, 
                ly                          =   None, 
                lz                          =   None, 
                Ns                          =   None,                   # manual size filters
                post_process_func           =   None,                   # function to modify params dict in-place 
                get_params_func             =   lambda r: r.params,     # function to extract params from LazyHDF5Entry
                # logging
                logger                      : 'Logger' =   None,
                **kwargs
                ) -> List[Union['LazyHDF5Entry', LazyDataEntry]]:
    r"""
    Load all results (HDF5, NPZ, Pickle, JSON) from directory.
    
    It searches recursively for supported files in data_dir, extracts parameters from filenames,
    applies optional filters, and returns a list of result objects.
    
    Parameters:
    -----------
    data_dir (str):
        Directory containing result files.
    *
    filters (dict, optional):
        Dictionary of parameter filters to apply.    
    lx, ly, lz, Ns (int, optional):
        Manual size filters to apply. These are optional and can be used to restrict
        results based on lattice dimensions or system size.
    get_params_func (callable):
        A function f(result) -> dict that extracts parameters from a result entry.
        Defaults to extracting the .params attribute.
    post_process_func (callable): 
        A function f(params) -> None that modifies the params dictionary in-place.
        Use this to inject derived parameters like theta/lambda.
    logger (Logger, optional):
        Logger for output messages.
    """
    
    # Try to import LazyHDF5Entry, but don't fail if strictly not needed (though it's usually needed for typing)
    try:
        from general_python.common.hdf5man  import LazyHDF5Entry
    except ImportError:
        LazyHDF5Entry = None
    
    results     = []
    data_path   = Path(data_dir)
    
    if not data_path.exists():
        if logger:
            logger.error(f"Data directory {data_dir} does not exist.", color='red')
        return results

    # Define supported extensions and their handlers
    # For HDF5 we delay instantiation until loop
    supported_extensions = {
        '.h5':      'hdf5',
        '.hdf5':    'hdf5',
        '.npz':     'npz',
        '.pkl':     'pickle',
        '.pickle':  'pickle',
        '.json':    'json'
    }

    found_files = []
    for ext in supported_extensions.keys():
        # globs are case sensitive usually, but here we assume standard lowercase extensions
        found_files.extend(data_path.glob(f'**/*{ext}'))

    # Remove duplicates if any (e.g. if globs overlap, though unlikely with distinct extensions)
    found_files = sorted(list(set(found_files)))

    size_pattern    = re.compile(r'(Ns|Lx|Ly|Lz)=(\d+)')    # e.g., Lx=4, Ns=16
    other_pattern   = re.compile(r'(\w+)=([\d\.\-]+)')      # e.g., J=1.0, hx=0.5
    
    if logger:
        logger.info(f"Found {len(found_files)} files", color='cyan')

    for filepath in found_files:
        try:
            # base params from filename
            # extract size info from any path component
            params      = parse_filename(filepath.name)
            size_info   = {}

            for part in filepath.parts:
                for match in size_pattern.finditer(part):
                    size_info[match.group(1)] = int(match.group(2))
                for match in other_pattern.finditer(part):
                    key = match.group(1)
                    try:
                        value = float(match.group(2))
                    except ValueError:
                        value = match.group(2)
                    params.setdefault(key, value)
                    
            # merge (do not overwrite if already present)
            for k, v in size_info.items():
                params.setdefault(k, v)

            # manual filters (if any)
            if lx is not None and size_info.get('Lx') != lx:
                continue
            if ly is not None and size_info.get('Ly') != ly:
                continue
            if lz is not None and size_info.get('Lz') != lz:
                continue
            if Ns is not None and size_info.get('Ns') != Ns:
                continue
            
            # Apply user-defined post-processing (e.g. compute theta/lambda)
            if post_process_func:
                post_process_func(params)

            # Instantiate appropriate entry
            suffix = filepath.suffix.lower()
            ftype  = supported_extensions.get(suffix)

            entry = None
            if ftype == 'hdf5':
                if LazyHDF5Entry is None:
                     raise ImportError("LazyHDF5Entry not found. Ensure general_python is installed properly.")
                entry = LazyHDF5Entry(str(filepath), params)
            elif ftype == 'npz':
                entry = LazyNpzEntry(str(filepath), params)
            elif ftype == 'pickle':
                entry = LazyPickleEntry(str(filepath), params)
            elif ftype == 'json':
                entry = LazyJsonEntry(str(filepath), params)

            if entry:
                results.append(entry)
        
        except Exception as e:
            if logger:
                logger.error(f"Error loading {filepath}: {e}", color='red')

    # generic filters
    if filters:
        try:
            from general_python.common.plot import Plotter
        except ImportError as e:
            raise ImportError("Required general_python modules not found. Please ensure general_python is installed and accessible.") from e
        
        results = Plotter.filter_results(results, filters, get_params_fun=get_params_func, **kwargs)
        if not results and logger:
            logger.warning("No results match the specified filters.")
    
    if logger:    
        logger.info(f"Loaded {len(results)} results after filtering", color='green')
    return results

# --------------------------------------------------------------
# Data transformation utilities
# --------------------------------------------------------------

class ResultProxy:
    """
    Lightweight proxy object that mimics LazyHDF5Entry interface.
    
    Allows direct data arrays to be used with plotting functions
    that expect LazyHDF5Entry objects.
    
    Parameters
    ----------
    data : dict or np.ndarray
        Data payload. If dict, should contain dataset keys.
        If array, will be stored under 'default' key.
    params : dict
        Parameter dictionary (e.g., {'J': 1.0, 'hx': 0.5, 'Lx': 4})
    filepath : str, optional
        Mock filepath for compatibility
        
    Attributes
    ----------
    params : dict
        Parameter dictionary
    data : dict
        Data dictionary
    filepath : str
        Mock filepath
        
    Examples
    --------
    >>> # From numpy array
    >>> akw     = np.random.rand(100, 50)  # (Nk, Nw)
    >>> result  = ResultProxy(akw, params={'J': 1.0, 'hx': 0.5})
    >>> print(result.params['J'])  # 1.0
    >>> print(result['default'].shape)  # (100, 50)
    
    >>> # From dict
    >>> data    = {'/spectral/akw': akw, '/spectral/omega': omega}
    >>> result  = ResultProxy(data, params={'J': 1.0})
    >>> print(result['/spectral/akw'].shape)  # (100, 50)
    """
    
    def __init__(self, data: Union[np.ndarray, dict], params: dict, filepath: str = '<memory>'):
        self.params     = params
        self.filepath   = filepath
        
        if isinstance(data, dict):
            self.data   = data
        elif isinstance(data, np.ndarray):
            self.data   = {'default': data}
        else:
            raise TypeError(f"Data must be dict or ndarray, got {type(data)}")
    
    def __getitem__(self, key):
        """Access data like LazyHDF5Entry: result['/path/to/data']."""
        return self.data.get(key, None)
    
    def keys(self):
        """Return available data keys."""
        return self.data.keys()
    
    def get(self, key, default=None):
        """Get data with default fallback."""
        return self.data.get(key, default)
    
    def __repr__(self):
        return f"ResultProxy(params={self.params}, keys={list(self.data.keys())})"
        
def create_result_from_data(
    data                : Union[np.ndarray, dict],
    params              : dict,
    filepath            : str = '<memory>'
) -> ResultProxy:
    """
    Create a result proxy object from raw data.
    
    Convenience wrapper around ResultProxy for creating single result objects.
    
    Parameters
    ----------
    data : np.ndarray or dict
        Data payload. Can be:
        - Single array   (e.g., spectral function)
        - Dict of arrays (e.g., {'/spectral/akw': akw, '/spectral/omega': omega})
    params : dict
        Parameter dictionary defining this data point
    filepath : str, optional
        Mock filepath for identification
        
    Returns
    -------
    ResultProxy
        Result object compatible with plotting functions
        
    Examples
    --------
    >>> akw = np.random.rand(100, 50)
    >>> result = create_result_from_data(
    ...     data=akw,
    ...     params={'J': 1.0, 'hx': 0.5, 'Lx': 4, 'Ly': 4}
    ... )
    >>> # Can now use with plotting functions expecting LazyHDF5Entry
    """
    return ResultProxy(data=data, params=params, filepath=filepath)

def prepare_results_for_plotting(
    data_array          : np.ndarray,
    x_param_values      : list,
    y_param_values      : list,
    *,
    x_param             : str = 'J',
    y_param             : str = 'hx',
    data_key            : str = 'default',
    fixed_params        : dict = None,
    post_process_func   : callable = None
) -> List[ResultProxy]:
    r"""
    Transform parameter-swept data arrays into result objects.
    
    Converts a grid of data (e.g., from parameter sweeps) into a list
    of ResultProxy objects that can be used with plotting functions.
    
    Parameters
    ----------
    data_array : np.ndarray
        Data array with shape matching parameter grid.
        Can be:
        - 2D: (n_x, n_y) for scalar values
        - 3D: (n_x, n_y, N) for vector/state data  
        - 4D: (n_x, n_y, Nk, Nw) for things like spectral functions
    x_param_values : list
        Values of x parameter (length n_x)
    y_param_values : list
        Values of y parameter (length n_y)
    x_param : str
        Name of x parameter
    y_param : str
        Name of y parameter
    data_key : str
        Key under which to store data in ResultProxy
    fixed_params : dict, optional
        Additional fixed parameters (e.g., {'Lx': 4, 'Ly': 4})
    post_process_func : callable, optional
        Function to modify params dict in-place for each result
        
    Returns
    -------
    List[ResultProxy]
        List of result objects, one per (x, y) grid point
        
    Examples
    --------
    >>> # Spectral function grid: shape (5, 3, 100, 50) = (n_J, n_hx, Nk, Nw)
    >>> akw_grid = np.random.rand(5, 3, 100, 50)
    >>> results = prepare_results_for_plotting(
    ...     data_array=akw_grid,
    ...     x_param_values=[0.5, 1.0, 1.5, 2.0, 2.5],
    ...     y_param_values=[0.0, 0.5, 1.0],
    ...     x_param='J',
    ...     y_param='hx',
    ...     data_key='/spectral/akw',
    ...     fixed_params={'Lx': 4, 'Ly': 4}
    ... )
    >>> print(len(results))  # 15 (5 * 3)
    >>> print(results[0].params)  # {'J': 0.5, 'hx': 0.0, 'Lx': 4, 'Ly': 4}
    """
    if fixed_params is None:
        fixed_params = {}
    
    results = []
    
    for ii, x_val in enumerate(x_param_values):
        for jj, y_val in enumerate(y_param_values):
            
            # Build params dict
            params  = {
                x_param : x_val,
                y_param : y_val,
                **fixed_params
            }
            
            # Apply post-processing if provided
            if post_process_func:
                post_process_func(params)
            
            # Extract data for this parameter point
            if data_array.ndim == 2:
                data_slice = data_array[ii, jj]
            elif data_array.ndim == 3:
                data_slice = data_array[ii, jj, :]
            elif data_array.ndim == 4:
                data_slice = data_array[ii, jj, :, :]
            else:
                data_slice = data_array[ii, jj, ...]
            
            # Create result object
            result = ResultProxy(
                data        =   {data_key: data_slice},
                params      =   params,
                filepath    =   f'<memory:{x_param}={x_val}_{y_param}={y_val}>'
            )
            results.append(result)
    
    return results

# ==============================================================
# Plotting helper functions
# ==============================================================

class PlotData:
    """Helper functions for plotting."""
    
    @staticmethod
    def from_input(
        directory       : str,
        data_values     : Union[np.ndarray, dict]   = None,
        x_parameters    : List[float]               = None,
        y_parameters    : List[float]               = None,
        *,
        x_param         : str                       = 'x',
        y_param         : str                       = 'y',
        data_key        : str                       = None,
        filters         : dict                      = None,
        # size
        logger          : 'Logger'                  = None,
        **kwargs
    ) -> List[Union['LazyHDF5Entry', 'ResultProxy', 'LazyDataEntry']]:
        """
        Unified result preparation from directory or direct data.
        
        Handles three input modes:
        1. directory != None: 
            Load from HDF5 files
        2. data_values is dict: 
            Already results
        3. data_values is array: 
            Transform to results
        
        Parameters
        ----------
        directory : str, optional
            Data directory for loading HDF5 files
        data_values : array/dict/list, optional
            Direct data input
        x_parameters, y_parameters : list
            Parameter grid values
        x_param, y_param : str
            Parameter names
        data_key : str, optional
            Key for storing array data in ResultProxy
        filters : optional
            Filters for HDF5 loading
        logger : optional
            Logger instance
        **kwargs
            Additional arguments for loading/processing.
            Can include Lx, Ly, Lz, Ns, post_process_func, 
            get_params_func, etc.
        
        Returns
        -------
        List[ResultProxy or LazyHDF5Entry]
            Results ready for plotting
        """
        
        # Mode 1: Load from directory
        if directory is not None and data_values is None:
            results = load_results(
                data_dir            =   directory,
                filters             =   filters,
                lx                  =   kwargs.pop('Lx', kwargs.pop('lx', None)),
                ly                  =   kwargs.pop('Ly', kwargs.pop('ly', None)),
                Ns                  =   kwargs.pop('Ns', kwargs.pop('ns', None)),
                post_process_func   =   kwargs.get('post_process_func', None),
                get_params_func     =   kwargs.get('get_params_func', None),
                logger              =   logger,
                **kwargs
            )
            return results
        
        # Mode 2: Data already as results
        if isinstance(data_values, (list, tuple)):
            # Assume list of result-like objects
            return list(data_values)
        
        if isinstance(data_values, dict):
            
            # Check if it's a single result dict or dict of results
            if 'params' in data_values or isinstance(next(iter(data_values.values()), None), (ResultProxy, dict)):
                
                # It's already results
                if isinstance(data_values, dict) and 'params' not in data_values:
                    # Dict mapping to results
                    return list(data_values.values())
                else:
                    # Single result
                    return [data_values]
            else:
                
                # It's raw data dict, create single result
                fixed_params = {k: v for k, v in kwargs.items() if k in ['Lx', 'Ly', 'Lz', 'Ns']}
                
                if len(x_parameters) == 1 and len(y_parameters) == 1:
                    params = {
                        x_param: x_parameters[0],
                        y_param: y_parameters[0],
                        **fixed_params
                    }
                    return [create_result_from_data(data_values, params)]
                else:
                    raise ValueError(
                        "For multi-parameter grid with dict data, use prepare_results_for_plotting explicitly"
                    )
        
        # Mode 3: Transform numpy array
        if isinstance(data_values, np.ndarray):
            if data_key is None:
                data_key = 'default'
            
            fixed_params    = {k: v for k, v in kwargs.items() if k in ['Lx', 'Ly', 'Lz', 'Ns']}
            results         = prepare_results_for_plotting(
                                data_array          =   data_values,
                                x_param_values      =   x_parameters,
                                y_param_values      =   y_parameters,
                                x_param             =   x_param,
                                y_param             =   y_param,
                                data_key            =   data_key,
                                fixed_params        =   fixed_params,
                                post_process_func   =   kwargs.get('post_process_func', None)
                            )
            return results
        
        raise ValueError(
            f"Cannot prepare results from data_values of type {type(data_values)}. "
            "Expected: directory path, list of results, dict, or numpy array."
        )
    
    @staticmethod
    def from_match(
        results         : List['LazyHDF5Entry'],
        x_param         : str,
        y_param         : str,
        x_val           : float,
        y_val           : float,
        tol             : float = 1e-5
    ) -> List['LazyHDF5Entry']:
        """
        Find result matching parameter values within tolerance.
        
        Parameters
        ----------
        results : List[ResultProxy or LazyHDF5Entry]
            List of result objects
        x_param, y_param : str
            Parameter names to match
        x_val, y_val : float
            Target parameter values
        tolerance : float
            Matching tolerance
            
        Returns
        -------
        result or None
            First matching result, or None if not found
        """
        for r in results:
            rx = r.params.get(x_param, np.nan)
            ry = r.params.get(y_param, np.nan)
            
            if abs(rx - x_val) < tol and abs(ry - y_val) < tol:
                return r
        
        return None

    # --------------------------------------------------------------
    # Parameter extraction
    # --------------------------------------------------------------
    
    @staticmethod
    def extract_parameter_arrays(filtered_results: List['LazyHDF5Entry'], x_param='J', y_param='hx', xlim=None, ylim=None):
        """
        Extract unique parameter values from filtered results. 
        Relies on load_results having already populated r.params (including derived ones). 
        This is useful for setting up axes in parameter space plots.
        
        Examples
        --------
        >>> x_vals, y_vals, unique_x, unique_y = PlotData.extract_parameter_arrays(
        ...     filtered_results,
        ...     x_param='J',
        ...     y_param='hx',
        ...     xlim=(0.5, 2.5),
        ...     ylim=(0.0, 1.0)
        ... )
        >>> print(unique_x)  # e.g., [0.5, 1.0, 1.5, 2.0, 2.5]
        >>> print(unique_y)  # e.g., [0.0, 0.5, 1.0]
        """
        x_vals = []
        y_vals = []
        
        for r in filtered_results:
            x_vals.append(r.params.get(x_param, np.nan))
            y_vals.append(r.params.get(y_param, np.nan))
        
        x_vals      = np.array(x_vals)
        y_vals      = np.array(y_vals)
        
        # Filter NaNs (often defaults for missing params)
        unique_x    = np.unique(x_vals[~np.isnan(x_vals)])
        unique_y    = np.unique(y_vals[~np.isnan(y_vals)])
        
        if xlim is not None:
            unique_x = unique_x[(unique_x >= xlim[0] if xlim[0] is not None else True) & (unique_x <= xlim[1] if xlim[1] is not None else True)]
        if ylim is not None:
            unique_y = unique_y[(unique_y >= ylim[0] if ylim[0] is not None else True) & (unique_y <= ylim[1] if ylim[1] is not None else True)]
            
        return x_vals, y_vals, unique_x, unique_y
    
    @staticmethod
    def sort_results_by_param(results: List['LazyHDF5Entry'], param_name: str):
        """
        Sort results list by a specific parameter.
        Returns:
            sorted_values (np.ndarray)
            sort_indices (np.ndarray)
        """
        vals = []
        for r in results:
            vals.append(r.params.get(param_name, np.nan))
        vals        = np.array(vals)
        sort_idx    = np.argsort(vals)
        return vals[sort_idx], sort_idx

    # --------------------------------------------------------------
    # vmin/vmax determination
    # --------------------------------------------------------------

    @staticmethod
    def determine_vmax_vmin(results: List['LazyHDF5Entry'], param_name: str, param_fun: Callable = lambda r, name: r.params[name], nstates: int = None):
        """
        Determine global vmin and vmax across all results for specified parameter.
        
        Parameters:
        -----------
        results (List[LazyHDF5Entry]):
            List of loaded results.
        param_name (str):
            Name of the parameter to extract.
        param_fun (Callable):
            Function to extract parameter values from a result. Defaults to lambda r, name: r.params[name].
            
        """
        all_values = []
        for r in results:
            try:
                values = param_fun(r, param_name)
                if isinstance(values, (list, np.ndarray)):
                    vals = np.array(values[:nstates]).flatten() if nstates is not None else np.array(values).flatten()
                    all_values.extend(vals.tolist())
                else:
                    all_values.append(values)
            except Exception:
                pass

        if not all_values:
            return np.nan, np.nan
        
        all_values  = np.array(np.real(all_values), dtype=float).flatten()
        vmin        = np.nanmin(all_values)
        vmax        = np.nanmax(all_values)
        return vmin, vmax

    # --------------------------------------------------------------
    # Grid plot setup
    # --------------------------------------------------------------
    
    @staticmethod
    def create_parameter_grid(
        results             : List['LazyHDF5Entry'],
        x_param             : str,
        y_param             : str,
        *,
        param_fun           : Callable  = lambda r, name: r.params[name],
        x_values            : list      = None,
        y_values            : list      = None,
        figsize_per_panel   : tuple     = (4, 3.5),
        sharex              : bool      = True,
        sharey              : bool      = True,
        **kwargs) -> tuple:
        """
        Generic setup for grid plots based on two varying parameters.
        Returns: fig, axes_grid, grid_iterator
        
        grid_iterator yields: 
            (ax, subset_results, x_val, y_val, row_idx, col_idx)
            
        Parameters:
        -----------
        results (List[LazyHDF5Entry]):
            List of loaded results.
        x_param (str):
            Parameter name for columns.
        y_param (str):
            Parameter name for rows.
        param_fun (Callable):
            Function to extract parameter values from a result. Defaults to lambda r, name: r.params[name].
        x_values (list, optional):
            Specific x values to include.
        y_values (list, optional):
            Specific y values to include.
        figsize_per_panel (tuple):
            Figure size per panel (width, height).
        sharex (bool):
            Whether to share x axes.
        sharey (bool):
            Whether to share y axes.
        **kwargs:
            Additional arguments for subplot creation.
        """
        
        # Extract and Sort Parameters
        _, _, unique_x, unique_y    = PlotData.extract_parameter_arrays(results, x_param, y_param)
        if x_values: unique_x       = [v for v in unique_x if any(abs(v - val) < 1e-5 for val in x_values)]
        if y_values: unique_y       = [v for v in unique_y if any(abs(v - val) < 1e-5 for val in y_values)]
        
        # Cols ascending, Rows descending (standard matrix layout)
        unique_x                    = np.sort(unique_x)
        unique_y                    = np.sort(unique_y)[::-1]
        n_cols, n_rows              = len(unique_x), len(unique_y)
        
        if n_cols == 0 or n_rows == 0:
            return None, None, []

        # Create Plot
        fig, axes, _, _ = FigureConfig.create_subplot_grid(
                            n_panels            =   n_rows * n_cols, 
                            max_cols            =   n_cols, 
                            figsize_per_panel   =   figsize_per_panel, 
                            sharex              =   sharex, 
                            sharey              =   sharey,
                            **kwargs
                        )
        
        # Enable constrained layout if available, this helps with spacing
        if hasattr(fig, 'set_constrained_layout'):
            fig.set_constrained_layout(True)
            
        axes_grid = np.array(axes).reshape((n_rows, n_cols))
        
        # Define Iterator
        def grid_iterator():
            for ii, y_val in enumerate(unique_y):
                for jj, x_val in enumerate(unique_x):
                    # Filter results for this cell
                    ax      = axes_grid[ii, jj]
                    subset  = []
                    
                    for r in results:
                        # Relies on params already being populated
                        rx = param_fun(r, x_param) if callable(param_fun) else r.params.get(x_param, np.nan)
                        ry = param_fun(r, y_param) if callable(param_fun) else r.params.get(y_param, np.nan)
                        
                        if abs(rx - x_val) < 1e-5 and abs(ry - y_val) < 1e-5:
                            subset.append(r)
                            
                    yield ax, subset, x_val, y_val, ii, jj

        return fig, axes_grid, grid_iterator()

    # --------------------------------------------------------------




# ------------------------------------------------------------------
#! End of file
# ------------------------------------------------------------------