''' 
Generic data loader for HDF5 results.

--------------------------------------------------------------
This module provides functions to load and filter HDF5 result files
from a specified directory. It supports parsing parameters from filenames,
applying manual and generic filters, and returning a list of LazyHDF5Entry objects.
--------------------------------------------------------------

--------------------------------------------------------------
Author          : Maksymilian Kliczkowski
Date            : 2025-12-10
--------------------------------------------------------------
'''

from    pathlib import Path
from    typing  import List, Callable, TYPE_CHECKING
import  numpy   as np
import  re

if TYPE_CHECKING:
    from QES.general_python.common.flog     import Logger
    from QES.general_python.common.hdf5man  import LazyHDF5Entry

# --------------------------------------------------------------
# Data loading functions
# --------------------------------------------------------------

def parse_filename(filename):
    """Parse parameters from HDF5 filename."""
    params = {}
    pattern = r'(\w+)=([\d\.\-]+)'
    matches = re.findall(pattern, filename)
    for key, value in matches:
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    return params

def load_hdf5_data(filepath, keys=None, verbose=False):
    """Load all data from an HDF5 file."""
    
    try:
        from QES.general_python.common.hdf5man  import HDF5Manager
    except ImportError as e:
        raise ImportError("Required QES modules not found. Please ensure QES is installed and accessible.") from e
    
    data = HDF5Manager.read_hdf5(filepath, keys=keys, verbose=verbose)
    return data

def load_results(data_dir: str, *, 
                filters                     =   None, 
                lx                          =   None, 
                ly                          =   None, 
                lz                          =   None, 
                Ns                          =   None,                   # manual size filters
                post_process_func           =   None,                   # function to modify params dict in-place 
                get_params_func             =   lambda r: r.params,     # function to extract params from LazyHDF5Entry
                logger: 'Logger'            =   None,
                **kwargs
                ) -> List['LazyHDF5Entry']:
    r"""
    Load all HDF5 results from directory.
    
    It searches recursively for .h5 files in data_dir, extracts parameters from filenames,
    applies optional filters, and returns a list of LazyHDF5Entry objects.
    
    Parameters:
    -----------
    data_dir (str):
        Directory containing HDF5 result files.
    *
    filters (dict, optional):
        Dictionary of parameter filters to apply.    
    lx, ly, lz, Ns (int, optional):
        Manual size filters to apply.
    post_process_func (callable): 
        A function f(params) -> None that modifies the params dictionary in-place.
        Use this to inject derived parameters like theta/lambda.
    logger (Logger, optional):
        Logger for output messages.
    """
    
    try:
        from QES.general_python.common.hdf5man  import LazyHDF5Entry
    except ImportError as e:
        raise ImportError("Required QES modules not found. Please ensure QES is installed and accessible.") from e
    
    results     = []
    data_path   = Path(data_dir)
    
    if not data_path.exists():
        if logger:
            logger.error(f"Data directory {data_dir} does not exist.")
        return results

    h5_files        = list(data_path.glob('**/*.h5'))
    size_pattern    = re.compile(r'(Ns|Lx|Ly|Lz)=(\d+)')
    if logger:
        logger.info(f"Found {len(h5_files)} HDF5 files", color='cyan')

    for filepath in h5_files:
        try:
            # base params from filename
            # extract size info from any path component
            params      = parse_filename(filepath.name)
            size_info   = {}

            for part in filepath.parts:
                for match in size_pattern.finditer(part):
                    size_info[match.group(1)] = int(match.group(2))
                    
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

            results.append(LazyHDF5Entry(str(filepath), params))
        
        except Exception as e:
            if logger:
                logger.error(f"Error loading {filepath}: {e}")

    # generic filters
    if filters:
        try:
            from QES.general_python.common.plot import Plotter
        except ImportError as e:
            raise ImportError("Required QES modules not found. Please ensure QES is installed and accessible.") from e
        
        results = Plotter.filter_results(results, filters, get_params_fun=get_params_func, **kwargs)
        if not results and logger:
            logger.warning("No results match the specified filters.")
    
    if logger:    
        logger.info(f"Loaded {len(results)} results after filtering", color='green')
    return results

class PlotDataHelpers:
    """Helper functions for plotting."""
    
    # --------------------------------------------------------------
    # Parameter extraction
    # --------------------------------------------------------------
    
    @staticmethod
    def extract_parameter_arrays(filtered_results: List['LazyHDF5Entry'], x_param='J', y_param='hx', xlim=None, ylim=None):
        """
        Extract unique parameter values from filtered results. 
        Relies on load_results having already populated r.params (including derived ones).
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
    # Figure saving
    # --------------------------------------------------------------

    @staticmethod
    def savefig(fig, directory, param_name, x_param, y_param=None, suffix='', **kwargs):
        """Helper to save figure with consistent naming convention."""
        try:
            from QES.general_python.common.plot import Plotter
        except ImportError as e:
            raise ImportError("Required QES modules not found. Please ensure QES is installed and accessible.") from e
        
        if y_param:
            filename = f"{param_name}_vs_{x_param}_{y_param}{suffix}"
        else:
            filename = f"{param_name}_vs_{x_param}{suffix}"
        
        Plotter.save_fig(directory, filename, **kwargs)
    
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
        
        all_values  = np.array(all_values, dtype=float).flatten()
        vmin        = np.nanmin(all_values)
        vmax        = np.nanmax(all_values)
        return vmin, vmax

    # --------------------------------------------------------------
    # Grid plot setup
    # --------------------------------------------------------------
    
    @staticmethod
    def create_subplot_grid(n_panels, 
                            max_cols            =   3, 
                            figsize_per_panel   =   (6, 5), 
                            **kwargs):
        """
        Create a subplot grid with proper axis handling.
        
        Parameters:
        -----------
        n_panels (int):
            Total number of panels to create.
        max_cols (int):
            Maximum number of columns in the grid.
        figsize_per_panel (tuple):
            Figure size per panel (width, height).
        
        Returns:
        --------
        fig, axes, n_rows, n_cols
        """
        try:
            from QES.general_python.common.plot import Plotter
        except ImportError as e:
            raise ImportError("Required QES modules not found. Please ensure QES is installed and accessible.") from e
        
        n_cols      = min(max_cols, n_panels) if n_panels > 0 else 1
        n_rows      = (n_panels + n_cols - 1) // n_cols if n_panels > 0 else 1
        figsize     = (figsize_per_panel[0]*n_cols, figsize_per_panel[1]*n_rows)
        fig, axes   = Plotter.get_subplots(n_rows, n_cols, figsize=figsize, **kwargs)
        
        if n_rows > 1 and n_cols > 1:
            axes    = list(np.array(axes).reshape((n_rows, n_cols)))
        elif not isinstance(axes, (list, np.ndarray)):
            axes    = [axes]
            
        return fig, np.array(axes), n_rows, n_cols
    
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
        _, _, unique_x, unique_y    = PlotDataHelpers.extract_parameter_arrays(results, x_param, y_param)
        if x_values: unique_x       = [v for v in unique_x if any(abs(v - val) < 1e-5 for val in x_values)]
        if y_values: unique_y       = [v for v in unique_y if any(abs(v - val) < 1e-5 for val in y_values)]
        
        # Cols ascending, Rows descending (standard matrix layout)
        unique_x                    = np.sort(unique_x)
        unique_y                    = np.sort(unique_y)[::-1]
        n_cols, n_rows              = len(unique_x), len(unique_y)
        
        if n_cols == 0 or n_rows == 0:
            return None, None, []

        # Create Plot
        fig, axes, _, _ = PlotDataHelpers.create_subplot_grid(
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