'''
This module contains common plotting utilities for Quantum EigenSolver.

Modular architecture for ED plotting with support for:
- Real-space and k-space correlation functions
- Dynamical spectral functions A(k,w) and S(k,w)
- Static structure factors S(k,n) for eigenstates
- Discrete k-point handling with path or grid modes
- High-symmetry path extraction with BZ extension
- Flexible visualization options

Key Features
------------
1. **Spectral Functions**
    - plot_spectral_function_2d: 
        Universal 2D spectral plotter (k,w) or (k,n)
    - plot_static_structure_factor: 
        Eigenstate-resolved structure factors
    - Supports path mode (band structure) or full grid mode
    - Optional BZ extension to show multiple copies

2. **K-Space Utilities**
    - select_kpoints_along_path: 
        Extract k-points along high-symmetry paths
    - extend_kspace_data: 
        Replicate data across BZ copies for visualization
    - Tolerance-based path matching for discrete k-grids

3. **Correlation Functions**
    - plot_correlation_grid: 
        Multi-mode correlation visualization (matrix, lattice, kspace, kpath)
    - plot_bz_path_from_corr: 
        Structure factor along paths
    - Automatic S(k) computation from real-space correlations

4. **Parameter Studies**
    - plot_phase_diagram_states: 
        Eigenstate properties across parameter space
    - plot_multistate_vs_param: 
        Multi-state line plots
    - plot_scaling_analysis: 
        System size scaling

Usage Patterns
--------------
All high-level plotters support:
- Path mode: 
    path_labels=['Gamma', 'K', 'M'] with use_extend=True
- Grid mode: 
    Full k-space with optional BZ extension
- Configurable styling via PlotStyle and KSpaceConfig dataclasses

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-01-15
License             : MIT
----------------------------------------------------------------
'''

from    typing              import Dict, List, Optional, Tuple, Union, Literal, Any, TYPE_CHECKING
from    scipy.interpolate   import griddata
from    scipy.spatial       import cKDTree

import  numpy               as np
import  matplotlib.pyplot   as plt

try:
    from .data_loader                       import load_results, prepare_results_for_plotting, create_result_from_data
    from .config                            import PlotStyle, KSpaceConfig, KPathConfig, SpectralConfig, FigureConfig
    from .kspace_utils                      import (
                                                    point_to_segment_distance_2d,
                                                    select_kpoints_along_path,
                                                    compute_structure_factor_from_corr,
                                                    label_high_sym_points,
                                                    format_pi_ticks
                                                )
    from .spectral_utils                    import compute_spectral_broadening, extract_spectral_data
    from .plot_helpers                      import plot_static_structure_factor, plot_kspace_intensity
    from .data_loader                       import PlotData
except ImportError:
    raise ImportError("Failed to import required modules from the current package.")

if TYPE_CHECKING:
    from general_python.lattices.lattice    import Lattice
    from general_python.common.flog         import Logger

# ==============================================================================
# Correlation function plotter along BZ path

def plot_bz_path_from_corr(
        ax,
        lattice         : "Lattice",
        corr_matrix     : np.ndarray,
        *,
        path                                    = None,
        points_per_seg  : int                   = None,
        value_label     : str                   = r"$S(\mathbf{k})$",
        line_kw         : dict                  = None,
        hsline_kw       : dict                  = None,
        print_vectors   : bool                  = False,
        kpath_config    : Optional[KPathConfig] = None,
        style           : Optional[PlotStyle]   = None,
    ):
    r"""
    Plot correlation-derived structure factor along high-symmetry path.
        
    Computes (site-based) structure factor:
        S(k) = (1/Ns) sum_{i,j} C_{ij} exp[-i k . (r_i - r_j)]
    
    Uses modular helper select_kpoints_along_path for path extraction.
    
    Parameters
    ----------
    ax : matplotlib Axes
    lattice : Lattice
        Must have .rvectors, .kvectors, .high_symmetry_points()
    corr_matrix : (Ns,Ns) array
        Correlation matrix in site basis
    path : optional
        Path specification (None = lattice default, or list like ['Gamma','K','M'])
    kpath_config : KPathConfig, optional
        Configuration object (overrides individual parameters)
    style : PlotStyle, optional
        Styling configuration
    
    Returns
    -------
    result : dict
        Path data with k_dist, values, label_positions, label_texts
    """
    
    # Setup configuration
    if kpath_config is None:
        kpath_config = KPathConfig(
                        path            =   path,
                        points_per_seg  =   points_per_seg
                    )
    if style is None:
        style       = PlotStyle()
    
    # Override config with explicit parameters if provided
    if path is not None:
        kpath_config.path           = path
    if points_per_seg is not None:
        kpath_config.points_per_seg = points_per_seg
    
    # Set up default styling - emphasize discrete points
    if line_kw is None:
        line_kw = {
                    "lw"            : style.linewidth,
                    "ls"            : "-",
                    "color"         : "C0",
                    "marker"        : style.marker,
                    "ms"            : style.markersize,
                    "mfc"           : "C0",
                    "mec"           : "white",
                    "mew"           : 0.5,
                    "alpha"         : style.alpha
                }
    
    if hsline_kw is None:
        hsline_kw   = kpath_config.separator_style
    
    C = np.asarray(corr_matrix, float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("corr_matrix must be square (Ns,Ns).")

    r_cart          = np.asarray(lattice.rvectors, float)
    k_cart          = np.asarray(lattice.kvectors, float)
    Ns              = C.shape[0]
    
    if r_cart.shape[0] != Ns:
        raise ValueError(f"Ns mismatch: corr {Ns}, lattice {r_cart.shape[0]}")

    # Compute S(k) at all k-points
    Sk              = compute_structure_factor_from_corr(C, r_cart, k_cart, normalize=True)

    # Use modular path selection helper
    k_spacing       = np.median(np.diff(np.sort(k_cart[:, 0])))
    tolerance       = kpath_config.tolerance if kpath_config.tolerance else k_spacing * 0.5
    path_labels     = kpath_config.path
    
    path_result = select_kpoints_along_path(
        lattice         =   lattice,
        k_vectors       =   k_cart,
        path_labels     =   path_labels,
        tolerance       =   tolerance,
        use_extend      =   False
    )
    
    if print_vectors:
        print("\n=== DEBUG: BZ path correlation plotting ===")
        print(f"Number of sites: {r_cart.shape[0]}")
        print(f"Number of k-points: {k_cart.shape[0]}")
        print(f"Path: {path_labels}")
        print(f"Selected k-points: {len(path_result['indices'])}")
        print("=" * 40 + "\n")
    
    # Extract values at selected k-points
    selected_indices    = path_result['indices']
    k_distances         = path_result['distances']
    label_positions     = path_result['label_positions']
    label_texts         = path_result['label_texts']
    
    if len(selected_indices) == 0:
        raise ValueError("No k-points found along path. Try increasing tolerance or checking path definition.")
    
    y = Sk[selected_indices]
    x = k_distances

    # Plot discrete k-points with markers
    ax.plot(x, y, **line_kw)
    ax.set_ylabel(value_label, fontsize=style.fontsize_label)
    ax.set_xlim(x.min(), x.max())

    # Add high-symmetry separators + ticks
    if kpath_config.show_separators:
        for xv in label_positions:
            ax.axvline(xv, **hsline_kw)
        ax.set_xticks(label_positions)
        ax.set_xticklabels(label_texts)

    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=style.fontsize_tick)
    
    # Return result with compatibility
    result = {
        'k_dist'            : x,
        'values'            : y,
        'label_positions'   : label_positions,
        'label_texts'       : label_texts
    }
    
    return result

# ==============================================================================
# Spectral function plotter
# ==============================================================================

def plot_spectral_function(
        directory               : str,
        param_name              : str,
        lattice                 : "Lattice",
        x_parameters            : list,
        y_parameters            : list,
        *,
        omega_key               : str                                   = '/spectral/omega',
        kvectors_key            : str                                   = '/spectral/kvectors',
        data_values             : Optional[np.ndarray | Dict[str, Any]] = None,
        x_param                 : str                                   = 'J',
        y_param                 : str                                   = 'hx',
        # plotting mode
        filters                 = None,
        state_idx               : int                                   = 0,
        mode                    : Literal['sum', 'kpath', 'grid', 'single'] = "sum",
        param_labels            : dict                                  = {},
        # k-path specific
        path_labels             : Optional[List[str]]                   = None,
        k_indices               : Optional[List[int]]                   = None,
        use_extend              : bool = False,
        extend_copies           : int = 2,
        # other
        spectral_config         : Optional[SpectralConfig]              = None,
        style                   : Optional[PlotStyle]                   = None,
        letter_x                : float                                 = 0.05,
        letter_y                : float                                 = 0.9,
        annotate_letter         : bool                                  = True,
        title                   : str                                   = '',
        **kwargs
    ):
    """
    Plot spectral function A(k,w) or S(k,w) from ED results.
    
    Universal plotter supporting multiple visualization modes:
    - 'sum'   : Sum over all k-points -> A(w) line plot         - useful for DOS
        needs the data to be at least (Nk, Nw)
    - 'kpath' : A(k_path, w) heatmap along high-symmetry path   - band structure style
        needs the data to be at least (Nk, Nw)
    - 'grid'  : Full k-space grid with optional BZ extension    - 2D intensity map
        needs the data to be at least (Nk, Nw)
    - 'single': A(k_i, w) line plots for specific k-points      - discrete point analysis
        needs the data to be at least (Nk, Nw)
    
    Parameters
    ----------
    directory : str
        Data directory
    param_name : str
        Spectral data key (e.g., '/spectral/akw', '/spectral/skw')
    lattice : Lattice
        Lattice object
    x_parameters, y_parameters : list
        Parameter grid values - will create subplots for each (x,y) pair
    x_param, y_param : str
        Parameter names - used to match results
    mode : str
        Visualization mode ('sum', 'kpath', 'grid', 'single')
    path_labels : list of str, optional
        High-symmetry path for mode='kpath' (e.g., ['Gamma', 'K', 'M', 'Gamma']).
        Depends on lattice high-symmetry points. See Lattice.high_symmetry_points().
    k_indices : list of int, optional
        K-point indices for mode='single'
    use_extend : bool
        Extend k-space to show multiple BZ copies (for kpath/grid modes)
    extend_copies : int
        Number of BZ copies in each direction
    spectral_config : SpectralConfig, optional
        Spectral function configuration
    style : PlotStyle, optional
        Styling configuration
    
    Returns
    -------
    fig, axes_grid : Figure and axes array
    
    Examples
    --------
    # Sum over k (or discrete points)
    >>> fig, axes = plot_spectral_function(
    ...     directory       = './data', 
    ...     param_name      = '/akw', 
    ...     lattice         = lat,
    ...     x_parameters    = [1.0], 
    ...     y_parameters    = [0.0, 0.5],
    ...     mode            = 'sum'
    ... )
    
    # K-path with extended BZ
    >>> fig, axes = plot_spectral_function(
    ...     directory       = './data', param_name='/akw', lattice=lat,
    ...     x_parameters    =   [1.0], 
    ...     y_parameters    =   [0.0],
    ...     mode            =   'kpath',
    ...     path_labels     =   ['Gamma', 'K', 'M', 'Gamma'],
    ...     use_extend      =   True, extend_copies =   2
    ... )
    """
    
    try:
        from ..plot     import Plotter
    except ImportError:
        raise ImportError("Failed to import Plotter from the current package.")
    
    spectral_config     = SpectralConfig()  if spectral_config is None else spectral_config
    style               = PlotStyle()       if style is None else style
    data_key            = param_name        if data_values is None else 'default'
    lx, ly              = lattice.lx, lattice.ly
    logger              = kwargs.pop('logger', None)
    
    try:
        results             = PlotData.from_input(
                                directory       =   directory,
                                data_values     =   data_values,
                                x_parameters    =   x_parameters,
                                y_parameters    =   y_parameters,
                                x_param         =   x_param,
                                y_param         =   y_param,
                                data_key        =   data_key,
                                filters         =   filters,
                                logger          =   logger,
                                lx              =   lx,
                                ly              =   ly,
                                **kwargs
                            )
    except Exception as e:
        if logger:
            logger.error(f"Error loading results: {e}")
        return None, None
    
    if not results:
        return None, None
    
    # Setup grid
    unique_x        = sorted(set(x_parameters))
    unique_y        = sorted(set(y_parameters))
    n_rows, n_cols  = len(unique_y), len(unique_x)
    fig, axes, _, _ = FigureConfig.setup_figure_grid(n_rows=n_rows, n_cols=n_cols, style=style, fig=kwargs.get('fig', None), axes=kwargs.get('axes', None))
    
    # ---------------
    # Plotting loop
    # ---------------
    
    iterator        = 0
    for ii, y_val in enumerate(unique_y):
        for jj, x_val in enumerate(unique_x):
            ax      = axes[ii, jj]
            subset  = PlotData.from_match(results=results, x_param=x_param, x_val=x_val, y_param=y_param, y_val=y_val)
            
            if not subset:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
                        
            try:
                omega, k_vectors, akw = extract_spectral_data(subset, param_name, 
                                            state_idx           =   state_idx, 
                                            component           =   kwargs.get('component', None),
                                            omega_key           =   omega_key,
                                            kvectors_key        =   kvectors_key,
                                            reshape_to_komega   =   True,
                                        )
            except Exception as e:
                ax.text(0.5, 0.5, 'Data Error', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Handle different modes
            if mode == "sum":
                # Sum over all k
                intensity_omega     = np.sum(akw, axis=0)
                Plotter.plot(ax, omega, intensity_omega, lw=style.linewidth, color='C0', alpha=style.alpha)
                Plotter.set_ax_params(
                    ax,
                    xlabel          =   spectral_config.omega_label,
                    ylabel          =   spectral_config.intensity_label,
                    fontsize        =   style.fontsize_label
                )
                Plotter.grid(ax, alpha=0.3)
                
            elif mode == 'single':
                # Single k-point(s)
                if k_indices is None:
                    k_indices = [0]
                
                for idx in k_indices:
                    if idx < len(k_vectors):
                        intensity_omega = akw[idx, :]
                        Plotter.plot(ax, omega, intensity_omega, lw=style.linewidth, marker=style.marker, ms=style.markersize, alpha=style.alpha, label=f'k{idx}')
                
                Plotter.set_ax_params(
                    ax,
                    xlabel          =   spectral_config.omega_label,
                    ylabel          =   spectral_config.intensity_label,
                    fontsize        =   style.fontsize_label
                )
                Plotter.grid(ax, alpha=0.3)
                
                if len(k_indices) > 1:
                    Plotter.set_legend(ax, loc='best', fontsize=style.fontsize_legend)
            
            elif mode == 'kpath':
                # K-path mode
                try:
                    from .spectral_utils import plot_spectral_function_2d
                
                    path_result     = select_kpoints_along_path(
                                        lattice         =   lattice,
                                        k_vectors       =   k_vectors,
                                        path_labels     =   path_labels,
                                        use_extend      =   use_extend,
                                        extend_copies   =   extend_copies
                                    )
                    
                    # Extract data at selected k-points
                    selected_indices    = path_result['indices']        # (Npath,)
                    k_distances         = path_result['distances']      # (Npath,)
                    intensity_kw        = akw[selected_indices, :]      # (Npath, Nw)
                    
                    # Use new modular plotter
                    im                  = plot_spectral_function_2d(
                                            ax, k_distances, omega, intensity_kw,
                                            mode            =   'kpath',
                                            path_info       =   path_result,
                                            style           =   style,
                                            spectral_config =   spectral_config,
                                            lattice         =   lattice,
                                            use_extend      =   False,  # Already extended in select_kpoints_along_path
                                            colorbar        =   False,
                                            fig             =   fig
                                        )
                        
                except Exception as e:
                    if logger:
                        logger.error(f"Error plotting k-path spectral function: {e}")
                    ax.text(0.5, 0.5, f'Path Error', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            elif mode == 'grid':
                # Full k-space grid mode
                try:
                    from .spectral_utils import plot_spectral_function_2d
                    
                    im              = plot_spectral_function_2d(
                                            ax, k_vectors, omega, akw,
                                            mode            =   'grid',
                                            style           =   style,
                                            spectral_config =   spectral_config,
                                            lattice         =   lattice,
                                            use_extend      =   use_extend,
                                            extend_copies   =   extend_copies,
                                            colorbar        =   False,
                                            fig             =   fig
                                        )
                except Exception as e:
                    ax.text(0.5, 0.5, f'Grid Error', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            # Annotation
            if annotate_letter:
                xparam_lbl  =   param_labels.get(x_param, x_param)
                yparam_lbl  =   param_labels.get(y_param, y_param)
                addit = '' if not kwargs.get('annotate_params', True) else f'{xparam_lbl}={x_val:.2g}, {yparam_lbl}={y_val:.2g}'
                Plotter.set_annotate_letter(ax, iter=iterator, x=letter_x, y=letter_y, boxaround=kwargs.get('boxaround', False), addit=addit)
                ax.tick_params(labelsize=style.fontsize_tick)
                
            iterator += 1
    
    if title:
        fig.suptitle(title, fontsize=style.fontsize_title)
    
    return fig, axes

# ==============================================================================
# Phase diagram plotter
# ==============================================================================

def plot_phase_diagram_states(
        directory           : str, 
        param_name          : str,
        x_param             : str,
        y_param             : str, 
        filters             = None, 
        *,
        lx                  = None, 
        ly                  = None,
        lz                  = None,
        Ns                  = None,
        post_process_func   = None,
        # other
        figsize_per_panel   : tuple             = (6, 5), 
        nstates                                 = 4,
        # parameter labels
        param_labels        : dict              = {},
        param_fun           : callable          = lambda r, param_name: r.get(param_name, []),
        param_x_fun         : callable          = None,
        param_lbl           : Optional[str]     = None,
        # plot limits
        xlim                : Optional[tuple]   = None,
        ylim                : Optional[tuple]   = None,
        # colormap
        vmin                = None, 
        vmax                = None, 
        # plot settings
        ylabel              : Optional[str]     = None,
        xlabel              : Optional[str]     = None,
        cmap                = 'viridis', 
        logger              : Optional['Logger'] = None,
        save                : bool = False,
        **kwargs):
    
    """Plot phase diagram from ED results in specified directory."""
    try:
        from ..plot     import Plotter
    except ImportError:
        raise ImportError("Failed to import Plotter from the current package.")


    results = load_results(data_dir=directory, filters=filters, lx=lx, ly=ly, lz=lz, Ns=Ns, post_process_func=post_process_func, logger=logger)
    if len(results) == 0:
        if logger: logger.warning("No results found for phase diagram.")
        return None, None
    
    # extract unique parameters
    x_vals, y_vals, unique_x, unique_y  = PlotData.extract_parameter_arrays(results, x_param=x_param, y_param=y_param)
    plot_map                            = len(unique_x) > 1 and len(unique_y) > 1
    if logger:                          logger.info(f"Plot type: {'Colormap' if plot_map else 'Line plots'}", color='cyan')
    
    if nstates <= 1:
        ncols, nrows, npanels = 1, 1, 1
    elif nstates <= 3:
        ncols   = 1 if not plot_map else nstates
        nrows   = 1
        npanels = nstates if plot_map else 1
    else:
        ncols   = 1 if not plot_map else nstates // 2
        nrows   = 1 if not plot_map else ((nstates + ncols - 1) // ncols)
        npanels = nstates if plot_map else 1
        
    fig, axes, _, _                     = FigureConfig.create_subplot_grid(n_panels=npanels , max_cols=ncols, figsize_per_panel=figsize_per_panel, sharex=True, sharey=True)
    axes                                = axes.flatten()
    
    if xlabel is None:
        xlabel                          = param_labels.get(x_param, x_param) if len(unique_x) > 1 else param_labels.get(y_param, y_param)
    if ylabel is None:
        ylabel                          = param_labels.get(y_param, y_param) if len(unique_y) > 1 else param_labels.get(x_param, x_param)
    
    if plot_map:
        label_cond = lambda state_idx: {'x': state_idx // ncols == nrows - 1, 'y': state_idx % ncols == 0}
    else:
        label_cond = lambda state_idx: {'x': True, 'y': True}
    
    if vmin is None or vmax is None:
        vmin_n, vmax_n                  = PlotData.determine_vmax_vmin(results, param_name, param_fun, nstates) if vmin is None or vmax is None else (vmin, vmax)
        vmin                            = vmin_n if vmin is None else vmin
        vmax                            = vmax_n if vmax is None else vmax
        
    letter_x, letter_y                  = kwargs.pop('letter_x', 0.05), kwargs.pop('letter_y', 0.85) # annotation position

    # a) Only X Variation (Line Plot)
    if len(unique_x) > 1 and len(unique_y) == 1 or len(unique_x) == 1 and len(unique_y) > 1:
        param_plot  = x_param if len(unique_x) > 1 else y_param
        ax          = axes[0]
        label       = xlabel if len(unique_x) > 1 else ylabel
        getcolor    = Plotter.get_colormap(values=np.linspace(0, nstates, nstates), cmap=cmap, elsecolor='black', get_mappable=False)[0]
        
        for ii in range(nstates):
            x_plot, y_plot = [], []
            for r in results:
                y   = param_fun(r, param_name)
                x   = param_x_fun(r, param_plot) if param_x_fun is not None else r.params.get(param_plot, 0.0)
                if len(y) > ii:
                    x_plot.append(x)
                    y_plot.append(y[ii])
            
            sort_idx    = np.argsort(x_plot)
            x_plot      = np.array(x_plot)[sort_idx]
            y_plot      = np.array(y_plot)[sort_idx]
            Plotter.plot(ax, x=x_plot, y=y_plot, ls='-', marker='o', ms=4, color=getcolor(ii), label=rf'$|\Psi_{{{ii}}}\rangle$', zorder=100-ii)
        
        Plotter.set_ax_params(ax, xlabel=label, ylabel=param_lbl if param_lbl is not None else param_labels.get(param_name, param_name), xlim=xlim, ylim=ylim, yscale=kwargs.get('yscale', 'linear'), xscale=kwargs.get('xscale', 'linear'),)
        Plotter.set_annotate_letter(ax, iter=0, x=letter_x, y=letter_y, boxaround=False)
        Plotter.set_legend(ax, loc=kwargs.get('legend_loc', 'lower right'), fontsize=kwargs.get('legend_fontsize', 8))
        Plotter.grid(ax, alpha=0.3)

    # b) Colormap - Both X and Y Variation
    elif len(unique_x) > 1 and len(unique_y) > 1:
        xi                          = np.linspace(min(unique_x), max(unique_x), 200)
        yi                          = np.linspace(min(unique_y), max(unique_y), 200)
        Xi, Yi                      = np.meshgrid(xi, yi)
        scale_type                  = kwargs.get('cbar_scale', 'linear')
        getcolor, _, _, mappable    = Plotter.get_colormap(vmin=vmin, vmax=vmax, cmap=cmap, elsecolor='black', get_mappable=True, scale=scale_type)
        
        for ii in range(nstates):
            ax                              = axes[ii]
            x_scatter, y_scatter, z_scatter = [], [], []
            
            for r in results:
                val     = param_fun(r, param_name)
                # X param
                x_val   = param_x_fun(r, x_param) if param_x_fun is not None else r.params.get(x_param, 0.0)
                
                # Y param
                y_val   = r.params.get(y_param, 0.0)

                if isinstance(val, (list, np.ndarray)) and len(val) > ii:
                    x_scatter.append(x_val)
                    y_scatter.append(y_val)
                    z_scatter.append(val[ii])
                    
                elif isinstance(val, (int, float)) and ii == 0:
                    x_scatter.append(x_val)
                    y_scatter.append(y_val)
                    z_scatter.append(val)
            
            if len(x_scatter) < 3: # Need at least 3 points for griddata
                continue

            try:
                if kwargs.get('gridmethod', 'cubic') is None:
                    # do not use griddata, just scatter points where they are, color by value
                    for x, y, z in zip(x_scatter, y_scatter, z_scatter):
                        ax.scatter(x, y, color=getcolor(z), s=100, edgecolor=None, linewidth=0.5, marker='o')
                else:
                    Zi = griddata((x_scatter, y_scatter), z_scatter, (Xi, Yi), method=kwargs.get('gridmethod', 'cubic'), rescale=True)
                    cf = ax.contourf(Xi, Yi, Zi, cmap=cmap, extend='both', vmin=vmin, vmax=vmax)
            
                    if not np.all(np.isnan(Zi)):
                        cs = ax.contour(Xi, Yi, Zi, levels=kwargs.get('levels', 10), colors=kwargs.get('c_levels', 'white'), linewidths=0.5, alpha=0.3, vmin=vmin, vmax=vmax)
                        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2e', colors=kwargs.get('c_labels', 'white'))
            except Exception as e:
                if logger: logger.error(f"Error interpolating data for state {ii}: {e}")

            text_color      = kwargs.get('text_color', 'white')
            text_fontsize   = kwargs.get('text_fontsize', 8)
            Plotter.set_ax_params(ax, xlabel=xlabel, ylabel=ylabel, label_cond=label_cond(ii))
            Plotter.set_annotate_letter(ax, iter=ii, x=letter_x, y=letter_y, boxaround=False, color=text_color, addit=rf'$|\Psi_{{{ii}}}\rangle$', fontsize=text_fontsize)
        
        # Colorbar for all panels
        cbar_pos    = kwargs.get('cbar_pos', [0.92, 0.15, 0.02, 0.7])
        cbar_scale  = kwargs.pop('cbar_scale', 'linear')
        vmin        = vmin if cbar_scale == 'linear' else None
        Plotter.add_colorbar(fig=fig, pos=cbar_pos, mappable=mappable, cmap=cmap, scale=cbar_scale,
                    vmin=vmin, vmax=vmax, label=param_labels.get(param_name, param_name) if param_lbl is None else param_lbl,
                    extend=None, format='%.1e' if cbar_scale == 'log' else '%.2f', remove_pdf_lines=True
                )
    else:
        if logger: logger.warning("Insufficient parameter variation for phase diagram plot.")
        return None, None
        
    for ax in axes:
        Plotter.set_tickparams(ax)
        
    title = kwargs.get('title', '')
    if title:
        fig.suptitle(title, fontsize=kwargs.get('title_fontsize', 12))

    Plotter.hide_unused_panels(axes, npanels)

        
    return fig, axes

# ==============================================================================
# Correlation grid plotter
# ==============================================================================

def plot_correlation_grid(
                        directory           : str,
                        param_name          : str,
                        lattice             : 'Lattice',
                        *,
                        x_parameters        : list,
                        y_parameters        : list,
                        x_param             : str       = 'J',
                        y_param             : str       = 'hx',
                        # plotting mode
                        mode                : str       = 'lattice',
                        filters                         = None,
                        # reference
                        state_idx           : int       = 0,
                        ref_site_idx        : int       = 0,
                        # figure
                        figsize_per_panel   : tuple     = (4, 3.5),
                        cmap                : str       = 'RdBu_r',
                        vmin                            = None,
                        vmax                            = None,
                        title               : str       = '',
                        # data extraction
                        param_fun           : callable  = lambda r, param_name: r.get(param_name, []),
                        param_labels        : dict      = {},
                        post_process_func               = None,
                        point_closeness     : float     = 1e-5,
                        **kwargs):
    r"""
    Plot correlation matrices from ED results in a parameter grid with multiple visualization modes.
    
    This function creates state-of-the-art visualizations of quantum correlations across parameter space,
    supporting real-space, momentum-space, and band-structure representations that work universally
    for any lattice geometry (square, triangular, honeycomb, etc.).

    Parameters
    ----------
    directory : str
        Path to directory containing ED result files
    param_name : str
        Name of correlation data parameter to extract from results (e.g., 'spin_corr', 'density_corr')
    lattice : Lattice
        Lattice object defining the geometry (must have rvectors, kvectors, and k-space methods)
    x_parameters : list
        Values of x_param to include in grid
    y_parameters : list
        Values of y_param to include in grid
    x_param : str
        Parameter name for grid x-axis (default 'J')
    y_param : str
        Parameter name for grid y-axis (default 'hx')
    mode : str
        Visualization mode (see Modes section)
    state_idx : int
        Which eigenstate to plot (default 0 for ground state)
    ref_site_idx : int
        Reference site for real-space correlations (default 0)
    figsize_per_panel : tuple
        (width, height) for each subplot panel
    cmap : str
        Matplotlib colormap name
    vmin, vmax : float or None
        Color scale limits (auto-determined if None)
    title : str
        Figure super-title
    param_fun : callable
        Function to extract parameter data from result object
    param_labels : dict
        Pretty labels for parameters (e.g., {'hx': r'$h_x$'})
    post_process_func : callable or None
        Function to process results before plotting

    Modes
    -----
    'matrix'    : Full correlation matrix C_{ij} as heatmap
    'lattice'   : Real-space correlations from reference site
                    - 1D: line plot of C(x) vs. displacement
                    - 2D: smooth field with scattered points on lattice   
    'kspace'    : Structure factor S(k) in full Brillouin zone
                    - Smooth interpolated map with optional BZ outline
                    - Wigner-Seitz masking for proper BZ shape
                    - High-symmetry point labels (Γ, K, M, etc.)
    'kpath'     : Structure factor along high-symmetry path
                    - Band-structure style plot with vertical separators
                    - Uses lattice's default path or custom specification

    Structure Factor
    ----------------
    The momentum-space structure factor is computed as:
        S(k) = (1/Ns) sum _{i,j} C_{ij} exp[-i k . (r_i - r_j)]
    This properly handles multi-site unit cells by using full position vectors r_i
    (including basis offsets), not just Bravais lattice vectors.

    Key Parameters (kwargs)
    -----------------------
    **K-Space 2D Map (mode='kspace')**
        ks_grid_n           : int, interpolation grid resolution (default 220)
        ks_interp           : 'linear'|'cubic'|'nearest', interpolation method (default 'linear')
        ks_show_points      : bool, overlay discrete k-points (default True)
        ks_point_size       : float, marker size for k-points (default 10)
        ks_alpha_points     : float, transparency of k-point overlay (default 0.35)
        ks_draw_bz          : bool, draw Brillouin zone outline (default True)
        ks_label_hs         : bool, label high-symmetry points (default True)
        ks_mask_outside_bz  : bool, mask regions outside BZ (default True)
        ks_im_interp        : str, imshow interpolation (default 'bilinear')
        ks_blob_radius      : float, masking radius scale (default 2.5)
        ks_xlabel, ks_ylabel: str, axis labels (default r'$k_x$', r'$k_y$')
        auto_vlim_kspace    : bool, auto color limits from S(k) (default True)
        bz_shells           : int, WS cell neighbor shells for masking (default 1)
        hs_color            : str, high-symmetry label color (default 'white')
        hs_fs               : int, high-symmetry label fontsize (default 10)
    
    **K-Path (mode='kpath')**
        kpath               : path specification (None→lattice default, or StandardBZPath, 
                              or list like ['Gamma', 'K', 'M', 'Gamma'])
        kpath_pps           : int or None, points per path segment (None=auto-detect from k-grid density)
        kpath_ylabel        : str, y-axis label (default r"$S(\mathbf{k})$")
        kpath_line_kw       : dict, line plot styling. Default shows discrete points with markers.
                              For markers only (no lines): {"ls": "none", "marker": "o", "ms": 6}
        kpath_hs_kw         : dict, vertical separator styling (default {"color": "k", "alpha": 0.3})
        print_vectors       : bool, print debug info about k-grid (default False, True for first panel)
    
    **Real-Space (mode='lattice')**
        rs_interp           : str, 2D interpolation method (default 'linear')
        rs_point_size       : float, scatter marker size (default 55)
        rs_blob_radius      : float, field masking radius (default 2.5)
        rs_xlabel, rs_ylabel: str, axis labels (default r'$\Delta x$', r'$C(\Delta x)$')
    
    **General Styling**
        logger              : Logger, optional logging object
        param_label         : str, colorbar label override
        suptitle_fontsize   : int, title fontsize (default 14)
        letter_x, letter_y  : float, panel annotation position (default 0.05, 0.9)
        text_color          : str, annotation color (default 'black')
        show_panel_labels   : bool, show parameter values on panels (default True)
        cbar_pos            : list, colorbar [left, bottom, width, height] (default [0.92, 0.15, 0.02, 0.7])
        cbar_scale          : 'linear'|'log', colorbar scale (default 'linear')

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    axes_grid : np.ndarray
        2D array of axes objects, shaped (n_y_params, n_x_params)

    Examples
    --------
    >>> # K-space structure factor for square lattice
    >>> from general_python.lattices.square import SquareLattice
    >>> lattice = SquareLattice(dim=2, lx=8, ly=8, bc='pbc')
    >>> fig, axes = plot_correlation_grid(
    ...     directory='./ed_results',
    ...     param_name='spin_corr',
    ...     lattice=lattice,
    ...     x_parameters=[0.5, 1.0, 1.5],
    ...     y_parameters=[0.0, 0.1, 0.2],
    ...     x_param='J',
    ...     y_param='hx',
    ...     mode='kspace',
    ...     ks_draw_bz=True,
    ...     ks_label_hs=True,
    ...     param_labels={'J': r'$J$', 'hx': r'$h_x$'}
    ... )
    
    >>> # Band structure along high-symmetry path
    >>> fig, axes = plot_correlation_grid(
    ...     directory='./ed_results',
    ...     param_name='density_corr',
    ...     lattice=lattice,
    ...     x_parameters=[1.0],
    ...     y_parameters=[0.0, 0.5, 1.0],
    ...     mode='kpath',
    ...     kpath=['Gamma', 'X', 'M', 'Gamma'],
    ...     kpath_pps=100
    ... )
    
    >>> # Real-space correlation on honeycomb lattice
    >>> from general_python.lattices.honeycomb import HoneycombLattice
    >>> lattice = HoneycombLattice(dim=2, lx=6, ly=6, bc='pbc')
    >>> fig, axes = plot_correlation_grid(
    ...     directory='./ed_results',
    ...     param_name='spin_corr',
    ...     lattice=lattice,
    ...     x_parameters=np.linspace(0, 2, 5),
    ...     y_parameters=[0.0],
    ...     mode='lattice',
    ...     ref_site_idx=0,
    ...     rs_point_size=80,
    ...     cmap='RdBu_r'
    ... )

    Notes
    -----
    - Works with any lattice type implementing the standard Lattice interface
    - Automatically handles multi-site unit cells (e.g., honeycomb has 2 sites per cell)
    - K-space modes require lattice.kvectors and lattice.rvectors attributes
    - For best results with kpath mode, lattice should implement high_symmetry_points()
    - Wigner-Seitz BZ masking uses general_python.lattices.tools.lattice_kspace.ws_bz_mask
    """
    try:
        from ..plot     import Plotter
    except ImportError:
        raise ImportError("Failed to import Plotter from the current package.")
    
    # Needs lattice
    if lattice is not None:
        lx, ly, lz, Ns  = lattice.lx, lattice.ly, lattice.lz, lattice.Ns
    else:
        lx, ly, lz, Ns  = kwargs.get('lx', None), kwargs.get('ly', None), kwargs.get('lz', None), kwargs.get('Ns', None)
    
    # Require lattice or lx, ly, Ns
    if lx is None or ly is None or Ns is None:
        raise ValueError("Lattice or lx, ly, Ns must be provided.")
    
    results             = load_results(
                            data_dir            =   directory,
                            filters             =   filters,
                            lx                  =   lx, 
                            ly                  =   ly, 
                            lz                  =   lz, 
                            Ns                  =   Ns,
                            logger              =   kwargs.get('logger', None),
                            post_process_func   =   post_process_func
                        )
    if not results:
        return None, None

    _, _, unique_x, unique_y = PlotData.extract_parameter_arrays(results, x_param, y_param)

    unique_x            = np.array([v for v in unique_x if any(abs(v - p) < 1e-5 for p in x_parameters)])
    unique_y            = np.array([v for v in unique_y if any(abs(v - p) < 1e-5 for p in y_parameters)])
    unique_x            = np.sort(unique_x)
    unique_y            = np.sort(unique_y)
    unique_y_plot       = unique_y[::-1]

    n_rows, n_cols      = len(unique_y_plot), len(unique_x)
    if n_rows == 0 or n_cols == 0:
        return None, None

    fig, axes, _, _     = FigureConfig.create_subplot_grid(
                            n_panels            = n_rows * n_cols,
                            max_cols            = n_cols,
                            figsize_per_panel   = figsize_per_panel,
                            sharex              = True,
                            sharey              = True
                        )
    axes_grid           = np.array(axes).reshape((n_rows, n_cols))

    # --------------------------------------------------------------
    # resolve plot_mode
    # --------------------------------------------------------------
    plot_mode   = 'matrix'
    mode_l      = (mode or '').lower().strip()
    if mode_l in ('matrix',):
        plot_mode = 'matrix'
    elif mode_l in ('lattice', 'real', 'realspace'):
        plot_mode = 'lattice'
    elif mode_l in ('kspace', 'bz', 'brillouin', 'brillouin_zone'):
        plot_mode = 'kspace'
    elif mode_l == 'kpath':
        plot_mode = 'kpath'
    else:
        plot_mode = 'matrix'

    # --------------------------------------------------------------
    # real-space geometry
    # --------------------------------------------------------------
    positions   = None
    d_dim       = None
    if lattice is not None:
        try:
            positions   = np.asarray(lattice.rvectors, float)
            d_dim       = positions.shape[1]
        except Exception:
            positions   = None
            d_dim       = None

    # --------------------------------------------------------------
    # k-space precomputes (shared across panels)
    # --------------------------------------------------------------
    k_cart              = None          # (Nk,3)
    k2                  = None          # (Nk,2)
    r2                  = None          # (Ns,2)
    phase               = None          # (Nk,Ns)
    phase_conj          = None          # (Nk,Ns)
    Ns_lat              = None          # number of sites in lattice, if needed
    Nk                  = None          # number of k-points
    
    interpolation       = kwargs.get('interpolation', 'linear')             # interpolation method

    ks_grid_n           = int(kwargs.get('ks_grid_n',           500))       # interpolation grid resolution
    ks_show_points      = bool(kwargs.get('ks_show_points',     True))      # show discrete k-points
    ks_point_size       = float(kwargs.get('ks_point_size',     10))        # k-point marker size
    ks_alpha_points     = float(kwargs.get('ks_alpha_points',   0.35))      # k-point alpha - transparency
    ks_draw_bz          = bool(kwargs.get('ks_draw_bz',         True))      # draw BZ outline
    ks_label_hs         = bool(kwargs.get('ks_label_hs',        True))      # label high-symmetry points
    ks_mask_outside_bz  = bool(kwargs.get('ks_mask_outside_bz', True))      # mask outside BZ
    auto_vlim_kspace    = bool(kwargs.get('auto_vlim_kspace',   True))      # auto vmin/vmax

    # If kspace: build phase once.
    if plot_mode == 'kspace' and lattice is not None:
        try:
            k_cart      = np.asarray(lattice.kvectors, float)   # (Nk,3) usually Nk = Nc
            r_cart      = np.asarray(lattice.rvectors, float)   # (Ns,3) Ns = Nc*Nb

            k2          = k_cart[:, :2]
            r2          = r_cart[:, :2]

            phase       = np.exp(-1j * (k2 @ r2.T))             # (Nk,Ns)
            phase_conj  = np.conjugate(phase)                   # (Nk,Ns) for S(k) calc

            Ns_lat      = r2.shape[0]
            Nk          = k2.shape[0]
        except Exception:
            plot_mode   = 'matrix'

    # --------------------------------------------------------------
    # determine vmin/vmax
    # --------------------------------------------------------------
    
    def _extract_corr_matrix(res):
        ns          = int(res.params.get('Ns', Ns))
        data_root   = param_fun(res, param_name)
        if data_root is None or getattr(data_root, "size", 0) == 0:
            return None

        if data_root.ndim == 3:
            return np.asarray(np.real(data_root[:, :, state_idx]), float)
        if data_root.ndim == 2 and data_root.shape == (ns, ns):
            return np.asarray(np.real(data_root), float)
        return None

    def _selected_results():
        out = []
        for y_val in unique_y_plot:
            for x_val in unique_x:
                for r in results:
                    rx = r.params.get(x_param, np.nan)
                    ry = r.params.get(y_param, np.nan)
                    if abs(rx - x_val) < 1e-5 and abs(ry - y_val) < 1e-5:
                        out.append(r)
                        break
        return out

    if (vmin is None or vmax is None):
        if plot_mode == 'kspace' and phase is not None and auto_vlim_kspace:
            sk_min, sk_max  = +np.inf, -np.inf
            for r in _selected_results():
                C           = _extract_corr_matrix(r)
                if C is None:
                    continue
                if C.shape[0] != Ns_lat:
                    continue
                tmp         = phase @ C
                Sk          = np.real((tmp * phase_conj).sum(axis=1) / C.shape[0])
                if np.all(np.isnan(Sk)):
                    continue
                sk_min      = min(sk_min, float(np.nanmin(Sk)))
                sk_max      = max(sk_max, float(np.nanmax(Sk)))
            if np.isfinite(sk_min) and np.isfinite(sk_max):
                if vmin is None: vmin = sk_min
                if vmax is None: vmax = sk_max
            else:
                # fallback
                if vmin is None: vmin = -1.0
                if vmax is None: vmax = +1.0
        else:
            vmin_n, vmax_n  = PlotData.determine_vmax_vmin(results, param_name, param_fun, nstates=1)
            if vmin is None: vmin = vmin_n
            if vmax is None: vmax = vmax_n

    # shared color normalization
    _, _, _, mappable = Plotter.get_colormap(values=[vmin, vmax], cmap=cmap, get_mappable=True)

    # axis label logic
    def _label_cond(i, j):
        return {'x': i == n_rows - 1, 'y': j == 0}

    def _set_axis_labels(ax, i, j):
        if plot_mode == 'matrix':
            Plotter.set_ax_params(
                ax,
                xlabel      = r'Site $j$',
                ylabel      = r'Site $i$',
                label_cond   = _label_cond(i, j),
                labelPos    = {'x': 'bottom', 'y': 'left'},
                tickPos     = {'x': 'bottom', 'y': 'left'},
            )
            return

        if plot_mode == 'lattice':
            # In 1D we show x; in 2D and lattice maps we usually hide axes.
            if positions is None:
                return
            if d_dim == 1:
                Plotter.set_ax_params(
                    ax,
                    xlabel      = kwargs.get('rs_xlabel', r'$\Delta x$'),
                    ylabel      = kwargs.get('rs_ylabel', r'$C(\Delta x)$'),
                    label_cond   = _label_cond(i, j),
                    labelPos    = {'x': 'bottom', 'y': 'left'},
                    tickPos     = {'x': 'bottom', 'y': 'left'},
                )
            else:
                ax.axis('off')
            return

        if plot_mode == 'kspace':
            # BZ panels style: only outer labels
            if _label_cond(i, j).get('x', False):
                ax.set_xlabel(kwargs.get('ks_xlabel', r'$k_x$'))
            if _label_cond(i, j).get('y', False):
                ax.set_ylabel(kwargs.get('ks_ylabel', r'$k_y$'))
            return

        if plot_mode == 'kpath':
            # k-path already sets its own labels via plot_bz_path_from_corr
            return

    # --------------------------------------------------------------
    # plotting loop
    # --------------------------------------------------------------
    
    for ii, y_val in enumerate(unique_y_plot):
        for jj, x_val in enumerate(unique_x):

            ax      = axes_grid[ii, jj]
            subset  = []

            for r in results:
                rx = r.params.get(x_param, np.nan)
                ry = r.params.get(y_param, np.nan)
                if abs(rx - x_val) < point_closeness and abs(ry - y_val) < point_closeness:
                    subset.append(r)
                    
            # if no result, skip
            if not subset:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            result      = subset[0]
            corr_matrix = _extract_corr_matrix(result)
            if corr_matrix is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            # ----------------------------------------------------------
            # plot per mode
            # ----------------------------------------------------------
            if plot_mode == 'matrix':
                ax.imshow(
                    corr_matrix,
                    cmap            = cmap,
                    vmin            = vmin,
                    vmax            = vmax,
                    interpolation   = interpolation,
                    origin          = 'lower'
                )

            elif plot_mode == 'lattice':
                if positions is None:
                    ax.imshow(
                        corr_matrix,
                        cmap            = cmap,
                        vmin            = vmin,
                        vmax            = vmax,
                        interpolation   = interpolation,
                        origin          = 'lower'
                    )
                    
                else:
                    ref_site_idx_use = ref_site_idx if (0 <= ref_site_idx < positions.shape[0]) else 0
                    site_vals        = corr_matrix[ref_site_idx_use, :]

                    # 1D: correlation vs displacement coordinate
                    if d_dim == 1 or (positions.shape[1] == 1):
                        x               = positions[:, 0] - positions[ref_site_idx_use, 0]
                        idx             = np.argsort(x)
                        Plotter.plot(ax, x[idx], site_vals[idx], marker='o', lw=1.5)
                        Plotter.vline(ax, 0.0, ls='--', lw=1, color='k', alpha=0.4)
                        Plotter.grid(ax, alpha=0.3)

                    # 2D: smooth "blob" field + point overlay
                    else:
                        pos             = positions[:, :2] - positions[ref_site_idx_use, :2]

                        x_min, x_max    = pos[:, 0].min(), pos[:, 0].max()
                        y_min, y_max    = pos[:, 1].min(), pos[:, 1].max()
                        pad_x           = 0.15 * (x_max - x_min + 1e-12)
                        pad_y           = 0.15 * (y_max - y_min + 1e-12)

                        xx              = np.linspace(x_min - pad_x, x_max + pad_x, 220)
                        yy              = np.linspace(y_min - pad_y, y_max + pad_y, 220)
                        X, Y            = np.meshgrid(xx, yy)

                        # Interpolate but keep it local (avoid filling big empty regions)
                        Z               = None
                        try:
                            Z           = griddata(pos, site_vals, (X, Y), method=kwargs.get('rs_interp', 'linear'))
                        except Exception:
                            Z           = None

                        if Z is None:
                            d2          = (X[..., None] - pos[:, 0])**2 + (Y[..., None] - pos[:, 1])**2
                            Z           = site_vals[np.argmin(d2, axis=2)]

                        # Local-support mask so you get blobs, not a filled convex hull
                        try:
                            tree        = cKDTree(pos)
                            d, _        = tree.query(np.column_stack([X.ravel(), Y.ravel()]), k=1)
                            d           = d.reshape(X.shape)
                            d0          = np.median(d[np.isfinite(d)])
                            dmax        = float(kwargs.get('rs_blob_radius', 2.5 * d0))
                            Z           = np.where(d <= dmax, Z, np.nan)
                        except Exception:
                            pass

                        ax.imshow(
                                Z,
                                extent  = (xx[0], xx[-1], yy[0], yy[-1]),
                                origin  = 'lower',
                                cmap    = cmap,
                                vmin    = vmin,
                                vmax    = vmax,
                                alpha   = 0.95,
                            )

                        Plotter.scatter(
                            ax,
                            pos[:, 0], pos[:, 1],
                            c           = site_vals,
                            cmap        = cmap,
                            vmin        = vmin,
                            vmax        = vmax,
                            edgecolor   = 'k',
                            linewidths  = 0.5,
                            s           = kwargs.get('rs_point_size', 55),
                            zorder      = 10
                        )
                        Plotter.scatter(ax, [0], [0], marker='*', s=120, c='yellow', edgecolor='k', zorder=11)
                        ax.set_aspect('equal')
                        ax.axis('off')

            elif plot_mode == 'kspace':
                if phase is None or k2 is None:
                    ax.text(0.5, 0.5, 'k-space unavailable', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                else:
                    # Compute S(k)
                    if corr_matrix.shape[0] != Ns_lat:
                        ax.text(0.5, 0.5, 'Ns mismatch', ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                    else:
                        C       = np.asarray(corr_matrix, float)
                        Ns0     = C.shape[0]

                        tmp     = phase @ C
                        Sk      = np.real((tmp * phase_conj).sum(axis=1) / Ns0) # (Nk,)

                        # Use the new modular k-space plotter
                        # Create configuration objects
                        style   = PlotStyle(
                                    cmap                    =   cmap,
                                    vmin                    =   vmin,
                                    vmax                    =   vmax,
                                    fontsize_label          =   kwargs.get('fontsize_label', 10),
                                    fontsize_tick           =   kwargs.get('fontsize_tick', 8),
                                    linewidth               =   1.5,
                                    markersize              =   3,
                                    marker                  =   'o',
                                    alpha                   =   ks_alpha_points
                                )
                        
                        ks_config = KSpaceConfig(
                                    grid_n                  =   ks_grid_n,
                                    interp_method           =   interpolation,
                                    show_discrete_points    =   ks_show_points,
                                    point_size              =   ks_point_size,
                                    point_alpha             =   ks_alpha_points,
                                    draw_bz_outline         =   ks_draw_bz,
                                    label_high_symmetry     =   ks_label_hs,
                                    mask_outside_bz         =   ks_mask_outside_bz,
                                    imshow_interp           =   kwargs.get('ks_im_interp', 'bilinear'),
                                    ws_shells               =   kwargs.get('bz_shells', 1)
                                )
                        
                        # Use new plot_kspace_intensity with extended BZ and Pi labels
                        plot_kspace_intensity(
                                    ax                      =   ax,
                                    k2                      =   k2,
                                    intensity               =   Sk,
                                    style                   =   style,
                                    ks_config               =   ks_config,
                                    lattice                 =   lattice,
                                    show_extended_bz        =   kwargs.get('show_extended_bz', True),
                                    bz_copies               =   kwargs.get('bz_copies', 2)
                                )

            elif plot_mode == 'kpath':
                # k-path mode: plot S(k) along high-symmetry path using EXACT k-points
                if corr_matrix is None:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                else:
                    plot_bz_path_from_corr(
                        ax              = ax,
                        lattice         = lattice,
                        corr_matrix     = corr_matrix,
                        path            = kwargs.get('kpath',           None),
                        points_per_seg  = kwargs.get('kpath_pps',       None),  # None = auto-detect based on k-grid
                        value_label     = kwargs.get('kpath_ylabel',    r"$S(\mathbf{k})$"),
                        line_kw         = kwargs.get('kpath_line_kw',   None),  # None = use default (discrete markers)
                        hsline_kw       = kwargs.get('kpath_hs_kw',     {"color": "k", "alpha": 0.3}),
                        print_vectors   = kwargs.get('print_vectors',   (ii == 0 and jj == 0)),  # print only once
                    )

            # ----------------------------------------------------------
            # per-panel annotations
            # ----------------------------------------------------------
            
            if kwargs.get('show_panel_labels', True):
                ann_color = kwargs.get('text_color', 'black')
                if plot_mode == 'kspace' and ks_mask_outside_bz:
                    ann_color   = kwargs.get('text_color', 'white')  # better contrast on BZ maps
                Plotter.set_annotate_letter(
                    ax,
                    iter        = 0,
                    x           = kwargs.get('letter_x', 0.05),
                    y           = kwargs.get('letter_y', 0.9),
                    boxaround   = False,
                    color       = ann_color,
                    addit       = rf'{param_labels.get(x_param, x_param)}$={x_val:.2g}$, '
                                  rf'{param_labels.get(y_param, y_param)}$={y_val:.2g}$'
                )

            if plot_mode not in ('lattice', 'kpath'):
                Plotter.grid(ax, alpha=0.25)
            elif plot_mode == 'lattice' and (positions is None or d_dim == 1):
                Plotter.grid(ax, alpha=0.25)

            _set_axis_labels(ax, ii, jj)
            Plotter.set_tickparams(ax)

    # --------------------------------------------------------------
    # colorbar + title
    # --------------------------------------------------------------
    cbar_scale              = kwargs.pop('cbar_scale', 'linear')
    vmin_cb                 = vmin if cbar_scale != 'linear' else None
    
    # For kpath mode, use fewer ticks (discrete k-points)
    cbar_format             = '%.2f'
    if plot_mode == 'kpath':
        cbar_format = '%.3f'  # more precision for discrete values

    Plotter.add_colorbar(
        fig                 = fig,
        pos                 = kwargs.get('cbar_pos', [0.92, 0.15, 0.02, 0.7]),
        mappable            = mappable,
        cmap                = cmap,
        vmin                = vmin_cb,
        vmax                = vmax,
        label               = kwargs.get('param_label', param_labels.get(param_name, param_name)),
        format              = cbar_format,
        scale               = cbar_scale,
        remove_pdf_lines    = True
    )

    # adjust the ticks
    if plot_mode == 'kspace' or plot_mode == 'kpath':
        # Format tick labels as multiples of Pi
        format_pi_ticks(ax, axis=kwargs.get('pi_tick_axis', 'both' if plot_mode == 'kspace' else 'x'))

    fig.suptitle(title, fontsize=kwargs.get('suptitle_fontsize', 14))
    plt.show()
    return fig, axes_grid

# c) Multi-state vs parameter plotter

def plot_multistate_vs_param(
        directory           : str,
        param_name          : str,
        x_param             : str = 'hx', 
        p_param             : str = 'Gz', 
        p_values            : list = None,
        filters             = None, 
        *,
        lx                  = None, 
        ly                  = None,
        lz                  = None,
        Ns                  = None,
        # plot settings
        nstates             : int   = 10, 
        figsize_per_panel   : tuple = (4, 3),
        xlim                = None, 
        ylim                = None,
        cmap                : str = 'viridis',
        # data extraction
        param_fun           : callable  = lambda r, key: r.get(key, []),
        param_labels        : dict      = {},
        trans_fun           : callable  = lambda raw_data_array, state_index: raw_data_array[state_index],
        post_process_func   : callable  = None,
        # labels
        ylabel              : str = None,
        title               : str = '',
        # advanced
        derivative_order    : int  = 0,
        susceptibility_sign : bool = False,
        **kwargs):

    results = load_results(data_dir=directory, filters=filters, lx=lx, ly=ly, lz=lz, Ns=Ns, post_process_func=post_process_func, logger=kwargs.get('logger', None))
    if not results: return None, None

    if p_values is None:
        _, _, _, unique_p       = PlotData.extract_parameter_arrays(results, x_param, p_param)
        p_values                = sorted(unique_p)
        if len(p_values) > 9 and kwargs.get('limit_panels', True):
            p_values = p_values[:9]

    n_panels                    = len(p_values)
    fig, axes, n_rows, n_cols   = FigureConfig.create_subplot_grid(n_panels, max_cols=3, figsize_per_panel=figsize_per_panel, sharex=True, sharey=True)
    if hasattr(fig, 'set_constrained_layout'):
        fig.set_constrained_layout(True)
    
    axes                        = axes.flatten()
    label_cond                   = lambda idx: {'x': idx // n_cols == n_rows - 1, 'y': idx % n_cols == 0}
    xlabel_str                  = param_labels.get(x_param, x_param)
    ylabel_str                  = ylabel if ylabel else param_labels.get(param_name, param_name)
    getcolor, _, _, mappable    = Plotter.get_colormap(values=np.arange(nstates), cmap=cmap, elsecolor='black', get_mappable=True)
    
    for idx, p_val in enumerate(p_values):
        # Filter for panel param
        ax              = axes[idx]
        panel_results   = []
        
        for r in results:
            val = r.params.get(p_param, np.nan)
            if abs(val - p_val) < 1e-6:
                panel_results.append(r)
        
        if not panel_results:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            continue

        x_vals, sort_idx    = PlotData.sort_results_by_param(panel_results, x_param)
        sorted_res          = [panel_results[i] for i in sort_idx]

        for state_i in range(nstates):
            y_vals, valid_x = [], []
            for k, r in enumerate(sorted_res):
                raw = param_fun(r, param_name)
                try:
                    is_valid = isinstance(raw, (list, np.ndarray)) and len(raw) > state_i
                    if is_valid:
                        val = trans_fun(raw, int(state_i))
                        if not np.isnan(val):
                            y_vals.append(val)
                            valid_x.append(x_vals[k])
                except Exception:
                    pass

            if len(valid_x) > 2:
                x_arr = np.array(valid_x)
                y_arr = np.array(y_vals)
                
                if derivative_order >= 1: y_arr = np.gradient(y_arr, x_arr)
                if derivative_order >= 2: y_arr = np.gradient(y_arr, x_arr) * (-1 if susceptibility_sign else 1)
                ax.plot(x_arr, y_arr, color=getcolor(state_i), marker='o', markersize=3, ls='-', zorder=100 - state_i, label=rf'$|\Psi_{{{state_i}}}\rangle$')

        p_label = param_labels.get(p_param, p_param)
        Plotter.set_annotate_letter(ax, iter=idx, x=kwargs.get('annotate_x', 0.05), y=kwargs.get('annotate_y', 0.9), addit=rf' {p_label}$={p_val:.2g}$', boxaround=False)
        Plotter.set_ax_params(ax, xlabel=xlabel_str, ylabel=ylabel_str, label_cond=label_cond(idx), xlim=xlim, ylim=ylim, yscale=kwargs.get('yscale', 'linear'), xscale=kwargs.get('xscale', 'linear'))
        Plotter.set_tickparams(ax)
        Plotter.grid(ax, alpha=0.3)

    Plotter.hide_unused_panels(axes, n_panels)
    
    cbar_pos    = kwargs.get('cbar_pos', [0.92, 0.2, 0.02, 0.6])
    cbar_scale  = kwargs.get('cbar_scale', 'linear')
    vmin        = 0 if cbar_scale == 'linear' else None
    if nstates > 6:
        Plotter.add_colorbar(fig=fig, pos=cbar_pos, mappable=mappable, cmap=cmap, scale=cbar_scale, vmin=vmin, vmax=nstates - 1, label=r'State Index $n$', extend=None, format='%d', remove_pdf_lines=True)
    else:
        Plotter.set_legend(axes[0], loc=kwargs.get('legend_loc', 'upper right'), fontsize=kwargs.get('legend_fontsize', 8))
        
    if title: 
        fig.suptitle(title, fontsize=kwargs.get('suptitle_fontsize', 12))
        
    return fig, axes

# d) Size scaling plotter

def plot_scaling_analysis(
        directory: str,
        param_name: str,
        scaling_param: str = 'Ns',
        series_param: str = 'hx',
        filters=None,
        state_idx: int = 0,
        *,
        param_fun: callable = lambda r, name: r.get(name, []),
        param_labels: dict = {},
        figsize: tuple = (6, 5),
        logger: Optional['Logger'] = None,
        **kwargs):
    """
    Plots a scaling analysis of a parameter (e.g., energy, gap) vs system size (Ns or 1/L),
    grouping lines by another parameter (e.g., field strength).
    """
    
    results = load_results(data_dir=directory, filters=filters, logger=logger)
    if not results: return None, None

    # Get unique series values
    _, _, _, unique_series = PlotData.extract_parameter_arrays(results, scaling_param, series_param)
    unique_series = sorted(unique_series)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = plt.get_cmap(kwargs.get('cmap', 'viridis'))
    norm = plt.Normalize(min(unique_series), max(unique_series))
    
    for s_val in unique_series:
        # Filter for this series
        subset = [r for r in results if abs(r.params.get(series_param, -999) - s_val) < 1e-5]
        if not subset: continue
        
        x_vals, sort_idx = PlotData.sort_results_by_param(subset, scaling_param)
        sorted_subset = [subset[i] for i in sort_idx]
        
        y_vals = []
        x_plot = []
        for r, x_v in zip(sorted_subset, x_vals):
            val = param_fun(r, param_name)
            
            # Handle list/array or scalar
            if isinstance(val, (list, np.ndarray)):
                if len(val) > state_idx:
                    y_vals.append(val[state_idx])
                    x_plot.append(x_v)
            elif isinstance(val, (float, int)):
                 y_vals.append(val)
                 x_plot.append(x_v)
        
        if len(x_plot) > 0:
            color = cmap(norm(s_val))
            label = f"{param_labels.get(series_param, series_param)}={s_val:.2g}"
            ax.plot(x_plot, y_vals, marker='o', linestyle='-', color=color, label=label)

    xlabel = param_labels.get(scaling_param, scaling_param)
    ylabel = param_labels.get(param_name, param_name)
    
    Plotter.set_ax_params(ax, xlabel=xlabel, ylabel=ylabel, title=kwargs.get('title', ''))
    Plotter.set_tickparams(ax)
    Plotter.grid(ax)
    
    # Legend
    if len(unique_series) <= 10:
        ax.legend(frameon=False, fontsize=8)
    else:
        # If too many lines, use colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(param_labels.get(series_param, series_param))

    PlotData.savefig(fig, directory, param_name, scaling_param, series_param, suffix='_scaling')
    return fig, ax

# ------------------------------------------------------------------
#! End of file
# ------------------------------------------------------------------
