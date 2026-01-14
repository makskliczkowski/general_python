'''
This module contains common plotting utilities for Quantum EigenSolver.

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-12-01
License             : MIT
----------------------------------------------------------------
'''

from    typing              import List, Callable, TYPE_CHECKING, Optional, Tuple, Union
from    scipy.interpolate   import griddata
import  numpy               as np
import  matplotlib.pyplot   as plt

try:
    from general_python.lattices.lattice    import Lattice
    from .data_loader                       import load_results, PlotDataHelpers
    from ..plot                             import Plotter
except ImportError:
    raise ImportError("Failed to import required modules from the current package.")

if TYPE_CHECKING:
    from general_python.common.flog         import Logger
    
# --------------------------------------------------------------

class PlotEDHelpers(PlotDataHelpers):
    """Helper functions for plotting ED results."""
    
    # --------------------------------------------------------------
    #! K-SPACE PATH PLOTTING HELPERS
    # --------------------------------------------------------------
    
    @staticmethod
    def format_k_ticks(ax, q_vectors: List[np.ndarray], step=None):
        """
        Sets x-ticks on the given axis based on q_vectors.
        Parameters:
        -----------
        ax:
            Matplotlib axis to set ticks on.
        q_vectors:
            List of q-vectors (each as [qx, qy, qz,...]).
        step:
            Step size for ticks. If None, it is determined automatically.
        """
        
        if q_vectors is None or len(q_vectors) == 0: 
            return
        
        n_q = len(q_vectors)
        if step is None:
            if n_q <= 10    : step = 1          # all ticks
            elif n_q <= 20  : step = 2          # approx 10 ticks
            else            : step = n_q // 6   # approx 6 ticks
            
        tick_idx    = np.arange(0, n_q, step)
        labels      = []
        for i in tick_idx:
            qx, qy      = q_vectors[i][:2] 
            str_x       = f"{int(qx)}" if float(qx).is_integer() else f"{qx:.2g}"
            str_y       = f"{int(qy)}" if float(qy).is_integer() else f"{qy:.2g}"
            labels.append(f"({str_x},{str_y})")
            
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # --------------------------------------------------------------

    @staticmethod
    def find_k_path_indices(q_vectors, path_points: List[np.ndarray], num_points=50):
        """
        Find indices of q_vectors closest to a path defined by waypoints.

        Parameters:
        -----------
        q_vectors: 
            list of q-vectors (each as [qx, qy, qz,...])
        path_points: 
            list of waypoints defining the path in q-space
        num_points: 
            total number of points to sample along the path
        Returns: 
            path_indices: 
                list of indices in q_vectors
            path_ticks: 
                list of (index_in_path, label) for axis ticks
        """
        if not path_points: 
            return [], []
        
        # We walk from point A to point B
        # Since q_vectors are discrete, we just find the closest ones along the line
        path_indices        = []
        path_ticks          = []
        q_vectors           = np.array(q_vectors)
        
        for i in range(len(path_points) - 1):
            start   = np.array(path_points[i])
            end     = np.array(path_points[i+1])
            
            # Record tick
            path_ticks.append((len(path_indices), f"P{i}"))
            
            # Interpolate
            # Use distances to find closest
            # This is a naive implementation. For small lattices, we might just jump.
            # For finite size, we pick the q-point minimizing distance to the line segment AND progress
            
            # Simple approach: Linear interpolation sample
            steps = np.linspace(0, 1, num_points // (len(path_points)-1))
            for t in steps:
                # Find closest q
                target  = start + t * (end - start)
                dists   = np.sum((q_vectors[:, :2] - target)**2, axis=1)
                closest = np.argmin(dists)
                if len(path_indices) == 0 or path_indices[-1] != closest:
                    path_indices.append(closest)
                    
        # Add last tick
        path_ticks.append((len(path_indices)-1, f"P{len(path_points)-1}"))
        
        return path_indices, path_ticks

# --------------------------------------------------------------
# Plotting functions
# --------------------------------------------------------------

# a) Phase diagram plotter

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
        figsize_per_panel   : tuple = (6, 5), 
        nstates             = 4,
        # parameter labels
        param_labels        : dict  = {},
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

    results = load_results(data_dir=directory, filters=filters, lx=lx, ly=ly, lz=lz, Ns=Ns, post_process_func=post_process_func, logger=logger)
    if len(results) == 0:
        if logger: logger.warning("No results found for phase diagram.")
        return None, None
    
    # extract unique parameters
    x_vals, y_vals, unique_x, unique_y  = PlotEDHelpers.extract_parameter_arrays(results, x_param=x_param, y_param=y_param)
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
        
    fig, axes, _, _                     = PlotEDHelpers.create_subplot_grid(n_panels=npanels , max_cols=ncols, figsize_per_panel=figsize_per_panel, sharex=True, sharey=True)
    axes                                = axes.flatten()
    
    if xlabel is None:
        xlabel                          = param_labels.get(x_param, x_param) if len(unique_x) > 1 else param_labels.get(y_param, y_param)
    if ylabel is None:
        ylabel                          = param_labels.get(y_param, y_param) if len(unique_y) > 1 else param_labels.get(x_param, x_param)
    
    if plot_map:
        labelcond = lambda state_idx: {'x': state_idx // ncols == nrows - 1, 'y': state_idx % ncols == 0}
    else:
        labelcond = lambda state_idx: {'x': True, 'y': True}
    
    if vmin is None or vmax is None:
        vmin_n, vmax_n                  = PlotEDHelpers.determine_vmax_vmin(results, param_name, param_fun, nstates) if vmin is None or vmax is None else (vmin, vmax)
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
            Plotter.set_ax_params(ax, xlabel=xlabel, ylabel=ylabel, labelCond=labelcond(ii))
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
    
    if save:
        PlotEDHelpers.savefig(fig, directory, param_name, x_param, y_param if plot_map else None, **kwargs)
        
    return fig, axes

# b) Correlation grid plotter

    
def plot_bz_path_from_corr(
        ax,
        lattice         : "Lattice",
        corr_matrix     : np.ndarray,
        *,
        path            =   None,
        points_per_seg  : int   = 60,
        value_label     : str   = r"$S(\mathbf{k})$",
        line_kw         : dict  = None,
        hsline_kw       : dict  = None,
        tol             : float = 1e-12,
    ):
    r"""
    Plot a correlation-derived structure factor along a high-symmetry path.

    Computes (site-based) structure factor:
        S(k) = (1/Ns) sum_{i,j} C_{ij} exp[-i k·(r_i - r_j)]
             = (1/Ns) * diag( P C P^\dagger ),
    where P_{k,i} = exp(-i k·r_i).

    Uses lattice utilities:
      - lattice.generate_bz_path(...) and/or lattice.extract_bz_path_data(...)

    Parameters
    ----------
    ax : matplotlib Axes
    lattice : Lattice
        Must have .rvectors, .kvectors, .kvectors_frac (recommended), and .extract_bz_path_data.
        If kvectors_frac is missing, a fallback is used (approx).
    corr_matrix : (Ns,Ns) array
        Correlation matrix in site basis.
    path : optional
        Path specification accepted by lattice.extract_bz_path_data:
        - None -> uses default high-symmetry path from lattice.high_symmetry_points()
        - list of names like ['Gamma','K','M','Gamma'] if supported
        - StandardBZPath enum or PathTypes or string, as your API allows
    points_per_seg : int
        Number of interpolated points per segment in the *ideal* path before nearest-k matching.
    value_label : str
        y-axis label.
    line_kw : dict
        kwargs forwarded to ax.plot for the line.
    hsline_kw : dict
        kwargs forwarded to ax.axvline for high-symmetry separators.
    tol : float
        Numerical tolerance.

    Returns
    -------
    result : KPathResult
        Whatever lattice.extract_bz_path_data returns (typically KPathResult).
    """
    line_kw   = {} if line_kw   is None else dict(line_kw)
    hsline_kw = {} if hsline_kw is None else dict(hsline_kw)

    C = np.asarray(corr_matrix, float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("corr_matrix must be square (Ns,Ns).")

    r_cart = np.asarray(lattice.rvectors, float)
    k_cart = np.asarray(lattice.kvectors, float)

    Ns = C.shape[0]
    if r_cart.shape[0] != Ns:
        raise ValueError(f"Ns mismatch: corr_matrix is {Ns}, lattice.rvectors is {r_cart.shape[0]}")

    # Use only in-plane components for 2D plotting/pathing
    r2 = r_cart[:, :2]
    k2 = k_cart[:, :2]

    # Phase matrix P_{k,i} = exp(-i k·r_i)
    P  = np.exp(-1j * (k2 @ r2.T))                  # (Nk,Ns)
    PC = P @ C                                      # (Nk,Ns)
    Sk = np.real((PC * np.conjugate(P)).sum(axis=1) / Ns)  # (Nk,)

    # Fractional k coords are needed for robust path extraction.
    # Your lattice.calculate_k_vectors sets kvectors_frac; use it if present.
    if hasattr(lattice, "kvectors_frac"):
        k_frac = np.asarray(lattice.kvectors_frac, float)
    else:
        # Fallback: approximate fractional coordinates by projecting onto reciprocal basis.
        # This is less robust; better to ensure lattice.calculate_k_vectors sets kvectors_frac.
        B = np.asarray(lattice.bvec, float)[:2, :2]       # (2,2) from b1,b2
        k_frac = np.zeros((k_cart.shape[0], 3), float)
        k_frac[:, :2] = (k2 @ np.linalg.inv(B).T)

    # Extract path data via your lattice API.
    # values must be shape (Nk, n_bands); treat Sk as single "band".
    values = Sk[:, None]                                  # (Nk,1)
    result = lattice.extract_bz_path_data(
        k_vectors      = k_cart,
        k_vectors_frac = k_frac,
        values         = values,
        path           = path,
        points_per_seg = points_per_seg,
        return_result  = True
    )

    # result.values likely shape (Npath,1)
    y = np.asarray(result.values).reshape(-1)
    x = np.asarray(result.k_dist).reshape(-1)

    ax.plot(x, y, **line_kw)
    ax.set_ylabel(value_label)
    ax.set_xlim(x.min(), x.max())

    # Add high-symmetry separators + ticks
    try:
        # Your KPathResult docstring suggests label_positions/label_texts exist.
        xs = np.asarray(result.label_positions)
        ls = list(result.label_texts)

        for xv in xs:
            ax.axvline(xv, **({"ls": "--", "lw": 1.0, "alpha": 0.35} | hsline_kw))

        ax.set_xticks(xs)
        ax.set_xticklabels(ls)
    except Exception:
        pass

    ax.grid(alpha=0.25)
    return result

def plot_correlation_grid(
                        directory           : str,
                        param_name          : str,
                        lattice             : 'Lattice',
                        *,
                        x_parameters        : list,
                        y_parameters        : list,
                        x_param             : str       = 'J',
                        y_param             : str       = 'hx',
                        mode                : str       = 'lattice',
                        filters             = None,
                        state_idx           : int       = 0,
                        ref_site_idx        : int       = 0,
                        figsize_per_panel   : tuple     = (4, 3.5),
                        cmap                : str       = 'RdBu_r',
                        vmin                = None,
                        vmax                = None,
                        title               : str       = '',
                        # data extraction
                        param_fun           : callable  = lambda r, param_name: r.get(param_name, []),
                        param_labels        : dict      = {},
                        post_process_func   = None,
                        **kwargs):
    r"""
    Uses ED results to plot correlation matrices in a grid defined by two parameters.
    If lattice is provided, can plot in lattice geometry or in k-space (BZ map).

    Modes
    -----
    mode = 'matrix'   : show full C_{ij} matrix (imshow).
    mode = 'lattice'  : show real-space correlations from a reference site (1D line / 2D map).
    mode = 'kspace'   : show S(k) in the Brillouin zone (reference site ignored).

    Notes on k-space
    ---------------
    Uses the standard site-based structure factor:
        S(k) = (1/Ns) sum_{i,j} C_{ij} exp[-i k·(r_i - r_j)]
    where r_i are full site positions including basis offsets (so multi-site unit cells are handled).

    kwargs (useful)
    ---------------
    logger                  : optional logger
    ks_grid_n               : int, grid resolution for interpolation (default 220)
    ks_interp               : 'linear'|'cubic'|'nearest' (default 'linear')
    ks_show_points          : bool, overlay discrete k points (default True)
    ks_point_size           : float (default 10)
    ks_alpha_points         : float (default 0.35)
    ks_draw_bz              : bool, try to draw BZ outline (default True)
    ks_label_hs             : bool, try to label Gamma/K/M if available (default True)
    ks_mask_outside_bz      : bool, try to mask outside BZ polygon if available (default True)
    auto_vlim_kspace        : bool, compute vmin/vmax from S(k) over selected panels (default True)
    """

    lx, ly, lz, Ns  = lattice.lx, lattice.ly, lattice.lz, lattice.Ns
    results         = load_results(
                        data_dir=directory,
                        filters=filters,
                        lx=lx, ly=ly, lz=lz, Ns=Ns,
                        logger=kwargs.get('logger', None),
                        post_process_func=post_process_func
                    )
    if not results:
        return None, None

    _, _, unique_x, unique_y = PlotEDHelpers.extract_parameter_arrays(results, x_param, y_param)

    unique_x        = np.array([v for v in unique_x if any(abs(v - p) < 1e-5 for p in x_parameters)])
    unique_y        = np.array([v for v in unique_y if any(abs(v - p) < 1e-5 for p in y_parameters)])
    unique_x        = np.sort(unique_x)
    unique_y        = np.sort(unique_y)
    unique_y_plot   = unique_y[::-1]

    n_rows, n_cols  = len(unique_y_plot), len(unique_x)
    if n_rows == 0 or n_cols == 0:
        return None, None

    fig, axes, _, _ = PlotEDHelpers.create_subplot_grid(
                        n_panels            = n_rows * n_cols,
                        max_cols            = n_cols,
                        figsize_per_panel   = figsize_per_panel,
                        sharex              = True,
                        sharey              = True
                    )
    axes_grid = np.array(axes).reshape((n_rows, n_cols))

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
        return plot_bz_path_from_corr(
            ax,
            lattice,
            corr_matrix,
            path            = kwargs.get('kpath',           None),   # None -> default lattice path
            points_per_seg  = kwargs.get('kpath_pps',       60),
            value_label     = kwargs.get('kpath_ylabel',    r"$S(\mathbf{k})$"),
            line_kw         = kwargs.get('kpath_line_kw',   {"lw": 2.0}),
            hsline_kw       = kwargs.get('kpath_hs_kw',     {"color": "k"})
        )
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
    k_cart      = None         # (Nk,3)
    k2          = None         # (Nk,2)
    r2          = None         # (Ns,2)
    phase       = None         # (Nk,Ns)
    phase_conj  = None         # (Nk,Ns)
    Ns_lat      = None
    Nk          = None

    ks_grid_n           = int(kwargs.get('ks_grid_n', 220))
    ks_interp           = kwargs.get('ks_interp', 'linear')
    ks_show_points      = bool(kwargs.get('ks_show_points', True))
    ks_point_size       = float(kwargs.get('ks_point_size', 10))
    ks_alpha_points     = float(kwargs.get('ks_alpha_points', 0.35))
    ks_draw_bz          = bool(kwargs.get('ks_draw_bz', True))
    ks_label_hs         = bool(kwargs.get('ks_label_hs', True))
    ks_mask_outside_bz  = bool(kwargs.get('ks_mask_outside_bz', True))
    auto_vlim_kspace    = bool(kwargs.get('auto_vlim_kspace', True))

    # Helpers for BZ: we try to draw outline / labels if your lattice.plot exists.
    def _try_draw_bz(ax):
        if not ks_draw_bz:
            return None
        try:
            if hasattr(lattice, "plot") and hasattr(lattice.plot, "brillouin_zone"):
                return lattice.plot.brillouin_zone(ax=ax)
        except Exception:
            pass
        return None

    def _try_label_hs(ax):
        if not ks_label_hs:
            return
        try:
            hs = lattice.high_symmetry_points() if hasattr(lattice, "high_symmetry_points") else None
            if hs is None:
                return

            # Convert fractional coords (f1,f2,f3) to cart using bvec (k1,k2,k3).
            bvec = np.asarray(lattice.bvec, float)  # (3,3)
            def frac_to_cart(frac):
                frac = np.asarray(frac, float).reshape(3)
                return frac[0] * bvec[0] + frac[1] * bvec[1] + frac[2] * bvec[2]

            # Common names
            pts = []
            if hasattr(hs, "Gamma"): pts.append(("Γ", frac_to_cart(hs.Gamma.frac_coords)))
            if hasattr(hs, "K"):     pts.append(("K", frac_to_cart(hs.K.frac_coords)))
            if hasattr(hs, "M"):     pts.append(("M", frac_to_cart(hs.M.frac_coords)))
            if not pts:
                return

            for lab, kc in pts:
                ax.text(kc[0], kc[1], lab, ha='center', va='center',
                        color=kwargs.get('hs_color', 'white'),
                        fontsize=kwargs.get('hs_fs', 10),
                        weight='bold',
                        zorder=30)
        except Exception:
            return

    # If kspace: build phase once.
    if plot_mode == 'kspace' and lattice is not None:
        try:
            k_cart  = np.asarray(lattice.kvectors, float)   # (Nk,3) usually Nk = Nc
            r_cart  = np.asarray(lattice.rvectors, float)   # (Ns,3) Ns = Nc*Nb

            k2      = k_cart[:, :2]
            r2      = r_cart[:, :2]

            phase       = np.exp(-1j * (k2 @ r2.T))         # (Nk,Ns)
            phase_conj  = np.conjugate(phase)

            Ns_lat  = r2.shape[0]
            Nk      = k2.shape[0]
        except Exception:
            plot_mode = 'matrix'

    # --------------------------------------------------------------
    # determine vmin/vmax
    # --------------------------------------------------------------
    def _extract_corr_matrix(res):
        ns          = int(res.params.get('Ns', Ns))
        data_root   = param_fun(res, param_name)
        if data_root is None or getattr(data_root, "size", 0) == 0:
            return None

        if data_root.ndim == 3:
            return np.asarray(data_root[:, :, state_idx], float)
        if data_root.ndim == 2 and data_root.shape == (ns, ns):
            return np.asarray(data_root, float)
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
            sk_min, sk_max = +np.inf, -np.inf
            for r in _selected_results():
                C = _extract_corr_matrix(r)
                if C is None:
                    continue
                if C.shape[0] != Ns_lat:
                    continue
                tmp = phase @ C
                Sk  = np.real((tmp * phase_conj).sum(axis=1) / C.shape[0])
                if np.all(np.isnan(Sk)):
                    continue
                sk_min = min(sk_min, float(np.nanmin(Sk)))
                sk_max = max(sk_max, float(np.nanmax(Sk)))
            if np.isfinite(sk_min) and np.isfinite(sk_max):
                if vmin is None: vmin = sk_min
                if vmax is None: vmax = sk_max
            else:
                # fallback
                if vmin is None: vmin = -1.0
                if vmax is None: vmax = +1.0
        else:
            vmin_n, vmax_n  = PlotEDHelpers.determine_vmax_vmin(results, param_name, param_fun, nstates=1)
            if vmin is None: vmin = vmin_n
            if vmax is None: vmax = vmax_n

    # shared color normalization
    _, _, _, mappable = Plotter.get_colormap(values=[vmin, vmax], cmap=cmap, get_mappable=True)

    # axis label logic
    def _labelcond(i, j):
        return {'x': i == n_rows - 1, 'y': j == 0}

    def _set_axis_labels(ax, i, j):
        if plot_mode == 'matrix':
            Plotter.set_ax_params(
                ax,
                xlabel      = 'j',
                ylabel      = 'i',
                labelCond   = _labelcond(i, j),
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
                    xlabel      = r'$x$',
                    ylabel      = r'$C_{ij}$',
                    labelCond   = _labelcond(i, j),
                    labelPos    = {'x': 'bottom', 'y': 'left'},
                    tickPos     = {'x': 'bottom', 'y': 'left'},
                )
            else:
                ax.axis('off')
            return

        if plot_mode == 'kspace':
            # mimic the "BZ panels" style: only outer labels
            if _labelcond(i, j).get('x', False):
                ax.set_xlabel(r'$k_x$')
            if _labelcond(i, j).get('y', False):
                ax.set_ylabel(r'$k_y$')

    # --------------------------------------------------------------
    # k-space interpolation grid (for smooth imshow-like maps)
    # --------------------------------------------------------------
    def _kspace_grid_and_mask():
        if k2 is None:
            return None

        kx_min, kx_max = float(np.min(k2[:, 0])), float(np.max(k2[:, 0]))
        ky_min, ky_max = float(np.min(k2[:, 1])), float(np.max(k2[:, 1]))

        pad_x = 0.05 * (kx_max - kx_min + 1e-12)
        pad_y = 0.05 * (ky_max - ky_min + 1e-12)

        kx = np.linspace(kx_min - pad_x, kx_max + pad_x, ks_grid_n)
        ky = np.linspace(ky_min - pad_y, ky_max + pad_y, ks_grid_n)
        KX, KY = np.meshgrid(kx, ky)

        # Optional polygon mask from lattice.plot.brillouin_zone is not guaranteed.
        # We do a robust fallback: mask points far from the discrete k-cloud.
        # This avoids filling the whole rectangle when k points live on a shape.
        mask = None
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(k2)
            d, _ = tree.query(np.column_stack([KX.ravel(), KY.ravel()]), k=1)
            d = d.reshape(KX.shape)

            # heuristic: nearest-neighbor distance scale
            d0 = np.median(d[np.isfinite(d)])
            dmax = float(kwargs.get('ks_blob_radius', 2.5 * d0))
            mask = (d > dmax) if ks_mask_outside_bz else None
        except Exception:
            mask = None

        return (kx, ky, KX, KY, mask)

    k_grid_pack = _kspace_grid_and_mask() if plot_mode == 'kspace' else None

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
                if abs(rx - x_val) < 1e-5 and abs(ry - y_val) < 1e-5:
                    subset.append(r)

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
                    interpolation   = 'nearest',
                    origin          = 'lower'
                )

            elif plot_mode == 'lattice':
                if positions is None:
                    ax.imshow(
                        corr_matrix,
                        cmap            = cmap,
                        vmin            = vmin,
                        vmax            = vmax,
                        interpolation   = 'nearest',
                        origin          = 'lower'
                    )
                else:
                    ref_site_idx_use = ref_site_idx if (0 <= ref_site_idx < positions.shape[0]) else 0
                    site_vals        = corr_matrix[ref_site_idx_use, :]

                    # 1D: correlation vs displacement coordinate
                    if d_dim == 1 or (positions.shape[1] == 1):
                        x   = positions[:, 0] - positions[ref_site_idx_use, 0]
                        idx = np.argsort(x)
                        Plotter.plot(ax, x[idx], site_vals[idx], marker='o', lw=1.5)
                        Plotter.vline(ax, 0.0, ls='--', lw=1, color='k', alpha=0.4)
                        Plotter.grid(ax, alpha=0.3)

                    # 2D: smooth “blob” field + point overlay
                    else:
                        pos = positions[:, :2] - positions[ref_site_idx_use, :2]

                        x_min, x_max    = pos[:, 0].min(), pos[:, 0].max()
                        y_min, y_max    = pos[:, 1].min(), pos[:, 1].max()
                        pad_x           = 0.15 * (x_max - x_min + 1e-12)
                        pad_y           = 0.15 * (y_max - y_min + 1e-12)

                        xx              = np.linspace(x_min - pad_x, x_max + pad_x, 220)
                        yy              = np.linspace(y_min - pad_y, y_max + pad_y, 220)
                        X, Y            = np.meshgrid(xx, yy)

                        # Interpolate but keep it local (avoid filling big empty regions)
                        Z = None
                        try:
                            from scipy.interpolate import griddata
                            Z = griddata(pos, site_vals, (X, Y), method=kwargs.get('rs_interp', 'linear'))
                        except Exception:
                            Z = None

                        if Z is None:
                            d2  = (X[..., None] - pos[:, 0])**2 + (Y[..., None] - pos[:, 1])**2
                            Z   = site_vals[np.argmin(d2, axis=2)]

                        # Local-support mask so you get blobs, not a filled convex hull
                        try:
                            from scipy.spatial import cKDTree
                            tree = cKDTree(pos)
                            d, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]), k=1)
                            d    = d.reshape(X.shape)
                            d0   = np.median(d[np.isfinite(d)])
                            dmax = float(kwargs.get('rs_blob_radius', 2.5 * d0))
                            Z    = np.where(d <= dmax, Z, np.nan)
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
                        C   = np.asarray(corr_matrix, float)
                        Ns0 = C.shape[0]

                        tmp = phase @ C
                        Sk  = np.real((tmp * phase_conj).sum(axis=1) / Ns0)  # (Nk,)

                        # Smooth BZ map like your reference figure: interpolate onto a regular grid,
                        # then overlay the BZ outline + HS labels if available.
                        im = None
                        if k_grid_pack is not None:
                            
                            from QES.general_python.lattices.tools.lattice_kspace import ws_bz_mask
                            
                            (kx, ky, KX, KY, mask) = k_grid_pack
                            b1 = lattice.k1   # or lattice.b1, but ensure it's the reciprocal primitive vector
                            b2 = lattice.k2

                            inside = ws_bz_mask(KX, KY, b1, b2, shells=1)  # shells=1 usually enough in 2D
                            Z = np.where(inside, Z, np.nan)
                            Z = None
                            try:
                                from scipy.interpolate import griddata
                                Z = griddata(k2, Sk, (KX, KY), method=ks_interp, rescale=False)
                            except Exception:
                                Z = None

                            if Z is None:
                                # nearest fallback
                                try:
                                    from scipy.spatial import cKDTree
                                    tree = cKDTree(k2)
                                    _, idx = tree.query(np.column_stack([KX.ravel(), KY.ravel()]), k=1)
                                    Z = Sk[idx].reshape(KX.shape)
                                except Exception:
                                    Z = np.full_like(KX, np.nan, dtype=float)

                            if mask is not None:
                                Z = np.where(mask, np.nan, Z)

                            im = ax.imshow(
                                Z,
                                extent  = (kx[0], kx[-1], ky[0], ky[-1]),
                                origin  = 'lower',
                                cmap    = cmap,
                                vmin    = vmin,
                                vmax    = vmax,
                                interpolation=kwargs.get('ks_im_interp', 'bilinear'),
                                alpha   = 0.98
                            )

                        if ks_show_points:
                            Plotter.scatter(
                                ax,
                                k2[:, 0], k2[:, 1],
                                c           = Sk,
                                cmap        = cmap,
                                vmin        = vmin,
                                vmax        = vmax,
                                s           = ks_point_size,
                                edgecolor   = 'none',
                                alpha       = ks_alpha_points,
                                zorder      = 15
                            )

                        _try_draw_bz(ax)
                        _try_label_hs(ax)

                        ax.set_aspect('equal')
                        ax.tick_params(direction='out', length=3, width=0.8)

            # ----------------------------------------------------------
            # per-panel annotations
            # ----------------------------------------------------------
            Plotter.set_annotate_letter(
                ax,
                iter        = 0,
                x           = kwargs.get('letter_x', 0.05),
                y           = kwargs.get('letter_y', 0.9),
                boxaround   = False,
                color       = kwargs.get('text_color', 'black'),
                addit       = rf' {param_labels.get(x_param, x_param)}$={x_val:.2g}$, '
                              rf'{param_labels.get(y_param, y_param)}$={y_val:.2g}$'
            )

            if plot_mode != 'lattice' or (positions is None or d_dim == 1):
                Plotter.grid(ax, alpha=0.25)

            Plotter.set_tickparams(ax)
            _set_axis_labels(ax, ii, jj)

    # --------------------------------------------------------------
    # colorbar + title
    # --------------------------------------------------------------
    cbar_scale  = kwargs.pop('cbar_scale', 'linear')
    vmin_cb     = vmin if cbar_scale != 'linear' else None

    Plotter.add_colorbar(
        fig                 = fig,
        pos                 = kwargs.get('cbar_pos', [0.92, 0.15, 0.02, 0.7]),
        mappable            = mappable,
        cmap                = cmap,
        vmin                = vmin_cb,
        vmax                = vmax,
        label               = kwargs.get('param_label', param_labels.get(param_name, param_name)),
        format              = '%.2f',
        scale               = cbar_scale,
        remove_pdf_lines    = True
    )

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
        _, _, _, unique_p       = PlotEDHelpers.extract_parameter_arrays(results, x_param, p_param)
        p_values                = sorted(unique_p)
        if len(p_values) > 9 and kwargs.get('limit_panels', True):
            p_values = p_values[:9]

    n_panels                    = len(p_values)
    fig, axes, n_rows, n_cols   = PlotEDHelpers.create_subplot_grid(n_panels, max_cols=3, figsize_per_panel=figsize_per_panel, sharex=True, sharey=True)
    if hasattr(fig, 'set_constrained_layout'):
        fig.set_constrained_layout(True)
    
    axes                        = axes.flatten()
    labelcond                   = lambda idx: {'x': idx // n_cols == n_rows - 1, 'y': idx % n_cols == 0}
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

        x_vals, sort_idx    = PlotEDHelpers.sort_results_by_param(panel_results, x_param)
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
        Plotter.set_ax_params(ax, xlabel=xlabel_str, ylabel=ylabel_str, labelCond=labelcond(idx), xlim=xlim, ylim=ylim, yscale=kwargs.get('yscale', 'linear'), xscale=kwargs.get('xscale', 'linear'))
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
    _, _, _, unique_series = PlotEDHelpers.extract_parameter_arrays(results, scaling_param, series_param)
    unique_series = sorted(unique_series)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = plt.get_cmap(kwargs.get('cmap', 'viridis'))
    norm = plt.Normalize(min(unique_series), max(unique_series))
    
    for s_val in unique_series:
        # Filter for this series
        subset = [r for r in results if abs(r.params.get(series_param, -999) - s_val) < 1e-5]
        if not subset: continue
        
        x_vals, sort_idx = PlotEDHelpers.sort_results_by_param(subset, scaling_param)
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

    PlotEDHelpers.savefig(fig, directory, param_name, scaling_param, series_param, suffix='_scaling')
    return fig, ax

# ------------------------------------------------------------------
#! End of file
# ------------------------------------------------------------------
