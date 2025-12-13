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
import  numpy as np
import  matplotlib.pyplot as plt

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
        param_lbl           : Optional[str]     = None,
        # plot limits
        xlim                : Optional[tuple]   = None,
        ylim                : Optional[tuple]   = None,
        # colormap
        vmin                = None, 
        vmax                = None, 
        # plot settings
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
    
    xlabel                              = param_labels.get(x_param, x_param) if len(unique_x) > 1 else param_labels.get(y_param, y_param)
    ylabel                              = param_labels.get(y_param, y_param) if len(unique_y) > 1 else param_labels.get(x_param, x_param)
    
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
                x   = r.params.get(param_plot, 0.0)
                
                if len(y) > ii:
                    x_plot.append(x)
                    y_plot.append(y[ii])
            
            sort_idx    = np.argsort(x_plot)
            x_plot      = np.array(x_plot)[sort_idx]
            y_plot      = np.array(y_plot)[sort_idx]
            Plotter.plot(ax, x=x_plot, y=y_plot, ls='-', marker='o', ms=4, color=getcolor(ii), label=rf'$|\Psi_{{{ii}}}\rangle$')
        
        Plotter.set_ax_params(ax, xlabel=label, ylabel=param_lbl if param_lbl is not None else param_labels.get(param_name, param_name), xlim=xlim, ylim=ylim, yscale=kwargs.get('yscale', 'linear'), xscale=kwargs.get('xscale', 'linear'),)
        Plotter.set_annotate_letter(ax, iter=0, x=letter_x, y=letter_y, boxaround=False)
        Plotter.set_legend(ax, loc=kwargs.get('legend_loc', 'lower right'), fontsize=kwargs.get('legend_fontsize', 8))
        Plotter.grid(ax, alpha=0.3)

    # b) Colormap - Both X and Y Variation
    else:
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
                x_val   = r.params.get(x_param, 0.0)
                
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
                Zi = griddata((x_scatter, y_scatter), z_scatter, (Xi, Yi), method='linear', rescale=True)
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
        cbar_pos = kwargs.get('cbar_pos', [0.92, 0.15, 0.02, 0.7])
        Plotter.add_colorbar(fig=fig, pos=cbar_pos, mappable=mappable, cmap=cmap, scale=scale_type,
                    vmin=vmin, vmax=vmax, label=param_labels.get(param_name, param_name) if param_lbl is None else param_lbl,
                    extend=None, format='%.1e' if scale_type == 'log' else '%.2f', remove_pdf_lines=True
                )
        
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

    lx, ly, lz, Ns      = lattice.lx, lattice.ly, lattice.lz, lattice.Ns
    results             = load_results(data_dir=directory, filters=filters, lx=lx, ly=ly, lz=lz, Ns=Ns, logger=kwargs.get('logger', None), post_process_func=post_process_func)
    
    if not results:
        return None, None

    _, _, unique_x, unique_y = PlotEDHelpers.extract_parameter_arrays(results, x_param, y_param)

    unique_x        = np.array([v for v in unique_x if any(abs(v-p)<1e-5 for p in x_parameters)])
    unique_y        = np.array([v for v in unique_y if any(abs(v-p)<1e-5 for p in y_parameters)])
    unique_x        = np.sort(unique_x)
    unique_y        = np.sort(unique_y)
    unique_y_plot   = unique_y[::-1]
    n_rows, n_cols  = len(unique_y_plot), len(unique_x)
    if n_rows == 0 or n_cols == 0:
        return None, None

    fig, axes, _, _ = PlotEDHelpers.create_subplot_grid(
                        n_panels            =   n_rows*n_cols,
                        max_cols            =   n_cols,
                        figsize_per_panel   =   figsize_per_panel,
                        sharex              =   True,
                        sharey              =   True
                    )
    axes_grid       = np.array(axes).reshape((n_rows, n_cols))

    if vmin is None or vmax is None:
        vmin_n, vmax_n  = PlotEDHelpers.determine_vmax_vmin(results, param_name, param_fun, nstates=1)
        vmin            = vmin_n if vmin is None else vmin
        vmax            = vmax_n if vmax is None else vmax

    labelcond = lambda i, j: {'x': i == n_rows-1, 'y': j == 0}

    # ------------------------------------------------------------------
    # determine plotting geometry
    # ------------------------------------------------------------------
    plot_mode           = 'matrix'
    positions           = None

    if mode == 'lattice' and lattice is not None:
        try:
            positions       = lattice.rvectors.copy()
            ref_site_idx    = ref_site_idx if ref_site_idx < positions.shape[0] else 0
            d               = positions.shape[1]

            if d == 1:
                plot_mode   = 'lattice_1d'
            else:
                plot_mode   = 'lattice_2d'
        except Exception:
            plot_mode       = 'matrix'

    # ------------------------------------------------------------------
    # colormap handle
    # ------------------------------------------------------------------
    _, _, _, mappable       = Plotter.get_colormap(values=[vmin, vmax], cmap=cmap, get_mappable=True)

    # ------------------------------------------------------------------
    # plotting loop
    # ------------------------------------------------------------------
    for ii, y_val in enumerate(unique_y_plot):
        for jj, x_val in enumerate(unique_x):

            ax      = axes_grid[ii, jj]
            subset  = []

            for r in results:
                rx = r.params.get(x_param, np.nan)
                ry = r.params.get(y_param, np.nan)
                if abs(rx-x_val)<1e-5 and abs(ry-y_val)<1e-5:
                    subset.append(r)

            if not subset:
                ax.text(0.5,0.5,'N/A',ha='center',va='center',transform=ax.transAxes)
                continue

            result      = subset[0]
            ns          = int(result.params.get('Ns', positions.shape[0] if positions is not None else 8))
            corr_matrix = np.full((ns, ns), np.nan)
            data_root   = param_fun(result, param_name)

            if data_root is None or data_root.size == 0:
                continue

            if data_root.ndim == 3:
                corr_matrix = data_root[:,:,state_idx]
            elif data_root.ndim == 2 and data_root.shape == (ns,ns):
                corr_matrix = data_root

            site_vals = corr_matrix[ref_site_idx, :]

            # ----------------------------------------------------------
            # plotting modes
            # ----------------------------------------------------------
            if plot_mode == 'matrix':

                im = ax.imshow(
                        corr_matrix,
                        cmap            =   cmap,
                        vmin            =   vmin,
                        vmax            =   vmax,
                        interpolation   =   'nearest',
                        origin          =   'lower'
                    )
                if ii == n_rows-1: ax.set_xlabel('j')
                if jj == 0:        ax.set_ylabel('i')

            elif plot_mode == 'lattice_1d':

                x   = positions[:,0] - positions[ref_site_idx,0]
                idx = np.argsort(x)

                Plotter.plot(ax, x[idx], site_vals[idx], marker='o', lw=1.5)
                Plotter.vline(ax, 0.0, ls='--', lw=1, color='k', alpha=0.4)
                ax.set_ylabel(r'$C_{ij}$')
                ax.grid(alpha=0.3)

            elif plot_mode == 'lattice_2d':

                pos             = positions[:,:2] - positions[ref_site_idx,:2]
                x_min, x_max    = pos[:,0].min(), pos[:,0].max()
                y_min, y_max    = pos[:,1].min(), pos[:,1].max()
                pad_x           = 0.15*(x_max-x_min+1e-12)
                pad_y           = 0.15*(y_max-y_min+1e-12)

                xx              = np.linspace(x_min-pad_x, x_max+pad_x, 200)
                yy              = np.linspace(y_min-pad_y, y_max+pad_y, 200)
                X, Y            = np.meshgrid(xx, yy)

                try:
                    from scipy.interpolate import griddata
                    Z = griddata(pos, site_vals, (X,Y), method='linear')
                except Exception:
                    d2 = (X[...,None]-pos[:,0])**2 + (Y[...,None]-pos[:,1])**2
                    Z  = site_vals[np.argmin(d2, axis=2)]

                im = ax.imshow(Z,
                        extent=(xx[0],xx[-1],yy[0],yy[-1]),
                        origin='lower',
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        alpha=0.9
                    )

                Plotter.scatter(ax,
                        pos[:,0], pos[:,1],
                        c           =   site_vals,
                        cmap        =   cmap,
                        vmin        =   vmin,
                        vmax        =   vmax,
                        edgecolor   =   'k',
                        linewidths  =   0.5,
                        s           =   60,
                        zorder      =   10
                    )

                Plotter.scatter(ax, [0], [0], marker='*', s=120, c='yellow', edgecolor='k', zorder=11)
                ax.set_aspect('equal')
                ax.axis('off')

            # ----------------------------------------------------------
            # annotations & cosmetics
            # ----------------------------------------------------------
            
            Plotter.set_annotate_letter(
                ax, 
                iter        =   0,
                x           =   kwargs.get('letter_x',0.05),
                y           =   kwargs.get('letter_y',0.9),
                boxaround   =   False,
                color       =   kwargs.get('text_color','black'),
                addit       =   rf' {param_labels.get(x_param,x_param)}$={x_val:.2g}$, '
                                rf'{param_labels.get(y_param,y_param)}$={y_val:.2g}$'
            )

            Plotter.set_tickparams(ax)
            Plotter.grid(ax, alpha=0.3)

            Plotter.set_ax_params(
                ax,
                xlabel      =   'i' if plot_mode=='matrix' else '',
                ylabel      =   'j' if plot_mode=='matrix' else '',
                labelCond   =   labelcond(ii,jj),
                labelPos    =   {'x':'bottom','y':'left'},
                tickPos     =   {'x':'bottom','y':'left'}
            )

    # ------------------------------------------------------------------
    # colorbar, title, save
    # ------------------------------------------------------------------
    Plotter.add_colorbar(
        fig                 =   fig,
        pos                 =   kwargs.get('cbar_pos',[0.92,0.15,0.02,0.7]),
        mappable            =   mappable,
        cmap                =   cmap,
        vmin                =   vmin,
        vmax                =   vmax,
        label               =   kwargs.get('param_label', param_labels.get(param_name, param_name)),
        format              =   '%.2f',
        scale               =   kwargs.get('cbar_scale','linear'),
        remove_pdf_lines    =   True
    )

    fig.suptitle(title, fontsize=kwargs.get('suptitle_fontsize',14))
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
        _, _, _, unique_p   = PlotEDHelpers.extract_parameter_arrays(results, x_param, p_param)
        p_values            = sorted(unique_p)
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
                ax.plot(x_arr, y_arr, color=getcolor(state_i), marker='o', markersize=3, ls='-')

        p_label = param_labels.get(p_param, p_param)
        Plotter.set_annotate_letter(ax, iter=idx, x=0.05, y=0.9, addit=rf' {p_label}$={p_val:.2g}$', boxaround=False)
        Plotter.set_ax_params(ax, xlabel=xlabel_str, ylabel=ylabel_str, labelCond=labelcond(idx), xlim=xlim, ylim=ylim, yscale=kwargs.get('yscale', 'linear'), xscale=kwargs.get('xscale', 'linear'))
        Plotter.set_tickparams(ax)
        Plotter.grid(ax, alpha=0.3)

    Plotter.hide_unused_panels(axes, n_panels)
    
    cbar_pos = kwargs.get('cbar_pos', [0.92, 0.2, 0.02, 0.6])
    Plotter.add_colorbar(fig=fig, pos=cbar_pos, mappable=mappable, cmap=cmap, scale='linear', vmin=0, vmax=nstates - 1, label=r'State Index $n$', extend=None, format='%d', remove_pdf_lines=True)

    if title: 
        fig.suptitle(title, fontsize=kwargs.get('suptitle_fontsize', 12))
        
    plt.show()
    return fig, axes

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
