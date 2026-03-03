'''
General plotting helpers for spectral and k-space data.

Framework-agnostic plotting functions for 2D spectral data visualization:
- Spectral function plots A(k,w) in path or grid mode
- Static structure factors S(k,n)
- K-space intensity maps with BZ extension

These functions accept preprocessed numpy arrays and can be used
with any calculation method (ED, DMRG, QMC, etc.).

----------------------------------------------------------------
Author              : Maksymilian Kliczkowski
Date                : 2025-01-15
License             : MIT
----------------------------------------------------------------
'''
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Literal, Any
import numpy as np
from scipy.interpolate  import griddata

try:
    from .config        import PlotStyle, KSpaceConfig, KPathConfig, SpectralConfig
    from .kspace_utils  import label_high_sym_points
except ImportError:
    raise ImportError("Failed to import plotting configurations or k-space utilities.")

if TYPE_CHECKING:
    from general_python.lattices.lattice import Lattice

_KSPACE_CONFIG_KEYS     = {
    "grid_n",
    "interp_method",
    "mask_outside_bz",
    "ws_shells",
    "draw_bz_outline",
    "show_discrete_points",
    "point_size",
    "point_alpha",
    "point_marker",
    "label_high_symmetry",
    "hs_marker_size",
    "hs_marker_color",
    "hs_marker_edge",
    "hs_label_offset_x",
    "hs_label_offset_y",
    "hs_label_fontsize",
    "hs_label_color",
    "blob_radius_factor",
    "imshow_interp",
    "extend_bz",
    "bz_copies",
}
_PLOT_STYLE_KEYS = {
    "cmap",
    "vmin",
    "vmax",
    "fontsize_label",
    "fontsize_tick",
    "fontsize_title",
    "fontsize_legend",
    "fontsize_annotation",
    "fontsize_colorbar",
    "marker",
    "markersize",
    "linewidth",
    "linestyle",
    "alpha",
    "edgecolor",
    "edgewidth",
    "grid_alpha",
    "grid_linestyle",
    "spine_width",
    "figsize_per_panel",
}

# --------------------------------------------------------------
#! Static structure factor S(k, n) plots
# --------------------------------------------------------------

def _flatten_kspace_points(k_values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return flattened 2D k-points together with the structured grid shape."""
    arr = np.asarray(k_values, dtype=float)
    if arr.ndim < 2:
        raise ValueError("k_values must have at least one k-grid axis and one coordinate axis")
    if arr.shape[-1] < 2:
        raise ValueError("k_values must provide at least two Cartesian components")
    return arr.reshape(-1, arr.shape[-1])[:, :2], tuple(int(v) for v in arr.shape[:-1])

def _extract_scalar_kfield(values: np.ndarray, grid_shape: tuple[int, ...], *,
        batch_index     : Optional[Any] = None,
        component_index : Optional[Any] = None,
        value_mode      : Literal['auto', 'real', 'imag', 'abs', 'phase'] = 'auto',
    ) -> np.ndarray:
    """Extract a scalar field on the sampled k-grid from batched/general values."""
    values_arr  = np.asarray(values)
    nk          = int(np.prod(grid_shape))
    grid_axis   = None
    grid_span   = None
    
    # Try to identify the k-grid axes by matching the grid shape as a contiguous block in the values shape
    for start in range(values_arr.ndim - len(grid_shape) + 1):
        if tuple(values_arr.shape[start:start + len(grid_shape)]) == grid_shape:
            grid_axis = start
            grid_span = len(grid_shape)
            break
    if grid_axis is None:
        for axis, axis_size in enumerate(values_arr.shape):
            if int(axis_size) == nk:
                grid_axis = axis
                grid_span = 1
                break
    if grid_axis is None or grid_span is None:
        raise ValueError(f"Could not identify k-grid axes {grid_shape} in intensity with shape {values_arr.shape}.")

    batch_shape     = values_arr.shape[:grid_axis]
    feature_shape   = values_arr.shape[grid_axis + grid_span:]
    values_flat     = values_arr.reshape(batch_shape + (nk,) + feature_shape)

    if len(batch_shape) > 0:
        if batch_index is None:
            batch_index = tuple(0 for _ in batch_shape)
        elif not isinstance(batch_index, tuple):
            batch_index = (batch_index,)
        if len(batch_index) != len(batch_shape):
            raise ValueError(f"batch_index must have length {len(batch_shape)}, got {len(batch_index)}")
        values_flat = values_flat[batch_index + (slice(None),) + tuple(slice(None) for _ in feature_shape)]

    if feature_shape:
        if component_index is None:
            component_index = tuple(0 for _ in feature_shape)
        elif not isinstance(component_index, tuple):
            component_index = (component_index,)
        if len(component_index) != len(feature_shape):
            raise ValueError(f"component_index must have length {len(feature_shape)}, got {len(component_index)}")
        values_flat = values_flat[(slice(None),) + component_index]

    field = np.asarray(values_flat)
    if field.ndim != 1 or field.shape[0] != nk:
        raise ValueError("Could not reduce intensity to a scalar field on the sampled k-grid")

    if value_mode == 'real':
        field       = np.real(field)
    elif value_mode == 'imag':
        field       = np.imag(field)
    elif value_mode == 'abs':
        field       = np.abs(field)
    elif value_mode == 'phase':
        field       = np.angle(field)
    elif value_mode == 'auto':
        real_close  = np.real_if_close(field)
        field       = np.real(real_close) if not np.iscomplexobj(real_close) else np.abs(field)
    else:
        raise ValueError(f"Unknown value_mode '{value_mode}'")

    return np.asarray(field, dtype=float)

def _ws_shifts_2d(lattice: Optional["Lattice"], show_extended_bz: bool, bz_copies: int) -> list[np.ndarray]:
    """Return 2D WS/BZ translation shifts including the origin."""
    
    if lattice is None:
        return [np.zeros(2, dtype=float)]
    if not show_extended_bz or bz_copies <= 0:
        return [np.zeros(2, dtype=float)]
    
    shifts = lattice.wigner_seitz_shifts(copies=(bz_copies, bz_copies), include_origin=True)
    shifts = np.asarray(shifts, dtype=float).reshape(-1, shifts.shape[-1] if np.asarray(shifts).ndim > 1 else 1)
    return [shift[:2] for shift in shifts] if len(shifts) > 0 else [np.zeros(2, dtype=float)]

# --------------------------------------------------------------
#! Config override helpers
# --------------------------------------------------------------

def _apply_kspace_config_overrides(ks_config: KSpaceConfig, overrides: dict[str, Any]) -> None:
    """Apply direct keyword overrides to a KSpaceConfig instance."""
    for key in list(overrides):
        if key in _KSPACE_CONFIG_KEYS:
            setattr(ks_config, key, overrides.pop(key))

def _apply_plot_style_overrides(style: PlotStyle, overrides: dict[str, Any]) -> None:
    """Apply direct keyword overrides to a PlotStyle instance."""
    for key in list(overrides):
        if key in _PLOT_STYLE_KEYS:
            setattr(style, key, overrides.pop(key))

# --------------------------------------------------------------
#! Correlation function k-space transformation and path plotting
# --------------------------------------------------------------

def plot_kspace_path(
        ax,
        lattice         : "Lattice",
        *,
        values          : Optional[np.ndarray] = None,
        k_values        : Optional[np.ndarray] = None,
        k_frac          : Optional[np.ndarray] = None,
        corr_matrix     : Optional[np.ndarray] = None,
        reduction       : Literal["none", "sum", "trace", "mean", "diag"] = "sum",
        norm            : Literal["none", "cell", "site"] = "site",
        # path information can be provided via kpath_config or directly via
        path                                    = None,
        points_per_seg  : Optional[int]         = None,
        value_label     : str                   = r"$S(\mathbf{k})$",
        line_kw         : Optional[dict]        = None,
        hsline_kw       : Optional[dict]        = None,
        style           : Optional[PlotStyle]   = None,
        kpath_config    : Optional[KPathConfig] = None,
        **kwargs
    ):
    r"""
    Plot a scalar k-space quantity along a high-symmetry path.

    Provide either ``values`` together with ``k_values``/``k_frac`` or provide
    ``corr_matrix`` and let the helper build the structure factor first.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    lattice : Lattice
        Lattice object for k-space path utilities
    values : array, optional
        Precomputed k-space values to plot along the path. If not provided, will be computed from ``corr_matrix``.
    k_values : array, optional
        Cartesian k-point coordinates corresponding to the provided values. Required if ``values`` is provided without ``corr_matrix``.
    k_frac : array, optional
        Fractional k-point coordinates corresponding to the provided values. If not provided, will attempt to use ``lattice.kvectors_frac`` if available.
    corr_matrix : array, optional
        Real-space correlation matrix to be transformed into k-space values if ``values`` is not provided.
    reduction : str
        Method to reduce the correlation matrix into a scalar value at each k-point (if ``values`` not provided). Options: "none", "sum", "trace", "mean", "diag".
    norm : str
        Normalization method for the structure factor if computed from ``corr_matrix``. Options: "none", "cell", "site".
    """
    if lattice is None:
        raise ValueError("Lattice information is required to compute k-space path data.")
    
    if style is None:
        style = PlotStyle()
    if kpath_config is None:
        kpath_config = KPathConfig(path=path, points_per_seg=points_per_seg)
    if path is not None:
        kpath_config.path = path
    if points_per_seg is not None:
        kpath_config.points_per_seg = points_per_seg

    # Compute k-space values if not provided, using the correlation matrix and lattice information
    if values is None:
        if corr_matrix is None:
            raise ValueError("Provide either values with k_values/k_frac or corr_matrix.")
        values, k_values, k_frac = lattice.structure_factor(corr_matrix, reduction=reduction, norm=norm)
    elif k_values is None:
        raise ValueError("k_values must be provided when plotting precomputed values.")

    # Use lattice's fractional k-vectors if k_frac not provided, otherwise rely on provided k_frac
    if k_frac is None:
        k_frac = getattr(lattice, "kvectors_frac", None)
        if k_frac is None:
            raise ValueError("k_frac must be provided when lattice.kvectors_frac is unavailable.")

    # Compute the path data using the lattice's BZ path utilities
    result          = lattice.bz_path_data(
                        k_vectors       =   k_values,
                        k_vectors_frac  =   k_frac,
                        values          =   values,
                        path            =   kpath_config.path,
                        points_per_seg  =   kpath_config.points_per_seg,
                        return_result   =   True,
                    )
    path_values     = np.real_if_close(np.asarray(result.values))
    if path_values.ndim != 1:
        raise ValueError(f"plot_kspace_path expects scalar path values, got shape {path_values.shape}.")

    if line_kw is None:
        line_kw = {
            "lw"        : style.linewidth,
            "ls"        : style.linestyle,
            "color"     : kwargs.get("line_color", "C0"),
            "marker"    : style.marker,
            "ms"        : style.markersize,
            "mfc"       : kwargs.get("marker_facecolor", "C0"),
            "mec"       : kwargs.get("marker_edgecolor", "white"),
            "mew"       : kwargs.get("marker_edgewidth", 0.5),
            "alpha"     : style.alpha,
        }
        
    if hsline_kw is None:
        hsline_kw = {
            "color"     : kwargs.get("hline_color", "k"),
            "ls"        : "--",
            "lw"        : 1.0,
            "alpha"     : 0.3,
        }

    # Plot the path values and add vertical lines for high-symmetry points
    ax.plot(result.k_dist, path_values, **line_kw)
    ax.set_ylabel(value_label, fontsize=style.fontsize_label)
    ax.set_xlim(float(np.min(result.k_dist)), float(np.max(result.k_dist)))

    for xv in result.label_positions:
        ax.axvline(xv, **hsline_kw)
        
    ax.set_xticks(result.label_positions)
    ax.set_xticklabels(result.label_texts)
    ax.grid(alpha=style.grid_alpha)
    ax.tick_params(labelsize=style.fontsize_tick)

    return result

def plot_static_structure_factor(
        ax,
        k_values        : np.ndarray,                                       # (Nk,) or (Nk, D) array of k-point coordinates
        intensity       : np.ndarray,
        *,
        state_indices   : Optional[np.ndarray]                              = None, # (Nstates,) array of eigenstate indices to show
        mode            : Literal['kpath', 'grid']                          = 'kpath',
        path_info       : Optional[dict]                                    = None,
        style           : Optional[PlotStyle]                               = None,
        lattice         : Optional["Lattice"]                               = None,
        use_extend      : bool                                              = False,
        extend_copies   : int                                               = 2,
        state_select    : Optional[int]                                     = None,
        value_mode      : Literal['auto', 'real', 'imag', 'abs', 'phase']   = 'auto',
        **kwargs
    ):
    r"""
    Plot static structure factor S(k, n) where n is eigenstate index.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Target axes
    k_values : (Nk,) or (Nk, D) array
        K-point coordinates, where D is the dimensionality corresponding to the states...
    state_indices : (Nstates,) array, optional
        Eigenstate indices to show. If None, all states are shown or no-state if intensity has no state axis.
    intensity : (Nk, Nstates) array
        Structure factor for each k-point and state
    mode : str
        'kpath' or 'grid'
    path_info : dict, optional
        Path information for mode='kpath'
    style : PlotStyle, optional
        Styling configuration
    lattice : Lattice, optional
        For grid mode with extension
    use_extend : bool
        Extend k-space
    extend_copies : int
        BZ copies
    state_select : int, optional
        For ``mode='grid'``, select which state axis entry to plot. Defaults to
        the first state if not provided.
    value_mode : {"auto", "real", "imag", "abs", "phase"}, default="auto"
        How to convert complex-valued k-space data to a scalar intensity.
    **kwargs
        Additional arguments for k-space plotting (e.g., k1_vec, k2_vec if lattice not provided)
    Returns
    -------
    im : plot object
        The plotted image
    """
    if style is None:
        style = PlotStyle()
    
    # Determine the state indices to show
    if state_indices is None:
        if intensity.ndim == 1:
            state_indices = np.array([0])  # No state axis, treat as single state
        elif intensity.ndim == 2:
            state_indices = np.arange(intensity.shape[1])  # Assume second axis is states
        else:
            raise ValueError(f"Intensity array has unexpected number of dimensions {intensity.ndim}.")
    else:
        state_indices = np.asarray(state_indices, dtype=int)
        
    if np.any(state_indices < 0) or np.any(state_indices >= intensity.shape[1]):
        raise ValueError(f"state_indices must be between 0 and {intensity.shape[1]-1}, got {state_indices}.")
    
    # Determine color scale limits
    vmin = style.vmin if style.vmin is not None else np.nanmin(intensity)
    vmax = style.vmax if style.vmax is not None else np.nanmax(intensity)
    
    # Select correct plotting mode
    if mode == 'kpath':
        # Path mode
        if k_values.ndim > 1:
            k_values = k_values[:, 0]
        
        if len(state_indices) == 1:
            im          = ax.plot(k_values, intensity[:, state_indices[0]], color=style.cmap(0.5), marker=style.marker, markersize=style.markersize, alpha=style.alpha)[0]
        else:
            # Create a 2D meshgrid for pcolormesh: K along x-axis, States along y-axis
            K, States   = np.meshgrid(k_values, state_indices, indexing='ij')
            im          = ax.pcolormesh(K, States, intensity,
                            cmap=style.cmap,
                            vmin=vmin,
                            vmax=vmax,
                            shading='auto'
                        )
            
            ax.set_ylabel(kwargs.get('ylabel', 'Eigenstate Index'), fontsize=style.fontsize_label)
            ax.set_xlim(k_values.min(), k_values.max())
        
        # Add high-symmetry separators
        if path_info is not None:
            label_positions = path_info.get('label_positions', [])
            label_texts     = path_info.get('label_texts', [])
            
            if len(label_positions) > 0:
                for xv in label_positions:
                    ax.axvline(xv, color='k', ls='--', lw=1.0, alpha=0.35)
                ax.set_xticks(label_positions)
                ax.set_xticklabels(label_texts)
    
    elif mode == 'grid':
        state_select = 0 if state_select is None else int(state_select)
        im = plot_kspace_intensity(
            ax,
            k_values,
            intensity,
            style=style,
            lattice=lattice,
            show_extended_bz=use_extend,
            bz_copies=extend_copies,
            batch_index=state_select,
            component_index=state_select,
            value_mode=value_mode,
        )
    
    ax.tick_params(labelsize=style.fontsize_tick)
    
    return im

# ==============================================================================
# K-SPACE INTENSITY PLOTS
# ==============================================================================

def plot_kspace_intensity(
        ax,
        k2                  : np.ndarray,
        intensity           : np.ndarray,       # e.g., S(k)
        *,
        style               : Optional[PlotStyle]       = None,
        ks_config           : Optional[KSpaceConfig]    = None,
        lattice             : Optional["Lattice"]       = None,
        show_extended_bz    : bool                      = True,
        bz_copies           : int                       = 2,
        batch_index         : Optional[Any]             = None,
        component_index     : Optional[Any]             = None,
        value_mode          : Literal['auto', 'real', 'imag', 'abs', 'phase'] = 'auto',
        **kwargs
    ):
    """
    Plot intensity(k) in 2D k-space.
    
    Can show:
    - Discrete scatter plot of k-points
    - Interpolated map with optional BZ masking
    - BZ boundary outline
    - High-symmetry point markers in all BZ copies
    - Extended k-space showing multiple Brillouin zones
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    k2 : np.ndarray
        K-point coordinates, either flattened ``(Nk, 2/3)`` or structured
        ``(Lx, Ly, Lz, 3)``.
    intensity : np.ndarray
        Values at each k-point. Can be a scalar field on the sampled grid or a
        batched field with leading batch axes and/or trailing component axes.
    style : PlotStyle, optional
        Styling configuration
    ks_config : KSpaceConfig, optional
        K-space plotting configuration
    lattice : Lattice, optional
        Lattice object for BZ info and high-symmetry points
    show_extended_bz : bool
        If True, replicate k-points to show multiple BZ copies from -2Pi to 2Pi
    bz_copies : int
        Number of BZ copies in each direction (default 2 for -2Pi to 2Pi)
    batch_index : int or tuple, optional
        Index used to choose leading batch axes before plotting a single panel.
    component_index : int or tuple, optional
        Index used to choose trailing component axes after the k-grid axes.
    value_mode : {"auto", "real", "imag", "abs", "phase"}, default="auto"
        How to convert complex-valued k-space data to a scalar intensity.
    **kwargs : dict
        Additional arguments (k1_vec, k2_vec if lattice not provided)
    """
    
    if style is None:
        style               = PlotStyle()
    if ks_config is None:
        ks_config           = KSpaceConfig()
    _apply_plot_style_overrides(style, kwargs)
    _apply_kspace_config_overrides(ks_config, kwargs)
    
    k2_orig, grid_shape     = _flatten_kspace_points(k2)
    intensity_field         = _extract_scalar_kfield(intensity, grid_shape, batch_index=batch_index, component_index=component_index, value_mode=value_mode)
    k1_vec                  = lattice.k1 if lattice is not None else kwargs.get('k1_vec', None)
    k2_vec                  = lattice.k2 if lattice is not None else kwargs.get('k2_vec', None)
    if k1_vec is None or k2_vec is None:
        raise ValueError("Reciprocal lattice vectors k1 and k2 must be provided via lattice or kwargs.")
    k1_vec                  = np.asarray(k1_vec, dtype=float).ravel()[:2]
    k2_vec                  = np.asarray(k2_vec, dtype=float).ravel()[:2]
        
    # Determine color scale
    vmin                    = style.vmin if style.vmin is not None else np.nanmin(intensity_field)
    vmax                    = style.vmax if style.vmax is not None else np.nanmax(intensity_field)
    
    if show_extended_bz:
        if lattice is not None:
            final_k, final_intensity    = lattice.wigner_seitz_extend(k_points=k2_orig, data=intensity_field, copies=(bz_copies, bz_copies))
        else:
            from ...lattices.tools.lattice_kspace import extend_kspace_data
            final_k, final_intensity    = extend_kspace_data(k2_orig, intensity_field, b1=k1_vec, b2=k2_vec, nx=bz_copies, ny=bz_copies)
    else:
        final_k, final_intensity        = k2_orig, intensity_field

    shifts = _ws_shifts_2d(lattice, show_extended_bz, bz_copies)
        
    # Interpolated background (optional)
    if ks_config.grid_n > 0:
        mins    = final_k.min(axis=0)
        maxs    = final_k.max(axis=0)
        span    = np.maximum(maxs - mins, 1.0)
        pad     = 0.08 * span
        KX, KY  = np.meshgrid(
            np.linspace(mins[0] - pad[0], maxs[0] + pad[0], ks_config.grid_n),
            np.linspace(mins[1] - pad[1], maxs[1] + pad[1], ks_config.grid_n)
        )
        
        try:
            Z = griddata(final_k, final_intensity, (KX, KY), method=ks_config.interp_method)
            if ks_config.mask_outside_bz and lattice is not None:
                mask = np.zeros_like(KX, dtype=bool)
                for shift in shifts:
                    mask |= lattice.wigner_seitz_mask(KX - shift[0], KY - shift[1], shells=ks_config.ws_shells)
                Z = np.where(mask, Z, np.nan)
            im = ax.imshow(
                Z,
                extent          =   (KX.min(), KX.max(), KY.min(), KY.max()),
                origin          =   'lower',
                cmap            =   style.cmap,
                vmin            =   vmin,
                vmax            =   vmax,
                interpolation   =   ks_config.imshow_interp,
                alpha           =   0.9
            )
        except Exception as e:
            raise RuntimeError(f"{e}")
    else:
        im = None
        
    # Discrete points overlay
    if ks_config.show_discrete_points:
        im = ax.scatter(
            final_k[:, 0], final_k[:, 1],
            c           =   final_intensity,
            cmap        =   style.cmap,
            vmin        =   vmin,
            vmax        =   vmax,
            s           =   ks_config.point_size,
            alpha       =   ks_config.point_alpha,
            edgecolors  =   'none',
            zorder      =   120
        )
    if ks_config.draw_bz_outline and lattice is not None:
        mins    = final_k.min(axis=0)
        maxs    = final_k.max(axis=0)
        span    = np.maximum(maxs - mins, 1.0)
        pad     = 0.08 * span
        GX, GY = np.meshgrid(
            np.linspace(mins[0] - pad[0], maxs[0] + pad[0], ks_config.grid_n),
            np.linspace(mins[1] - pad[1], maxs[1] + pad[1], ks_config.grid_n)
        )
        for shift in shifts:
            mask = lattice.wigner_seitz_mask(GX - shift[0], GY - shift[1], shells=ks_config.ws_shells)
            ax.contour(GX, GY, mask.astype(float), levels=[0.5], colors=['0.35'], linewidths=1.0, zorder=40)
    
    # Label high-symmetry points in all BZ copies
    if ks_config.label_high_symmetry and lattice is not None:
        try:
            label_high_sym_points(ax, lattice, bz_copies if show_extended_bz else 0, show_labels=True, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to label high-symmetry points: {e}")
    
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$', fontsize=style.fontsize_label)
    ax.set_ylabel(r'$k_y$', fontsize=style.fontsize_label)
    ax.tick_params(labelsize=style.fontsize_tick)

    mins    = final_k.min(axis=0)
    maxs    = final_k.max(axis=0)
    span    = np.maximum(maxs - mins, 1e-12)
    pad     = 0.08 * span
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])

    return im

# ==============================================================================
#! EOF
# ==============================================================================
