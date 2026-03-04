r'''
Axis-level plotting helpers for correlations, k-space fields, and spectral data.

The helpers in this module operate on already prepared arrays. They accept a
provided axis or create a single panel when no axis is supplied. Loader or
parameter-grid orchestration belongs in ``ed.py``.

Direct keyword overrides are supported for the dataclass configs:
- ``PlotStyle`` fields, e.g. ``cmap='magma'``, ``linewidth=2.0``
- ``KSpaceConfig`` fields, e.g. ``grid_n=260``, ``point_size=24``
- ``KPathConfig`` fields, e.g. ``points_per_seg=80``, ``separator_style={...}``
- ``SpectralConfig`` fields, e.g. ``vmin_omega=0.0``, ``omega_value=0.3``

If a keyword matches one of those config fields, it is applied to the
corresponding config instance before plotting. Plot-specific keywords that are
not config fields are still forwarded to the local plotting logic.

------------------------------------------------------------------------------
Author              : Maks Kliczkowski
Version             : 3.0.0
Last modified       : 2026-03-04
------------------------------------------------------------------------------
'''
from    __future__ import annotations

from    typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import  matplotlib.pyplot as plt
import  matplotlib.tri as mtri
import  numpy as np
from    scipy.interpolate import griddata
from    scipy.spatial import cKDTree

try:
    from ..plot         import Plotter
    from .config        import KPathConfig, KSpaceConfig, PlotStyle, SpectralConfig
    from .kspace_utils  import label_high_sym_points
except ImportError as exc:
    raise ImportError("Failed to import plotting helpers dependencies.") from exc

if TYPE_CHECKING:
    from general_python.lattices.lattice import Lattice

_PLOT_STYLE_KEYS        = set(PlotStyle.__dataclass_fields__.keys())
_KSPACE_CONFIG_KEYS     = set(KSpaceConfig.__dataclass_fields__.keys())
_KPATH_CONFIG_KEYS      = set(KPathConfig.__dataclass_fields__.keys())
_SPECTRAL_CONFIG_KEYS   = set(SpectralConfig.__dataclass_fields__.keys())


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _apply_overrides(config: Any, overrides: dict[str, Any], allowed_keys: set[str]) -> None:
    """Apply matching keyword arguments to a dataclass config instance."""
    for key in list(overrides):
        if key in allowed_keys:
            setattr(config, key, overrides.pop(key))

def _prepare_axes(ax=None, *, figsize: tuple[float, float] = (4.0, 3.5), dpi: Optional[int] = None) -> tuple[plt.Figure, plt.Axes, bool]:
    """Return ``(fig, ax, created)`` using ``Plotter`` for new panels."""
    if ax is not None:
        return ax.figure, ax, False
    
    fig, ax = Plotter.get_subplots(nrows=1, ncols=1, sizex=figsize[0], sizey=figsize[1], single_if_1=True, dpi=dpi)
    return fig, ax, True

# kspace points

def _find_kgrid_axes(shape: Sequence[int], grid_shape: tuple[int, ...]) -> tuple[int, int]:
    """Locate the contiguous or flattened k-grid axes inside ``shape``."""
    
    # total number of k-points in the grid
    nk = int(np.prod(grid_shape))
    
    # first look for a contiguous block of axes matching the grid shape
    for start in range(len(shape) - len(grid_shape) + 1):
        if tuple(shape[start : start + len(grid_shape)]) == grid_shape:
            return start, len(grid_shape)
        
    for axis, axis_size in enumerate(shape):
        if int(axis_size) == nk:
            return axis, 1
    raise ValueError(f"Could not identify k-grid axes {grid_shape} in array shape {tuple(shape)}.")

def _flatten_kspace_points(k_values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return flattened k-points together with the original grid shape."""
    arr = np.asarray(k_values, dtype=float)
    if arr.ndim < 2:
        raise ValueError("k_values must be shaped like (Nk, D) or (..., D).")
    if arr.shape[-1] < 1:
        raise ValueError("k_values must include Cartesian coordinates on the last axis.")
    grid_shape = tuple(int(v) for v in arr.shape[:-1])
    if len(grid_shape) == 0:
        grid_shape = (int(arr.shape[0]),)
    return arr.reshape(-1, arr.shape[-1]), grid_shape

# planar coordinates and interpolation

def _planar_xy(points: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    """
    Extract planar ``(x, y)`` coordinates.

    Grid heatmaps are only supported for planar k-space data. Embedded 2D data
    in 3D is allowed if all higher Cartesian components are negligible.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("Planar plots require k-points with at least two Cartesian components.")
    if pts.shape[1] > 2 and not np.all(np.abs(pts[:, 2:]) <= tol):
        raise ValueError("Grid k-space plots are only supported for planar data. For higher-dimensional lattices use a k-path plot instead.")
    return pts[:, :2]

# k-space field extraction and normalization

def _extract_scalar_kfield(values: np.ndarray, grid_shape: tuple[int, ...], *,
    batch_index         : Optional[Any] = None,
    component_index     : Optional[Any] = None,
    value_mode          : Literal["auto", "real", "imag", "abs", "phase"] = "auto",
) -> np.ndarray:
    """Reduce a general k-space array to one scalar value per sampled k-point."""
    values_arr              = np.asarray(values)
    nk                      = int(np.prod(grid_shape))
    grid_axis, grid_span    = _find_kgrid_axes(values_arr.shape, grid_shape)

    batch_shape             = values_arr.shape[:grid_axis]
    feature_shape           = values_arr.shape[grid_axis + grid_span :]
    values_flat             = values_arr.reshape(batch_shape + (nk,) + feature_shape)

    if len(batch_shape) > 0:
        if batch_index is None:
            batch_index = tuple(0 for _ in batch_shape)
        elif not isinstance(batch_index, tuple):
            batch_index = (batch_index,)
        if len(batch_index) != len(batch_shape):
            raise ValueError(f"batch_index must have length {len(batch_shape)}.")
        values_flat = values_flat[batch_index + (slice(None),) + tuple(slice(None) for _ in feature_shape)]

    if len(feature_shape) > 0:
        if component_index is None:
            component_index = tuple(0 for _ in feature_shape)
        elif not isinstance(component_index, tuple):
            component_index = (component_index,)
        if len(component_index) != len(feature_shape):
            raise ValueError(f"component_index must have length {len(feature_shape)}.")
        values_flat = values_flat[(slice(None),) + component_index]

    field = np.asarray(values_flat)
    if field.ndim != 1 or field.shape[0] != nk:
        raise ValueError("Could not reduce values to a scalar field on the sampled k-grid.")

    if value_mode == "real":
        field = np.real(field)
    elif value_mode == "imag":
        field = np.imag(field)
    elif value_mode == "abs":
        field = np.abs(field)
    elif value_mode == "phase":
        field = np.angle(field)
    elif value_mode == "auto":
        real_close = np.real_if_close(field)
        field = np.real(real_close) if not np.iscomplexobj(real_close) else np.abs(field)
    else:
        raise ValueError(f"Unknown value_mode '{value_mode}'.")

    return np.asarray(field, dtype=float)

def _extract_spectral_matrix(
    intensity   : np.ndarray,
    omega       : np.ndarray,
    *,
    grid_shape  : Optional[tuple[int, ...]] = None,
    batch_index : Optional[Any] = None,
) -> np.ndarray:
    """
    Canonicalize spectral data to shape ``(Nk, Nomega)``.

    Supported common layouts include:
    - ``(Nk, Nw)``
    - ``(Nw, Nk)``
    - ``(Lx, Ly, Lz, Nw)``
    - ``(Nw, Lx, Ly, Lz)``
    - the same with leading batch axes, selected through ``batch_index``
    """
    arr     = np.asarray(intensity)
    n_omega = int(np.asarray(omega).size)
    if arr.ndim < 2:
        raise ValueError("Spectral intensity must have at least k and omega axes.")

    if grid_shape is None:
        omega_axes = [axis for axis, size in enumerate(arr.shape) if int(size) == n_omega]
        if not omega_axes:
            raise ValueError(f"Could not identify an omega axis of length {n_omega} in shape {arr.shape}.")
        omega_axis  = omega_axes[-1]
        arr         = np.moveaxis(arr, omega_axis, -1) # move omega to the last axis 
        return arr.reshape(-1, n_omega)

    grid_axis, grid_span    = _find_kgrid_axes(arr.shape, grid_shape)
    outside_axes            = [axis for axis in range(arr.ndim) if not (grid_axis <= axis < grid_axis + grid_span)]
    omega_candidates        = [axis for axis in outside_axes if int(arr.shape[axis]) == n_omega]
    if not omega_candidates:
        raise ValueError(f"Could not identify an omega axis of length {n_omega} outside k-grid axes {grid_shape} in intensity shape {arr.shape}.")
    omega_axis              = omega_candidates[-1]
    extra_axes              = [axis for axis in outside_axes if axis != omega_axis]
    if extra_axes:
        if batch_index is None:
            batch_index = tuple(0 for _ in extra_axes)
        elif not isinstance(batch_index, tuple):
            batch_index = (batch_index,)
        if len(batch_index) != len(extra_axes):
            raise ValueError(f"batch_index must have length {len(extra_axes)} for intensity shape {arr.shape}.")
        
        indexer = [slice(None)] * arr.ndim
        for axis, idx in zip(extra_axes, batch_index):
            indexer[axis] = idx
        arr = arr[tuple(indexer)]
        return _extract_spectral_matrix(arr, omega, grid_shape=grid_shape, batch_index=None)

    arr = np.moveaxis(arr, omega_axis, -1)
    nk  = int(np.prod(grid_shape))
    return arr.reshape(nk, n_omega)

def _deduplicate_xy_samples(points: np.ndarray, values: np.ndarray, *, decimals: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """
    Average duplicate planar samples before interpolation. This is used to clean up the extended k-space samples, 
    which may include multiple points mapping to the same planar coordinates due to the Wigner-Seitz shifts.    
    """
    pts     = np.asarray(points, dtype=float)
    vals    = np.asarray(values)
    if len(pts) != len(vals):
        raise ValueError("points and values must have the same length.")
    if len(pts) == 0:
        return pts.reshape(0, 2), vals

    rounded             = np.round(pts[:, :2], decimals=decimals)
    _, inverse, counts  = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    pts_sum             = np.zeros((len(counts), 2), dtype=float)
    vals_sum            = np.zeros(len(counts), dtype=np.result_type(vals.dtype, np.float64))
    np.add.at(pts_sum, inverse, pts[:, :2])
    np.add.at(vals_sum, inverse, vals)
    return pts_sum / counts[:, None], vals_sum / counts

def _compress_path_runs(k_dist: np.ndarray, values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compress consecutive repeated path matches to a single representative sample."""
    x       = np.asarray(k_dist, dtype=float).ravel()
    idx     = np.asarray(indices, dtype=int).ravel()
    vals    = np.asarray(values)
    if len(idx) == 0:
        return x, vals
    if vals.shape[0] != len(idx):
        raise ValueError("values must have the path axis first for path compression.")

    x_out   = []
    val_out = []
    start   = 0
    for pos in range(1, len(idx) + 1):
        if pos == len(idx) or idx[pos] != idx[start]:
            x_out.append(float(np.mean(x[start:pos])))
            val_out.append(np.mean(vals[start:pos], axis=0))
            start = pos
    return np.asarray(x_out, dtype=float), np.asarray(val_out)

# -----------------------------------------------------------------------------

def _masked_tripcolor(
    ax,
    points: np.ndarray,
    values: np.ndarray,
    *,
    lattice: Optional["Lattice"] = None,
    shift: Optional[np.ndarray] = None,
    shifts: Optional[Sequence[np.ndarray]] = None,
    shells: int = 1,
    shading: str = "gouraud",
    **kwargs,
):
    """Draw a triangulated planar field, optionally masked to one or more WS cells."""
    pts     = np.asarray(points, dtype=float)
    vals    = np.asarray(values, dtype=float)
    if len(pts) < 3:
        return None

    # create a triangulation and mask out triangles whose centroids do not fall within any of the specified WS cells
    tri             = mtri.Triangulation(pts[:, 0], pts[:, 1])
    active_shifts   = None
    if shifts is not None:
        active_shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
    elif shift is not None:
        active_shifts = np.asarray(shift, dtype=float).reshape(1, 2)

    if lattice is not None and active_shifts is not None and tri.triangles.size > 0:
        centroids = pts[tri.triangles].mean(axis=1)
        keep = np.zeros(len(centroids), dtype=bool)
        for one_shift in active_shifts:
            keep |= np.asarray(
                lattice.wigner_seitz_mask(centroids[:, 0] - float(one_shift[0]), centroids[:, 1] - float(one_shift[1]), shells=shells),
                dtype=bool,
            )
        tri.set_mask(~np.asarray(keep, dtype=bool))
    return Plotter.tripcolor_field(ax, pts, vals, triangles=tri.triangles, mask=tri.mask, shading=shading, **kwargs)

# -----------------------------------------------------------------------------

def compute_correlation_kspace(lattice: "Lattice", corr_matrix: np.ndarray, *,
    reduction: Literal["none", "sum", "trace", "mean", "diag"] = "sum",
    norm: Literal["none", "cell", "site"] = "site",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a real-space correlation matrix to k-space on the lattice grid.

    Parameters
    ----------
    lattice:
        Lattice instance providing ``structure_factor`` and the sampled k-grid.
    corr_matrix:
        Real-space correlation matrix shaped like ``(Ns, Ns)`` or batched
        ``(..., Ns, Ns)``.
    reduction:
        Block reduction forwarded to ``lattice.structure_factor``.
    norm:
        Normalization forwarded to ``lattice.structure_factor``.

    Returns
    -------
    values, k_grid, k_frac:
        Reduced k-space values, Cartesian k-grid, and fractional k-grid.
    """
    if lattice is None:
        raise ValueError("lattice is required to compute k-space correlations.")
    return lattice.structure_factor(corr_matrix, reduction=reduction, norm=norm)

# -----------------------------------------------------------------------------
# Real-space correlation plots
# -----------------------------------------------------------------------------

def plot_realspace_correlations(
    ax=None,
    *,
    corr_matrix             : np.ndarray,
    lattice                 : Optional["Lattice"] = None,
    ref_site_idx            : int = 0,
    style                   : Optional[PlotStyle] = None,
    mode                    : Literal["matrix", "lattice"] = "lattice",
    interpolation           : str = "linear",
    relative_coordinates    : bool = False,
    point_size              : float = 55.0,
    figsize                 : tuple[float, float] = (4.0, 3.5),
    dpi                     : Optional[int] = None,
    **kwargs,
):
    """
    Plot a real-space correlation matrix directly or on lattice geometry.
    
    Parameters
    ----------
    ax: 
        Optional Matplotlib axis to plot on. If None, a new single panel will be created.
    corr_matrix:
        Square matrix of shape (N, N) containing the correlations between N sites. For lattice plots, the reference site is determined by `ref_site_idx` and the correlations of that site to all others are plotted as a function of distance.
    lattice:
        Optional lattice object providing site positions for lattice plots. If None, the correlation matrix will be plotted directly as an image.
    ref_site_idx:
        Index of the reference site for lattice plots. Must be between 0 and N-1, where N is the number of sites in `corr_matrix`.
    style:
        Optional `PlotStyle` dataclass instance for styling the plot. Plot-specific overrides can be passed as keyword arguments.
    mode:
        Plotting mode. If "matrix", the correlation matrix will be visualized as a heatmap. If "lattice", the correlations of the reference site to all others will be plotted as a function of their positions on the lattice.
    interpolation:
        Interpolation method for visualizing the correlation values. For "matrix" mode, this is passed to `imshow`. For "lattice" mode with 2D lattices, this controls whether an interpolated background map of the correlations is shown. Supported values include "none", "nearest", "linear", and "cubic". The default is "linear" for lattice mode and "none" for matrix mode.
    relative_coordinates:
        If True and lattice is provided, the site positions will be plotted relative to the reference site. 
        Otherwise, absolute positions will be used.
    point_size:
        Base size of the points in lattice mode. This can be overridden with the `point_size` keyword argument.
    figsize:
        Size of the figure to create if `ax` is None. Ignored if an axis is provided.
    dpi:
        Resolution of the figure to create if `ax` is None. Ignored if an axis is provided.
    **kwargs:
        Supported keyword overrides are:
        - any ``PlotStyle`` field, e.g. ``cmap``, ``vmin``, ``vmax``,
          ``linewidth``, ``marker``, ``fontsize_label``, ``fontsize_tick``
        - common axis labels / limits:
          ``xlabel``, ``ylabel``, ``r_limits``, ``x_limits``, ``y_limits``,
          ``xmin``, ``xmax``, ``ymin``, ``ymax``
        - real-space styling:
          ``line_color``, ``map_alpha``, ``color_points_by_value``,
          ``point_size``, ``point_alpha``, ``point_facecolor``, ``edgecolor``,
          ``linewidths``, ``ref_point_size``, ``ref_point_color``, ``axis_off``
    """
    _, ax, _    = _prepare_axes(ax, figsize=figsize, dpi=dpi)
    style       = PlotStyle() if style is None else style
    overrides   = dict(kwargs)
    
    # apply style and lattice overrides before any validation or plotting logic
    _apply_overrides(style, overrides, _PLOT_STYLE_KEYS)

    # validate and prepare the correlation matrix for plotting
    C = np.asarray(np.real_if_close(corr_matrix))
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("corr_matrix must be square.")

    if mode == "matrix" or lattice is None:
        imshow_interp = "none" if interpolation is None else str(interpolation).lower().strip()
        if imshow_interp == "linear":
            imshow_interp = "bilinear"
        artist = ax.imshow(C, cmap=style.cmap, vmin=style.vmin, vmax=style.vmax, interpolation=imshow_interp, origin="lower")
        Plotter.set_ax_params(ax, xlabel=overrides.pop("xlabel", r"Site $j$"), ylabel=overrides.pop("ylabel", r"Site $i$"), fontsize=style.fontsize_label,)
        Plotter.set_tickparams(ax, labelsize=style.fontsize_tick)
        return artist

    # for lattice plots, we need the site positions to plot the correlations as a function of distance
    positions_source    = lattice.coordinates if hasattr(lattice, "coordinates") else lattice.rvectors
    positions           = np.asarray(positions_source, dtype=float)
    # reference
    ref_site_idx        = int(np.clip(ref_site_idx, 0, positions.shape[0] - 1))
    site_vals           = np.asarray(C[ref_site_idx], dtype=float)
    ref_pos             = np.asarray(positions[ref_site_idx], dtype=float)

    # plot 1D correlations if the lattice is effectively 1D or all sites lie on a line
    if positions.shape[1] == 1 or np.allclose(positions[:, 1:], 0.0):
        x       = positions[:, 0] - ref_pos[0] if relative_coordinates else positions[:, 0]
        order   = np.argsort(x)
        (line,) = ax.plot(x[order], site_vals[order],
            marker=style.marker, ms=style.markersize, lw=style.linewidth, ls=style.linestyle, alpha=style.alpha,
            color=overrides.pop("line_color", "C0"),
        )
        
        if relative_coordinates:
            ax.axvline(0.0, color="0.35", ls="--", lw=1.0, alpha=0.5)
        Plotter.set_ax_params(ax,
            xlabel=overrides.pop("xlabel", r"$\Delta x$" if relative_coordinates else r"$x$"),
            ylabel=overrides.pop("ylabel", r"$C(\Delta x)$" if relative_coordinates else r"$C(x)$"),
            fontsize=style.fontsize_label,
            grid=True, grid_alpha=style.grid_alpha, grid_color="0.8",
        )
        Plotter.set_tickparams(ax, labelsize=style.fontsize_tick)
        return line

    # for truly 2D lattices, plot the correlations as a function of the planar coordinates, optionally with an interpolated background map
    pos         = positions[:, :2] - ref_pos[:2] if relative_coordinates else positions[:, :2]
    xlim, ylim  = Plotter.resolve_planar_limits(
        pos,
        limits=overrides.pop("r_limits", None),
        x_limits=overrides.pop("x_limits", None),
        y_limits=overrides.pop("y_limits", None),
        xmin=overrides.pop("xmin", None),
        xmax=overrides.pop("xmax", None),
        ymin=overrides.pop("ymin", None),
        ymax=overrides.pop("ymax", None),
    )
    artist      = None
    use_interp  = str(interpolation).lower() != "none"
    if use_interp:
        shading = "flat" if str(interpolation).lower() == "nearest" else "gouraud"
        artist  = _masked_tripcolor(ax, pos, site_vals,
            cmap=style.cmap, vmin=style.vmin, vmax=style.vmax, alpha=overrides.pop("map_alpha", 0.95), shading=shading)

    # if not showing an interpolated map, color the points by their value by default to make them more visible. 
    color_points_by_value = bool(overrides.pop("color_points_by_value", not use_interp))
    if color_points_by_value:
        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=site_vals, cmap=style.cmap, vmin=style.vmin, vmax=style.vmax,
            s=overrides.pop("point_size", point_size), edgecolors=overrides.pop("edgecolor", "k"), linewidths=overrides.pop("linewidths", 0.5),
            zorder=20,
        )
    else:
        scatter = ax.scatter(pos[:, 0], pos[:, 1], s=overrides.pop("point_size", point_size),
            facecolors=overrides.pop("point_facecolor", "none"),
            edgecolors=overrides.pop("edgecolor", "k"),
            linewidths=overrides.pop("linewidths", 0.8),
            alpha=overrides.pop("point_alpha", 1.0),
            zorder=20,
        )
        
    # mark the reference site with a star, using the same relative coordinates as the other points if applicable
    ref_xy = np.zeros(2, dtype=float) if relative_coordinates else ref_pos[:2]
    ax.scatter(ref_xy[0], ref_xy[1], marker="*", s=overrides.pop("ref_point_size", point_size * 1.8), c=overrides.pop("ref_point_color", "yellow"),
        edgecolors="k", linewidths=0.8, zorder=21)
    
    # set axis parameters and return the artist for the interpolated map if it exists, otherwise the scatter plot artist
    Plotter.set_ax_params(ax,
        xlabel=overrides.pop("xlabel", r"$\Delta x$" if relative_coordinates else r"$x$"),
        ylabel=overrides.pop("ylabel", r"$\Delta y$" if relative_coordinates else r"$y$"),
        fontsize=style.fontsize_label, aspect="equal", xlim=xlim, ylim=ylim,
        show_spines=not bool(overrides.pop("axis_off", True)),
    )
    if kwargs.get("axis_off", True):
        ax.axis("off")
    Plotter.set_tickparams(ax, labelsize=style.fontsize_tick)
    return artist if artist is not None else scatter

# -----------------------------------------------------------------------------
# Static k-space plots
# -----------------------------------------------------------------------------

def plot_kspace_intensity(
    ax,
    k_values                : np.ndarray,
    intensity               : np.ndarray,
    *,
    # style
    style                   : Optional[PlotStyle] = None,
    ks_config               : Optional[KSpaceConfig] = None,
    # lattice info
    lattice                 : Optional["Lattice"] = None,
    # field extraction
    batch_index             : Optional[Any] = None,
    component_index         : Optional[Any] = None,
    value_mode              : Literal["auto", "real", "imag", "abs", "phase"] = "auto",
    # bz info
    show_extended_bz        : Optional[bool] = None,
    bz_copies               : Optional[int] = None,
    # plotting
    show_interpolated_map   : Optional[bool] = None,
    interpolation           : Optional[str] = None,
    limit_to_pi             : Optional[bool] = None,
    k_limits                : Optional[Sequence[float]] = None,
    figsize                 : tuple[float, float] = (4.0, 3.5),
    dpi                     : Optional[int] = None,
    **kwargs,
):
    """
    Plot a scalar intensity field on a planar k-grid.

    ``k_values`` may be flattened ``(Nk, D)`` or structured ``(..., D)``. Grid
    heatmaps are supported only for planar data. For higher-dimensional lattices,
    use ``plot_kspace_path`` or ``plot_spectral_function(..., mode='kpath')``.
    
    Parameters
    ----------
    ax:
        Optional Matplotlib axis to plot on. If None, a new single panel will be created.
    k_values:
        Array of shape (Nk, D) or (..., D) containing the Cartesian coordinates of the sampled k-points. The last axis must include at least two Cartesian components for planar plotting.
    intensity:
        Array containing the scalar field values corresponding to the k-points. The shape must include the k-grid shape as a contiguous block or flattened segment, and may include leading batch axes and/or trailing
        feature axes. The k-grid axes are identified by matching the shape of the grid to the shape of the k-values. The field values are reduced to a single scalar per k-point by selecting a batch index and/or component index if necessary, and applying the specified value mode (real, imag, abs, phase, or auto).
    style:
        Optional plotting style configuration.
    ks_config:
        Optional k-space plotting configuration. Recognized fields include `grid_n`, `point_size`, `extend_bz`, `bz_copies`, `interp_method`, `ws_shells`, 
        'limit_to_pi`, and `k_limits`. These can also be overridden with direct keyword arguments.
    lattice:
        Optional lattice object providing information about the Brillouin zone for extended zone plotting and interpolation masking. If None, the k-space points are treated as abstract Cartesian coordinates without any lattice structure.
    batch_index:
        Optional index or tuple of indices to select a specific batch from the intensity array if it includes leading batch axes. The length of the tuple must match the number of batch axes. If None, the first index (0) is used for all batch axes.
    component_index:
        Optional index or tuple of indices to select a specific component from the intensity array if it includes trailing feature axes. The length of the tuple must match the number of feature axes. If None, the first index (0) is used for all feature axes.
    value_mode:
        Mode for reducing complex values to real scalars. Supported values are "real", "imag", "abs", "phase", and "auto". The default "auto" will use the real part if the values are close to real, and the absolute value otherwise.
    show_extended_bz:
        Whether to show the extended Brillouin zone by applying Wigner-Seitz shifts to the k-points. If None, this defaults to the value of `ks_config.extend_bz` if `ks_config` is provided, otherwise False.
    bz_copies:
        Number of Brillouin zone copies to show in each direction when `show_extended_bz` is True. If None, this defaults to the value of `ks_config.bz_copies` if `ks_config` is provided, otherwise 0 (no copies).
    show_interpolated_map:
        Whether to show an interpolated background map of the field. This is only supported for 2D planar data with a regular grid. If None, this defaults to True if `ks_config.grid_n > 0`, otherwise False.
    interpolation:
        Interpolation method for the background map. Supported values include
        ``"none"``, ``"nearest"``, ``"linear"``, and ``"cubic"``.
    limit_to_pi:
        If True, use ``[-pi, pi]`` limits on both axes unless explicit limits
        are provided.
    k_limits:
        Explicit visible limits as ``(min, max)`` or
        ``(xmin, xmax, ymin, ymax)``.
    figsize, dpi:
        Figure creation parameters used only when ``ax`` is None.
    **kwargs:
        Supported keyword overrides are:
        - any ``PlotStyle`` field
        - any ``KSpaceConfig`` field, e.g. ``grid_n``, ``point_size``,
          ``point_alpha``, ``point_marker``, ``show_discrete_points``,
          ``draw_bz_outline``, ``label_high_symmetry``, ``ws_shells``
        - common axis labels / limits:
          ``xlabel``, ``ylabel``, ``x_limits``, ``y_limits``, ``xmin``,
          ``xmax``, ``ymin``, ``ymax``
        - point / map styling:
          ``extend_points``, ``map_alpha``, ``color_points_by_value``,
          ``point_facecolor``, ``point_edgecolor``, ``point_edgewidth``
        - Brillouin-zone styling:
          ``bz_edgecolor``, ``bz_linewidth``, ``bz_alpha``, ``show_hs_labels``
        - no-lattice fallback reciprocal vectors:
          ``k1_vec``, ``k2_vec``
    """
    fig, ax, _  = _prepare_axes(ax, figsize=figsize, dpi=dpi)
    del fig
    style       = PlotStyle() if style is None else style
    ks_config   = KSpaceConfig() if ks_config is None else ks_config
    overrides   = dict(kwargs)
    _apply_overrides(style, overrides, _PLOT_STYLE_KEYS)
    _apply_overrides(ks_config, overrides, _KSPACE_CONFIG_KEYS)

    if show_extended_bz is None:
        show_extended_bz = ks_config.extend_bz
    if bz_copies is None:
        bz_copies = ks_config.bz_copies
    if show_interpolated_map is None:
        show_interpolated_map = ks_config.grid_n > 0
    if interpolation is None:
        interpolation = ks_config.interp_method
    if limit_to_pi is None:
        limit_to_pi = getattr(ks_config, "limit_to_pi", False)
    if k_limits is None:
        k_limits = getattr(ks_config, "k_limits", None)

    # get k-space points to (Nk, D) and the corresponding scalar field values on the grid
    flat_k, grid_shape  = _flatten_kspace_points(k_values)
    xy                  = _planar_xy(flat_k) # raises if not planar
    field               = _extract_scalar_kfield(intensity, grid_shape, batch_index=batch_index,
                            component_index=component_index, value_mode=value_mode)
    
    extend_points       = bool(overrides.pop("extend_points", False))
    if lattice is not None:
        if show_extended_bz and int(bz_copies) > 0:
            shifts_arr  = np.asarray(lattice.wigner_seitz_shifts(copies=(int(bz_copies), int(bz_copies)), include_origin=True), dtype=float,)
            if shifts_arr.size == 0:
                shifts  = [np.zeros(2, dtype=float)]
            else:
                shifts_arr      = shifts_arr.reshape(-1, shifts_arr.shape[-1])[:, :2]
                rounded         = np.round(shifts_arr, decimals=12)
                _, unique_idx   = np.unique(rounded, axis=0, return_index=True)
                shifts          = [shifts_arr[idx] for idx in np.sort(unique_idx)]
        else:
            shifts = [np.zeros(2, dtype=float)]
    else:
        shifts = [np.zeros(2, dtype=float)]

    if show_extended_bz:
        if lattice is not None:
            final_k, final_field = lattice.wigner_seitz_extend(k_points=xy, data=field, copies=(int(bz_copies), int(bz_copies)))
        else:
            from ...lattices.tools.lattice_kspace import extend_kspace_data

            k1_vec = np.asarray(overrides.pop("k1_vec"), dtype=float)[:2]
            k2_vec = np.asarray(overrides.pop("k2_vec"), dtype=float)[:2]
            final_k, final_field = extend_kspace_data(xy, field, b1=k1_vec, b2=k2_vec, nx=int(bz_copies), ny=int(bz_copies))
    else:
        final_k, final_field = xy, field

    # for point plotting and axis limit calculations, use the extended points if applicable and requested, otherwise the original points
    point_k         = final_k if (show_extended_bz and extend_points) else xy
    point_field     = final_field if (show_extended_bz and extend_points) else field
    xlim, ylim      = Plotter.resolve_planar_limits(final_k, limits=k_limits, x_limits=overrides.pop("x_limits", None), y_limits=overrides.pop("y_limits", None),
        xmin=overrides.pop("xmin", None), xmax=overrides.pop("xmax", None), ymin=overrides.pop("ymin", None), ymax=overrides.pop("ymax", None), limit_to_pi=bool(limit_to_pi),
    )

    vmin            = style.vmin if style.vmin is not None else float(np.nanmin(final_field))
    vmax            = style.vmax if style.vmax is not None else float(np.nanmax(final_field))
    image_artist    = None

    # show an interpolated background map of the field if requested and supported, 
    # using a masked triangulation if lattice WS cell information is available, otherwise using griddata interpolation on a regular grid. The original or extended points can be used for interpolation depending on the show_extended_bz and extend_points options, but the final extended points are always used for masking to ensure consistency.
    if show_interpolated_map and ks_config.grid_n > 0:
        shading     = "flat" if str(interpolation).lower() == "nearest" else "gouraud"
        map_alpha   = overrides.pop("map_alpha", 0.95)
        if lattice is not None:
            ext_xy, ext_field   = np.asarray(final_k, dtype=float), np.asarray(final_field, dtype=float)
            ext_xy, ext_field   = _deduplicate_xy_samples(ext_xy[:, :2], ext_field)
            image_artist        = _masked_tripcolor(ax, ext_xy, ext_field, lattice=lattice,
                shifts=shifts, shells=ks_config.ws_shells, shading=shading, cmap=style.cmap, vmin=vmin, vmax=vmax, alpha=map_alpha, zorder=15,
            )
        else:
            raise ValueError("Grid interpolation requires lattice information for masking. Provide a lattice or disable interpolation.")

    # if not showing an interpolated map, color the points by their value by default to make them more visible. If showing an interpolated map, use a single color for the points by default to avoid overplotting the colormap.
    scatter_artist = None
    if ks_config.show_discrete_points:
        color_points_by_value = bool(overrides.pop("color_points_by_value", not show_interpolated_map))
        if color_points_by_value:
            scatter_artist = ax.scatter(point_k[:, 0], point_k[:, 1],
                c=point_field, cmap=style.cmap, vmin=vmin, vmax=vmax, s=ks_config.point_size,
                alpha=ks_config.point_alpha, marker=ks_config.point_marker,
                edgecolors=overrides.pop("point_edgecolor", "none"), linewidths=overrides.pop("point_edgewidth", 0.0),
                zorder=120,
            )
        else:
            scatter_artist = ax.scatter(
                point_k[:, 0], point_k[:, 1],
                s=ks_config.point_size, alpha=ks_config.point_alpha,
                marker=ks_config.point_marker, facecolors=overrides.pop("point_facecolor", "none"),
                edgecolors=overrides.pop("point_edgecolor", "k"), linewidths=overrides.pop("point_edgewidth", 0.8),
                zorder=120,
            )

    if ks_config.draw_bz_outline and lattice is not None:
        bz_edgecolor    = overrides.pop("bz_edgecolor", "0.35")
        bz_linewidth    = overrides.pop("bz_linewidth", 1.0)
        bz_alpha        = overrides.pop("bz_alpha", 0.9)
        outline_n       = max(int(ks_config.grid_n), 160)
        GX, GY          = np.meshgrid(
            np.linspace(xlim[0], xlim[1], outline_n),
            np.linspace(ylim[0], ylim[1], outline_n),
        )
        for shift in shifts:
            mask = lattice.wigner_seitz_mask(GX - shift[0], GY - shift[1], shells=ks_config.ws_shells)
            ax.contour(GX, GY, mask.astype(float), levels=[0.5], colors=[bz_edgecolor], linewidths=bz_linewidth, alpha=bz_alpha, zorder=40,)

    if ks_config.label_high_symmetry and lattice is not None:
        label_high_sym_points(
            ax,
            lattice,
            bz_copies=int(bz_copies) if show_extended_bz else 0,
            show_labels=bool(overrides.pop("show_hs_labels", True)),
            markersize=ks_config.hs_marker_size,
            markerfacecolor=ks_config.hs_marker_color,
            markeredgecolor=ks_config.hs_marker_edge,
            label_offset_x=ks_config.hs_label_offset_x,
            label_offset_y=ks_config.hs_label_offset_y,
            label_fontsize=ks_config.hs_label_fontsize,
            label_color=ks_config.hs_label_color,
            **overrides,
        )

    Plotter.set_ax_params(
        ax, xlabel=overrides.pop("xlabel", r"$k_x$"), ylabel=overrides.pop("ylabel", r"$k_y$"), fontsize=style.fontsize_label,
        aspect="equal", xlim=xlim, ylim=ylim,
    )
    Plotter.set_tickparams(ax, labelsize=style.fontsize_tick)
    return image_artist if image_artist is not None else scatter_artist

def plot_kspace_path(
    ax,
    lattice: "Lattice",
    *,
    values: Optional[np.ndarray] = None,
    k_values: Optional[np.ndarray] = None,
    k_frac: Optional[np.ndarray] = None,
    corr_matrix: Optional[np.ndarray] = None,
    reduction: Literal["none", "sum", "trace", "mean", "diag"] = "sum",
    norm: Literal["none", "cell", "site"] = "site",
    path=None,
    points_per_seg: Optional[int] = None,
    value_label: str = r"$S(\mathbf{k})$",
    sampled_only: bool = True,
    line_kw: Optional[dict] = None,
    hsline_kw: Optional[dict] = None,
    style: Optional[PlotStyle] = None,
    kpath_config: Optional[KPathConfig] = None,
    figsize: tuple[float, float] = (4.0, 3.5),
    dpi: Optional[int] = None,
    **kwargs,
):
    """
    Plot a scalar quantity along a high-symmetry k-path.

    Parameters
    ----------
    ax:
        Optional Matplotlib axis to plot on.
    lattice:
        Lattice instance providing the path geometry and k-grid matching.
    values:
        Precomputed scalar k-space values. If omitted, ``corr_matrix`` is
        transformed with ``compute_correlation_kspace``.
    k_values, k_frac:
        Cartesian and fractional k-point coordinates for ``values``.
    corr_matrix:
        Real-space correlation matrix used when ``values`` is not provided.
    reduction, norm:
        Forwarded to ``compute_correlation_kspace`` when ``corr_matrix`` is
        provided.
    path, points_per_seg:
        High-symmetry path specification and sampling density.
    value_label:
        Y-axis label.
    sampled_only:
        If True, compress consecutive repeated path matches so only the actual
        sampled path points remain.
    line_kw, hsline_kw:
        Explicit Matplotlib style dictionaries for the path curve and separator
        lines.
    style, kpath_config:
        Optional style/config objects.
    figsize, dpi:
        Figure creation parameters used only when ``ax`` is None.
    **kwargs:
        Supported keyword overrides are:
        - any ``PlotStyle`` field
        - any ``KPathConfig`` field, e.g. ``path``, ``points_per_seg``,
          ``tolerance``, ``separator_style``
        - line styling when ``line_kw`` is not given:
          ``line_color``, ``marker_facecolor``, ``marker_edgecolor``,
          ``marker_edgewidth``
        - axis labels:
          ``xlabel``, ``ylabel``
    """
    fig, ax, _ = _prepare_axes(ax, figsize=figsize, dpi=dpi)
    del fig
    style = PlotStyle() if style is None else style
    kpath_config = KPathConfig() if kpath_config is None else kpath_config
    overrides = dict(kwargs)
    _apply_overrides(style, overrides, _PLOT_STYLE_KEYS)
    _apply_overrides(kpath_config, overrides, _KPATH_CONFIG_KEYS)

    if path is not None:
        kpath_config.path = path
    if points_per_seg is not None:
        kpath_config.points_per_seg = points_per_seg

    if values is None:
        if corr_matrix is None:
            raise ValueError("Provide either precomputed values or corr_matrix.")
        values, k_values, k_frac = compute_correlation_kspace(
            lattice,
            corr_matrix,
            reduction=reduction,
            norm=norm,
        )
    elif k_values is None:
        raise ValueError("k_values must be provided with precomputed path values.")

    if k_frac is None:
        lattice_frac = getattr(lattice, "kvectors_frac", None)
        lattice_cart = getattr(lattice, "kvectors", None)
        if lattice_frac is not None and lattice_cart is not None and np.shape(lattice_cart) == np.shape(k_values):
            k_frac = lattice_frac
        else:
            raise ValueError(
                "k_frac must be provided with precomputed k-space values unless k_values "
                "matches lattice.kvectors."
            )

    result = lattice.bz_path_data(
        k_vectors=k_values,
        k_vectors_frac=k_frac,
        values=values,
        path=kpath_config.path,
        points_per_seg=kpath_config.points_per_seg or 40,
        return_result=True,
    )

    path_values = np.real_if_close(np.asarray(result.values))
    if path_values.ndim != 1:
        raise ValueError(f"plot_kspace_path expects scalar path values, got shape {path_values.shape}.")

    x_values = np.asarray(result.k_dist, dtype=float)
    if sampled_only and len(result.indices) > 0:
        x_values, path_values = _compress_path_runs(x_values, path_values, result.indices)

    if line_kw is None:
        line_kw = {
            "lw": style.linewidth,
            "ls": style.linestyle,
            "color": overrides.pop("line_color", "C0"),
            "marker": style.marker,
            "ms": style.markersize,
            "mfc": overrides.pop("marker_facecolor", "C0"),
            "mec": overrides.pop("marker_edgecolor", "white"),
            "mew": overrides.pop("marker_edgewidth", 0.5),
            "alpha": style.alpha,
        }
    if hsline_kw is None:
        hsline_kw = kpath_config.get_separator_kwargs()

    ax.plot(x_values, path_values, **line_kw)
    for xv in result.label_positions:
        ax.axvline(float(xv), **hsline_kw)

    Plotter.set_ax_params(
        ax,
        xlabel=overrides.pop("xlabel", r"$k$"),
        ylabel=overrides.pop("ylabel", value_label),
        fontsize=style.fontsize_label,
        xlim=(float(np.min(result.k_dist)), float(np.max(result.k_dist))),
        grid=True,
        grid_alpha=style.grid_alpha,
        grid_style=style.grid_linestyle,
        grid_color="0.8",
    )
    Plotter.set_tickparams(
        ax,
        labelsize=style.fontsize_tick,
        xticks=result.label_positions,
        xticklabels=result.label_texts,
    )
    return result

def plot_correlation(
    ax=None,
    *,
    corr_matrix: np.ndarray,
    lattice: Optional["Lattice"] = None,
    mode: Literal["matrix", "lattice", "kspace", "kpath"] = "lattice",
    interpolation: str = "linear",
    ref_site_idx: int = 0,
    reduction: Literal["none", "sum", "trace", "mean", "diag"] = "sum",
    norm: Literal["none", "cell", "site"] = "site",
    path=None,
    style: Optional[PlotStyle] = None,
    ks_config: Optional[KSpaceConfig] = None,
    kpath_config: Optional[KPathConfig] = None,
    value_label: Optional[str] = None,
    figsize: tuple[float, float] = (4.0, 3.5),
    dpi: Optional[int] = None,
    **kwargs,
):
    """
    Generic single-axis plotter for correlation matrices and their k-space views.

    Parameters
    ----------
    ax:
        Optional Matplotlib axis to plot on.
    corr_matrix:
        Real-space correlation matrix.
    lattice:
        Optional lattice instance. Required for ``mode='lattice'``,
        ``'kspace'``, and ``'kpath'``.
    mode:
        One of ``"matrix"``, ``"lattice"``, ``"kspace"``, or ``"kpath"``.
    interpolation:
        Default interpolation forwarded to the selected lower-level helper.
    ref_site_idx:
        Reference site for ``mode='lattice'``.
    reduction, norm:
        Forwarded to ``compute_correlation_kspace`` for k-space modes.
    path:
        High-symmetry path specification used for ``mode='kpath'``.
    style, ks_config, kpath_config:
        Optional style/config objects.
    value_label:
        Optional y-axis label for ``mode='kpath'``.
    figsize, dpi:
        Figure creation parameters used only when ``ax`` is None.
    **kwargs:
        This function forwards keywords to the selected helper.
        Supported aliases are:
        - ``rs_interp`` -> ``interpolation``
        - ``rs_point_size`` -> ``point_size``
        - ``rs_blob_radius`` -> ``blob_radius``
        - ``rs_show_interpolated_map`` -> ``show_interpolated_map``
        For the full accepted kwargs, see the docstrings of
        ``plot_realspace_correlations``, ``plot_kspace_intensity``, and
        ``plot_kspace_path``.
    """
    _, ax, _ = _prepare_axes(ax, figsize=figsize, dpi=dpi)
    style = PlotStyle() if style is None else style
    mode_key = mode.lower().strip()
    local_kwargs = dict(kwargs)

    if "rs_interp" in local_kwargs and "interpolation" not in local_kwargs:
        local_kwargs["interpolation"] = local_kwargs.pop("rs_interp")
    if "rs_point_size" in local_kwargs and "point_size" not in local_kwargs:
        local_kwargs["point_size"] = local_kwargs.pop("rs_point_size")
    if "rs_blob_radius" in local_kwargs and "blob_radius" not in local_kwargs:
        local_kwargs["blob_radius"] = local_kwargs.pop("rs_blob_radius")
    if "rs_show_interpolated_map" in local_kwargs and "show_interpolated_map" not in local_kwargs:
        local_kwargs["show_interpolated_map"] = local_kwargs.pop("rs_show_interpolated_map")

    if mode_key == "matrix":
        return plot_realspace_correlations(
            ax=ax,
            corr_matrix=corr_matrix,
            style=style,
            mode="matrix",
            interpolation=local_kwargs.pop("interpolation", interpolation),
            **local_kwargs,
        )
    if mode_key == "lattice":
        return plot_realspace_correlations(
            ax=ax,
            corr_matrix=corr_matrix,
            lattice=lattice,
            ref_site_idx=ref_site_idx,
            style=style,
            mode="lattice",
            interpolation=local_kwargs.pop("interpolation", interpolation),
            **local_kwargs,
        )
    if lattice is None:
        raise ValueError(f"lattice is required for mode='{mode_key}'.")
    if mode_key == "kspace":
        values, k_grid, _ = compute_correlation_kspace(
            lattice,
            corr_matrix,
            reduction=reduction,
            norm=norm,
        )
        return plot_kspace_intensity(
            ax=ax,
            k_values=k_grid,
            intensity=values,
            style=style,
            ks_config=ks_config,
            lattice=lattice,
            interpolation=interpolation,
            **local_kwargs,
        )
    if mode_key == "kpath":
        return plot_kspace_path(
            ax=ax,
            lattice=lattice,
            corr_matrix=corr_matrix,
            reduction=reduction,
            norm=norm,
            path=path,
            value_label=value_label or r"$S(\mathbf{k})$",
            style=style,
            kpath_config=kpath_config,
            **local_kwargs,
        )
    raise ValueError(f"Unknown correlation plot mode '{mode}'.")


# -----------------------------------------------------------------------------
# Spectral plots
# -----------------------------------------------------------------------------

def plot_spectral_function(
    ax=None,
    *,
    omega: np.ndarray,
    intensity: np.ndarray,
    k_values: Optional[np.ndarray] = None,
    lattice: Optional["Lattice"] = None,
    mode: Literal["sum", "single", "kpath", "grid"] = "sum",
    sum_axis: int = 0,
    k_indices: Optional[list[int]] = None,
    k_labels: Optional[list[str]] = None,
    path_labels: Optional[list[str]] = None,
    use_extend: bool = False,
    extend_copies: int = 2,
    sampled_only: bool = True,
    batch_index: Optional[Any] = None,
    style: Optional[PlotStyle] = None,
    ks_config: Optional[KSpaceConfig] = None,
    kpath_config: Optional[KPathConfig] = None,
    spectral_config: Optional[SpectralConfig] = None,
    figsize: tuple[float, float] = (4.0, 3.5),
    dpi: Optional[int] = None,
    colorbar: bool = False,
    fig=None,
    **kwargs,
):
    """
    Plot a spectral function on one axis.

    Modes
    -----
    ``sum``:
        Sum over a chosen k-axis and plot intensity vs omega.
    ``single``:
        Plot selected sampled k-points vs omega.
    ``kpath``:
        Plot a ``k-path x omega`` heatmap using the nearest sampled k-points.
    ``grid``:
        Plot a planar k-space intensity map at a selected omega or integrated
        over omega if no specific ``omega_value`` is requested.

    Parameters
    ----------
    ax:
        Optional Matplotlib axis to plot on.
    omega:
        Frequency / energy grid shaped like ``(Nomega,)``.
    intensity:
        Spectral data. Supported common layouts include ``(Nk, Nw)``,
        ``(Nw, Nk)``, ``(...grid..., Nw)``, ``(Nw, ...grid...)``, and the same
        with leading batch axes.
    k_values:
        Sampled Cartesian k-points, required for ``mode='kpath'`` and
        ``mode='grid'``.
    lattice:
        Lattice instance used for high-symmetry path selection and WS/BZ
        geometry.
    mode:
        One of ``"sum"``, ``"single"``, ``"kpath"``, or ``"grid"``.
    sum_axis:
        Axis reduced in ``mode='sum'`` after canonicalizing the spectral matrix.
    k_indices, k_labels:
        Selected sampled k-points and labels for ``mode='single'``.
    path_labels:
        High-symmetry path specification for ``mode='kpath'``.
    use_extend, extend_copies:
        Extended-zone matching options for path/grid views.
    sampled_only:
        If True, compress consecutive repeated path matches in ``mode='kpath'``.
    batch_index:
        Batch selector for spectral arrays with extra leading axes.
    style, ks_config, kpath_config, spectral_config:
        Optional style/config objects.
    figsize, dpi:
        Figure creation parameters used only when ``ax`` is None.
    colorbar, fig:
        Colorbar control forwarded to the 2D spectral renderer.
    **kwargs:
        Supported keyword overrides are:
        - any ``PlotStyle`` field
        - any ``KSpaceConfig`` field
        - any ``KPathConfig`` field
        - any ``SpectralConfig`` field, e.g. ``omega_value``,
          ``intensity_label``, ``colorbar_label``, ``vmin_omega``,
          ``vmax_omega``
        - line styling for ``mode='sum'`` / ``'single'``:
          ``line_color``, ``legend_loc``, ``xlabel``, ``ylabel``
        - remaining kwargs are forwarded to ``plot_spectral_function_2d`` for
          ``mode='kpath'`` and ``'grid'``
    """
    from .spectral_utils import plot_spectral_function_2d

    fig_local, ax, _ = _prepare_axes(ax, figsize=figsize, dpi=dpi)
    if fig is None:
        fig = fig_local

    style = PlotStyle() if style is None else style
    ks_config = KSpaceConfig() if ks_config is None else ks_config
    kpath_config = KPathConfig() if kpath_config is None else kpath_config
    spectral_config = SpectralConfig() if spectral_config is None else spectral_config
    overrides = dict(kwargs)
    _apply_overrides(style, overrides, _PLOT_STYLE_KEYS)
    _apply_overrides(ks_config, overrides, _KSPACE_CONFIG_KEYS)
    _apply_overrides(kpath_config, overrides, _KPATH_CONFIG_KEYS)
    _apply_overrides(spectral_config, overrides, _SPECTRAL_CONFIG_KEYS)

    omega_arr = np.asarray(omega, dtype=float).ravel()
    mode_key = mode.lower().strip()

    if mode_key in {"sum", "single"}:
        spec_matrix = _extract_spectral_matrix(np.asarray(intensity), omega_arr, batch_index=batch_index)
        if mode_key == "sum":
            y = np.sum(np.real_if_close(spec_matrix), axis=sum_axis)
            if np.asarray(y).ndim != 1 or len(np.asarray(y).ravel()) != len(omega_arr):
                raise ValueError(
                    "sum mode must reduce the k-axis and leave one omega-resolved curve. "
                    "Use the canonical spectral layout or choose a sum_axis that leaves length Nomega."
                )
            (line,) = ax.plot(
                omega_arr,
                y,
                lw=style.linewidth,
                color=overrides.pop("line_color", "C0"),
                alpha=style.alpha,
            )
            Plotter.set_ax_params(
                ax,
                xlabel=overrides.pop("xlabel", spectral_config.omega_label),
                ylabel=overrides.pop(
                    "ylabel",
                    spectral_config.colorbar_label
                    if getattr(spectral_config, "colorbar_label", None)
                    else spectral_config.intensity_label,
                ),
                fontsize=style.fontsize_label,
                grid=True,
                grid_alpha=style.grid_alpha,
                grid_color="0.8",
            )
            Plotter.set_tickparams(ax, labelsize=style.fontsize_tick)
            return line

        indices = [0] if k_indices is None else list(k_indices)
        lines = []
        for ii, idx in enumerate(indices):
            if idx >= spec_matrix.shape[0]:
                continue
            label = (
                f"k{idx}"
                if k_labels is None or ii >= len(k_labels)
                else k_labels[ii]
            )
            (line,) = ax.plot(
                omega_arr,
                np.real_if_close(spec_matrix[idx]),
                lw=style.linewidth,
                marker=style.marker,
                ms=style.markersize,
                alpha=style.alpha,
                label=label,
            )
            lines.append(line)
        Plotter.set_ax_params(
            ax,
            xlabel=overrides.pop("xlabel", spectral_config.omega_label),
            ylabel=overrides.pop(
                "ylabel",
                spectral_config.colorbar_label
                if getattr(spectral_config, "colorbar_label", None)
                else spectral_config.intensity_label,
            ),
            fontsize=style.fontsize_label,
            grid=True,
            grid_alpha=style.grid_alpha,
            grid_color="0.8",
            legend=len(lines) > 1,
        )
        Plotter.set_tickparams(ax, labelsize=style.fontsize_tick)
        if len(lines) > 1:
            ax.legend(loc=overrides.pop("legend_loc", "best"), fontsize=style.fontsize_legend)
        return lines

    if k_values is None:
        raise ValueError(f"k_values is required for mode='{mode_key}'.")

    flat_k, grid_shape = _flatten_kspace_points(k_values)
    spec_matrix = _extract_spectral_matrix(
        intensity,
        omega_arr,
        grid_shape=grid_shape,
        batch_index=batch_index,
    )

    if mode_key == "kpath":
        if lattice is None:
            raise ValueError("lattice is required for spectral k-path plots.")
        if path_labels is not None:
            kpath_config.path = path_labels

        selection = lattice.bz_path_points(
            path=kpath_config.path,
            points_per_seg=kpath_config.points_per_seg or 40,
            k_vectors=flat_k,
            tol=kpath_config.tolerance,
            periodic=not (use_extend or kpath_config.use_extend),
        )
        if not selection.has_matches:
            raise ValueError("Could not match the requested path to the sampled k-grid.")

        path_dist = np.asarray(selection.k_dist, dtype=float)
        path_indices = np.asarray(selection.matched_indices, dtype=int)
        path_matrix = spec_matrix[path_indices]
        if sampled_only:
            path_dist, path_matrix = _compress_path_runs(path_dist, path_matrix, path_indices)

        path_info = {
            "label_positions": np.asarray(
                [
                    selection.k_dist[min(idx, len(selection.k_dist) - 1)]
                    for idx, _ in selection.labels
                ],
                dtype=float,
            ),
            "label_texts": [label for _, label in selection.labels],
        }
        return plot_spectral_function_2d(
            ax,
            k_values=path_dist,
            omega=omega_arr,
            intensity=path_matrix,
            mode="kpath",
            path_info=path_info,
            style=style,
            kpath_config=kpath_config,
            spectral_config=spectral_config,
            colorbar=colorbar,
            fig=fig,
            **overrides,
        )

    if mode_key == "grid":
        return plot_spectral_function_2d(
            ax,
            k_values=k_values,
            omega=omega_arr,
            intensity=spec_matrix,
            mode="grid",
            style=style,
            ks_config=ks_config,
            spectral_config=spectral_config,
            lattice=lattice,
            use_extend=use_extend,
            extend_copies=extend_copies,
            colorbar=colorbar,
            fig=fig,
            **overrides,
        )

    raise ValueError(f"Unknown spectral plot mode '{mode}'.")


__all__ = [
    "compute_correlation_kspace",
    "plot_realspace_correlations",
    "plot_kspace_intensity",
    "plot_kspace_path",
    "plot_correlation",
    "plot_spectral_function",
]
