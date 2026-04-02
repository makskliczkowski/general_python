"""
Matplotlib-based visualisation helpers for lattice objects.

This module provides a comprehensive set of plotting utilities for visualizing
lattice structures, reciprocal space vectors, Brillouin zones, and other
geometric properties. It supports 1D, 2D, and 3D lattices.

Functions:
    - plot_real_space           : Scatter plot of real-space sites.
    - plot_reciprocal_space     : Scatter plot of reciprocal lattice vectors.
    - plot_brillouin_zone       : Visualization of the Brillouin Zone.
    - plot_lattice_structure    : Detailed connectivity plot with boundaries.

Classes:
    - LatticePlotter            : Convenience wrapper for lattice plotting.

----------------------------------------------------------------------
File    : general_python/lattices/visualization/plotting.py
Author  : Maksymilian Kliczkowski
Date    : 2025-02-01
----------------------------------------------------------------------
"""

from    __future__          import annotations
import  numpy               as np
import  matplotlib.pyplot   as plt

from    collections         import defaultdict
from    dataclasses         import dataclass
from    typing              import Optional, Tuple, List, Set, Dict, Union, Any, Iterable

from    matplotlib.axes     import Axes
from    matplotlib.figure   import Figure

# Optional dependencies for 3D and ConvexHull
try:
    from mpl_toolkits.mplot3d           import Axes3D
    from mpl_toolkits.mplot3d.art3d     import Poly3DCollection
except ImportError:
    Poly3DCollection = None

try:
    from scipy.spatial                  import ConvexHull
except ImportError:
    ConvexHull = None

from ..lattice import Lattice

# ==============================================================================
# Helpers
# ==============================================================================

def _ensure_numpy(vectors) -> np.ndarray:
    """ Ensure input is a 2D numpy array of float vectors. """
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, arr.shape[0])
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of vectors, got shape {arr.shape!r}.")
    return arr

def _init_axes(ax: Optional[Axes], dim: int, projection: Optional[str] = None) -> Tuple[Figure, Axes]:
    """ Initialize figure and axes if not provided. """
    if ax is not None:
        return ax.figure, ax
    
    if dim >= 3:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection=projection or "3d")
    else:
        fig, ax = plt.subplots()
        
    return fig, ax

def _annotate_indices(ax: Axes, coords: np.ndarray, *, zorder: int = 5, color: str = 'k', fontsize: int = 8, padding: float = 0.0) -> None:
    """ Annotate sites with their indices. """
    for idx, point in enumerate(coords):
        
        true_point = point + padding * np.sign(point) if padding != 0 else point
        
        if true_point.size >= 3:
            ax.text(true_point[0], true_point[1], true_point[2], str(idx), zorder=zorder, color=color, fontsize=fontsize)
        elif true_point.size == 2:
            ax.text(true_point[0], true_point[1], str(idx), zorder=zorder, color=color, fontsize=fontsize)
        else:
            ax.text(true_point[0], 0.0, str(idx), zorder=zorder, color=color, fontsize=fontsize)

def _finalise_figure(fig: Figure, *, top_padding: float = 0.88) -> None:
    """ Apply layout adjustments. """
    try:
        # Avoid subplots_adjust if a layout engine (e.g. constrained_layout) is active
        if hasattr(fig, 'get_layout_engine') and fig.get_layout_engine() is not None:
            return
            
        fig.subplots_adjust(top=top_padding)
        # Tight layout can be problematic with 3D axes in some mpl versions
        # fig.tight_layout() 
    except Exception:
        pass

def _apply_planar_aspect(axis: Axes, *, fix_aspect: bool = True) -> None:
    """Apply or release equal aspect for 2D axes."""
    axis.set_aspect("equal" if fix_aspect else "auto", adjustable="box")

def _apply_spatial_limits(
    axis        : Axes, 
    coords      : np.ndarray, 
    dim         : int, 
    show_axes   : bool, 
    margin      : float = 0.08,
    labels      : Optional[Tuple[str, ...]] = None,
    fix_aspect  : bool = True,
) -> None:
    """ 
    Uniformly apply spatial limits with padding and aspect ratio. 
    Ensures that nodes and annotations near the edges are not clipped.
    """
    if coords.size == 0:
        return

    # Calculate data bounds
    mins    = coords.min(axis=0)
    maxs    = coords.max(axis=0)
    
    # Calculate span-based padding
    spans   = maxs - mins
    
    # 1D case
    if dim == 1:
        span    = spans[0]
        pad     = margin * (span if span > 0 else 1.0)
        axis.set_xlim(mins[0] - pad, maxs[0] + pad)
        axis.set_ylim(-0.5, 0.5)
        
    # 2D case
    elif dim == 2:
        diag    = np.sqrt(spans[0]**2 + spans[1]**2)
        pad     = margin * (diag if diag > 0 else 1.0)
        
        axis.set_xlim(mins[0] - pad, maxs[0] + pad)
        axis.set_ylim(mins[1] - pad, maxs[1] + pad)
        _apply_planar_aspect(axis, fix_aspect=fix_aspect)
        
    # 3D case
    else:
        diag    = np.sqrt(np.sum(spans[:3]**2))
        pad     = margin * (diag if diag > 0 else 1.0)
        
        axis.set_xlim(mins[0] - pad, maxs[0] + pad)
        axis.set_ylim(mins[1] - pad, maxs[1] + pad)
        axis.set_zlim(mins[2] - pad, maxs[2] + pad)

    # Handle axis visibility
    if not show_axes:
        if dim == 1:
            axis.get_yaxis().set_visible(False)
            for sp in axis.spines.values(): sp.set_visible(False)
        else:
            # Instead of set_axis_off(), we hide spines and ticks to keep 
            # the axes object useful for layout engines (like constrained_layout)
            axis.set_xticks([])
            axis.set_yticks([])
            if hasattr(axis, 'set_zticks'):
                axis.set_zticks([])
            for sp in axis.spines.values():
                sp.set_visible(False)
            axis.patch.set_alpha(0.0) # Hide background patch
    else:
        lbls = labels or ("x", "y", "z")
        axis.set_xlabel(lbls[0])
        if dim >= 2: axis.set_ylabel(lbls[1])
        if dim >= 3: axis.set_zlabel(lbls[2])

# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_real_space(
    lattice         : Lattice,
    *,
    ax              : Optional[Axes]                = None,
    show_indices    : bool                          = False,
    show_axes       : bool                          = True,
    color           : str                           = "C0",
    marker          : str                           = "o",
    figsize         : Optional[Tuple[float, float]] = None,
    fix_aspect      : bool                          = True,
    title           : Optional[str]                 = None,
    title_kwargs    : Optional[Dict[str, object]]   = None,
    tight_layout    : bool                          = True,
    elev            : Optional[float]               = None,
    azim            : Optional[float]               = None,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    r"""
    Scatter-plot of real-space lattice vectors.

    Parameters
    ----------
    lattice : Lattice
        The lattice object to plot.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    show_indices : bool, default=False
        If True, annotate each site with its index.
    show_axes : bool, default=True
        If False, hides the coordinate axes.
    color : str, default="C0"
        Color of the site markers.
    marker : str, default="o"
        Marker style.
    figsize : tuple, optional
        Figure size in inches (width, height).
    fix_aspect : bool, default=True
        If True, preserve equal axis scaling in 2D plots. Set to ``False`` to
        let the requested ``figsize`` control the on-screen aspect.
    title : str, optional
        Title of the plot.
    elev, azim : float, optional
        Elevation and azimuth angles for 3D plots.
    **scatter_kwargs
        Additional arguments passed to `ax.scatter`.

    Returns
    -------
    fig, ax : Tuple[Figure, Axes]
        The figure and axes objects.
    """
    coords      = _ensure_numpy(lattice.rvectors)
    target_dim  = lattice.dim if lattice.dim else coords.shape[1]
    dim         = max(1, min(coords.shape[1], target_dim, 3))
    coords      = coords[:, :dim]
    
    fig, axis   = _init_axes(ax, dim)
    
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)
        
    if dim == 3 and (elev is not None or azim is not None):
        current_elev = elev if elev is not None else getattr(axis, "elev", None)
        current_azim = azim if azim is not None else getattr(axis, "azim", None)
        axis.view_init(elev=current_elev, azim=current_azim)

    # Plotting based on dimension
    if dim == 1:
        axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), color=color, marker=marker, **scatter_kwargs)
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], color=color, marker=marker, **scatter_kwargs)
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=color, marker=marker, **scatter_kwargs)

    # Spatial Limits & Visibility
    _apply_spatial_limits(axis, coords, dim, show_axes, fix_aspect=fix_aspect)

    # Title
    if title:
        kw = {"pad": 12}
        if title_kwargs: kw.update(title_kwargs)
        axis.set_title(title, **kw)

    # Annotations
    if show_indices:
        _annotate_indices(axis, coords)

    if tight_layout:
        _finalise_figure(fig)
        
    return fig, axis

# -------------------------------------------------------------------------------
#! Reciprocal Space Plotter
# -------------------------------------------------------------------------------

def plot_reciprocal_space(
    lattice         : Lattice,
    *,
    ax              : Optional[Axes]                = None,
    show_indices    : bool                          = False,
    show_axes       : bool                          = True,
    color           : str                           = "C1",
    marker          : str                           = "o",
    figsize         : Optional[Tuple[float, float]] = None,
    fix_aspect      : bool                          = True,
    title           : Optional[str]                 = None,
    title_kwargs    : Optional[Dict[str, object]]   = None,
    tight_layout    : bool                          = True,
    elev            : Optional[float]               = None,
    azim            : Optional[float]               = None,
    # extension
    extend_kpoints  : bool                          = False,
    extend_copies   : Union[int, Iterable[int]]     = 2,
    extend_tol      : float                         = 1e-10,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Scatter-plot of reciprocal lattice vectors (k-points).
    
    Parameters mirror :func:`plot_real_space`
    --------------------------------------------------------------------------
    lattice : Lattice
        The lattice object to plot.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    show_indices : bool, default=False
        If True, annotate each k-point with its index.
    show_axes : bool, default=True
        If False, hides the coordinate axes.
    color : str, default="C1"
        Color of the k-point markers.
    marker : str, default="o"
        Marker style.
    figsize : tuple, optional
        Figure size in inches (width, height).
    fix_aspect : bool, default=True
        If True, preserve equal axis scaling in 2D plots. Set to ``False`` to
        let the requested ``figsize`` control the on-screen aspect.
    title : str, optional
        Title of the plot.
    elev, azim : float, optional
        Elevation and azimuth angles for 3D plots.
    extend_kpoints : bool, default=False
        If True, draw translated reciprocal-space copies around the original mesh.
    extend_copies : int or iterable of int, default=2
        Number of copies per reciprocal direction used when ``extend_kpoints=True``.
        Scalars are applied to all active reciprocal directions.
    extend_tol : float, default=1e-10
        Tolerance used to identify which extended points are already present in
        the original reciprocal mesh.
    **scatter_kwargs
        Include:
        - point_edgecolor: Color of the marker edges (default "white").
        - point_zorder: Z-order for the scatter points (default 5).
        - color_extended: Color for translated copies (default "C2").
        - edgecolor_extended: Edge color for translated copies (default "gray").
        - marker_extended: Marker for translated copies (default ``marker``).
        - Any other valid arguments for `ax.scatter`.
    """
    coords      = _ensure_numpy(lattice.kvectors)
    target_dim  = lattice.dim if lattice.dim else coords.shape[1]
    dim         = max(1, min(coords.shape[1], target_dim, 3))
    coords      = coords[:, :dim]
    
    fig, axis   = _init_axes(ax, dim)
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)
        
    # Set 3D view angles if specified
    if dim == 3 and (elev is not None or azim is not None):
        current_elev = elev if elev is not None else getattr(axis, "elev", None)
        current_azim = azim if azim is not None else getattr(axis, "azim", None)
        axis.view_init(elev=current_elev, azim=current_azim)

    # Scatter plot of k-points -> a simple scatter with appropriate axis labels
    point_edgecolor     = scatter_kwargs.pop("edgecolor", "white")
    point_zorder        = scatter_kwargs.pop("zorder", 5)
    point_color         = color if color else scatter_kwargs.get("color", "C1")
    color_extended      = scatter_kwargs.pop("color_extended", "C2")
    edgecolor_extended  = scatter_kwargs.pop("edgecolor_extended", "gray")
    marker_extended     = scatter_kwargs.pop("marker_extended", marker)
    if dim == 1:
        axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), color=point_color, marker=marker, edgecolor=point_edgecolor, zorder=point_zorder, **scatter_kwargs)
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], color=point_color, marker=marker, edgecolor=point_edgecolor, zorder=point_zorder, **scatter_kwargs)
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=point_color, marker=marker, edgecolor=point_edgecolor, zorder=point_zorder, **scatter_kwargs)

    plotted_coords = coords

    # Set the title if necessary
    if title:
        kw = {"pad": 12}
        if title_kwargs: 
            kw.update(title_kwargs)
        axis.set_title(title, **kw)

    if show_indices:
        _annotate_indices(axis, coords)
        
    if extend_kpoints:
        active_dim = max(1, min(dim, 3))
        if np.isscalar(extend_copies):
            copy_spec   = int(extend_copies)
        else:
            copy_values = [int(copy) for copy in extend_copies]
            if len(copy_values) < active_dim:
                raise ValueError("extend_copies must provide at least one value per plotted reciprocal direction")
            copy_spec   = tuple(copy_values[:active_dim])

        extended_k_points, _    = lattice.wigner_seitz_extend(k_points=coords, copies=copy_spec)

        # Compare by rounded row keys to avoid quadratic allclose scans.
        scale                   = max(float(extend_tol), np.finfo(float).eps)
        original_keys           = {tuple(np.rint(row / scale).astype(np.int64)) for row in np.asarray(coords, dtype=float)}
        extended_keys           = [tuple(np.rint(row / scale).astype(np.int64)) for row in np.asarray(extended_k_points, dtype=float)]
        extended_k_points_mask  = np.array([key not in original_keys for key in extended_keys], dtype=bool)
        
        # plot other k-points in a different style
        if np.any(extended_k_points_mask):
            extended_coords     = extended_k_points[extended_k_points_mask]
            plotted_coords      = np.vstack((coords, extended_coords))
            if dim == 1:
                axis.scatter(extended_coords[:, 0], np.zeros_like(extended_coords[:, 0]), color=color_extended, marker=marker_extended, edgecolor=edgecolor_extended, zorder=point_zorder-1, **scatter_kwargs)
            elif dim == 2:
                axis.scatter(extended_coords[:, 0], extended_coords[:, 1], color=color_extended, marker=marker_extended, edgecolor=edgecolor_extended, zorder=point_zorder-1, **scatter_kwargs)
            else:
                axis.scatter(extended_coords[:, 0], extended_coords[:, 1], extended_coords[:, 2], color=color_extended, marker=marker_extended, edgecolor=edgecolor_extended, zorder=point_zorder-1, **scatter_kwargs)

    # Spatial Limits & Visibility
    k_labels = (r"$k_x$", r"$k_y$", r"$k_z$")
    _apply_spatial_limits(axis, plotted_coords, dim, show_axes, labels=k_labels, fix_aspect=fix_aspect)

    if tight_layout:
        _finalise_figure(fig)
        
    return fig, axis

# -------------------------------------------------------------------------------
#! Brillouin Zone Plotter
# -------------------------------------------------------------------------------

def _draw_bz_region(axis: Axes, points: np.ndarray, *,
    dim             : int,
    lattice         : Optional[Lattice] = None,
    offset          : Optional[np.ndarray] = None,
    shells          : int = 2,
    facecolor       : str,
    edgecolor       : str,
    alpha           : float,
    linewidth       : float = 1.5,
    fix_aspect      : bool = True,
    show_points     : bool = True,
    point_kwargs    : Optional[Dict[str, Any]] = None,
    zorder          : float = 0.0,
    **kwargs,
) -> None:
    """
    Draw a Brillouin-zone region from sampled boundary points. It allows
    to visualize the Wigner-Seitz cell of the reciprocal lattice, which is the
    Brillouin zone, by plotting the region defined by the given boundary points.
    
    Parameters
    ----------
    axis : Axes
        The matplotlib axes to draw on.
    points : array-like
        An array of shape (N, D) containing the coordinates of the boundary points.
    dim : int
        The dimensionality of the points (1, 2, or 3).
    lattice : Lattice, optional
        If provided, the lattice's Wigner-Seitz mask will be used to draw the Brillouin zone region. This can provide a more accurate representation of the BZ shape.
    offset : array-like, optional
        An optional offset to apply to the points when using the lattice's Wigner-Seitz mask. This can be used to visualize the BZ region around a specific k-point.
    shells : int, default=2
        When using the lattice's Wigner-Seitz mask, this parameter controls how many shells of reciprocal lattice vectors to consider when determining the mask. A larger number of shells can provide a more accurate BZ shape but may increase computation time.
    facecolor : str
        The color to fill the Brillouin zone region.
    edgecolor : str
        The color to use for the edges of the Brillouin zone region.
    alpha : float
        The transparency level for the filled region (0.0 transparent, 1.0 opaque).
    linewidth : float, default=1.5
        The width of the edges of the Brillouin zone region.
    show_points : bool, default=True
        If True, the original boundary points will be plotted on top of the filled region.
    point_kwargs : dict, optional
        Additional keyword arguments to pass to the scatter function when plotting the boundary points (e.g., marker style, size).
    zorder : float, default=0.0
        The z-order for the filled region and edges. Points will be plotted at zorder + 0.2 to ensure they are on top.
    """
    pts             = np.asarray(points, dtype=float)
    point_kwargs    = {} if point_kwargs is None else dict(point_kwargs)

    if dim == 1:
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        axis.axvspan(x_min, x_max, ymin=0.25, ymax=0.75, facecolor=facecolor, alpha=alpha, zorder=zorder)
        axis.plot([x_min, x_max], [0.5, 0.5], color=edgecolor, linewidth=linewidth, zorder=zorder + 0.1, **kwargs)
        if show_points:
            axis.scatter([x_min, x_max], [0.5, 0.5], color=edgecolor, edgecolor="white", zorder=zorder + 0.2, **point_kwargs)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xlabel("k")
        return

    if dim == 2:
        if lattice is not None:
            # Use the lattice's Wigner-Seitz mask to draw the Brillouin zone region
            b1      = np.asarray(lattice.k1, dtype=float).ravel()[:2]
            b2      = np.asarray(lattice.k2, dtype=float).ravel()[:2]
            # pad to ensure we cover the entire region even if the Wigner-Seitz cell is slightly larger than the sampled points
            pad     = 1.35
            kmax    = max(np.linalg.norm(b1), np.linalg.norm(b2)) * pad
            
            # Generate a grid of points around the origin (or offset) to evaluate the Wigner-Seitz mask
            gx      = np.linspace(-kmax, kmax, 500)
            gy      = np.linspace(-kmax, kmax, 500)
            
            # Apply offset if provided
            if offset is not None:
                shift   = np.asarray(offset, dtype=float).ravel()[:2]
                gx     += shift[0]
                gy     += shift[1]
            else:
                shift   = np.zeros(2, dtype=float)
            GX, GY  = np.meshgrid(gx, gy)
            mask    = lattice.wigner_seitz_mask(GX - shift[0], GY - shift[1], shells=shells)
            
            # Plot the filled contour for the Wigner-Seitz cell and its edges
            axis.contourf(GX, GY, mask.astype(float), levels=[0.5, 1.5], colors=[facecolor], alpha=alpha, zorder=zorder)
            # The contour for the edge is plotted with a slightly higher zorder to ensure it appears on top of the filled region
            axis.contour(GX, GY, mask.astype(float), levels=[0.5], colors=[edgecolor], linewidths=linewidth, zorder=zorder + 0.1)
        else:
            # If no lattice is provided, we can attempt to draw a convex hull around the points to represent the BZ region. 
            # This is a fallback and may not accurately capture the true BZ shape, especially if 
            # the points are not sampled densely or if the BZ has a complex shape.
            polygon = None
            if ConvexHull is not None:
                try:
                    hull    = ConvexHull(pts[:, :2])
                    polygon = pts[hull.vertices, :2]
                except Exception:
                    pass
            if polygon is None:
                x_min, y_min    = pts[:, :2].min(axis=0)
                x_max, y_max    = pts[:, :2].max(axis=0)
                polygon         = np.array([
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max],
                ])
            axis.fill(*polygon.T, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
            axis.plot(*polygon.T, color=edgecolor, linewidth=linewidth, zorder=zorder + 0.1)
            axis.plot([polygon[-1, 0], polygon[0, 0]], [polygon[-1, 1], polygon[0, 1]], color=edgecolor, linewidth=linewidth, zorder=zorder + 0.1)
        if show_points:
            axis.scatter(pts[:, 0], pts[:, 1], color=edgecolor, edgecolor="white", zorder=zorder + 0.2, **point_kwargs)
        _apply_planar_aspect(axis, fix_aspect=fix_aspect)
        axis.set_xlabel(r"$k_x$")
        axis.set_ylabel(r"$k_y$")
        return

    # ---------------
    # For 3D, we attempt to create a convex hull of the points to represent the BZ region.
    # ---------------

    if Poly3DCollection is None:
        raise RuntimeError("3D plotting support requires mpl_toolkits.mplot3d.")

    faces = None
    if ConvexHull is not None:
        try:
            hull    = ConvexHull(pts[:, :3])
            faces   = [pts[simplex, :3] for simplex in hull.simplices]
        except Exception:
            pass
    if faces is None:
        mins    = pts[:, :3].min(axis=0)
        maxs    = pts[:, :3].max(axis=0)
        corners = np.array([
            [mins[0], mins[1], mins[2]], [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]], [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]], [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]], [mins[0], maxs[1], maxs[2]],
        ])
        faces = [
            corners[[0, 1, 2, 3]], corners[[4, 5, 6, 7]],
            corners[[0, 1, 5, 4]], corners[[2, 3, 7, 6]],
            corners[[1, 2, 6, 5]], corners[[3, 0, 4, 7]],
        ]
    collection = Poly3DCollection(
        faces, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
        linewidths=linewidth, zorder=zorder
    )
    axis.add_collection3d(collection)
    if show_points:
        axis.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=edgecolor, edgecolor="white", zorder=zorder + 0.2, **point_kwargs)
    axis.set_xlabel(r"$k_x$")
    axis.set_ylabel(r"$k_y$")
    axis.set_zlabel(r"$k_z$")

def _plot_1d_bz(axis: Axes, bounds: Tuple[float, float], *, facecolor: str, alpha: float, **kwargs) -> None:
    ''' Plot a 1D Brillouin Zone as a horizontal band. '''
    _draw_bz_region(axis, points=np.array(bounds).reshape(1, 2), dim=1, facecolor=facecolor, alpha=alpha, **kwargs)

def _plot_2d_bz(axis: Axes, points: np.ndarray, *, facecolor: str, edgecolor: str, alpha: float, **kwargs) -> None:
    ''' Plot a 2D Brillouin Zone as a filled polygon. '''
    _draw_bz_region(axis, points, dim=2, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs)
    
def _plot_3d_bz(axis: Axes, points: np.ndarray, *, facecolor: str, edgecolor: str, alpha: float, elev: float = 30.0, azim: float = 45.0, **kwargs) -> None:
    ''' Plot a 3D Brillouin Zone as a convex hull polyhedron. '''
    _draw_bz_region(axis, points, dim=3, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, **kwargs)
    axis.view_init(elev=elev, azim=azim)
    
def plot_brillouin_zone(
    lattice         : Lattice,
    *,
    ax              : Optional[Axes]                = None,
    facecolor       : str                           = "tab:blue",
    edgecolor       : str                           = "black",
    alpha           : float                         = 0.25,
    figsize         : Optional[Tuple[float, float]] = None,
    fix_aspect      : bool                          = True,
    title           : Optional[str]                 = None,
    title_kwargs    : Optional[Dict[str, object]]   = None,
    tight_layout    : bool                          = True,
    elev            : Optional[float]               = None,
    azim            : Optional[float]               = None) -> Tuple[Figure, Axes]:
    """
    Plot the Brillouin Zone approximation based on sampled k-vectors.
    
    Parameters
    ----------
    lattice : Lattice
        The lattice object containing k-vectors.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    facecolor : str, default="tab:blue"
        Color to fill the Brillouin Zone area.
    edgecolor : str, default="black"
        Color for the Brillouin Zone boundary.
    alpha : float, default=0.25 
        Transparency level for the Brillouin Zone fill.
    figsize : tuple, optional   
        Figure size in inches (width, height).
    fix_aspect : bool, default=True
        If True, preserve equal axis scaling in 2D plots. Set to ``False`` to
        let the requested ``figsize`` control the on-screen aspect.
    title : str, optional
        Title of the plot.
    elev, azim : float, optional
        Elevation and azimuth angles for 3D plots.
    """
    coords      = _ensure_numpy(lattice.kvectors)
    target_dim  = lattice.dim if lattice.dim else coords.shape[1]
    dim         = max(1, min(coords.shape[1], target_dim, 3))
    coords      = coords[:, :dim]
    
    fig, axis   = _init_axes(ax, dim)
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)

    if dim == 1:
        _plot_1d_bz(axis, (coords[:, 0].min(), coords[:, 0].max()), facecolor=facecolor, alpha=alpha)
    elif dim == 2:
        _plot_2d_bz(axis, coords[:, :2], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, fix_aspect=fix_aspect)
    else:
        _plot_3d_bz(axis, coords[:, :3], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, elev=elev, azim=azim)

    if title:
        kw = {"pad": 12}
        if title_kwargs: 
            kw.update(title_kwargs)
        axis.set_title(title, **kw)

    if tight_layout:
        _finalise_figure(fig)
        
    return fig, axis

# ==============================================================================
#! Structural Plotting Helpers
# ==============================================================================

def _gather_nn_edges(lattice: Lattice) -> List[Tuple[int, int]]:
    """ Extract nearest-neighbor edges from lattice. """
    edges = set()
    
    for i in range(lattice.Ns):
        
        neighbors = lattice.get_nn(i)
        
        if not neighbors: 
            continue
        
        for j in neighbors:
            if lattice.wrong_nei(j): 
                continue
            
            # Canonical edge (min, max) to avoid duplicates
            a, b = sorted((int(i), int(j)))
            if a != b:
                edges.add((a, b))
    return sorted(edges)

def _infer_bipartite_coloring(adjacency: List[List[int]]) -> Optional[List[int]]:
    """ Try to 2-color the graph. Returns list of 0/1 colors or None if not bipartite. """
    
    ns      = len(adjacency)
    colors  = [-1] * ns
    
    for start in range(ns):
        if colors[start] != -1 or not adjacency[start]:
            continue
            
        colors[start]   = 0
        queue           = [start]
        while queue:
            node        = queue.pop(0)
            for neigh in adjacency[node]:
                if neigh < 0: continue
                
                if colors[neigh] == -1:
                    colors[neigh] = colors[node] ^ 1
                    queue.append(neigh)
                    
                elif colors[neigh] == colors[node]:
                    return None # Not bipartite
                    
    # Fill any disconnected single nodes
    for idx, neighbours in enumerate(adjacency):
        if colors[idx] == -1:
            colors[idx] = 0
            
    return colors

def _boundary_masks(positions: np.ndarray, lattice: Lattice, *, tol_factor: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """ Identify sites on the spatial boundaries of the lattice. """
    if lattice.dim == 0 or positions.size == 0:
        return np.zeros(positions.shape[0], dtype=bool), np.ones(positions.shape[0], dtype=bool)

    mins            = positions.min(axis=0)
    maxs            = positions.max(axis=0)
    span            = np.maximum(maxs - mins, tol_factor)
    tol             = span * tol_factor

    boundary_mask   = np.zeros(positions.shape[0], dtype=bool)
    
    # Check boundaries in each dimension
    for axis in range(min(positions.shape[1], 3)):
        boundary_axis = (np.isclose(positions[:, axis], mins[axis], atol=tol[axis]) |
                         np.isclose(positions[:, axis], maxs[axis], atol=tol[axis]))
        boundary_mask |= boundary_axis

    interior_mask = ~boundary_mask
    return boundary_mask, interior_mask

def _draw_primitive_cell(axis: Axes, origin: np.ndarray, basis_vectors: List[np.ndarray], dim: int, **kwargs) -> None:
    """ Draw the primitive unit cell vectors from an origin. """
    
    if not basis_vectors: 
        return
    
    color       = kwargs.get("color",       "0.4")
    linestyle   = kwargs.get("linestyle",   ":")
    linewidth   = kwargs.get("linewidth",   1.0)

    if dim == 1 and len(basis_vectors) >= 1:
        points  = np.vstack([origin, origin + basis_vectors[0]])
        axis.plot(points[:, 0], np.zeros_like(points[:, 0]), color=color, linestyle=linestyle, linewidth=linewidth)
        
    elif dim == 2 and len(basis_vectors) >= 2:
        a1, a2  = basis_vectors[:2]
        corners = np.array([origin, origin + a1, origin + a1 + a2, origin + a2, origin])
        axis.plot(corners[:, 0], corners[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)
    
    elif dim == 3 and len(basis_vectors) >= 3:
        raise NotImplementedError("3D primitive cell plotting is not implemented yet.")

def _draw_boundary_annotations(
    axis            : Axes,
    positions       : np.ndarray,
    lattice         : Lattice,
    *,
    periodic_color  : str,
    open_color      : str,
    offset_fraction : float,
) -> None:
    """ Draw annotations indicating OBC/PBC on the plot axes. """
    if positions.shape[1] < 2: return

    mins    = positions.min(axis=0)
    maxs    = positions.max(axis=0)
    mid     = (mins + maxs) / 2.0
    diag    = np.linalg.norm(maxs[:2] - mins[:2])
    padding = offset_fraction * (diag if diag > 0 else 1.0)

    flags   = lattice.periodic_flags()
    labels  = ("x", "y", "z")

    def _annotate_axis(axis_index: int, label: str) -> None:
        is_periodic = bool(flags[axis_index])
        color       = periodic_color if is_periodic else open_color
        
        # Helper for common styles
        style_kw    = dict(color=color, lw=1.2, linestyle="--", alpha=0.8)
        arrow_kw    = dict(arrowstyle="->", color=color, lw=1.5, linestyle="--")
        text_kw     = dict(color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

        if axis_index == 0:  # X-direction boundaries
            y = mid[1] if positions.shape[1] > 1 else 0.0
            
            if is_periodic:
                axis.annotate(f"PBC {label}", xy=(maxs[0], y), xytext=(maxs[0] + padding, y),
                              ha="left", va="center", arrowprops=arrow_kw, **text_kw)
                axis.annotate("", xy=(mins[0], y), xytext=(mins[0] - padding, y), arrowprops=arrow_kw)
            else:
                # Draw lines indicating open boundaries
                axis.plot([mins[0], mins[0]], [mins[1], maxs[1]], **style_kw)
                axis.plot([maxs[0], maxs[0]], [mins[1], maxs[1]], **style_kw)
                axis.text(mid[0], maxs[1] + padding, f"Open {label}", ha="center", va="bottom", **text_kw)
                
        elif axis_index == 1:  # Y-direction boundaries
            x = mid[0]
            if is_periodic:
                axis.annotate(f"PBC {label}", xy=(x, maxs[1]), xytext=(x, maxs[1] + padding), ha="center", va="bottom", arrowprops=arrow_kw, **text_kw)
                axis.annotate("", xy=(x, mins[1]), xytext=(x, mins[1] - padding), arrowprops=arrow_kw)
            else:
                axis.plot([mins[0], maxs[0]], [mins[1], mins[1]], **style_kw)
                axis.plot([mins[0], maxs[0]], [maxs[1], maxs[1]], **style_kw)
                axis.text(mins[0] - padding, mid[1], f"Open {label}", ha="right", va="center", rotation=90, **text_kw)

    for idx in range(min(positions.shape[1], 2)):
        _annotate_axis(idx, labels[idx])

def plot_lattice_structure(
    lattice                     : Lattice,
    *,
    ax                          : Optional[Axes]                = None,
    show_indices                : bool                          = False,
    highlight_boundary          : bool                          = True,
    # related to boundary highlighting
    show_axes                   : bool                          = False,
    edge_color                  : str                           = "0.5",
    node_color                  : str                           = "tab:blue",
    boundary_node_color         : str                           = "tab:red",
    periodic_color              : str                           = "tab:orange",
    open_color                  : str                           = "tab:green",
    bond_colors                 : dict                          = { 0 : "tab:red", 1 : "tab:blue", 2: "tab:green" },
    # styling
    node_size                   : int                           = 30,
    edge_alpha                  : float                         = 0.7,
    label_padding               : float                         = 0.05,
    label_fontsize              : int                           = 8,
    boundary_offset             : float                         = 0.05,
    # general plot settings
    figsize                     : Optional[Tuple[float, float]] = None,
    fix_aspect                  : bool                          = True,
    title                       : Optional[str]                 = None,
    title_kwargs                : Optional[Dict[str, object]]   = None,
    tight_layout                : bool                          = True,
    elev                        : Optional[float]               = None,
    azim                        : Optional[float]               = None,
    partition_colors            : Optional[Tuple[str, ...]]     = None,
    show_periodic_connections   : bool                          = True,
    show_primitive_cell         : bool                          = True,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    r"""
    Visualise lattice geometry with connectivity, boundary cues, and sublattices.

    This function draws nodes and edges based on nearest-neighbor connectivity.
    It highlights boundaries, annotates PBCs, and can color nodes by bipartite
    partitioning if applicable.

    Parameters
    ----------
    lattice : Lattice
        The lattice model.
    show_indices : bool
        If True, annotates nodes with their site indices.
    highlight_boundary : bool
        If True, draws boundary nodes with a distinct color/edge.
    show_axes : bool
        If False, hides the coordinate axes for a cleaner diagram.
    partition_colors : tuple of str, optional
        Colors to use for bipartite/sublattice coloring. If provided, nodes are
        colored based on sublattice parity.
    show_periodic_connections : bool
        If True, indicates wrap-around connections textually or graphically.
    show_primitive_cell : bool
        If True, overlays the primitive unit cell vectors/box.
    fix_aspect : bool, default=True
        If True, preserve equal axis scaling in 2D plots. Set to ``False`` to
        let the requested ``figsize`` control the on-screen aspect.
    
    ... other parameters mirror plot_real_space ...
    """
    coords      = _ensure_numpy(lattice.rvectors)
    target_dim  = lattice.dim if lattice.dim else coords.shape[1]
    dim         = max(1, min(coords.shape[1], target_dim, 3))
    coords      = coords[:, :dim]

    fig, axis   = _init_axes(ax, dim)
    
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)
        
    if dim == 3 and (elev is not None or azim is not None):
        current_elev = elev if elev is not None else getattr(axis, "elev", None)
        current_azim = azim if azim is not None else getattr(axis, "azim", None)
        axis.view_init(elev=current_elev, azim=current_azim)

    # Compute Connectivity
    edges       = _gather_nn_edges(lattice)
    adjacency   = [[] for _ in range(lattice.Ns)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    # Periodic Edges Detection
    periodic_neighbors      = defaultdict(list)
    periodic_label_counts   = defaultdict(int)
    typical_distance        = None
    
    if edges:
        # Heuristic: distances significantly larger than min distance are likely PBC wraps
        all_dists           = [np.linalg.norm(coords[j] - coords[i]) for i, j in edges]
        valid_dists         = [d for d in all_dists if d > 1e-9]
        if valid_dists:
            typical_distance = min(valid_dists)

    # Draw edges
    for i, j in edges:
        start       = coords[i]
        end         = coords[j]
        dist        = np.linalg.norm(end - start)
        bond_type   = lattice.bond_type(i, j)
        
        # Check if this edge wraps around the boundary
        is_periodic = False
        if typical_distance is not None:
            # 1.5x factor is a safe heuristic for regular lattices
            if dist > typical_distance * 1.5:
                is_periodic = True

        # Draw logic
        linestyle = "--" if is_periodic else "-"
        
        if is_periodic:
            periodic_neighbors[i].append(j)
            periodic_neighbors[j].append(i)
        
        # Plot the line
        line_args = dict(color      =   bond_colors.get(bond_type, edge_color), 
                         alpha      =   edge_alpha,
                         linestyle  =   linestyle, 
                         linewidth  =   1.0,
                         zorder     =   2)
        
        if dim == 3:
            axis.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **line_args)
        elif dim == 2:
            axis.plot([start[0], end[0]], [start[1], end[1]], **line_args)
        else: # 1D
            axis.plot([start[0], end[0]], [0.0, 0.0], **line_args)

    # Node Coloring
    node_face_colors    = [node_color] * lattice.Ns
    
    if partition_colors:
        partitions = _infer_bipartite_coloring(adjacency)
        if partitions is not None:
            palette             = partition_colors
            node_face_colors    = [palette[partitions[i] % len(palette)] for i in range(lattice.Ns)]

    # Draw Nodes
    scatter_defaults = dict(s=node_size, zorder=3, **scatter_kwargs)
    
    if dim == 1:
        axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), c=node_face_colors, **scatter_defaults)
        axis.set_ylim(-0.5, 0.5)
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], c=node_face_colors, **scatter_defaults)
        _apply_planar_aspect(axis, fix_aspect=fix_aspect)
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=node_face_colors, **scatter_defaults)

    # Boundary Highlight
    boundary_mask, _ = _boundary_masks(coords, lattice)
    if highlight_boundary and np.any(boundary_mask):
        b_coords = coords[boundary_mask]
        b_args   = dict(facecolors="none", edgecolors=boundary_node_color, s=node_size*1.2, linewidths=1.2, zorder=4)
        
        if dim == 3:
            axis.scatter(b_coords[:, 0], b_coords[:, 1], b_coords[:, 2], **b_args)
        elif dim == 2:
            axis.scatter(b_coords[:, 0], b_coords[:, 1], **b_args)
        else:
            axis.scatter(b_coords[:, 0], np.zeros_like(b_coords[:, 0]), **b_args)

    # Spatial Limits & Visibility
    # Use a larger margin if we have boundary annotations or periodic labels
    margin = 0.08
    if dim == 2:
        margin = max(margin, boundary_offset * 1.5)
    
    _apply_spatial_limits(axis, coords, dim, show_axes, margin=margin, fix_aspect=fix_aspect)

    if title:
        kw = {"pad": 15}
        if title_kwargs: kw.update(title_kwargs)
        axis.set_title(title, **kw)

    # Indices & Annotations
    node_label_positions = {}
    
    if show_indices:
        mins    = coords.min(axis=0)
        maxs    = coords.max(axis=0)
        diag    = np.linalg.norm(maxs - mins) if coords.size else 1.0
        offset  = label_padding * (diag if diag > 0 else 1.0)
        
        for idx, point in enumerate(coords):
            label_pos = point.copy()
            if dim >= 1: label_pos[0] += offset
            if dim >= 2: label_pos[1] += offset
            if dim >= 3: label_pos[2] += offset
            
            txt_args = dict(ha="center", va="center", color="black", fontsize=label_fontsize,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1), zorder=6)
            
            if dim == 3:
                axis.text(label_pos[0], label_pos[1], label_pos[2], str(idx), **txt_args)
            elif dim == 2:
                axis.text(label_pos[0], label_pos[1], str(idx), **txt_args)
            else:
                axis.text(label_pos[0], offset, str(idx), **txt_args)
                
            node_label_positions[idx] = label_pos

    # Boundary Annotations (2D only)
    if dim == 2:
        _draw_boundary_annotations(axis, coords, lattice,
                                   periodic_color=periodic_color,
                                   open_color=open_color,
                                   offset_fraction=boundary_offset)

    # Periodic Connections Text
    if show_periodic_connections and periodic_neighbors:
        diag_extent = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) or 1.0
        base_offset = label_padding * diag_extent * 0.6
        
        for idx, neighbours in periodic_neighbors.items():
            if not neighbours: continue
            
            anchor  = node_label_positions.get(idx, coords[idx])
            label   = "↔ " + ",".join(str(n) for n in sorted(set(neighbours)))
            
            count   = periodic_label_counts[idx]
            periodic_label_counts[idx] += 1
            
            shift   = base_offset * (count + 1)
            pos     = anchor.copy()
            if dim == 1:
                pos[0] = pos[0] # Usually stack vertically in 1D? Or just use y-offset
                y_pos  = shift
            elif dim == 2:
                pos[1] += shift # Shift up y
            
            txt_args = dict(color=periodic_color, fontsize=8, ha="center", va="bottom",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.4), zorder=5)

            if dim == 3:
                pos[2] += shift
                axis.text(pos[0], pos[1], pos[2], label, **txt_args)
            elif dim == 2:
                axis.text(pos[0], pos[1], label, **txt_args)
            else:
                axis.text(pos[0], shift, label, **txt_args)

    # Primitive Cell
    if show_primitive_cell:
        # Try to find basis vectors
        basis_vectors = []
        
        for attr in ("a1", "a2", "a3"):
            
            vec = getattr(lattice, attr, None)
            if vec is not None:
                vec = np.asarray(vec).flatten()
                if vec.size >= dim and np.linalg.norm(vec[:dim]) > 1e-9:
                    basis_vectors.append(vec[:dim])
        
        if basis_vectors:
            origin = coords.min(axis=0)
            _draw_primitive_cell(axis, origin, basis_vectors, dim)

    if tight_layout:
        _finalise_figure(fig, top_padding=0.92)
        
    return fig, axis

# ==============================================================================
#! Region Plotting
# ==============================================================================

def _region_palette(n: int) -> List:
    """
    Return *n* high-contrast, distinguishable colours for region plots.

    Uses a hand-picked palette for small *n* (≤8) and falls back to the
    ``tab20`` colour-map for larger numbers.
    """
    # High-contrast hand-picked palette (colour-blind friendly order)
    _BASE = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
    ]
    if n <= len(_BASE):
        return _BASE[:n]
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", max(n, 20))
    return [cmap(i) for i in range(n)]

def _normalize_site_indices(indices_like: Any, n_sites: int) -> List[int]:
    """Convert supported index containers to sorted unique valid site indices."""
    if indices_like is None:
        return []
    if isinstance(indices_like, np.ndarray):
        raw = indices_like.ravel().tolist()
    elif isinstance(indices_like, (list, tuple, set)):
        raw = list(indices_like)
    else:
        return []

    out: Set[int] = set()
    for item in raw:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if 0 <= idx < n_sites:
                out.add(idx)
    return sorted(out)

def _extract_region_indices(region_spec: Any, n_sites: int, component: str = "A") -> List[int]:
    """
    Normalize one region descriptor to a site-index list.

    Supports:
    - plain lists/arrays of indices,
    - dicts containing region components,
    - Region-like objects (with .get/.A/.to_dict()).
    """
    direct = _normalize_site_indices(region_spec, n_sites)
    if direct:
        return direct

    comp = str(component).upper()

    if isinstance(region_spec, dict):
        if comp in region_spec:
            return _normalize_site_indices(region_spec.get(comp), n_sites)
        merged: List[int] = []
        for value in region_spec.values():
            v = _normalize_site_indices(value, n_sites)
            if v:
                merged.extend(v)
        return sorted(set(merged))

    get_fn = getattr(region_spec, "get", None)
    if callable(get_fn):
        try:
            cand = get_fn(comp, None)
        except Exception:
            cand = None
        cand_norm = _normalize_site_indices(cand, n_sites)
        if cand_norm:
            return cand_norm

    if hasattr(region_spec, comp):
        cand_norm = _normalize_site_indices(getattr(region_spec, comp), n_sites)
        if cand_norm:
            return cand_norm

    to_dict_fn = getattr(region_spec, "to_dict", None)
    if callable(to_dict_fn):
        try:
            mapping = to_dict_fn()
        except Exception:
            mapping = None
        if isinstance(mapping, dict):
            if comp in mapping:
                return _normalize_site_indices(mapping.get(comp), n_sites)
            merged: List[int] = []
            for value in mapping.values():
                v = _normalize_site_indices(value, n_sites)
                if v:
                    merged.extend(v)
            return sorted(set(merged))
    return []

def plot_regions(
    lattice             : Lattice,
    regions             : Union[Dict[str, Any], Any],
    *,
    ax                  : Optional[Axes]                = None,
    # showers
    show_indices        : bool                          = False,
    show_system         : bool                          = True,
    show_complement     : bool                          = False,
    show_labels         : bool                          = True,
    show_overlaps       : bool                          = True,
    show_bonds          : bool                          = False,
    show_legend         : bool                          = True,
    # Other points
    origin              : Optional[np.ndarray]          = None,
    system_color        : str                           = 'lightgray',
    system_alpha        : float                         = 0.25,
    region_colors       : Optional[Dict[str, str]]      = None,
    region_alpha        : float                         = 0.6,
    complement_color    : str                           = 'lightgray',
    complement_alpha    : float                         = 0.3,
    overlap_color       : str                           = 'red',
    fill                : bool                          = False,
    fill_alpha          : float                         = 0.2,
    # region styling
    blob_radius         : Optional[float]               = None,
    blob_alpha          : float                         = 0.12,
    marker_size         : int                           = 60,
    edge_width          : float                         = 1.5,
    # general plot settings
    figsize             : Optional[Tuple[float, float]] = None,
    fix_aspect          : bool                          = True,
    title               : Optional[str]                 = None,
    title_kwargs        : Optional[Dict[str, object]]   = None,
    tight_layout        : bool                          = True,
    elev                : Optional[float]               = None,
    azim                : Optional[float]               = None,
    # Region labels and legend
    region_descriptions : Optional[Dict[str, str]]      = None,
    legend_loc          : str                           = 'best',
    legend_fontsize     : int                           = 9,
    legend_bbox         : tuple                         = (1.05, 1),
    # Label styling
    label_fontsize      : int                           = 11,
    label_offset        : float                         = 1.2,
    # Indices and axes
    indices_padding     : float                         = 0.05,
    show_axes           : bool                          = False,
    region_component    : str                           = "A",
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot labelled lattice regions with distinct colours and informative legend.

    Features
    --------
    - High-contrast colours that are distinguishable for every region.
    - Labels placed *radially outward* from the plot centre so they never
      overlap even for Kitaev-Preskill-style pie-slice sectors.
    - Legend entries include the region name, site count, and optional
      human-readable description.
    - Optional convex-hull fill, translucent per-site blobs, and intra-region
      bond drawing.

    Parameters
    ----------
    lattice : Lattice
        The lattice object.
    regions : Dict[str, Any] or Region-like
        Region mapping or Region-like objects. Region-like values are
        resolved using ``region_component`` (default ``"A"``).
    region_descriptions : dict[str, str], optional
        Optional human-readable description per region key that is appended
        to the legend entry (e.g. ``{'A': 'sector 0°-120°'}``).
    legend_loc : str
        Matplotlib legend location string (default ``'best'``).
    legend_fontsize : int
        Font size for legend entries (default 9).
    label_fontsize : int
        Font size for region labels drawn on the plot (default 11).
    label_offset : float
        Controls how far the label is pushed radially outward from the
        region centroid (default 1.2 x distance from plot centre to
        centroid).  Values > 1 push the label outside the region.
    show_bonds : bool
        Draw NN bonds coloured by region.
    region_component : str
        Component to extract from Region-like entries (default ``"A"``).
        Ignored for plain index lists.
    blob_radius, blob_alpha : float
        Per-site circle patches (2D only).
    fill, fill_alpha : bool, float
        Convex-hull polygon fill (2D, requires scipy).
    (other parameters identical to previous version)
    show_axes : bool
        If False (default), hides the coordinate axes for a cleaner diagram.
    fix_aspect : bool, default=True
        If True, preserve equal axis scaling in 2D plots. Set to ``False`` to
        let the requested ``figsize`` control the on-screen aspect.
    """
    coords      = _ensure_numpy(lattice.rvectors)
    target_dim  = lattice.dim if lattice.dim else coords.shape[1]
    dim         = max(1, min(coords.shape[1], target_dim, 3))
    coords      = coords[:, :dim]
    fig, axis   = _init_axes(ax, dim)

    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)

    # Set 3D view if requested and applicable
    if dim == 3 and (elev is not None or azim is not None):
        axis.view_init(
            elev=elev if elev is not None else getattr(axis, "elev", None),
            azim=azim if azim is not None else getattr(axis, "azim", None),
        )

    # background: all system sites
    if show_system:
        _sc = dict(color=system_color, alpha=system_alpha, marker='o', s=30, zorder=0)
        if dim <= 2:
            y = coords[:, 1] if dim == 2 else np.zeros(len(coords))
            axis.scatter(coords[:, 0], y, **_sc)
        else:
            axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **_sc)
            
        # bonds 
        if show_bonds and dim <= 2:
            for i in range(len(coords)):
                for j in lattice.get_nn(i):
                    if lattice.wrong_nei(j):
                        continue
                    j = int(j)
                    if i < j: # Avoid double counting
                        ri, rj = coords[i, :2], coords[j, :2]
                        axis.plot([ri[0], rj[0]], [ri[1], rj[1]], color=system_color, lw=0.8, alpha=system_alpha * 0.8, zorder=0)

    if origin is not None:
        _sc = dict(color='black', alpha=0.8, marker='X', s=100, zorder=5)
        if dim <= 2:
            y = origin[1] if dim == 2 else 0.0
            axis.scatter(origin[0], y, **_sc)
        else:
            axis.scatter(origin[0], origin[1], origin[2], **_sc)

    # Normalize inputs: convert Region/dict/list descriptors to list[int]
    if not isinstance(regions, dict):
        regions = {"region": regions}

    normalized_regions  : Dict[str, List[int]] = {}
    n_sites             = len(coords)
    for name, spec in regions.items():
        normalized_regions[str(name)] = _extract_region_indices(spec, n_sites=n_sites, component=region_component)
    regions             = normalized_regions

    # site membership bookkeeping 
    all_region_sites            = set()
    site_counts: Dict[int, int] = {}
    for indices in regions.values():
        for idx in indices:
            all_region_sites.add(idx)
            site_counts[idx] = site_counts.get(idx, 0) + 1

    # complement and overlap sites for optional distinct styling
    complement_sites = [i for i in range(len(coords)) if i not in all_region_sites]
    overlap_sites    = [i for i, c in site_counts.items() if c > 1]

    # complement — small faint dots
    if show_complement and complement_sites:
        cc  = coords[complement_sites]
        _ca = dict(color=complement_color, alpha=complement_alpha, marker='o',
                   s=marker_size * 0.25, edgecolors='none', zorder=0)
        if dim <= 2:
            y = cc[:, 1] if dim == 2 else np.zeros(len(cc))
            axis.scatter(cc[:, 0], y, **_ca)
        else:
            axis.scatter(cc[:, 0], cc[:, 1], cc[:, 2], **_ca)

    # Global centroid (used for radial label placement)
    # colour palette
    palette             = _region_palette(len(regions))
    global_com          = np.mean(coords, axis=0)
    region_descriptions = region_descriptions or {}

    # draw each region
    for i, (name, indices) in enumerate(regions.items()):
        if not indices:
            continue

        rc    = coords[indices]
        color = palette[i % len(palette)] if region_colors is None else region_colors.get(name, palette[i % len(palette)])
        alpha = region_alpha
        n_pts = len(indices)

        # Build informative legend text
        desc  = region_descriptions.get(name, "")
        lbl   = f"{name}: {n_pts} sites"
        if desc:
            lbl += f" — {desc}"

        # Convex-hull fill (2D) - allows visualising the overall shape of the region even if the sites are sparse
        if fill and dim == 2 and ConvexHull is not None and len(rc) >= 3:
            try:
                hull   = ConvexHull(rc)
                hp     = rc[hull.vertices]
                axis.fill(hp[:, 0], hp[:, 1], color=color, alpha=alpha * fill_alpha, zorder=1)
            except Exception:
                pass

        # Per-site blobs (2D) - gives a visual sense of the site density and extent of the region
        if blob_radius is not None and dim == 2:
            from matplotlib.patches     import Circle as _Circle
            from matplotlib.collections import PatchCollection as _PC
            circles     = [_Circle((x, y), blob_radius) for x, y in rc[:, :2]]
            pc          = _PC(circles, facecolors=color, edgecolors='none', alpha=alpha * blob_alpha, zorder=1)
            axis.add_collection(pc)

        # Intra-region NN bonds (2D)
        if show_bonds and dim == 2:
            idx_set = set(indices)
            for si in indices:
                for nj in lattice.get_nn(si):
                    if lattice.wrong_nei(nj):
                        continue
                    nj = int(nj)
                    if nj in idx_set and nj > si:
                        ri, rj = coords[si, :2], coords[nj, :2]
                        axis.plot([ri[0], rj[0]], [ri[1], rj[1]], color=color, lw=1.2, alpha=alpha * 0.55, zorder=2)

        # Scatter markers
        sc_kw   = dict(color=color, marker='o', s=marker_size, edgecolors='black', linewidths=edge_width * 0.5, label=lbl, zorder=3, alpha=alpha)
        sc_kw.update(scatter_kwargs)
        
        if dim <= 2:
            y = rc[:, 1] if dim == 2 else np.zeros(len(rc))
            axis.scatter(rc[:, 0], y, **sc_kw)
        else:
            axis.scatter(rc[:, 0], rc[:, 1], rc[:, 2], **sc_kw)

        # Region label — placed radially outward from global centroid
        # Place near one of the first sites, not in centroid
        if show_labels and dim == 2 and len(rc) > 0:
            # Determine system scale for relative padding
            spans       = rc.max(axis=0) - rc.min(axis=0)
            diag        = np.sqrt(np.sum(spans**2))
            rel_pad     = 0.15 * (diag if diag > 1e-9 else 1.0)
            
            com         = np.mean(rc[:, :2], axis=0)
            direc       = com - global_com[:2]
            norm        = np.linalg.norm(direc)
            if norm < 1e-9:
                direc   = np.array([0.0, 1.0])
            else:
                direc   = direc / norm
            lbl_pos = com + direc * norm * (label_offset - 1.0) + direc * rel_pad

            axis.annotate(
                name,
                xy=com, xytext=lbl_pos,
                fontsize=label_fontsize, fontweight='bold', color=color,
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color,
                          alpha=0.85, lw=1.0),
                zorder=5,
            )
            
        # Fallback label placement for non-2D or if no sites (just put at centroid)
        elif show_labels and dim != 2 and len(rc) > 0:
            com     = np.mean(rc, axis=0)
            txt_kw  = dict(fontsize=label_fontsize, fontweight='bold', color=color,
                          ha='center', va='center', bbox=dict(fc='white', ec=color, alpha=0.8, pad=1.0))
            if dim == 1:
                axis.text(com[0], 0, name, **txt_kw)
            else:
                axis.text(com[0], com[1], com[2], name, **txt_kw)

    # overlap highlight -> draw on top of everything else with a distinct style
    if show_overlaps and overlap_sites:
        oc  = coords[overlap_sites]
        _oa = dict(color='none', edgecolors=overlap_color, marker='o',
                   s=marker_size * 1.5, linewidths=edge_width * 1.5,
                   label=f'Overlaps ({len(overlap_sites)})', zorder=4)
        if dim <= 2:
            y = oc[:, 1] if dim == 2 else np.zeros(len(oc))
            axis.scatter(oc[:, 0], y, **_oa)
        else:
            axis.scatter(oc[:, 0], oc[:, 1], oc[:, 2], **_oa)

    # site-index annotations
    if show_indices:
        _annotate_indices(axis, coords, padding=indices_padding)

    # Spatial Limits & Visibility
    # Use a larger margin if we have region labels to avoid clipping
    margin = 0.08
    if show_labels:
        # Increase margin to accommodate labels and their leader lines
        margin = max(margin, (label_offset - 1.0) * 0.4 + 0.1)
    
    _apply_spatial_limits(axis, coords, dim, show_axes, margin=margin, fix_aspect=fix_aspect)

    # title
    if title:
        kw = {"pad": 12}
        if title_kwargs:
            kw.update(title_kwargs)
        axis.set_title(title, **kw)
        
    elif ax is None:
        n_total    = len(coords)
        n_covered  = len(all_region_sites)
        cov_pct    = n_covered / n_total * 100 if n_total else 0
        parts      = [f"Regions ({len(regions)})  —  {n_covered}/{n_total} sites ({cov_pct:.0f}%)"]
        if overlap_sites:
            parts.append(f",  {len(overlap_sites)} overlaps")

    # legend (deduplicated, compact)
    if show_legend:
        handles, labels = axis.get_legend_handles_labels()
        by_label        = dict(zip(labels, handles))
        if by_label:
            axis.legend(
                by_label.values(), by_label.keys(),
                loc             =legend_loc, 
                fontsize        =legend_fontsize,
                bbox_to_anchor  =legend_bbox,
                framealpha      =0.90, 
                edgecolor       ='lightgray', 
                fancybox        =True,
                handletextpad   =0.3, 
                labelspacing    =0.25,
                borderpad       =0.4, 
                handlelength    =1.2,
                markerscale     =0.7,
                **scatter_kwargs,
            )

    if tight_layout:
        _finalise_figure(fig)

    return fig, axis

# ==============================================================================
# K-space / Brillouin-zone with High-Symmetry Points
# ==============================================================================

def plot_high_symmetry_points(
    lattice                 : Lattice,
    *,
    path                    : Optional[Union[List[str], str, Iterable[Tuple[str, Iterable[float]]]]] = None,
    selection               : Optional[Any]                 = None,
    ax                      : Optional[Axes]                = None,
    show_kpoints            : bool                          = True,
    show_bz                 : bool                          = True,
    show_path               : bool                          = True,
    show_matched_kpoints    : bool                          = True,
    bz_facecolor            : str                           = "lavender",
    bz_edgecolor            : str                           = "slategrey",
    bz_alpha                : float                         = 0.20,
    kpoint_color            : str                           = "C0",
    kpoint_alpha            : float                         = 0.35,
    kpoint_size             : int                           = 15,
    path_color              : str                           = "crimson",
    path_linewidth          : float                         = 1.8,
    matched_kpoint_color    : str                           = "goldenrod",
    matched_kpoint_alpha    : float                         = 1.0,
    matched_kpoint_size     : int                           = 44,
    matched_kpoint_marker   : str                           = "o",
    matched_kpoint_edgecolor: str                           = "black",
    hs_marker_size          : int                           = 90,
    hs_marker_facecolor     : str                           = "white",
    hs_marker_edgecolor     : str                           = "black",
    hs_font_size            : int                           = 13,
    hs_label_kwargs         : Optional[Dict[str, object]]   = None, 
    hs_plot                 : str                           = "markers", # "none", "markers", "labels", or "both"
    points_per_seg          : int                           = 40,
    path_match_tol          : Optional[float]               = None,
    figsize                 : Optional[Tuple[float, float]] = None,
    fix_aspect              : bool                          = True,
    title                   : Optional[str]                 = None,
    title_kwargs            : Optional[Dict[str, object]]   = None,
    tight_layout            : bool                          = True,
    extend                  : bool                          = False,
    extend_copies           : Optional[Union[int, Iterable[int]]] = None,
    nx                      : int                           = 1,
    ny                      : int                           = 1,
    nz                      : int                           = 1,
    extended_kpoint_color   : str                           = "gray",
    extended_kpoint_alpha   : float                         = 0.15,
    extended_bz_facecolor   : str                           = "lightgray",
    extended_bz_edgecolor   : str                           = "dimgray",
    extended_bz_alpha       : float                         = 0.10,
    show_background_bz      : bool                          = False,
    # legend
    legend_kwargs           : Optional[Dict[str, object]]   = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    r"""
    Plot the Brillouin zone, high-symmetry path, and sampled reciprocal mesh.

    This view combines exact reciprocal-space geometry from the lattice with an
    ideal high-symmetry path and, optionally, the subset of sampled k-points
    that genuinely match that path within a configurable tolerance.

    Parameters
    ----------
    lattice : Lattice
        Lattice object providing reciprocal vectors, sampled ``kvectors``, and
        optionally ``kvectors_frac`` and ``high_symmetry_points()``.
    path : list[str], str, or iterable[(label, frac)], optional
        High-symmetry path specification. If omitted, the lattice default path
        is used.
    selection : object, optional
        Precomputed output of ``lattice.bz_path_points(...)``. When provided,
        this exact matched set is used for path and matched-point rendering.
    ax : Axes, optional
        Existing matplotlib axes. If omitted, a new figure and axes are created.
    show_kpoints : bool, default=True
        Draw sampled reciprocal-space mesh points.
    show_bz : bool, default=True
        Draw the first Brillouin zone.
    show_path : bool, default=True
        Draw the ideal high-symmetry path.
    show_matched_kpoints : bool, default=True
        Highlight sampled k-points whose distance to the path is within the
        matching tolerance.
    bz_facecolor, bz_edgecolor, bz_alpha
        Style of the first Brillouin zone.
    kpoint_color, kpoint_alpha, kpoint_size
        Style of the original sampled k-mesh.
    path_color, path_linewidth
        Style of the ideal path.
    matched_kpoint_color, matched_kpoint_alpha, matched_kpoint_size,
    matched_kpoint_marker, matched_kpoint_edgecolor
        Style of valid matched mesh points.
    hs_marker_size, hs_marker_facecolor, hs_marker_edgecolor
        Style of exact high-symmetry vertices.
    hs_font_size, hs_label_kwargs
        Style of high-symmetry labels.
    hs_plot : {"none", "markers", "labels", "both"}, default="markers"
        Whether to draw exact high-symmetry markers, labels, or both.
    points_per_seg : int, default=40
        Number of interpolation points per path segment for the ideal path.
    path_match_tol : float, optional
        Distance tolerance for highlighting mesh points near the drawn path.
        If omitted, a mesh-based Cartesian tolerance is estimated from the
        sampled reciprocal points.
    fix_aspect : bool, default=True
        If True, preserve equal axis scaling in 2D plots. Set to ``False`` to
        let the requested ``figsize`` control the on-screen aspect.
    extend : bool, default=False
        Draw translated copies of the sampled k-mesh.
    extend_copies : int or iterable[int], optional
        Number of reciprocal-cell copies per direction. In 2D,
        ``extend_copies=1`` includes the first shell around the first Brillouin
        zone and ``extend_copies=2`` includes the second shell as well.
    nx, ny, nz : int, default=(1, 1, 1)
        Legacy per-direction extension counts used when ``extend_copies`` is
        not specified.
    bz_upscale : float, default=1.1
        Factor by which the maximum reciprocal-vector norm is multiplied to
        determine the plot limits when ``show_bz=True``.
    extended_kpoint_color, extended_kpoint_alpha
        Style of translated mesh points.
    extended_bz_facecolor, extended_bz_edgecolor, extended_bz_alpha
        Style of translated Brillouin-zone copies.
    show_background_bz : bool, default=False
        Draw translated Brillouin-zone copies behind the mesh.
    legend_kwargs : dict, optional
        Extra keyword arguments passed to ``axis.legend``.
    **kwargs
        Low-level style overrides such as ``zorder_path``, ``zorder_kpoints``,
        ``marker_hs`` or ``marker_extend``.

    Returns
    -------
    fig, ax : Figure, Axes
        Matplotlib figure and axes containing the plot.
    """

    kvecs_full  = _ensure_numpy(lattice.kvectors)
    target_dim  = lattice.dim if lattice.dim else kvecs_full.shape[1]
    dim         = max(1, min(kvecs_full.shape[1], target_dim, 3))
    coords      = kvecs_full[:, :dim]
    kfrac       = getattr(lattice, "kvectors_frac", None)

    fig, axis   = _init_axes(ax, dim)
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)

    # Set 3D view if applicable
    if dim == 3:
        axis.view_init(elev=getattr(axis, "elev", 30.0), azim=getattr(axis, "azim", 45.0))

    # Determine how many extended copies to generate in each direction
    if extend_copies is None:
        copy_spec = nx if dim == 1 else ((nx, ny) if dim == 2 else (nx, ny, nz))
    elif np.isscalar(extend_copies):
        copy_spec = int(extend_copies)
    else:
        copy_values = tuple(int(v) for v in extend_copies)
        if len(copy_values) < dim:
            raise ValueError("extend_copies must provide at least one value per reciprocal-space dimension")
        copy_spec = copy_values[:dim]

    # Check the BZ path and find the nearest k-points along it.
    if selection is None:
        selection = lattice.bz_path_points(
            path=path, points_per_seg=points_per_seg, k_vectors=kvecs_full, k_vectors_frac=kfrac, tol=path_match_tol, periodic=False,
        )
    hs              = lattice.high_symmetry_points()
    resolved_path   = lattice.default_resolve_path(path if path is not None else hs)
    legend_kwargs   = {} if legend_kwargs is None else dict(legend_kwargs)
    label_kmesh     = kwargs.get("label_kmesh", "k-mesh")
    label_extended  = kwargs.get("label_extended", "extended mesh")
    label_matched   = kwargs.get("label_matched", "matched path points")
    plotted_coords  = [coords]
    round_scale     = 1e-10

    # Determine the BZ extent for auto-scaling
    bz_upscale      = kwargs.pop("bz_upscale", 1.1)

    if show_bz:
        _draw_bz_region(axis, coords, dim=dim, lattice=lattice, facecolor=bz_facecolor, edgecolor=bz_edgecolor, alpha=bz_alpha, fix_aspect=fix_aspect, show_points=False, zorder=0.0, **kwargs)

        # Ensure the BZ is fully contained in the auto-scaling plotted_coords
        b_norms = []
        for i in range(1, 4):
            # Reciprocal vectors are always 3D in the Lattice class, but may be 2D/1D for others
            vec = getattr(lattice, f"k{i}", getattr(lattice, f"b{i}", None))
            if vec is not None:
                b_norms.append(np.linalg.norm(np.asarray(vec)[:dim]))

        if b_norms:
            # Use a factor that ensures BZ and some margin is visible.
            # 0.5*b is the BZ face, 0.75*b covers corners, 1.1*b is a generous margin.
            kmax         = max(b_norms) * bz_upscale
            bbox         = np.eye(dim) * kmax
            plotted_coords.append(bbox)
            plotted_coords.append(-bbox)

    if show_background_bz:
        bz_centers       = lattice.wigner_seitz_shifts(copies=copy_spec, include_origin=False)
        seen_shifts     = set()
        shifts          = []
        for shift in bz_centers:
            key         = tuple(np.rint(shift / round_scale).astype(np.int64))
            if key in seen_shifts:
                continue
            seen_shifts.add(key)
            if np.allclose(shift, 0.0):
                continue
            shifts.append(shift)
        
        # Draw the extended BZ regions first (if requested) so they appear behind the original points and path
        if show_bz:
            for shift in shifts:
                _draw_bz_region(axis, coords + shift, dim=dim, lattice=lattice, offset=shift, facecolor=extended_bz_facecolor,
                    edgecolor=extended_bz_edgecolor, alpha=extended_bz_alpha, fix_aspect=fix_aspect, show_points=False, zorder=-0.1, **kwargs
                )
            if len(shifts) > 0:
                plotted_coords.append(np.vstack([coords + shift for shift in shifts]))

    # Get the path points in Cartesian coordinates and plot the path segments
    if show_path:
        path_cart   = selection.path_cart[:, :dim]
        plotted_coords.append(path_cart)
        zorder_path = kwargs.get("zorder_path", 2)
        
        if dim == 1:
            axis.plot(path_cart[:, 0], np.zeros(len(path_cart)), color=path_color, linewidth=path_linewidth, zorder=zorder_path)
        elif dim == 2:
            axis.plot(path_cart[:, 0], path_cart[:, 1], color=path_color, linewidth=path_linewidth, zorder=zorder_path)
        else:
            axis.plot(path_cart[:, 0], path_cart[:, 1], path_cart[:, 2], color=path_color, linewidth=path_linewidth, zorder=zorder_path)

    # Optionally extend the k-point mesh and plot the extended points with a distinct style
    if extend and show_kpoints:
        extended_k, _   = lattice.wigner_seitz_extend(k_points=coords, copies=copy_spec)
        original_keys   = {tuple(np.rint(row / round_scale).astype(np.int64)) for row in coords}
        seen_ext_keys   = set()
        ext_mask_list   = []
        for row in extended_k:
            key = tuple(np.rint(row / round_scale).astype(np.int64))
            keep = key not in original_keys and key not in seen_ext_keys
            ext_mask_list.append(keep)
            if keep:
                seen_ext_keys.add(key)
        ext_mask        = np.array(ext_mask_list, dtype=bool)
        
        # Plot the extended k-points with a distinct style, ensuring we only plot the new points and not duplicates of the original mesh
        if np.any(ext_mask):
            ext_coords          = extended_k[ext_mask]
            plotted_coords.append(ext_coords)
            marker_extend       = kwargs.get("marker_extend", "o")
            edgecolor_extend    = kwargs.get("edgecolor_extend", "none")
            zorder_extend       = kwargs.get("zorder_extend", 3)
            
            if dim == 1:
                axis.scatter(ext_coords[:, 0], np.zeros(len(ext_coords)), s=kpoint_size, color=extended_kpoint_color,
                             alpha=extended_kpoint_alpha, marker=marker_extend, edgecolors=edgecolor_extend, zorder=zorder_extend, label=label_extended)
            elif dim == 2:
                axis.scatter(ext_coords[:, 0], ext_coords[:, 1], s=kpoint_size, color=extended_kpoint_color,
                             alpha=extended_kpoint_alpha, marker=marker_extend, edgecolors=edgecolor_extend, zorder=zorder_extend, label=label_extended)
            else:
                axis.scatter(ext_coords[:, 0], ext_coords[:, 1], ext_coords[:, 2], s=kpoint_size, color=extended_kpoint_color,
                             alpha=extended_kpoint_alpha, marker=marker_extend, edgecolors=edgecolor_extend, zorder=zorder_extend, label=label_extended)

    # Plot the original k-points on top of everything else (if requested) so they are visible even if they overlap with the path or extended points
    if show_kpoints:
        zorder_kpoints      = kwargs.get("zorder_kpoints", 6)
        marker_kpoints      = kwargs.get("marker_kpoints", "o")
        edgecolor_kpoints   = kwargs.get("edgecolor_kpoints", "none")
        
        if dim == 1:
            axis.scatter(coords[:, 0], np.zeros(len(coords)), s=kpoint_size, color=kpoint_color, alpha=kpoint_alpha, marker=marker_kpoints, edgecolors=edgecolor_kpoints, zorder=zorder_kpoints, label=label_kmesh)
        elif dim == 2:
            axis.scatter(coords[:, 0], coords[:, 1], s=kpoint_size, color=kpoint_color, alpha=kpoint_alpha, marker=marker_kpoints, edgecolors=edgecolor_kpoints, zorder=zorder_kpoints, label=label_kmesh)
        else:
            axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=kpoint_size, color=kpoint_color, alpha=kpoint_alpha, marker=marker_kpoints, edgecolors=edgecolor_kpoints, zorder=zorder_kpoints, label=label_kmesh)

    # Plot the matched k-points along the path with a distinct style, ensuring we only plot them if there are matches and if the option is enabled
    if show_matched_kpoints and selection.has_matches:
        valid_mask          = selection.matched_distances <= (selection.match_tolerance + 1e-14)
        valid_positions     = np.flatnonzero(valid_mask)
        seen_match_indices  = set()
        keep_positions      = []
        for pos in valid_positions:
            key_array = selection.matched_grid_indices if len(selection.matched_grid_indices) > 0 else selection.matched_indices
            idx = int(key_array[pos])
            if idx in seen_match_indices:
                continue
            seen_match_indices.add(idx)
            keep_positions.append(pos)

        matched = selection.matched_cart[np.asarray(keep_positions, dtype=int), :dim] if keep_positions else np.zeros((0, dim), dtype=float)
        if len(matched) > 0:
            plotted_coords.append(matched)
            
            if dim == 1:
                axis.scatter(matched[:, 0], np.zeros(len(matched)), s=matched_kpoint_size, color=matched_kpoint_color,
                             alpha=matched_kpoint_alpha, marker=matched_kpoint_marker, edgecolors=matched_kpoint_edgecolor,
                             linewidths=0.9, zorder=7, label=label_matched)
            elif dim == 2:
                axis.scatter(matched[:, 0], matched[:, 1], s=matched_kpoint_size, color=matched_kpoint_color,
                             alpha=matched_kpoint_alpha, marker=matched_kpoint_marker, edgecolors=matched_kpoint_edgecolor,
                             linewidths=0.9, zorder=7, label=label_matched)
            else:
                axis.scatter(matched[:, 0], matched[:, 1], matched[:, 2], s=matched_kpoint_size, color=matched_kpoint_color,
                             alpha=matched_kpoint_alpha, marker=matched_kpoint_marker, edgecolors=matched_kpoint_edgecolor,
                             linewidths=0.9, zorder=7, label=label_matched)

    # Plot exact high-symmetry vertices, not the nearest interpolated path samples.
    if hs_plot != "none":
        hs_points   = []
        seen_hs     = set()
        b1          = lattice.b1
        b2          = lattice.b2 if lattice.dim >= 2 else np.zeros(3, dtype=float)
        b3          = lattice.b3 if lattice.dim >= 3 else np.zeros(3, dtype=float)
        
        for lbl, frac in resolved_path:
            frac_arr                = np.zeros(3, dtype=float)
            frac_arr[:len(frac)]    = np.asarray(frac, dtype=float)
            pt_obj                  = hs.get(lbl) if hs is not None and hasattr(hs, "get") else None
            cart3                   = pt_obj.to_cartesian(b1, b2, b3) if pt_obj is not None else frac_arr[0] * b1 + frac_arr[1] * b2 + frac_arr[2] * b3
            cart                    = cart3[:dim]
            key                     = tuple(np.rint(cart / round_scale).astype(np.int64))
            if key in seen_hs:
                continue
            seen_hs.add(key)
            hs_points.append((lbl, cart))

        if hs_points:
            hs_coords = np.array([cart for _, cart in hs_points], dtype=float)
            plotted_coords.append(hs_coords)
            
            if hs_plot in ["markers", "both"]:
                
                marker_hs   = kwargs.get("marker_hs", "o")
                zorder_hs   = kwargs.get("zorder_hs", 5)
                lw_hs       = kwargs.get("lw_hs", 1.4)
                
                if dim == 1:
                    axis.scatter(hs_coords[:, 0], np.zeros(len(hs_coords)), s=hs_marker_size, color=hs_marker_facecolor, edgecolors=hs_marker_edgecolor, linewidths=lw_hs, zorder=zorder_hs, marker=marker_hs)
                elif dim == 2:
                    axis.scatter(hs_coords[:, 0], hs_coords[:, 1], s=hs_marker_size, color=hs_marker_facecolor, edgecolors=hs_marker_edgecolor, linewidths=lw_hs, zorder=zorder_hs, marker=marker_hs)
                else:
                    axis.scatter(hs_coords[:, 0], hs_coords[:, 1], hs_coords[:, 2], s=hs_marker_size, color=hs_marker_facecolor, edgecolors=hs_marker_edgecolor, linewidths=lw_hs, zorder=zorder_hs, marker=marker_hs)

            if hs_plot in ["labels", "both"]:
                for lbl, cart in hs_points:
                    pt      = hs.get(lbl) if hs is not None and hasattr(hs, "get") else None
                    text    = pt.latex_label if pt is not None else str(lbl)
                    hs_dict = hs_label_kwargs.copy() if hs_label_kwargs else {}
                    hs_fw   = hs_dict.pop("fontweight", "bold")
                    hs_zord = hs_dict.pop("zorder", 8)
                    bbox    = hs_dict.pop("bbox", None)
                    hs_xy   = hs_dict.pop("xy", (8, 8))
                    hs_xy   = hs_xy if isinstance(hs_xy, tuple) else (hs_xy.get(lbl, (8, 8)) if isinstance(hs_xy, dict) else (8, 8))
                    if dim == 3:
                        text_kwargs = dict(fontsize=hs_font_size, fontweight=hs_fw, zorder=hs_zord, ha='center', va='center', **hs_dict)
                        if bbox is not None:
                            text_kwargs["bbox"] = bbox
                        axis.text(cart[0], cart[1], cart[2], text, **text_kwargs)
                    else:
                        ann_kwargs = dict(
                            xy=(cart[0], cart[1] if dim > 1 else 0.0),
                            textcoords='offset points',
                            xytext=hs_xy,
                            fontsize=hs_font_size,
                            fontweight=hs_fw,
                            zorder=hs_zord,
                            **hs_dict,
                        )
                        if bbox is not None:
                            ann_kwargs["bbox"] = bbox
                        axis.annotate(text, **ann_kwargs)

    # Final spatial limits and legend
    plotted_stack = np.vstack([np.asarray(arr, dtype=float).reshape(-1, dim) for arr in plotted_coords if np.asarray(arr).size > 0])
    _apply_spatial_limits(axis, plotted_stack, dim, True, labels=(r'$k_x$', r'$k_y$', r'$k_z$'), fix_aspect=fix_aspect)

    if dim == 2:
        axis.axhline(0, color='grey', lw=0.4, zorder=-1)
        axis.axvline(0, color='grey', lw=0.4, zorder=-1)

    if title:
        kw = {"pad": 12}
        if title_kwargs:
            kw.update(title_kwargs)
        axis.set_title(title, **kw)

    handles, labels_ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc=legend_kwargs.pop("loc", "best"), 
            fontsize=legend_kwargs.pop("fontsize", 9), 
            framealpha=legend_kwargs.pop("framealpha", 0.90), 
            edgecolor=legend_kwargs.pop("edgecolor", "lightgray"), 
            fancybox=legend_kwargs.pop("fancybox", True), **legend_kwargs)

    if tight_layout:
        _finalise_figure(fig)

    return fig, axis

# ==============================================================================
# Plotter Class
# ==============================================================================

@dataclass
class LatticePlotter:
    """
    Convenience wrapper bundling plotting helpers for a single lattice.
    
    Usage:
        lattice.plot.real_space()
        lattice.plot.structure(show_indices=True)
        lattice.plot.regions(regions_dict)
    """

    lattice: Lattice

    def real_space(self, **kwargs) -> Tuple[Figure, Axes]:
        """ Plot real-space sites. """
        kwargs.setdefault("figsize", (5.0, 5.0))
        return plot_real_space(self.lattice, **kwargs)

    def reciprocal_space(self, **kwargs) -> Tuple[Figure, Axes]:
        """
        Scatter-plot of reciprocal lattice vectors (k-points).
        
        Parameters mirror :func:`plot_real_space`
        --------------------------------------------------------------------------
        lattice : Lattice
            The lattice object to plot.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new figure is created.
        show_indices : bool, default=False
            If True, annotate each k-point with its index.
        show_axes : bool, default=True
            If False, hides the coordinate axes.
        color : str, default="C1"
            Color of the k-point markers.
        marker : str, default="o"
            Marker style.
        figsize : tuple, optional
            Figure size in inches (width, height).
        title : str, optional
            Title of the plot.
        elev, azim : float, optional
            Elevation and azimuth angles for 3D plots.
        extend_kpoints : bool, default=False
            If True, draw translated reciprocal-space copies around the original mesh.
        extend_copies : int or iterable of int, default=2
            Number of copies per reciprocal direction used when ``extend_kpoints=True``.
            Scalars are applied to all active reciprocal directions.
        extend_tol : float, default=1e-10
            Tolerance used to identify which extended points are already present in
            the original reciprocal mesh.
        **scatter_kwargs
            Include:
            - point_edgecolor: Color of the marker edges (default "white").
            - point_zorder: Z-order for the scatter points (default 5).
            - color_extended: Color for translated copies (default "C2").
            - edgecolor_extended: Edge color for translated copies (default "gray").
            - marker_extended: Marker for translated copies (default ``marker``).
            - Any other valid arguments for `ax.scatter`.
        """
        kwargs.setdefault("figsize", (5.0, 5.0))
        return plot_reciprocal_space(self.lattice, **kwargs)

    def brillouin_zone(self, **kwargs) -> Tuple[Figure, Axes]:
        """ Plot the Brillouin Zone. """
        kwargs.setdefault("figsize", (5.0, 4.0))
        return plot_brillouin_zone(self.lattice, **kwargs)

    def structure(self, **kwargs) -> Tuple[Figure, Axes]:
        r""" 
        Plot detailed lattice structure with connectivity. 
        
        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, a new figure is created.
        show_indices : bool
            If True, annotates nodes with their site indices.
        highlight_boundary : bool
            If True, draws boundary nodes with a distinct color/edge.
        show_axes : bool
            If False, hides the coordinate axes for a cleaner diagram.
        edge_color : str
            Color of the edges.
        node_color : str
            Color of the nodes.
        boundary_node_color : str
            Color of the boundary node edges.
        periodic_color : str
            Color for periodic boundary annotations.
        open_color : str
            Color for open boundary annotations.
        node_size : int
            Size of the node markers.
        edge_alpha : float
            Transparency of the edges.
        label_padding : float
            Fractional padding for node index labels.
        boundary_offset : float
            Fractional offset for boundary annotations.
        figsize : tuple, optional
            Figure size in inches (width, height).
        title : str, optional
            Title of the plot.
        title_kwargs : dict, optional
            Additional keyword arguments for the title.
        tight_layout : bool
            If True, applies tight layout to the figure.
        elev, azim : float, optional
            Elevation and azimuth angles for 3D plots.
        partition_colors : tuple of str, optional
            Colors to use for bipartite/sublattice coloring. If provided, nodes are
            colored based on sublattice parity.
        show_periodic_connections : bool
            If True, indicates wrap-around connections textually or graphically.
        show_primitive_cell : bool
            If True, overlays the primitive unit cell vectors/box.
        **scatter_kwargs
            Additional arguments passed to `ax.scatter`.
        """
        kwargs.setdefault("figsize", (5.5, 5.5))
        return plot_lattice_structure(self.lattice, **kwargs)

    def regions(self, regions: Dict[str, List[int]], **kwargs) -> Tuple[Figure, Axes]:
        """
        Plot specific regions on the lattice.
        
        Parameters
        ----------
        regions : Dict[str, List[int]]
            Dictionary mapping region names to lists of site indices.
        show_system : bool
            If True, plot all lattice sites faintly in the background.
        system_color : str
            Color for background system sites.
        cmap : str
            Colormap name for distinct regions.
        blob_radius : float, optional
            If given, draw a translucent circle around each site.
        show_bonds : bool
            If True, draw intra-region NN bonds.
        ... other args mirror plot_real_space ...
        """
        kwargs.setdefault("figsize", (6.0, 6.0))
        
        if isinstance(regions, str):
            regions = {regions: self.lattice.get_region(regions, **kwargs)}
        
        return plot_regions(self.lattice, regions, **kwargs)

    def bz_high_symmetry(self, **kwargs) -> Tuple[Figure, Axes]:
        """
        Plot the Brillouin zone, high-symmetry path, and sampled reciprocal mesh.

        Parameters
        ----------
        path : list[str], str, or iterable[(label, frac)], optional
            High-symmetry path specification. If omitted, the lattice default path
            is used.
        show_kpoints : bool, default=True
            Draw sampled reciprocal-space mesh points.
        show_bz : bool, default=True
            Draw the first Brillouin zone.
        show_path : bool, default=True
            Draw the ideal high-symmetry path.
        show_matched_kpoints : bool, default=True
            Highlight sampled k-points whose distance to the path is within the
            matching tolerance.
        points_per_seg : int, default=40
            Number of interpolation points per path segment for the ideal path.
        path_match_tol : float, optional
            Distance tolerance used when highlighting mesh points near the
            drawn path.
        extend : bool, default=False
            Draw translated copies of the sampled k-mesh.
        extend_copies : int or iterable[int], optional
            Number of reciprocal-cell copies per direction. In 2D,
            ``extend_copies=1`` includes the first shell around the first Brillouin
            zone and ``extend_copies=2`` includes the second shell as well.
        show_background_bz : bool, default=False
            Draw translated Brillouin-zone copies behind the mesh.
        hs_plot : {"none", "markers", "labels", "both"}, default="markers"
            Whether to draw exact high-symmetry markers, labels, or both.
        legend_kwargs : dict, optional
            Extra keyword arguments passed to ``axis.legend``.
        **kwargs
            Additional style overrides forwarded to ``plot_high_symmetry_points``.
        """
        kwargs.setdefault("figsize", (5.5, 5.5))
        return plot_high_symmetry_points(self.lattice, **kwargs)

    def subsystem(
        self,
        sites           : List[int],
        *,
        show_boundary   : bool = True,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot a single subsystem with its boundary highlighted.
        
        Parameters
        ----------
        sites : list of int
            Site indices in the subsystem.
        show_boundary : bool, default=True
            If True, highlight the boundary bonds crossing A/B.
        **kwargs
            Passed to plot_regions.
            
        Returns
        -------
        fig, ax : Figure, Axes
            
        Examples
        --------
        >>> lattice.plot.subsystem([0, 1, 4, 5])
        >>> lattice.plot.subsystem(range(8), show_bonds=True)
        """
        kwargs.setdefault("figsize", (5.0, 5.0))
        kwargs.setdefault("show_bonds", True)
        
        # Compute boundary if requested
        title = kwargs.pop("title", None)
        if show_boundary and hasattr(self.lattice, 'regions'):
            dA = self.lattice.regions.subsystem_boundary(sites)
            if title is None:
                title = f"Subsystem: |A|={len(sites)}, ∂A={dA}"
        elif title is None:
            title = f"Subsystem: |A|={len(sites)}"
        
        return plot_regions(self.lattice, {"A": list(sites)}, title=title, **kwargs)

    def sweep(
        self,
        direction       : Optional[str] = None,
        *,
        rectangular     : bool = False,
        max_panels      : int = 6,
        figsize         : Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot subsystem sweep showing cuts with different boundary sizes.
        
        Creates a grid of subplots showing subsystems grouped by ∂A.
        
        Parameters
        ----------
        direction : str, optional
            Direction for sweep ('x', 'y', 'z'). Creates full-width cuts.
        rectangular : bool, default=False
            If True and direction is None, use rectangular subsystems (various shapes).
            If False, use lexicographic sweep (sequential site addition).
        max_panels : int, default=6
            Maximum number of panels to show.
        figsize : tuple, optional
            Figure size. Auto-computed if None.
        **kwargs
            Passed to plot_regions for each panel.
            
        Returns
        -------
        fig, axes : Figure, ndarray of Axes
            
        Examples
        --------
        >>> lattice.plot.sweep(rectangular=True)  # Various rectangular shapes
        >>> lattice.plot.sweep(direction='x')     # Full-width column cuts
        >>> lattice.plot.sweep()                  # Lexicographic sweep
        """
        import matplotlib.pyplot as plt
        
        # Get sweep data (logic is now in sweep_subsystems)
        by_dA = self.lattice.regions.sweep_subsystems(
            direction=direction, rectangular=rectangular
        )
        
        # Collect one representative per dA
        panels = []
        for dA in sorted(by_dA.keys()):
            # Pick middle-sized subsystem as representative
            subs        = by_dA[dA]
            subs_sorted = sorted(subs, key=len)
            rep         = subs_sorted[len(subs_sorted) // 2]
            panels.append((dA, rep))
            if len(panels) >= max_panels:
                break
        
        n = len(panels)
        if n == 0:
            raise ValueError("No subsystems generated")
        
        # Grid layout - compact
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (2.8 * ncols, 2.5 * nrows)
        
        fig, axes   = plt.subplots(nrows, ncols, figsize=figsize)
        axes        = np.atleast_1d(axes).flatten()
        
        # Plot each panel with clean defaults
        kwargs.setdefault("show_bonds", True)
        kwargs.setdefault("show_legend", False)
        kwargs.setdefault("show_labels", False)
        kwargs.setdefault("show_system", True)
        kwargs.setdefault("show_axes", False)
        kwargs.setdefault("marker_size", 40)
        kwargs.setdefault("tight_layout", False)  # We do our own
        
        for i, (dA, sites) in enumerate(panels):
            ax = axes[i]
            plot_regions(
                self.lattice, {"A": sites}, ax=ax,
                title=f"∂A={dA}, |A|={len(sites)}",
                title_kwargs={"fontsize": 10},
                **kwargs,
            )
            ax.set_aspect('equal', adjustable='box')
        
        # Hide unused axes
        for i in range(n, len(axes)):
            axes[i].set_visible(False)
        
        fig.tight_layout(pad=0.5)
        return fig, axes

# ------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------
