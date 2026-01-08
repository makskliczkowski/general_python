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
File    : QES/general_python/lattices/visualization/plotting.py
Author  : Maksymilian Kliczkowski
Date    : 2025-02-01
----------------------------------------------------------------------
"""

from    __future__          import annotations
import  math
import  numpy               as np
import  matplotlib.pyplot   as plt

from    collections         import defaultdict
from    dataclasses         import dataclass
from    typing              import Optional, Tuple, List, Set, Dict, Union, Any

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

def _annotate_indices(ax: Axes, coords: np.ndarray, *, zorder: int = 5, color: str = 'k', fontsize: int = 8) -> None:
    """ Annotate sites with their indices. """
    for idx, point in enumerate(coords):
        if point.size >= 3:
            ax.text(point[0], point[1], point[2], str(idx), zorder=zorder, color=color, fontsize=fontsize)
        elif point.size == 2:
            ax.text(point[0], point[1], str(idx), zorder=zorder, color=color, fontsize=fontsize)
        else:
            ax.text(point[0], 0.0, str(idx), zorder=zorder, color=color, fontsize=fontsize)

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

# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_real_space(
    lattice         : Lattice,
    *,
    ax              : Optional[Axes]                = None,
    show_indices    : bool                          = False,
    color           : str                           = "C0",
    marker          : str                           = "o",
    figsize         : Optional[Tuple[float, float]] = None,
    title           : Optional[str]                 = None,
    title_kwargs    : Optional[Dict[str, object]]   = None,
    tight_layout    : bool                          = True,
    elev            : Optional[float]               = None,
    azim            : Optional[float]               = None,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Scatter-plot of real-space lattice vectors.

    Parameters
    ----------
    lattice : Lattice
        The lattice object to plot.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, a new figure is created.
    show_indices : bool, default=False
        If True, annotate each site with its index.
    color : str, default="C0"
        Color of the site markers.
    marker : str, default="o"
        Marker style.
    figsize : tuple, optional
        Figure size in inches (width, height).
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
        axis.set_ylim(-0.5, 0.5)
        axis.set_ylabel("y (projected)")
        axis.set_xlabel("x")
        
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], color=color, marker=marker, **scatter_kwargs)
        axis.set_ylabel("y")
        axis.set_xlabel("x")
        axis.set_aspect("equal", adjustable="datalim")
        
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=color, marker=marker, **scatter_kwargs)
        axis.set_zlabel("z")
        axis.set_ylabel("y")
        axis.set_xlabel("x")

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


def plot_reciprocal_space(
    lattice         : Lattice,
    *,
    ax              : Optional[Axes]                = None,
    show_indices    : bool                          = False,
    color           : str                           = "C1",
    marker          : str                           = "o",
    figsize         : Optional[Tuple[float, float]] = None,
    title           : Optional[str]                 = None,
    title_kwargs    : Optional[Dict[str, object]]   = None,
    tight_layout    : bool                          = True,
    elev            : Optional[float]               = None,
    azim            : Optional[float]               = None,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Scatter-plot of reciprocal lattice vectors (k-points).
    
    Parameters mirror :func:`plot_real_space`.
    """
    coords      = _ensure_numpy(lattice.kvectors)
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

    if dim == 1:
        axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), color=color, marker=marker, **scatter_kwargs)
        axis.set_ylim(-0.5, 0.5)
        axis.set_ylabel(r"$k_y$ (projected)")
        axis.set_xlabel(r"$k_x$")
        
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], color=color, marker=marker, **scatter_kwargs)
        axis.set_ylabel(r"$k_y$")
        axis.set_xlabel(r"$k_x$")
        axis.set_aspect("equal", adjustable="datalim")
        
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=color, marker=marker, **scatter_kwargs)
        axis.set_zlabel(r"$k_z$")
        axis.set_ylabel(r"$k_y$")
        axis.set_xlabel(r"$k_x$")

    if title:
        kw = {"pad": 12}
        if title_kwargs: kw.update(title_kwargs)
        axis.set_title(title, **kw)

    if show_indices:
        _annotate_indices(axis, coords)

    if tight_layout:
        _finalise_figure(fig)
        
    return fig, axis

# ==============================================================================
# Brillouin Zone Helpers
# ==============================================================================

def _plot_1d_bz(axis: Axes, bounds: Tuple[float, float], *, facecolor: str, alpha: float) -> None:
    x_min, x_max = bounds
    axis.axvspan(x_min, x_max, ymin=0.25, ymax=0.75, facecolor=facecolor, alpha=alpha)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlabel("k")

def _plot_2d_bz(axis: Axes, points: np.ndarray, *, facecolor: str, edgecolor: str, alpha: float) -> None:
    polygon = None
    if ConvexHull is not None:
        try:
            hull    = ConvexHull(points)
            polygon = points[hull.vertices]
        except Exception:
            pass

    if polygon is None:
        # Fallback to bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        polygon = np.array([
            [x_min, y_min], [x_max, y_min],
            [x_max, y_max], [x_min, y_max],
        ])

    axis.fill(*polygon.T, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=1.5)
    axis.plot(*polygon.T, color=edgecolor, linewidth=1.5)
    # close the loop
    axis.plot([polygon[-1, 0], polygon[0, 0]], [polygon[-1, 1], polygon[0, 1]], color=edgecolor, linewidth=1.5)
    
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel(r"$k_x$")
    axis.set_ylabel(r"$k_y$")

def _plot_3d_bz(axis: Axes, points: np.ndarray, *, facecolor: str, edgecolor: str, alpha: float) -> None:
    if Poly3DCollection is None:
        raise RuntimeError("3D plotting support requires mpl_toolkits.mplot3d.")

    faces = None
    if ConvexHull is not None:
        try:
            hull    = ConvexHull(points)
            faces   = [points[simplex] for simplex in hull.simplices]
        except Exception:
            pass
            
    if faces is None:
        # Fallback to bounding box cube
        mins    = points.min(axis=0)
        maxs    = points.max(axis=0)
        corners = np.array([
            [mins[0], mins[1], mins[2]], [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]], [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]], [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]], [mins[0], maxs[1], maxs[2]],
        ])
        
        # Simple cube faces (indices of corners)
        faces = [
            corners[[0, 1, 2, 3]], corners[[4, 5, 6, 7]], 
            corners[[0, 1, 5, 4]], corners[[2, 3, 7, 6]],
            corners[[1, 2, 6, 5]], corners[[3, 0, 4, 7]],
        ]

    collection = Poly3DCollection(faces, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    axis.add_collection3d(collection)
    axis.set_xlabel(r"$k_x$")
    axis.set_ylabel(r"$k_y$")
    axis.set_zlabel(r"$k_z$")

def plot_brillouin_zone(
    lattice         : Lattice,
    *,
    ax              : Optional[Axes]                = None,
    facecolor       : str                           = "tab:blue",
    edgecolor       : str                           = "black",
    alpha           : float                         = 0.25,
    figsize         : Optional[Tuple[float, float]] = None,
    title           : Optional[str]                 = None,
    title_kwargs    : Optional[Dict[str, object]]   = None,
    tight_layout    : bool                          = True,
    elev            : Optional[float]               = None,
    azim            : Optional[float]               = None,
) -> Tuple[Figure, Axes]:
    """
    Plot the Brillouin Zone approximation based on sampled k-vectors.
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
        _plot_2d_bz(axis, coords[:, :2], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    else:
        _plot_3d_bz(axis, coords[:, :3], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)

    if title:
        kw = {"pad": 12}
        if title_kwargs: kw.update(title_kwargs)
        axis.set_title(title, **kw)

    if tight_layout:
        _finalise_figure(fig)
        
    return fig, axis

# ==============================================================================
# Structural Plotting Helpers
# ==============================================================================

def _gather_nn_edges(lattice: Lattice) -> List[Tuple[int, int]]:
    """ Extract nearest-neighbor edges from lattice. """
    edges = set()
    for i in range(lattice.Ns):
        neighbors = lattice.get_nn(i)
        if not neighbors: continue
        
        for j in neighbors:
            if lattice.wrong_nei(j): continue
            
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
            
        colors[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
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

def _draw_primitive_cell(axis: Axes, origin: np.ndarray, basis_vectors: List[np.ndarray], dim: int) -> None:
    """ Draw the primitive unit cell vectors from an origin. """
    if not basis_vectors: return
    
    color       = "0.4"
    linestyle   = ":"
    linewidth   = 1.0

    if dim == 1 and len(basis_vectors) >= 1:
        points = np.vstack([origin, origin + basis_vectors[0]])
        axis.plot(points[:, 0], np.zeros_like(points[:, 0]), color=color, linestyle=linestyle, linewidth=linewidth)
        
    elif dim == 2 and len(basis_vectors) >= 2:
        a1, a2  = basis_vectors[:2]
        corners = np.array([origin, origin + a1, origin + a1 + a2, origin + a2, origin])
        axis.plot(corners[:, 0], corners[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)
        
    elif dim == 3 and len(basis_vectors) >= 3:
        a1, a2, a3 = basis_vectors[:3]
        base       = origin
        # vertices of the parallelepiped
        corners    = [
            base, base + a1, base + a2, base + a3,
            base + a1 + a2, base + a1 + a3, base + a2 + a3,
            base + a1 + a2 + a3
        ]
        # edges to draw
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5), (2, 4), (2, 6),
            (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)
        ]
        for i, j in edges:
            start, end = corners[i], corners[j]
            axis.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                      color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7)

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
                axis.annotate(f"PBC {label}", xy=(x, maxs[1]), xytext=(x, maxs[1] + padding),
                              ha="center", va="bottom", arrowprops=arrow_kw, **text_kw)
                axis.annotate("", xy=(x, mins[1]), xytext=(x, mins[1] - padding), arrowprops=arrow_kw)
            else:
                axis.plot([mins[0], maxs[0]], [mins[1], mins[1]], **style_kw)
                axis.plot([mins[0], maxs[0]], [maxs[1], maxs[1]], **style_kw)
                axis.text(mins[0] - padding, mid[1], f"Open {label}", ha="right", va="center", rotation=90, **text_kw)

    for idx in range(min(positions.shape[1], 2)):
        _annotate_axis(idx, labels[idx])

# ==============================================================================
# Main Structure Plotter
# ==============================================================================

def plot_lattice_structure(
    lattice                     : Lattice,
    *,
    ax                          : Optional[Axes]                = None,
    show_indices                : bool                          = False,
    highlight_boundary          : bool                          = True,
    show_axes                   : bool                          = False,
    edge_color                  : str                           = "0.5",
    node_color                  : str                           = "tab:blue",
    boundary_node_color         : str                           = "tab:red",
    periodic_color              : str                           = "tab:orange",
    open_color                  : str                           = "tab:green",
    node_size                   : int                           = 30,
    edge_alpha                  : float                         = 0.7,
    label_padding               : float                         = 0.05,
    boundary_offset             : float                         = 0.05,
    figsize                     : Optional[Tuple[float, float]] = None,
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
    """
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

    # --- 1. Compute Connectivity ---
    edges       = _gather_nn_edges(lattice)
    adjacency   = [[] for _ in range(lattice.Ns)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    # --- 2. Periodic Edges Detection ---
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
        start   = coords[i]
        end     = coords[j]
        dist    = np.linalg.norm(end - start)
        
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
        line_args = dict(color=edge_color, alpha=edge_alpha, linestyle=linestyle, linewidth=1.0, zorder=2)
        
        if dim == 3:
            axis.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], **line_args)
        elif dim == 2:
            axis.plot([start[0], end[0]], [start[1], end[1]], **line_args)
        else: # 1D
            axis.plot([start[0], end[0]], [0.0, 0.0], **line_args)

    # --- 3. Node Coloring ---
    node_face_colors    = [node_color] * lattice.Ns
    
    if partition_colors:
        partitions = _infer_bipartite_coloring(adjacency)
        if partitions is not None:
            palette = partition_colors
            node_face_colors = [palette[partitions[i] % len(palette)] for i in range(lattice.Ns)]

    # --- 4. Draw Nodes ---
    scatter_defaults = dict(s=node_size, zorder=3, **scatter_kwargs)
    
    if dim == 1:
        axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), c=node_face_colors, **scatter_defaults)
        axis.set_ylim(-0.5, 0.5)
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], c=node_face_colors, **scatter_defaults)
        axis.set_aspect("equal", adjustable="datalim")
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=node_face_colors, **scatter_defaults)

    # --- 5. Boundary Highlight ---
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

    # --- 6. Axes & Titles ---
    if not show_axes:
        if dim == 1:
            axis.get_yaxis().set_visible(False)
            for sp in axis.spines.values(): sp.set_visible(False)
        elif dim == 2:
            axis.set_axis_off()
        else:
            axis.set_axis_off()
    else:
        axis.set_xlabel("x")
        if dim >= 2: axis.set_ylabel("y")
        if dim >= 3: axis.set_zlabel("z")

    if title:
        kw = {"pad": 15}
        if title_kwargs: kw.update(title_kwargs)
        axis.set_title(title, **kw)

    # --- 7. Indices & Annotations ---
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
            
            txt_args = dict(ha="center", va="center", color="black", 
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1), zorder=6)
            
            if dim == 3:
                axis.text(label_pos[0], label_pos[1], label_pos[2], str(idx), **txt_args)
            elif dim == 2:
                axis.text(label_pos[0], label_pos[1], str(idx), **txt_args)
            else:
                axis.text(label_pos[0], offset, str(idx), **txt_args)
                
            node_label_positions[idx] = label_pos

    # --- 8. Boundary Annotations (2D only) ---
    if dim == 2:
        _draw_boundary_annotations(axis, coords, lattice,
                                   periodic_color=periodic_color,
                                   open_color=open_color,
                                   offset_fraction=boundary_offset)

    # --- 9. Periodic Connections Text ---
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

    # --- 10. Primitive Cell ---
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

def plot_regions(
    lattice             : Lattice,
    regions             : Dict[str, List[int]],
    *,
    ax                  : Optional[Axes]                = None,
    show_indices        : bool                          = False,
    show_system         : bool                          = True,
    show_complement     : bool                          = True,
    show_labels         : bool                          = True,
    show_connections    : bool                          = False,
    show_overlaps       : bool                          = True,
    system_color        : str                           = 'lightgray',
    system_alpha        : float                         = 0.3,
    complement_color    : str                           = 'wheat',
    complement_alpha    : float                         = 0.6,
    overlap_color       : str                           = 'red',
    cmap                : str                           = 'Set2',
    fill                : bool                          = False,
    fill_alpha          : float                         = 0.2,
    marker_size         : int                           = 100,
    edge_width          : float                         = 2.0,
    figsize             : Optional[Tuple[float, float]] = None,
    title               : Optional[str]                 = None,
    title_kwargs        : Optional[Dict[str, object]]   = None,
    tight_layout        : bool                          = True,
    elev                : Optional[float]               = None,
    azim                : Optional[float]               = None,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot specific regions on the lattice with enhanced visualization.

    This function provides comprehensive region visualization including:
    - Individual regions with distinct colors
    - Complement region (sites not in any region)
    - Overlap detection and highlighting
    - Region labels and statistics
    - Optional connectivity between regions

    Parameters
    ----------
    lattice : Lattice
        The lattice object.
    regions : Dict[str, List[int]]
        Dictionary mapping region names to lists of site indices.
    ax : Axes, optional
        Matplotlib axes.
    show_indices : bool
        If True, annotate sites with indices.
    show_system : bool
        If True, plot all lattice sites faintly in the background.
    show_complement : bool
        If True, highlight sites not in any region.
    show_labels : bool
        If True, add text labels for each region.
    show_connections : bool
        If True, draw lines between adjacent regions.
    show_overlaps : bool
        If True, highlight sites belonging to multiple regions.
    system_color : str
        Color for background system sites.
    system_alpha : float
        Alpha for background system sites.
    complement_color : str
        Color for complement region.
    complement_alpha : float
        Alpha for complement region.
    overlap_color : str
        Color to highlight overlapping sites.
    cmap : str
        Colormap name for distinct regions.
    fill : bool
        If True and dim=2, fill the region with a polygon (requires scipy).
    fill_alpha : float
        Transparency of filled regions.
    marker_size : int
        Size of region markers.
    edge_width : float
        Width of marker edges.
    ... other args mirror plot_real_space ...
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

    # Plot full system background
    if show_system:
        bg_args = dict(color=system_color, alpha=system_alpha, marker='o', s=30, label='System')
        if dim == 1:
            axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), **bg_args)
        elif dim == 2:
            axis.scatter(coords[:, 0], coords[:, 1], **bg_args)
        else:
            axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **bg_args)

    # Compute complement and overlaps
    all_region_sites    = set()
    site_counts         = {}
    for indices in regions.values():
        for idx in indices:
            all_region_sites.add(idx)
            site_counts[idx] = site_counts.get(idx, 0) + 1
    
    complement_sites    = [i for i in range(len(coords)) if i not in all_region_sites]
    overlap_sites       = [i for i, count in site_counts.items() if count > 1]
    
    # Plot complement region
    if show_complement and complement_sites:
        comp_coords = coords[complement_sites]
        comp_args   = dict(color=complement_color, alpha=complement_alpha, marker='x', 
                        s=marker_size*0.8, edgecolors='black', linewidths=edge_width*0.5, 
                        label='Complement')
        if dim == 1:
            axis.scatter(comp_coords[:, 0], np.zeros_like(comp_coords[:, 0]), **comp_args)
        elif dim == 2:
            axis.scatter(comp_coords[:, 0], comp_coords[:, 1], **comp_args)
        else:
            axis.scatter(comp_coords[:, 0], comp_coords[:, 1], comp_coords[:, 2], **comp_args)
    
    # Plot regions
    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap, max(len(regions), 10))
    except ImportError:
        def colormap(i): return f"C{i}"

    for i, (name, indices) in enumerate(regions.items()):
        if not indices: continue
        
        region_coords = coords[indices]
        color_val = colormap(i)
        
        # 1. Fill polygon (2D only)
        if fill and dim == 2 and ConvexHull is not None and len(region_coords) >= 3:
            try:
                hull = ConvexHull(region_coords)
                hull_points = region_coords[hull.vertices]
                axis.fill(hull_points[:, 0], hull_points[:, 1], color=color_val, 
                         alpha=fill_alpha, label=f'{name} (Area)')
            except Exception:
                pass

        # 2. Scatter points
        sc_args = dict(color=color_val, marker='o', s=marker_size, edgecolors='black', 
                      linewidths=edge_width, label=f'{name}', **scatter_kwargs)
        
        if dim == 1:
            axis.scatter(region_coords[:, 0], np.zeros_like(region_coords[:, 0]), **sc_args)
        elif dim == 2:
            axis.scatter(region_coords[:, 0], region_coords[:, 1], **sc_args)
        else:
            axis.scatter(region_coords[:, 0], region_coords[:, 1], region_coords[:, 2], **sc_args)
            
        # 3. Label center of region
        if show_labels and len(region_coords) > 0:
            com = np.mean(region_coords, axis=0)
            txt_args = dict(fontsize=12, fontweight='bold', color='black', ha='center', va='center', 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', pad=1.0))
            if dim == 1:
                axis.text(com[0], 0, name, **txt_args)
            elif dim == 2:
                axis.text(com[0], com[1], name, **txt_args)
            else:
                axis.text(com[0], com[1], com[2], name, **txt_args)
    
    # Highlight overlapping sites
    if show_overlaps and overlap_sites:
        overlap_coords  = coords[overlap_sites]
        overlap_args    = dict(color='none', edgecolors=overlap_color, marker='o', 
                           s=marker_size*1.5, linewidths=edge_width*2, 
                           label=f'Overlaps ({len(overlap_sites)})')
        if dim == 1:
            axis.scatter(overlap_coords[:, 0], np.zeros_like(overlap_coords[:, 0]), **overlap_args)
        elif dim == 2:
            axis.scatter(overlap_coords[:, 0], overlap_coords[:, 1], **overlap_args)
        else:
            axis.scatter(overlap_coords[:, 0], overlap_coords[:, 1], overlap_coords[:, 2], **overlap_args)

    # Axis setup
    if dim == 1:
        axis.set_ylim(-0.5, 0.5)
        axis.set_yticks([])
        axis.set_xlabel("x")
    elif dim == 2:
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    else:
        axis.set_zlabel("z")
        axis.set_ylabel("y")
        axis.set_xlabel("x")

    # Build informative title
    if title:
        kw = {"pad": 12}
        if title_kwargs: kw.update(title_kwargs)
        axis.set_title(title, **kw)
    elif not title and ax is None:
        n_total         = len(coords)
        n_covered       = len(all_region_sites)
        n_complement    = len(complement_sites)
        n_overlaps      = len(overlap_sites)
        coverage_pct    = (n_covered / n_total * 100) if n_total > 0 else 0
        
        title_parts     = [f"Lattice Regions ({len(regions)} regions)"]
        title_parts.append(f"\nCoverage: {n_covered}/{n_total} ({coverage_pct:.1f}%)")
        if n_complement > 0:
            title_parts.append(f", Complement: {n_complement}")
        if n_overlaps > 0:
            title_parts.append(f", Overlaps: {n_overlaps}")
        
        axis.set_title(''.join(title_parts), fontsize=10, pad=12)

    # Deduplicate legend labels
    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axis.legend(by_label.values(), by_label.keys(), loc='best')

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
        """ Plot reciprocal-space k-points. """
        kwargs.setdefault("figsize", (5.0, 5.0))
        return plot_reciprocal_space(self.lattice, **kwargs)

    def brillouin_zone(self, **kwargs) -> Tuple[Figure, Axes]:
        """ Plot the Brillouin Zone. """
        kwargs.setdefault("figsize", (5.0, 4.0))
        return plot_brillouin_zone(self.lattice, **kwargs)

    def structure(self, **kwargs) -> Tuple[Figure, Axes]:
        """ 
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
        ... other args mirror plot_real_space ...
        """
        kwargs.setdefault("figsize", (6.0, 6.0))
        return plot_regions(self.lattice, regions, **kwargs)

# ------------------------------------------------------------------------------
#! EOF
# ------------------------------------------------------------------------------
