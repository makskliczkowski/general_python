"""
Matplotlib-based visualisation helpers for lattice objects.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List, Set, Dict

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - imported for side-effects
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:  # pragma: no cover - matplotlib without mplot3d
    Poly3DCollection = None

try:
    from scipy.spatial import ConvexHull  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ConvexHull = None

from ..lattice import Lattice


def _ensure_numpy(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, arr.shape[0])
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of vectors, got shape {arr.shape!r}.")
    return arr


def _init_axes(ax: Optional[Axes], dim: int, projection: Optional[str] = None) -> Tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    if dim >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection or "3d")
    else:
        fig, ax = plt.subplots()
    return fig, ax


def _annotate_indices(ax: Axes, coords: np.ndarray, *, zorder: int = 5) -> None:
    for idx, point in enumerate(coords):
        if point.size == 3:
            ax.text(point[0], point[1], point[2], str(idx), zorder=zorder)
        elif point.size == 2:
            ax.text(point[0], point[1], str(idx), zorder=zorder)
        else:
            ax.text(point[0], 0.0, str(idx), zorder=zorder)


def _finalise_figure(fig: Figure, *, top_padding: float = 0.88) -> None:
    """
    Apply a consistent layout so that titles and labels do not overlap the data.
    """
    try:
        fig.tight_layout()
        fig.subplots_adjust(top=top_padding)
    except Exception:
        # Some projections (e.g. 3D) do not play nicely with tight_layout; ignore.
        pass


def plot_real_space(
    lattice: Lattice,
    *,
    ax: Optional[Axes] = None,
    show_indices: bool = False,
    color: str = "C0",
    marker: str = "o",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    title_kwargs: Optional[Dict[str, object]] = None,
    tight_layout: bool = True,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    """Scatter-plot of real-space lattice vectors.

    Parameters
    ----------
    show_indices:
        Annotate each site index next to the marker.
    figsize:
        Optional figure size in inches.
    title:
        Title text. Pass ``None`` to suppress the title entirely.
    tight_layout:
        If ``True`` (default) call :func:`tight_layout` to avoid overlaps.
    elev, azim:
        Optional spherical angles for 3D projections.
    scatter_kwargs:
        Forwarded to :func:`Axes.scatter`.
    """
    coords = _ensure_numpy(lattice.rvectors)
    target_dim = lattice.dim if lattice.dim else coords.shape[1]
    dim = max(1, min(coords.shape[1], target_dim, 3))
    coords = coords[:, :dim]
    fig, axis = _init_axes(ax, dim)
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)
    if dim == 3 and (elev is not None or azim is not None):
        current_elev = elev if elev is not None else getattr(axis, "elev", None)
        current_azim = azim if azim is not None else getattr(axis, "azim", None)
        axis.view_init(elev=current_elev, azim=current_azim)

    if dim == 1:
        axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), color=color, marker=marker, **scatter_kwargs)
        axis.set_ylim(-0.5, 0.5)
        axis.set_ylabel("y (projected)")
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], color=color, marker=marker, **scatter_kwargs)
        axis.set_ylabel("y")
        axis.set_aspect("equal", adjustable="datalim")
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=color, marker=marker, **scatter_kwargs)
        axis.set_zlabel("z")

    axis.set_xlabel("x")
    if title:
        kw: Dict[str, object] = {"pad": 12}
        if title_kwargs:
            kw.update(title_kwargs)
        axis.set_title(title, **kw)

    if show_indices:
        _annotate_indices(axis, coords)

    if tight_layout:
        _finalise_figure(fig)
    return fig, axis


def plot_reciprocal_space(
    lattice: Lattice,
    *,
    ax: Optional[Axes] = None,
    show_indices: bool = False,
    color: str = "C1",
    marker: str = "o",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    title_kwargs: Optional[Dict[str, object]] = None,
    tight_layout: bool = True,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    """Scatter-plot of reciprocal lattice vectors.

    Parameters mirror :func:`plot_real_space`.
    """
    coords = _ensure_numpy(lattice.kvectors)
    target_dim = lattice.dim if lattice.dim else coords.shape[1]
    dim = max(1, min(coords.shape[1], target_dim, 3))
    coords = coords[:, :dim]
    fig, axis = _init_axes(ax, dim)
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)
    if dim == 3 and (elev is not None or azim is not None):
        current_elev = elev if elev is not None else getattr(axis, "elev", None)
        current_azim = azim if azim is not None else getattr(axis, "azim", None)
        axis.view_init(elev=current_elev, azim=current_azim)

    if dim == 1:
        axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), color=color, marker=marker, **scatter_kwargs)
        axis.set_ylim(-0.5, 0.5)
        axis.set_ylabel("k_y (projected)")
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], color=color, marker=marker, **scatter_kwargs)
        axis.set_ylabel("k_y")
        axis.set_aspect("equal", adjustable="datalim")
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=color, marker=marker, **scatter_kwargs)
        axis.set_zlabel("k_z")

    axis.set_xlabel("k_x")
    if title:
        kw: Dict[str, object] = {"pad": 12}
        if title_kwargs:
            kw.update(title_kwargs)
        axis.set_title(title, **kw)

    if show_indices:
        _annotate_indices(axis, coords)

    if tight_layout:
        _finalise_figure(fig)
    return fig, axis


def _plot_1d_bz(axis: Axes, bounds: Tuple[float, float], *, facecolor: str, alpha: float) -> None:
    x_min, x_max = bounds
    axis.axvspan(x_min, x_max, ymin=0.25, ymax=0.75, facecolor=facecolor, alpha=alpha)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlabel("k")


def _plot_2d_bz(axis: Axes, points: np.ndarray, *, facecolor: str, edgecolor: str, alpha: float) -> None:
    if ConvexHull is not None:
        try:
            hull = ConvexHull(points)
            polygon = points[hull.vertices]
        except Exception:
            polygon = None
    else:
        polygon = None

    if polygon is None:
        # Fallback to axis-aligned bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        polygon = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ])

    axis.fill(*polygon.T, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=1.5)
    axis.plot(*polygon.T, color=edgecolor, linewidth=1.5)
    # close polygon for plotting
    axis.plot([polygon[-1, 0], polygon[0, 0]], [polygon[-1, 1], polygon[0, 1]], color=edgecolor)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("k_x")
    axis.set_ylabel("k_y")


def _plot_3d_bz(axis: Axes, points: np.ndarray, *, facecolor: str, edgecolor: str, alpha: float) -> None:
    if Poly3DCollection is None:
        raise RuntimeError("3D plotting support requires mpl_toolkits.mplot3d.")

    faces = None
    if ConvexHull is not None:
        try:
            hull = ConvexHull(points)
            faces = [points[simplex] for simplex in hull.simplices]
        except Exception:
            faces = None
    if faces is None:
        # Fallback to bounding box
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        corners = np.array([
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ])
        if ConvexHull is not None:
            hull = ConvexHull(corners)
            faces = [corners[simplex] for simplex in hull.simplices]
        else:
            # simple rectangular faces
            faces = [
                corners[[0, 1, 2, 3]],
                corners[[4, 5, 6, 7]],
                corners[[0, 1, 5, 4]],
                corners[[2, 3, 7, 6]],
                corners[[1, 2, 6, 5]],
                corners[[3, 0, 4, 7]],
            ]

    collection = Poly3DCollection(faces, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    axis.add_collection3d(collection)
    axis.set_xlabel("k_x")
    axis.set_ylabel("k_y")
    axis.set_zlabel("k_z")


def plot_brillouin_zone(
    lattice: Lattice,
    *,
    ax: Optional[Axes] = None,
    facecolor: str = "tab:blue",
    edgecolor: str = "black",
    alpha: float = 0.25,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    title_kwargs: Optional[Dict[str, object]] = None,
    tight_layout: bool = True,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
) -> Tuple[Figure, Axes]:
    """Plot an approximate Brillouin zone from the sampled reciprocal vectors."""
    coords = _ensure_numpy(lattice.kvectors)
    target_dim = lattice.dim if lattice.dim else coords.shape[1]
    dim = max(1, min(coords.shape[1], target_dim, 3))
    coords = coords[:, :dim]
    fig, axis = _init_axes(ax, dim)
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)

    if dim == 1:
        _plot_1d_bz(axis, (coords[:, 0].min(), coords[:, 0].max()), facecolor=facecolor, alpha=alpha)
    elif dim == 2:
        _plot_2d_bz(axis, coords[:, :2], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    else:
        _plot_3d_bz(axis, coords[:, :3], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)

    if title:
        kw: Dict[str, object] = {"pad": 12}
        if title_kwargs:
            kw.update(title_kwargs)
        axis.set_title(title, **kw)

    if tight_layout:
        _finalise_figure(fig)
    return fig, axis


def _gather_nn_edges(lattice: Lattice) -> List[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for i in range(lattice.Ns):
        neighbors = lattice.get_nn(i)
        if not neighbors:
            continue
        for j in neighbors:
            if j is None:
                continue
            if isinstance(j, float) and (math.isnan(j) or not math.isfinite(j)):
                continue
            try:
                j_idx = int(j)
            except (TypeError, ValueError):
                continue
            if j_idx < 0:
                continue
            a, b = sorted((int(i), j_idx))
            if a != b:
                edges.add((a, b))
    return sorted(edges)


def _infer_bipartite_coloring(adjacency: List[List[int]]) -> Optional[List[int]]:
    ns = len(adjacency)
    colors = [-1] * ns
    for start in range(ns):
        if colors[start] != -1 or not adjacency[start]:
            continue
        colors[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neigh in adjacency[node]:
                if neigh < 0:
                    continue
                if colors[neigh] == -1:
                    colors[neigh] = colors[node] ^ 1
                    queue.append(neigh)
                elif colors[neigh] == colors[node]:
                    return None
    for idx, neighbours in enumerate(adjacency):
        if colors[idx] == -1:
            colors[idx] = 0 if not neighbours else 1
    return colors


def _boundary_masks(positions: np.ndarray, lattice: Lattice, *, tol_factor: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return boolean masks for boundary versus interior sites based on spatial extent.
    """
    if lattice.dim == 0 or positions.size == 0:
        return np.zeros(positions.shape[0], dtype=bool), np.ones(positions.shape[0], dtype=bool)

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    span = np.maximum(maxs - mins, tol_factor)
    tol = span * tol_factor

    boundary_mask = np.zeros(positions.shape[0], dtype=bool)
    for axis in range(min(positions.shape[1], 3)):
        boundary_axis = (np.isclose(positions[:, axis], mins[axis], atol=tol[axis]) |
                         np.isclose(positions[:, axis], maxs[axis], atol=tol[axis]))
        boundary_mask |= boundary_axis

    interior_mask = ~boundary_mask
    return boundary_mask, interior_mask


def _draw_primitive_cell(axis: Axes, origin: np.ndarray, basis_vectors: List[np.ndarray], dim: int) -> None:
    if not basis_vectors:
        return
    color = "0.4"
    linestyle = ":"
    linewidth = 1.0

    if dim == 1 and len(basis_vectors) >= 1:
        points = np.vstack([origin, origin + basis_vectors[0]])
        axis.plot(points[:, 0], np.zeros_like(points[:, 0]), color=color, linestyle=linestyle, linewidth=linewidth)
    elif dim == 2 and len(basis_vectors) >= 2:
        a1, a2 = basis_vectors[:2]
        corners = np.array([
            origin,
            origin + a1,
            origin + a1 + a2,
            origin + a2,
            origin,
        ])
        axis.plot(corners[:, 0], corners[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)
    elif dim == 3 and len(basis_vectors) >= 3:
        a1, a2, a3 = basis_vectors[:3]
        base = origin
        corners = [
            base,
            base + a1,
            base + a2,
            base + a3,
            base + a1 + a2,
            base + a1 + a3,
            base + a2 + a3,
            base + a1 + a2 + a3,
        ]
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]
        for i, j in edges:
            start, end = corners[i], corners[j]
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.7,
            )


def _draw_boundary_annotations(
    axis: Axes,
    positions: np.ndarray,
    lattice: Lattice,
    *,
    periodic_color: str,
    open_color: str,
    offset_fraction: float,
) -> None:
    if positions.shape[1] < 2:
        return

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    mid = (mins + maxs) / 2.0
    diag = np.linalg.norm(maxs[:2] - mins[:2])
    padding = offset_fraction * (diag if diag > 0 else 1.0)

    flags = lattice.periodic_flags()
    labels = ("x", "y", "z")

    def _annotate_axis(axis_index: int, label: str) -> None:
        is_periodic = bool(flags[axis_index])
        color = periodic_color if is_periodic else open_color
        if axis_index == 0:  # x-direction
            y = mid[1] if positions.shape[1] > 1 else 0.0
            if is_periodic:
                axis.annotate(
                    f"Periodic {label}",
                    xy=(maxs[0], y),
                    xytext=(maxs[0] + padding, y),
                    ha="left",
                    va="center",
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, linestyle="--"),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )
                axis.annotate(
                    "",
                    xy=(mins[0], y),
                    xytext=(mins[0] - padding, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, linestyle="--"),
                )
            else:
                axis.plot([mins[0], mins[0]], [mins[1], maxs[1]], color=color, lw=1.2, linestyle="--", alpha=0.8)
                axis.plot([maxs[0], maxs[0]], [mins[1], maxs[1]], color=color, lw=1.2, linestyle="--", alpha=0.8)
                axis.text(
                    mid[0],
                    maxs[1] + padding,
                    f"Open {label}",
                    color=color,
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )
        elif axis_index == 1:  # y-direction
            x = mid[0]
            if is_periodic:
                axis.annotate(
                    f"Periodic {label}",
                    xy=(x, maxs[1]),
                    xytext=(x, maxs[1] + padding),
                    ha="center",
                    va="bottom",
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, linestyle="--"),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )
                axis.annotate(
                    "",
                    xy=(x, mins[1]),
                    xytext=(x, mins[1] - padding),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, linestyle="--"),
                )
            else:
                axis.plot([mins[0], maxs[0]], [mins[1], mins[1]], color=color, lw=1.2, linestyle="--", alpha=0.8)
                axis.plot([mins[0], maxs[0]], [maxs[1], maxs[1]], color=color, lw=1.2, linestyle="--", alpha=0.8)
                axis.text(
                    mins[0] - padding,
                    mid[1],
                    f"Open {label}",
                    color=color,
                    ha="right",
                    va="center",
                    rotation=90,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )

    for idx in range(min(positions.shape[1], 2)):
        _annotate_axis(idx, labels[idx])


def plot_lattice_structure(
    lattice: Lattice,
    *,
    ax: Optional[Axes] = None,
    show_indices: bool = False,
    highlight_boundary: bool = True,
    show_axes: bool = False,
    edge_color: str = "0.5",
    node_color: str = "tab:blue",
    boundary_node_color: str = "tab:red",
    periodic_color: str = "tab:orange",
    open_color: str = "tab:green",
    node_size: int = 50,
    edge_alpha: float = 0.7,
    label_padding: float = 0.05,
    boundary_offset: float = 0.08,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    title_kwargs: Optional[Dict[str, object]] = None,
    tight_layout: bool = True,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    partition_colors: Optional[Tuple[str, ...]] = None,
    show_periodic_connections: bool = True,
    show_primitive_cell: bool = True,
    **scatter_kwargs,
) -> Tuple[Figure, Axes]:
    """Visualise lattice geometry (Tenpy-style) with boundary cues.

    Parameters are similar to :func:`plot_real_space`, with the addition of
    ``highlight_boundary`` and ``show_axes`` to control stylistic choices and
    ``boundary_offset`` for the placement of open/periodic annotations. Titles
    can be customised or suppressed entirely via ``title``/``title_kwargs``.
    Passing ``partition_colors`` enables bipartite colouring when the lattice
    graph allows it. ``show_periodic_connections`` toggles textual annotation of
    wrap-around links, and ``show_primitive_cell`` draws the primitive cell
    generated by ``(a1, a2, a3)`` for context.
    """
    coords = _ensure_numpy(lattice.rvectors)
    target_dim = lattice.dim if lattice.dim else coords.shape[1]
    dim = max(1, min(coords.shape[1], target_dim, 3))
    coords = coords[:, :dim]

    fig, axis = _init_axes(ax, dim)
    if figsize is not None and axis is fig.axes[0]:
        fig.set_size_inches(*figsize, forward=True)
    if dim == 3 and (elev is not None or azim is not None):
        current_elev = elev if elev is not None else getattr(axis, "elev", None)
        current_azim = azim if azim is not None else getattr(axis, "azim", None)
        axis.view_init(elev=current_elev, azim=current_azim)

    edges = _gather_nn_edges(lattice)
    adjacency: List[List[int]] = [[] for _ in range(lattice.Ns)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    basis_vectors: List[np.ndarray] = []
    for attr in ("a1", "a2", "a3"):
        vec = getattr(lattice, attr, None)
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float).flatten()
        if arr.size == 0:
            continue
        if arr.size < dim:
            arr = np.pad(arr, (0, dim - arr.size))
        if np.linalg.norm(arr[:dim]) > 1e-9:
            basis_vectors.append(arr[:dim])

    periodic_neighbors: Dict[int, List[int]] = {i: [] for i in range(lattice.Ns)}
    periodic_label_counts = defaultdict(int)

    typical_distance = None
    if edges:
        distances = [
            np.linalg.norm(coords[j] - coords[i])
            for i, j in edges
            if np.linalg.norm(coords[j] - coords[i]) > 1e-9
        ]
        if distances:
            typical_distance = min(distances)

    for i, j in edges:
        start = coords[i]
        end = coords[j]
        dist = np.linalg.norm(end - start)
        is_periodic = typical_distance is not None and dist > typical_distance * 1.5
        linestyle = "--" if is_periodic else "-"

        if dim == 3:
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=edge_color,
                alpha=edge_alpha,
                linestyle=linestyle,
                linewidth=1.0,
            )
        elif dim == 2:
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=edge_color,
                alpha=edge_alpha,
                linestyle=linestyle,
                linewidth=1.0,
            )
        else:
            axis.plot(
                [start[0], end[0]],
                [0.0, 0.0],
                color=edge_color,
                alpha=edge_alpha,
                linestyle=linestyle,
                linewidth=1.0,
            )

        if is_periodic:
            periodic_neighbors[i].append(j)
            periodic_neighbors[j].append(i)

    boundary_mask, _ = _boundary_masks(coords, lattice)

    node_face_colors: List[str]
    partitions = _infer_bipartite_coloring(adjacency)
    if partitions is not None:
        palette = partition_colors or ("tab:blue", "tab:orange")
        unique_parts = sorted(set(partitions))
        color_map = {part: palette[idx % len(palette)] for idx, part in enumerate(unique_parts)}
        node_face_colors = [color_map[partitions[i]] for i in range(lattice.Ns)]
    else:
        node_face_colors = [node_color] * lattice.Ns

    # Remove axes by default for a cleaner "lattice diagram" look
    if not show_axes:
        if dim == 1:
            axis.get_yaxis().set_visible(False)
            axis.spines["left"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
        elif dim == 2:
            axis.set_axis_off()
        else:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([])
            axis.set_box_aspect((1, 1, 1))

    if dim == 1:
        axis.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), c=node_face_colors, s=node_size, **scatter_kwargs)
        axis.set_ylim(-0.5, 0.5)
        if show_axes:
            axis.set_ylabel("y (projected)")
    elif dim == 2:
        axis.scatter(coords[:, 0], coords[:, 1], c=node_face_colors, s=node_size, **scatter_kwargs)
        axis.set_aspect("equal", adjustable="datalim")
        if show_axes:
            axis.set_ylabel("y")
    else:
        axis.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=node_face_colors, s=node_size, **scatter_kwargs)
        if show_axes:
            axis.set_zlabel("z")

    if highlight_boundary and np.any(boundary_mask):
        boundary_coords = coords[boundary_mask]
        if dim == 3:
            axis.scatter(
                boundary_coords[:, 0],
                boundary_coords[:, 1],
                boundary_coords[:, 2],
                facecolors="none",
                edgecolors=boundary_node_color,
                s=node_size * 1.2,
                linewidths=1.2,
            )
        elif dim == 2:
            axis.scatter(
                boundary_coords[:, 0],
                boundary_coords[:, 1],
                facecolors="none",
                edgecolors=boundary_node_color,
                s=node_size * 1.2,
                linewidths=1.2,
            )
        else:
            axis.scatter(
                boundary_coords[:, 0],
                np.zeros_like(boundary_coords[:, 0]),
                facecolors="none",
                edgecolors=boundary_node_color,
                s=node_size * 1.2,
                linewidths=1.2,
            )

    if title:
        kw: Dict[str, object] = {"pad": 15}
        if title_kwargs:
            kw.update(title_kwargs)
        axis.set_title(title, **kw)

    if show_axes:
        axis.set_xlabel("x")
        if dim >= 2:
            axis.set_ylabel("y")
        if dim == 3:
            axis.set_zlabel("z")

    node_label_positions: Dict[int, np.ndarray] = {}

    if show_indices:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        diag = np.linalg.norm(maxs - mins) if coords.size else 1.0
        offset = label_padding * (diag if diag > 0 else 1.0)
        for idx, point in enumerate(coords):
            if dim == 3:
                label_pos = point + np.array([offset, offset, offset])
                axis.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    str(idx),
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )
            elif dim == 2:
                label_pos = point + np.array([offset, offset])
                axis.text(
                    label_pos[0],
                    label_pos[1],
                    str(idx),
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )
            else:
                label_pos = point + np.array([offset])
                axis.text(
                    label_pos[0],
                    offset,
                    str(idx),
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                )
            node_label_positions[idx] = label_pos

    if dim == 2:
        _draw_boundary_annotations(
            axis,
            coords,
            lattice,
            periodic_color=periodic_color,
            open_color=open_color,
            offset_fraction=boundary_offset,
        )

    if show_periodic_connections and periodic_neighbors:
        diag_extent = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) or 1.0
        base_offset = label_padding * diag_extent * 0.6
        for idx, neighbours in periodic_neighbors.items():
            if not neighbours:
                continue
            anchor = node_label_positions.get(idx, coords[idx])
            label = "↔ " + ",".join(str(n) for n in sorted(set(neighbours)))
            count = periodic_label_counts[idx]
            periodic_label_counts[idx] += 1

            offset_vec = np.zeros(dim)
            if dim >= 2:
                offset_vec[:2] = np.array([0.0, base_offset * (count + 1)])
            else:
                offset_vec[0] = base_offset * (count + 1)

            label_pos = anchor + offset_vec
            if dim == 3:
                axis.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    label,
                    color=periodic_color,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.4),
                )
            elif dim == 2:
                axis.text(
                    label_pos[0],
                    label_pos[1],
                    label,
                    color=periodic_color,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.4),
                )
            else:
                axis.text(
                    label_pos[0],
                    base_offset * (count + 1),
                    label,
                    color=periodic_color,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.4),
                )

    if show_primitive_cell and basis_vectors:
        origin = coords.min(axis=0)
        _draw_primitive_cell(axis, origin, basis_vectors, dim)

    if tight_layout:
        _finalise_figure(fig, top_padding=0.92)
    return fig, axis


@dataclass
class LatticePlotter:
    """
    Convenience wrapper bundling plotting helpers for a single lattice.
    """

    lattice: Lattice

    def real_space(self, **kwargs) -> Tuple[Figure, Axes]:
        kwargs.setdefault("figsize", (5.0, 5.0))
        kwargs.setdefault("title", None)
        return plot_real_space(self.lattice, **kwargs)

    def reciprocal_space(self, **kwargs) -> Tuple[Figure, Axes]:
        kwargs.setdefault("figsize", (5.0, 5.0))
        kwargs.setdefault("title", None)
        return plot_reciprocal_space(self.lattice, **kwargs)

    def brillouin_zone(self, **kwargs) -> Tuple[Figure, Axes]:
        kwargs.setdefault("figsize", (5.0, 4.0))
        kwargs.setdefault("title", None)
        return plot_brillouin_zone(self.lattice, **kwargs)

    def structure(self, **kwargs) -> Tuple[Figure, Axes]:
        kwargs.setdefault("figsize", (5.5, 5.5))
        kwargs.setdefault("title", None)
        return plot_lattice_structure(self.lattice, **kwargs)

# -------------------------
#! EOF
# -------------------------