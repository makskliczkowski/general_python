"""
Human-friendly formatting helpers for lattice objects.

The functions in this module intentionally avoid any side-effects on the passed
`Lattice` instances.  They simply transform existing lattice data into plain
Python strings so the caller can direct the output to terminals, loggers, or
documentation tooling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ..lattice import Lattice, LatticeDirection


@dataclass(frozen=True)
class _VectorTableConfig:
    """Configuration holder for vector table formatting."""

    max_rows: int = 10
    precision: int = 3
    column_labels: Optional[Sequence[str]] = None
    index_label: str = "#"
    indentation: str = ""


def _as_float_array(vectors: Iterable[Sequence[float]]) -> np.ndarray:
    """
    Convert arbitrary vector-like input into a 2D float64 array.
    """
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, arr.shape[0])
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of vectors, got shape {arr.shape!r}.")
    return arr


def _default_labels(dimension: int) -> Sequence[str]:
    axes = ("x", "y", "z")
    return axes[:dimension]


def format_vector_table(
    vectors: Iterable[Sequence[float]],
    *,
    max_rows: int = 10,
    precision: int = 3,
    column_labels: Optional[Sequence[str]] = None,
    index_label: str = "#",
    indentation: str = "",
) -> str:
    """
    Return a tabular string representation of an array of vectors.

    Parameters
    ----------
    vectors:
        Any iterable producing coordinate sequences.
    max_rows:
        Maximum number of rows to include in the formatted output.
    precision:
        Number of decimal places to use when printing floating point values.
    column_labels:
        Optional axis labels.  Defaults to Cartesian axes based on vector dimension.
    index_label:
        Label for the index column.
    indentation:
        Optional indentation prefix applied to each line of the table.
    """
    cfg = _VectorTableConfig(
        max_rows=max_rows,
        precision=precision,
        column_labels=column_labels,
        index_label=index_label,
        indentation=indentation,
    )

    array = _as_float_array(vectors)
    dim = array.shape[1]
    labels = cfg.column_labels or _default_labels(dim)
    if len(labels) != dim:
        raise ValueError(f"Expected {dim} column labels, received {len(labels)}.")

    header = cfg.indentation + "\t".join([cfg.index_label, *labels])
    lines = [header]
    row_limit = min(cfg.max_rows, array.shape[0])
    fmt = "{:." + str(cfg.precision) + "f}"

    for idx in range(row_limit):
        values = "\t".join(fmt.format(val) for val in array[idx])
        lines.append(f"{cfg.indentation}{idx:>4d}\t{values}")

    if array.shape[0] > row_limit:
        remaining = array.shape[0] - row_limit
        lines.append(f"{cfg.indentation}... ({remaining} more rows)")

    return "\n".join(lines)


def _format_flux_lines(lattice: Lattice, precision: int) -> Sequence[str]:
    flux = getattr(lattice, "flux", None)
    if flux is None or not getattr(flux, "values", None):
        return ("Boundary flux: none",)

    lines = ["Boundary flux:"]
    for direction in LatticeDirection:
        if direction in flux.values:
            phi = flux.values[direction]
            phase = flux.phase(direction)
            amp = abs(phase)
            argument = math.atan2(phase.imag, phase.real)
            lines.append(
                f"  {direction.name}: {phi:.{precision}f} rad "
                f"(phase = exp(i*{phi:.{precision}f}) "
                f"|phase|={amp:.2f}, arg={argument:.{precision}f})"
            )
    return tuple(lines)


def _format_basis(prefix: str, vector: Optional[Sequence[float]], precision: int) -> str:
    if vector is None:
        return f"{prefix}: unavailable"
    arr = np.asarray(vector, dtype=float).flatten()
    if arr.size == 0:
        return f"{prefix}: unavailable"
    comps = ", ".join(f"{val:.{precision}f}" for val in arr[:3])
    return f"{prefix}: ({comps})"


def format_lattice_summary(lattice: Lattice, *, precision: int = 3) -> str:
    """
    Produce a multi-line summary describing key lattice metadata.
    """
    lines: list[str] = []

    type_name = getattr(lattice, "typek", None)
    lattice_kind = type_name.name if type_name is not None else lattice.__class__.__name__
    lines.append(f"Lattice type: {lattice_kind}")
    lines.append(
        f"Dimensions: d={lattice.dim} "
        f"(Lx={lattice.Lx}, Ly={lattice.Ly}, Lz={lattice.Lz}) sites={lattice.Ns}"
    )
    lines.append(f"Boundary: {lattice.bc.name} periodic flags={lattice.periodic_flags()}")

    # Primitive real-space vectors when available
    lines.append(_format_basis("a1", getattr(lattice, "a1", None), precision))
    lines.append(_format_basis("a2", getattr(lattice, "a2", None), precision))
    if lattice.dim >= 3:
        lines.append(_format_basis("a3", getattr(lattice, "a3", None), precision))

    # Reciprocal primitive vectors when available
    if hasattr(lattice, "k1"):
        lines.append(_format_basis("b1", getattr(lattice, "k1", None), precision))
    if hasattr(lattice, "k2"):
        lines.append(_format_basis("b2", getattr(lattice, "k2", None), precision))
    if lattice.dim >= 3 and hasattr(lattice, "k3"):
        lines.append(_format_basis("b3", getattr(lattice, "k3", None), precision))

    lines.extend(_format_flux_lines(lattice, precision))

    coords = getattr(lattice, "coordinates", None)
    if coords:
        lines.append(f"Stored coordinates: {len(coords)} entries")
    return "\n".join(lines)


def format_real_space_vectors(
    lattice: Lattice,
    *,
    max_rows: int = 10,
    precision: int = 3,
    indentation: str = "",
) -> str:
    """
    Format a table of lattice real-space vectors.
    """
    arr = _as_float_array(lattice.rvectors)
    target_dim = lattice.dim if lattice.dim and lattice.dim > 0 else arr.shape[1]
    dim = max(1, min(arr.shape[1], target_dim))
    trimmed = arr[:, :dim]
    column_labels = _default_labels(dim)
    return format_vector_table(
        trimmed,
        max_rows=max_rows,
        precision=precision,
        column_labels=column_labels,
        indentation=indentation,
    )


def format_reciprocal_space_vectors(
    lattice: Lattice,
    *,
    max_rows: int = 10,
    precision: int = 3,
    indentation: str = "",
) -> str:
    """
    Format a table of reciprocal (k-space) vectors.
    """
    arr = _as_float_array(lattice.kvectors)
    target_dim = lattice.dim if lattice.dim and lattice.dim > 0 else arr.shape[1]
    dim = max(1, min(arr.shape[1], target_dim))
    trimmed = arr[:, :dim]
    column_labels = _default_labels(dim)
    return format_vector_table(
        trimmed,
        max_rows=max_rows,
        precision=precision,
        column_labels=column_labels,
        indentation=indentation,
    )


def format_brillouin_zone_overview(
    lattice: Lattice,
    *,
    precision: int = 3,
) -> str:
    """
    Provide a textual overview of the sampled Brillouin zone.

    The function reports bounding box limits and attempts to compute the
    convex hull measure (length/area/volume) when SciPy is available.
    """
    kvectors = getattr(lattice, "kvectors", None)
    if kvectors is None or len(kvectors) == 0:
        return "No reciprocal-space vectors available."

    arr = _as_float_array(kvectors)
    target_dim = lattice.dim if lattice.dim and lattice.dim > 0 else arr.shape[1]
    dim = max(1, min(arr.shape[1], target_dim))
    arr = arr[:, :dim]

    labels = _default_labels(min(3, dim))
    bounds = []
    for axis, label in enumerate(labels):
        comp = arr[:, axis]
        bounds.append(
            f"{label}: [{comp.min():.{precision}f}, {comp.max():.{precision}f}]"
        )

    lines = [f"Reciprocal-space bounds: {', '.join(bounds)}"]

    # Attempt to compute convex hull measure when applicable.
    try:
        from scipy.spatial import ConvexHull  # type: ignore

        if dim >= 2 and arr.shape[0] >= dim + 1:
            hull = ConvexHull(arr[:, :dim])
            measure_name = {2: "area", 3: "volume"}.get(dim, "measure")
            lines.append(f"Convex hull {measure_name}: {hull.volume:.{precision}f}")
    except ImportError:
        lines.append("Convex hull metrics unavailable (scipy not installed).")
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(f"Convex hull computation failed: {exc}")

    return "\n".join(lines)
