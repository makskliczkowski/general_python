"""Lattice factories and geometry utilities.

Purpose
-------
Provide lattice classes, boundary-condition handling, and registry-based
constructors for real-space and reciprocal-space geometry.

Input/output contracts
----------------------
- Factory helpers expect lattice dimensions, sizes, and boundary-condition
  settings. Returned objects implement the ``Lattice`` interface.
- Adjacency and neighbor utilities return integer index arrays or dense/sparse
  adjacency matrices with a consistent site ordering.

Dtype and shape expectations
----------------------------
- Coordinates are float arrays of shape ``(ns, dim)``.
- Adjacency matrices are shape ``(ns, ns)`` with integer or float entries.
- Reciprocal-space vectors are float arrays of shape ``(dim,)`` or ``(dim, dim)``.

Numerical stability notes
-------------------------
- Geometry routines are mostly integer-based, but reciprocal-space transforms
  can accumulate floating-point error for large lattices; keep units consistent.

Determinism notes
-----------------
- Deterministic given the same lattice parameters.
- Random graph lattices (if used) require explicit seeding for reproducibility.
"""

from collections                import OrderedDict
from typing                     import TYPE_CHECKING, Optional, Tuple, Type, Union
import numpy                    as np
if TYPE_CHECKING:
    import matplotlib.axes      as pltAxes
    import matplotlib.pyplot    as plt
    
from .lattice   import (
    BoundaryFlux,
    Lattice,
    Backend as LatticeBackend,
    LatticeBC,
    LatticeDirection,
    LatticeType,
    handle_boundary_conditions,
    handle_dim,
    HighSymmetryPoints,
    HighSymmetryPoint,
    KPathResult,
    StandardBZPath,
)
from .square        import SquareLattice
from .hexagonal     import HexagonalLattice
from .honeycomb     import HoneycombLattice
from .triangular    import TriangularLattice
from .graph         import GraphLattice
from ..common.plot  import colorsCycle

# Import visualization utilities
from .visualization import (
    format_lattice_summary,
    format_vector_table,
    format_real_space_vectors,
    format_reciprocal_space_vectors,
    format_brillouin_zone_overview,
    LatticePlotter,
    plot_real_space,
    plot_reciprocal_space,
    plot_brillouin_zone,
)

__all__ = [
    "BoundaryFlux",
    "Lattice",
    "LatticeBC",
    "LatticeDirection",
    "LatticeType",
    # Core lattice classes
    "SquareLattice",
    "HexagonalLattice",
    "HoneycombLattice",
    "TriangularLattice",
    "GraphLattice",
    # Factory functions - symmetry registry
    "HighSymmetryPoints",
    "HighSymmetryPoint",
    "KPathResult",
    "StandardBZPath",
    # Factory functions - registry
    "register_lattice",
    "available_lattices",
    # Factory function - main entry point
    "choose_lattice",
    # Visualization utilities
    "plot_bonds",
    "format_lattice_summary",
    "format_vector_table",
    "format_real_space_vectors",
    "format_reciprocal_space_vectors",
    "format_brillouin_zone_overview",
    "LatticePlotter",
    "plot_real_space",
    "plot_reciprocal_space",
    "plot_brillouin_zone",
    "plot_lattice_structure",
    # Testing utilities
    "run_lattice_tests",
]

LatticeFactory      = Type[Lattice]
_LATTICE_REGISTRY   : "OrderedDict[str, LatticeFactory]" = OrderedDict()

def register_lattice(name: str, lattice_cls: LatticeFactory, *aliases: str, overwrite: bool = False):
    """
    Register a lattice class under ``name`` and optional ``aliases``.
    """
    if not issubclass(lattice_cls, Lattice):
        raise TypeError(f"Registered lattice must inherit from Lattice; got {lattice_cls!r}")

    keys = (name, *aliases)
    for key in keys:
        if not overwrite and key in _LATTICE_REGISTRY:
            raise KeyError(f"Lattice '{key}' already registered. Pass overwrite=True to replace.")
    for key in keys:
        _LATTICE_REGISTRY[key] = lattice_cls


def available_lattices() -> Tuple[str, ...]:
    """
    Return tuple of registered lattice identifiers.
    """
    return tuple(_LATTICE_REGISTRY.keys())

# Default registrations
register_lattice("square",      SquareLattice,      "SquareLattice")
register_lattice("hexagonal",   HexagonalLattice,   "HexagonalLattice")
register_lattice("honeycomb",   HoneycombLattice,   "HoneycombLattice")
register_lattice("triangular",  TriangularLattice,  "TriangularLattice")
register_lattice("graph",       GraphLattice,       "GraphLattice")

# --------------------------------------------------------------------------------------------------

def plot_bonds(lattice      : Lattice,
            ax              : 'pltAxes' = None,
            **line_kwargs) -> 'pltAxes':
    '''
    Plot physical bonds of the lattice using primitive vectors (a1,a2,a3).

    Args:
        ax (Axes):
            existing matplotlib Axes; new one if None.
        include_nnn  (bool):
            include next-nearest bonds if True.
        **line_kwargs: passed to ax.plot and ax.scatter.
    '''
    if lattice is None:
        raise ValueError("Lattice cannot be None")
    dim         = lattice.dim
    Ns          = lattice.ns
    colors      = [next(colorsCycle) for _ in range(10)]
    if ax is None:
        if dim == 3:
            fig     = plt.figure()
            ax      = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()
        
    #! lattice indices
    coords      = np.asarray(lattice.coordinates)
    if coords.ndim != 2 or coords.shape[1] < dim:
        raise ValueError(f"Coordinates must have at least {dim} columns, got {coords.shape}")
    coords_idx  = coords[:, :dim]
    
    #! primitive vectors
    a1          = np.array(lattice.a1, dtype=float)[:dim]
    a2          = np.array(lattice.a2, dtype=float)[:dim]
    a3          = np.array(lattice.a3, dtype=float)[:dim]
    prims       = np.vstack((a1, a2, a3))   # shape (3, >=dim)
    M           = prims[:dim, :dim]         # shape (dim, dim)
    
    #! real-space positions
    pos         = coords_idx.dot(M)      # shape (Ns, dim)
    
    #! adjacency
    A           = lattice.adjacency_matrix(sparse=False, save=False)

    #! draw bonds
    if dim == 1:
        xs      = pos[:, 0]
        ys      = np.zeros_like(xs)
        for i in range(Ns):
            nonzero = np.nonzero(A[i])[0]
            for ctr, item in enumerate(nonzero):
                color   = colors[ctr % len(colors)]
                j       = item
                if j > i:
                    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], **line_kwargs, color=color)
        ax.scatter(xs, ys, **line_kwargs)
        ax.set_xlabel('$x$')
        ax.set_yticks([])
    elif dim == 2:
        xs, ys = pos[:, 0], pos[:, 1]
        for i in range(Ns):
            nonzero = np.nonzero(A[i])[0]
            for ctr, item in enumerate(nonzero):
                color = colors[ctr % len(colors)]
                ax.plot([xs[i], xs[item]], [ys[i], ys[item]], **line_kwargs, color=color)
        ax.scatter(xs, ys, **line_kwargs)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal', 'box')
    elif dim == 3:
        xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
        for i in range(Ns):
            nonzero = np.nonzero(A[i])[0]
            for ctr, item in enumerate(nonzero):
                color = colors[ctr % len(colors)]
                ax.plot([xs[i], xs[item]], [ys[i], ys[item]], [zs[i], zs[item]], **line_kwargs, color=color)
        ax.scatter(xs, ys, zs, **line_kwargs)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.set_box_aspect((1, 1, 1))
    else:
        raise ValueError(f"Unsupported lattice dimension: {dim}")
    return ax

def plot_lattice_structure(lattice, **kwargs):
    """
    Wrapper for the visualization module's lattice structure plotter.
    """
    from .visualization import _plot_lattice_structure_visual
    return _plot_lattice_structure_visual(lattice, **kwargs)

####################################################################################################
#! Tests
####################################################################################################

def run_lattice_tests(dim=2, lx=5, ly=5, lz=1, bc=None, typek="square"):
    """
    Run automated tests for a lattice in 1D, 2D, or 3D.
    
    Args:
        dim   (int): Lattice dimension (1, 2, or 3)
        lx    (int): Number of sites in the x-direction
        ly    (int): Number of sites in the y-direction (ignored if dim=1)
        lz    (int): Number of sites in the z-direction (ignored if dim < 3)
        bc          : Boundary condition (e.g., LatticeBC.PBC or LatticeBC.OBC)
        typek (str) : Type of lattice ("square", "hexagonal", or "honeycomb")
    """
    # If no boundary condition is provided, default to periodic
    if bc is None:
        bc = LatticeBC.PBC

    lattice = choose_lattice(typek, dim=dim, lx=lx, ly=ly, lz=lz, bc=bc)
    print(f"Running tests for {lattice}")

    ## Test 1: Nearest Neighbors
    print("\n1) Testing nearest neighbors...")
    for i in range(lattice.Ns):
        neighbors = lattice.get_nei(i)
        print(f"\tSite {i}: Nearest Neighbors: {neighbors}")

    ## Test 2: Forward Nearest Neighbors
    print("\n2) Testing forward nearest neighbors...")
    for i in range(lattice.Ns):
        forward_neighbors = lattice.get_nn_forward(i)
        print(f"\tSite {i}: Forward Neighbors: {forward_neighbors}")

    ## Test 3: Coordinate Mapping
    print("\n3) Testing coordinate mapping...")
    for i in range(lattice.Ns):
        coords  = lattice.get_coordinates(i)
        idx     = lattice.site_index(*coords)
        print(f"\tSite {i}: Coordinates {coords} -> Index {idx}")
    print("\tCoordinate mapping test passed!")

    ## Test 4: Performance (for large lattices)
    if lattice.Ns > 1000:
        print("\n4) Running performance test (large lattice)...")
        try:
            start_time = time.time()
            lattice.calculate_dft_matrix()
            end_time = time.time()
            print(f"\tPerformance test passed! Time taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"\tPerformance test failed: {e}")

    ## Generate Lattice Plot
    plot_lattice_structure(lattice)
    print(f"\n(ok) All tests completed successfully for {lattice}!")

####################################################################################################
#! Test the lattice module
####################################################################################################

def _handle_type(typek):
    """
    Resolve an identifier (string or ``LatticeType``) to a registered lattice class.
    """
    if typek is None:
        return SquareLattice

    if isinstance(typek, str):
        cls = _LATTICE_REGISTRY.get(typek)
        if cls is None:
            raise ValueError(f"Unknown lattice type '{typek}'. Available: {available_lattices()}.")
        return cls

    if isinstance(typek, LatticeType):
        mapping = {
            LatticeType.SQUARE: "square",
            LatticeType.HEXAGONAL: "hexagonal",
            LatticeType.HONEYCOMB: "honeycomb",
            LatticeType.GRAPH: "graph",
        }
        key = mapping.get(typek)
        if key is None:
            raise ValueError(f"Unsupported lattice type enum {typek!r}.")
        return _LATTICE_REGISTRY[key]

    raise ValueError(f"Unknown lattice type: {typek!r}")

def choose_lattice(typek    : Optional[str]         = 'square',
                dim         : Optional[int]         = None,
                lx          : Optional[int]         = 1,
                ly          : Optional[int]         = 1,
                lz          : Optional[int]         = 1,
                bc          : Optional[Union[str, LatticeBC]]   = None,
                flux        : Optional[Union[float, BoundaryFlux, dict]] = None,
                **kwargs):
    """
    Returns an instance of a lattice of the desired type.

    Args:
        typek (str):
            Type of lattice ("square", "hexagonal", or "honeycomb")
        dim (int):
            Dimension (1, 2, or 3)
        lx (int):
            Number of sites in x-direction
        ly (int):
            Number of sites in y-direction
        lz (int):
            Number of sites in z-direction (ignored if dim < 3)
        bc: Boundary condition (e.g., LatticeBC.PBC or LatticeBC.OBC)
        flux:
            Optional boundary flux specification forwarded to the lattice
            constructor.  Accepts a scalar phase, :class:`BoundaryFlux`, or a
            mapping from directions to phases (in radians).

    Returns:
        Lattice: An instance of the desired lattice.
    """
    #! handle boundary conditions
    _bc             = handle_boundary_conditions(bc)
    #! handle dimensions
    dim, lx, ly, lz = handle_dim(lx, ly, lz)
    #! handle type
    _class          = _handle_type(typek)
    if issubclass(_class, GraphLattice):
        return _class(bc=_bc, flux=flux, **kwargs)
    return _class(dim, lx, ly, lz, _bc, flux=flux, **kwargs)

####################################################################################################
