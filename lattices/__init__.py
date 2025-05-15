"""
A module for creating and managing lattices. This module provides a factory function to create a
lattice objects based on the specified lattice type. The supported lattice types are:
- "square"      : A square lattice.
- "hexagonal"   : A hexagonal lattice.

@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""

from typing import Union, Optional
from .lattice   import (
    Lattice, Backend as LatticeBackend, LatticeBC, LatticeDirection, LatticeType,
    handle_boundary_conditions, handle_dim
)
from .square    import SquareLattice
from .hexagonal import HexagonalLattice
from .honeycomb import HoneycombLattice
from general_python.common.plot import colorsCycle

# Export the lattice classes
__all__ = ["Lattice", "SquareLattice", "HexagonalLattice", "HoneycombLattice"]
_SWITCH = {
    "square"            : SquareLattice,
    "SquareLattice"     : SquareLattice,
    "hexagonal"         : HexagonalLattice,
    "HexagonalLattice"  : HexagonalLattice,
    "honeycomb"         : HoneycombLattice,
    "HoneycombLattice"  : HoneycombLattice
}

####################################################################################################
# Test the lattice module
####################################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def _handle_type(typek):
    """
    Handles and normalizes the input for lattice type.
    Parameters:
        typek (str): The lattice type to handle. Can be "square", "hexagonal", or "honeycomb".
    Returns:
        str: The corresponding lattice type in lowercase.
    Raises:
        ValueError: If the provided lattice type is not recognized.
    """
    if typek is None:
        return SquareLattice
    
    if isinstance(typek, str):
        if typek in _SWITCH:
            _out = _SWITCH[typek]
        else:
            raise ValueError(f"Unknown lattice type: {typek}")
    elif isinstance(typek, LatticeType):
        if typek == LatticeType.SQUARE:
            _out = SquareLattice
        elif typek == LatticeType.HEXAGONAL:
            _out = HexagonalLattice
        elif typek == LatticeType.HONEYCOMB:
            _out = HoneycombLattice
        else:
            raise ValueError(f"Unknown lattice type: {typek}")
    else:
        raise ValueError(f"Unknown lattice type: {typek}")
    return _out

def choose_lattice(typek    : Optional[str]         = 'square',
                dim         : Optional[int]         = None,
                lx          : Optional[int]         = 1,
                ly          : Optional[int]         = 1,
                lz          : Optional[int]         = 1,
                bc          : Optional[LatticeBC]   = None,
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

    Returns:
        Lattice: An instance of the desired lattice.
    """
    #! handle boundary conditions
    _bc             = handle_boundary_conditions(bc)
    #! handle dimensions
    dim, lx, ly, lz = handle_dim(lx, ly, lz)
    #! handle type
    _class          = _handle_type(typek)
    return _class(dim, lx, ly, lz, _bc, **kwargs)

# ------------------------------------------------------------------------

def plot_bonds(lattice      : Lattice,
            ax              : plt.Axes  = None,
            **line_kwargs) -> plt.Axes:
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

###################################
#! Tests
###################################

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
    print(f"\n✅ All tests completed successfully for {lattice}!")

def plot_lattice_structure(lattice):
    """
    Generate plots for the lattice structure and neighbor connections using real-space positions.
    For lattices other than square, the real-space positions (obtained from lattice.rvectors or via
    lattice.get_real_vec) are used to display the lattice. Bond direction labels (displaying the angle,
    or azimuth/elevation in 3D) are added above the connecting lines.

    Args:
        lattice: A lattice instance with methods such as get_coordinates, get_nei, site_index,
                 and (optionally) attributes like Lx, Ly, Lz, dim, Ns, and rvectors.
    """
    # Helper: try to use the lattice's real-space vectors if available.
    def get_pos(i):
        '''Get the real-space position of a site i.'''
        if lattice.rvectors is not None and len(lattice.rvectors) >= lattice.Ns:
            return lattice.rvectors[i]
        # Fall back to the coordinate mapping (which might be in unit cell space)
        return lattice.get_coordinates(i)
    
    if lattice.dim == 1:
        # 1D Plot
        positions   = np.array([get_pos(i) for i in range(lattice.Ns)])     # get the positions in 1D
        x_coords    = positions[:, 0]                                       # x-coordinates                
        y_coords    = positions[:, 1]                                       # may be zeros or computed from lattice vectors
        plt.figure(figsize=(6, 3))
        plt.scatter(x_coords, y_coords, color='blue', zorder=2, label="Lattice Sites")
        
        # go through the coordinates and plot the connections
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.text(x, y + 0.1, f'{i}', fontsize=10, ha='center', color='red') # label each site
            
            # get the neighbors (forward only) and plot the connections
            for j in lattice.get_nn_forward(i):
                if j != -1:
                    pos_j = get_pos(j)
                    plt.plot([x, pos_j[0]], [y, pos_j[1]], 'k-', lw=1)      # plot the connection line
                    
                    # Compute bond vector and angle
                    mid = ((x + pos_j[0]) / 2, (y + pos_j[1]) / 2)
                    plt.text(mid[0], mid[1] + 0.1, f"{i}-{j}", fontsize=8, ha='center', color='green')
        plt.title(f"1D Lattice: {lattice.Lx} sites")
        plt.xlabel("x")
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    elif lattice.dim == 2:
        # 2D Plot
        fig, ax     = plt.subplots(figsize=(6, 6))
        positions   = [get_pos(i) for i in range(lattice.Ns)]
        xs          = [pos[0] for pos in positions]
        ys          = [pos[1] for pos in positions]
        ax.scatter(xs, ys, color='blue', s=50, zorder=2, label="Lattice Sites")
        # Label each site
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x, y + 0.1, f'{i}', fontsize=8, ha='center', color='red',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
        # Draw neighbor connections and add bond direction labels
        segments    = []
        for i in range(lattice.Ns):
            pos_i = get_pos(i)
            for j in lattice.get_nn_forward(i):
                if j != -1 and j > i:  # avoid duplicates
                    pos_j = get_pos(j)
                    segments.append([(pos_i[0], pos_i[1]), (pos_j[0], pos_j[1])])

                    # add connection between sites i and j
                    ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'k-', lw=1)
                    ax.text((pos_i[0] + pos_j[0]) / 2,
                            (pos_i[1] + pos_j[1]) / 2 + 0.1,
                            f"{i}-{j}",
                            fontsize=8,
                            ha='center',
                            color='green',
                            bbox=dict(facecolor='white', edgecolor='none', pad=1))
            lc = LineCollection(segments, colors='gray', linewidths=1)
            ax.add_collection(lc)
            
        ax.set_title(f"2D Lattice: {lattice.Lx} x {lattice.Ly}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    elif lattice.dim == 3:
        # 3D Plot
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        positions = [get_pos(i) for i in range(lattice.Ns)]
        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        zs = [pos[2] for pos in positions]
        ax.scatter(xs, ys, zs, color='blue', s=50, zorder=2)
        for i, (x, y, z) in enumerate(positions):
            ax.text(x, y, z + 0.1, f'{i}', fontsize=8, ha='center', color='red')
        # Draw forward neighbor connections
        for i in range(lattice.Ns):
            pos_i = get_pos(i)
            for j in lattice.get_nn_forward(i):
                if j != -1 and j > i:
                    pos_j = get_pos(j)
                    ax.plot([pos_i[0], pos_j[0]],
                            [pos_i[1], pos_j[1]],
                            [pos_i[2], pos_j[2]], 'k-', lw=1)
                    mid = ((pos_i[0] + pos_j[0]) / 2,
                           (pos_i[1] + pos_j[1]) / 2,
                           (pos_i[2] + pos_j[2]) / 2)
                    ax.text(mid[0], mid[1], mid[2] + 0.1, f"{i}-{j}",
                            fontsize=8, ha='center', color='green')
        ax.set_title(f"3D Lattice: {lattice.Lx} x {lattice.Ly} x {lattice.Lz}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

####################################################################################################