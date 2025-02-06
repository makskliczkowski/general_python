"""
A module for creating and managing lattices. This module provides a factory function to create a
lattice objects based on the specified lattice type. The supported lattice types are:
- "square"      : A square lattice.
- "hexagonal"   : A hexagonal lattice.

@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""

from .__lattice__ import Lattice, Backend as LatticeBackend, LatticeBC, LatticeDirection, LatticeType
from .__square_lattice__ import SquareLattice
from .__hexagonal_lattice__ import HexagonalLattice
from .__honeycomb_lattice__ import HoneycombLattice

# other imports

# Export the lattice classes
__all__ = ["Lattice", "SquareLattice", "HexagonalLattice", "HoneycombLattice"]

def choose_lattice(lattice_type, *args, **kwargs):
    """
    Choose and create a lattice object based on the specified lattice type.

    Parameters:
    lattice_type (str): The type of lattice to create. Supported types are "square" and "honeycomb".
    *args: Variable length argument list to pass to the lattice constructor.
    **kwargs: Arbitrary keyword arguments to pass to the lattice constructor.

    Returns:
    object: An instance of the specified lattice type.

    Raises:
    ValueError: If the specified lattice type is not supported.
    """
    if lattice_type == "square":
        return SquareLattice(*args, **kwargs)
    elif lattice_type == "hexagonal":
        return HexagonalLattice(*args, **kwargs)
    # elif lattice_type == "honeycomb":
        # return HoneycombLattice(*args, **kwargs)
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    
####################################################################################################

# Test the lattice module

####################################################################################################

import time
import matplotlib.pyplot as plt

def run_lattice_tests(dim=2, lx=5, ly=5, lz=1, bc=LatticeBC.PBC, typek = "square"):
    """
    Run automated tests for SquareLattice in 1D, 2D, or 3D.
    
    Args:
        - dim   : Lattice dimension (1D, 2D, or 3D)
        - lx    : Number of sites in the x-direction
        - ly    : Number of sites in the y-direction (ignored if dim=1)
        - lz    : Number of sites in the z-direction (ignored if dim<3)
        - bc    : Boundary condition (PBC or OBC)
        - typek : Type of lattice
    """
    lattice = choose_lattice(typek, dim=dim, lx=lx, ly=ly, lz=lz, bc=bc)
    print(f"Running tests for {lattice}")

    ## **Test: Nearest Neighbors**
    print("\t1) Testing nearest neighbors...")
    for i in range(lattice.Ns):
        neighbors = lattice.get_nei(i)
        for j in neighbors:
            print(f"\t\t\tSite {i}: Nearest Neighbor {j}")
        print(f"\t\tSite {i}: Nearest Neighbors {neighbors}")

    ## **Test: Forward Nearest Neighbors**
    print("\t2) Testing forward nearest neighbors...")
    for i in range(lattice.Ns):
        forward_neighbors = lattice.get_nn_forward(i)
        for j in forward_neighbors:
            print(f"\t\t\tSite {i}: Forward Nearest Neighbor {j}")
        print(f"\t\tSite {i}: Forward Nearest Neighbors {forward_neighbors}")

    ## **Test: Coordinate Mapping**
    print("\t3) Testing coordinate mapping...")
    for i in range(lattice.Ns):
        x, y, z     = lattice.get_coordinates(i)
        index       = lattice.site_index(x, y, z)
        print(f"\t\tSite {i}: Coordinates ({x}, {y}, {z}) -> Index {index}")
        assert index == i, f"Site {i}: Expected index {i}, got {index}."
    print("\t\t\tCoordinate mapping test passed!")

    ## **Performance Test for Large Lattice**
    if lattice.Ns > 1000:
        print("\t4) Running performance test (large lattice)...")
        try:
            start_time  = time.time()
            lattice.calculate_dft_matrix()
            end_time    = time.time()
            print(f"\t\t\tPerformance test passed! Time taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"\t\t\tPerformance test failed: {e}")

    ## **Generate Lattice Plots**
    plot_lattice_structure(lattice)
    print(f"✅ All tests completed successfully for {lattice}!")

def plot_lattice_structure(lattice):
    """
    Generate plots for lattice structure & nearest neighbors.
    """
    import matplotlib.pyplot as plt
    
    if lattice.dim == 1:
        x_coords = LatticeBackend.arange(lattice.Lx)
        y_coords = LatticeBackend.zeros(lattice.Lx)
        
        # scatter plot of lattice sites
        plt.scatter(x_coords, y_coords, c='blue', label="Lattice Sites")
        # plot nearest neighbors
        for i in range(lattice.Lx):
            plt.text(x_coords[i], y_coords[i] + 0.1, f'{i}', fontsize=8, ha='center')
            for j in lattice.nn_forward[i]:
                if j != -1:
                    if j < i:
                        arc_x = [x_coords[i], (x_coords[i] + x_coords[j]) / 2, x_coords[j]]
                        arc_y = [y_coords[i], max(y_coords) + 1, y_coords[j]]
                        plt.plot(arc_x, arc_y, 'k--')
                        plt.text((x_coords[i] + x_coords[j]) / 2, max(y_coords) + 1, f'{i}-{j}', fontsize=8, ha='center')
                    else:
                        plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 'k-')
                        plt.text((x_coords[i] + x_coords[j]) / 2, (y_coords[i] + y_coords[j]) / 2, f'{i}-{j}', fontsize=8, ha='center')
        plt.title(f"1D Lattice: {lattice.Lx} sites")
    elif lattice.dim == 2:
        fig, ax = plt.subplots()
        x_coords = []
        y_coords = []
        
        for i in range(lattice.Ns):
            x, y, _ = lattice.get_coordinates(i)
            x_coords.append(x)
            y_coords.append(y)
            ax.scatter(x, y, c='blue')
            ax.text(x_coords[i], y_coords[i] + 0.1, f'{i}', fontsize=8, ha='center')
            # for j in lattice.nn_forward[i]:
            #     if j != -1:
            #         x2, y2, _ = lattice.get_coordinates(j)
            #         ax.plot([x, x2], [y, y2], 'k-')
            #         ax.text((x + x2) / 2, (y + y2) / 2, f'{i}-{j}', fontsize=8, ha='center')

        plt.title(f"2D Square Lattice: {lattice.Lx} x {lattice.Ly}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

    elif lattice.dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(lattice.Ns):
            x, y, z = lattice.get_coordinates(i)
            ax.scatter(x, y, z, c='blue')
            for j in lattice.get_nn(i):
                if j != -1:
                    x2, y2, z2 = lattice.get_coordinates(j)
                    ax.plot([x, x2], [y, y2], [z, z2], 'k-')

        ax.set_title(f"3D Square Lattice: {lattice.Lx} x {lattice.Ly} x {lattice.Lz}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.show()

