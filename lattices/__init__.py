"""
A module for creating and managing lattices. This module provides a factory function to create a
lattice objects based on the specified lattice type. The supported lattice types are:
- "square"      : A square lattice.
- "hexagonal"   : A hexagonal lattice.

@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""


from .__lattice__ import Lattice
from .__square_lattice__ import SquareLattice
from .__hexagonal_lattice__ import HexagonalLattice

# Export the lattice classes
__all__ = ["Lattice", "SquareLattice", "HexagonalLattice"]

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
    elif lattice_type == "honeycomb":
        return HexagonalLattice(*args, **kwargs)
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")