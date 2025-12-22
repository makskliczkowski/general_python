"""
Lattice Tools Module

This module provides utility functions for working with lattice structures in Python.

--------------------------------
File            : lattices/tools/lattice_tools.py
Author          : Maksymilian Kliczkowski
--------------------------------
"""

from enum import Enum, auto
from typing import Any

# -----------------------------------------------------------------------------------------------------------
# LATTICE ENUMERATIONS
# -----------------------------------------------------------------------------------------------------------

class LatticeDirection(Enum):
    '''
    Enumeration for the lattice directions
    '''
    X = 0
    Y = 1
    Z = 2
    
    def __str__(self):      return str(self.name).lower()
    def __repr__(self):     return f"LatticeDirection.{self.name}"

# -----------------------------------------------------------------------------------------------------------

class LatticeBC(Enum):
    '''
    Enumeration for the boundary conditions in the lattice model.
    '''
    PBC         = auto()    # Periodic Boundary Conditions
    OBC         = auto()    # Open Boundary Conditions      
    MBC         = auto()    # Mixed Boundary Conditions     - periodic in X direction, open in Y direction
    SBC         = auto()    # Special Boundary Conditions   - periodic in Y direction, open in X direction
    
    def __str__(self):      return str(self.name).lower()
    def __repr__(self):     return self.__str__()

# -----------------------------------------------------------------------------------------------------------

class LatticeType(Enum):
    '''
    Contains all the implemented lattice types for the lattice model. 
    '''
    SQUARE      = auto()    # Square lattice
    HEXAGONAL   = auto()    # Hexagonal lattice
    HONEYCOMB   = auto()    # Honeycomb lattice
    GRAPH       = auto()    # Generic graph lattice (adjacency-defined)
    TRIANGULAR  = auto()    # Triangular lattice
    
    def __str__(self):      return str(self.name).lower()
    def __repr__(self):     return self.__str__()

# -----------------------------------------------------------------------------------------------------------
#! Boundary Conditions, Lattice Types, Directions
# -----------------------------------------------------------------------------------------------------------

def handle_boundary_conditions(bc: Any):
    """
    Handles and normalizes the input for boundary conditions.
    Parameters:
    -----------
        bc (str, LatticeBC, or None):
            The boundary condition to handle. Can be a string
            ("pbc", "obc", "mbc", "sbc"), an instance of LatticeBC, or None.
    Returns:
    --------
        LatticeBC: The corresponding LatticeBC enum value for the given boundary condition.
    Raises:
        ValueError: If the provided boundary condition is not recognized.
    """

    if bc is None:
        bc = LatticeBC.PBC
    elif isinstance(bc, str):
        if bc.lower() == "pbc":
            bc = LatticeBC.PBC
        elif bc.lower() == "obc":
            bc = LatticeBC.OBC
        elif bc.lower() == "mbc":
            bc = LatticeBC.MBC
        elif bc.lower() == "sbc":
            bc = LatticeBC.SBC
        else:
            raise ValueError(f"Unknown boundary condition: {bc}")
    elif not isinstance(bc, LatticeBC):
        raise ValueError(f"Unknown boundary condition: {bc}")
    return bc

def handle_dim(lx, ly, lz):
    """
    Handles and normalizes the input for lattice dimensions.
    Parameters:
        lx (int):
            Number of sites in the x-direction.
        ly (int):
            Number of sites in the y-direction.
        lz (int):
            Number of sites in the z-direction.
    Returns:
        tuple: A tuple containing the dimensions (lx, ly, lz).
    """
    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError("Lattice dimensions must be positive integers.")
    
    dim = 1
    if ly > 1:
        dim += 1
    if lz > 1:
        dim += 1
    return dim, lx, ly, lz

# -----------------------------------------------------------------------------------------------------------
#! END OF FILE
# ----------------------------------------------------------------------------------------------------------