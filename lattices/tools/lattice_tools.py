"""
Lattice Tools Module

This module provides utility functions for working with lattice structures in Python.

--------------------------------
File            : lattices/tools/lattice_tools.py
Author          : Maksymilian Kliczkowski
--------------------------------
"""

from enum   import Enum, auto
from typing import Any, Union, Tuple

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
    TWISTED     = auto()    # Twisted Boundary Conditions   - twisted boundary conditions with fluxes
    
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

def handle_twist_flux(flux: Union[None, dict, float, complex]):
    r"""
    Handles and normalizes the input for flux piercing the boundaries, which is relevant for twisted boundary conditions.
    Parameters:
    -----------
        flux (dict or float, optional):
            Flux piercing the boundaries. This can be a dictionary specifying the
            flux in each direction, or a single value applied to all directions.
    Returns:
    --------        
        dict: A dictionary containing the flux values for each direction (x, y, z).
    Raises:
        ValueError: If the provided flux is not in a recognized format.
    """
    if flux is None:
        return None # No flux specified, return None to indicate no twist
    
    elif isinstance(flux, (int, float, complex)):
        return {"x": flux, "y": flux, "z": flux}  # Same flux for all directions
    
    elif isinstance(flux, dict):
        # Normalise keys – accept LatticeDirection enums, their string names, or plain "x"/"y"/"z"
        def _get(d, direction_enum, str_key):
            """Look up a flux value by LatticeDirection enum **or** plain string key."""
            if direction_enum in d:
                return float(d[direction_enum])
            upper = str_key.upper()
            for k, v in d.items():
                if isinstance(k, str) and k.upper() == upper:
                    return float(v)
                if isinstance(k, LatticeDirection) and k.name == upper:
                    return float(v)
            return 0.0
        return {
            "x": _get(flux, LatticeDirection.X, "x"),
            "y": _get(flux, LatticeDirection.Y, "y"),
            "z": _get(flux, LatticeDirection.Z, "z"),
        }
    else:
        raise ValueError(f"Invalid flux format: {flux}")

def handle_boundary_conditions(bc: Union[LatticeBC, Any], flux: Union[None, dict, float, complex] = None) -> Union[LatticeBC, Tuple[LatticeBC, dict]]:
    """
    Handles and normalizes the input for boundary conditions.
    Parameters:
    -----------
        bc (str, LatticeBC, or None):
            The boundary condition to handle. Can be a string
            ("pbc", "obc", "mbc", "sbc"), an instance of LatticeBC, or None.
        flux (dict or float, optional):
            Flux piercing the boundaries. This can be a dictionary specifying the
            flux in each direction, or a single value applied to all directions. Importantly, 
            this automatically implies **TWISTED** boundary conditions, so the `bc` parameter can be left as None or set to 'TWISTED' for clarity.        
    Returns:
    --------
        LatticeBC: The corresponding LatticeBC enum value for the given boundary condition.
    Raises:
        ValueError: If the provided boundary condition is not recognized.
    """

    # First, handle the flux to determine if we have twisted BCs
    twist_flux = handle_twist_flux(flux)

    if bc is None:
        bc = LatticeBC.PBC
    elif isinstance(bc, str):
        if bc.lower() == "pbc":     # PBC - X and Y directions are periodic (Z direction is periodic too)
            bc = LatticeBC.PBC
        elif bc.lower() == "obc":   # OBC - X and Y directions are open (Z direction is open too)
            bc = LatticeBC.OBC
        elif bc.lower() == "mbc":   # MBC - X direction is periodic, Y direction is open (Z direction is periodic)
            bc = LatticeBC.MBC
        elif bc.lower() == "sbc":   # SBC - X direction is open, Y direction is periodic (Z direction is periodic)
            bc = LatticeBC.SBC
        elif bc.lower() == "twisted" and flux is not None: # TWISTED - Twisted Boundary Conditions with specified fluxes
            bc = LatticeBC.TWISTED
        else:
            raise ValueError(f"Unknown boundary condition: {bc}")
    elif not isinstance(bc, LatticeBC):
        raise ValueError(f"Unknown boundary condition: {bc}")
    
    if twist_flux is not None:
        bc = LatticeBC.TWISTED  # If flux is specified, we automatically have twisted BCs
        return bc, twist_flux   # Return both the BC and the flux information for further processing
    
    return bc

def handle_boundary_conditions_detailed(bc: Union[LatticeBC, Any], flux: Union[None, dict, float, complex] = None) -> dict:
    """
    Handles and normalizes the input for boundary conditions, providing detailed
    information about the periodicity in each direction.
    
    Parameters
    ----------
    bc : str, LatticeBC, or None
        The boundary condition to handle.
    flux : dict or float, optional
        Flux piercing the boundaries.  Implies TWISTED BCs when not *None*.

    Returns
    -------
    dict
        ``{'x': bool_or_float, 'y': ..., 'z': ...}``
        For PBC/OBC/MBC/SBC directions are ``True``/``False``.
        For TWISTED directions the numeric flux value is stored instead.

    Raises
    ------
    ValueError
        If the provided boundary condition is not recognized.
    """

    bc = handle_boundary_conditions(bc, flux=flux)  # Normalize first

    if isinstance(bc, tuple):
        # TWISTED BCs with flux information
        bc, twist_flux = bc
        return {
            "x": twist_flux.get("x", 0.0),
            "y": twist_flux.get("y", 0.0),
            "z": twist_flux.get("z", 0.0)
        }

    if bc == LatticeBC.PBC:
        return {"x": True,  "y": True,  "z": True   }
    elif bc == LatticeBC.OBC:
        return {"x": False, "y": False, "z": False  }
    elif bc == LatticeBC.MBC:
        return {"x": True,  "y": False, "z": True   }
    elif bc == LatticeBC.SBC:
        return {"x": False, "y": True,  "z": True   }
    elif bc == LatticeBC.TWISTED:
        # TWISTED without flux dict -> treat all directions as periodic
        return {"x": True, "y": True, "z": True}
    else:
        raise ValueError(f"Unknown boundary condition: {bc}")

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