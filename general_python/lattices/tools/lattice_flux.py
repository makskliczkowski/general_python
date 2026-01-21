"""
A module for handling lattice fluxes.

--------------------------------
File            : lattices/tools/lattice_flux.py
Author          : Maksymilian Kliczkowski
--------------------------------
"""

from    dataclasses import dataclass
from    typing      import Mapping, Union, Optional, Dict, TYPE_CHECKING
import  numpy as np

if TYPE_CHECKING:
    from .lattice_tools import LatticeDirection

@dataclass(frozen=True)
class BoundaryFlux:
    """
    Collection of magnetic fluxes piercing lattice boundary loops.

    The value associated with a direction is interpreted as the phase 
    ``phi``
    (in radians) acquired upon wrapping around the boundary once along that
    direction. The corresponding hopping phase factor is ``exp(1j * phi)``.
    
    The fluxes are stored as a mapping from :class:`LatticeDirection` to corresponding
    complex phase values.
    
    Options for specifying fluxes include:
    - Uniform flux in all directions (single float value).
    - Direction-specific fluxes (mapping from direction to float).
    - Zero flux (empty mapping).
    
    Physically, these fluxes correspond to magnetic fluxes threading
    the holes of a torus formed by periodic boundary conditions.
    
    Example:
    >>> flux = BoundaryFlux({LatticeDirection.X: np.pi/2, LatticeDirection.Y: np.pi})
    >>> flux.phase(LatticeDirection.X)
    (6.123233995736766e-17+1j)
    >>> flux.phase(LatticeDirection.Y)
    (-1+0j)
    
    For non-abelian gauge fields, more complex structures are needed.
    """

    values  : Mapping['LatticeDirection', float]

    def phase(self, direction: 'LatticeDirection', winding: int = 1) -> complex:
        """
        Return ``exp(1j * winding * phi_direction)``.
        
        Parameters:
        -----------
        direction : LatticeDirection
            The lattice direction for which to get the phase factor.
        winding : int, optional
            The winding number for the phase factor. Defaults to 1.
        """
        phi = float(self.values.get(direction, 0.0))
        return np.exp(1j * winding * phi)

def _normalize_flux_dict(flux: Optional[Union[float, Mapping[Union[str, 'LatticeDirection'], float]]]) -> BoundaryFlux:
    """
    Normalize flux input into a :class:`BoundaryFlux` instance.
    
    Parameters:
    -----------
    flux : float or Mapping[Union[str, LatticeDirection], float] or None
        If a float, interpreted as uniform flux in all directions.
        If a mapping, keys can be either :class:`LatticeDirection` members
        or their string names (case-insensitive).  Values are fluxes in radians.
        If None, interpreted as zero flux in all directions.
    """
    if flux is None:
        return BoundaryFlux({})
    
    if isinstance(flux, (int, float)):
        phi = float(flux)
        return BoundaryFlux({direction: phi for direction in 'LatticeDirection'})
    
    if isinstance(flux, Mapping):
        out: Dict['LatticeDirection', float] = {}
        
        # parse mapping
        for key, value in flux.items():
            
            if isinstance(key, LatticeDirection):
                direction = key
                
            elif isinstance(key, str):
                try:
                    direction = LatticeDirection[key.upper()]
                except KeyError as exc:
                    raise ValueError(f"Unknown lattice direction '{key}' for flux specification.") from exc
            else:
                raise TypeError(f"Unsupported flux key type: {type(key)!r}")
            out[direction] = float(value)
        return BoundaryFlux(out)
    raise TypeError(f"Unsupported flux specification of type {type(flux)!r}.")

# -----------------------------------------------------------------------------------------------------------
#! END OF FILE
# ----------------------------------------------------------------------------------------------------------