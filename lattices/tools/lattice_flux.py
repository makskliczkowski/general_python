"""
A module for handling lattice fluxes.

Boundary fluxes model magnetic flux threading through the holes of a torus
(periodic boundary conditions). When a particle hops across a periodic
boundary in direction :math:`\\mu`, it acquires a phase
:math:`\\exp(i \\phi_\\mu)`. In reciprocal space this is equivalent to
shifting the allowed k-points:

.. math::
    f_\\mu \\;\\to\\; f_\\mu + \\frac{\\phi_\\mu}{2\\pi\\,L_\\mu},

where :math:`f_\\mu = n_\\mu / L_\\mu` is the standard fractional k-coordinate.

The flux values are always stored in **radians**.

--------------------------------
File            : lattices/tools/lattice_flux.py
Author          : Maksymilian Kliczkowski
Version         : 1.0
Date            : 2026-02-13
--------------------------------
"""

from    dataclasses import dataclass
from    typing      import Mapping, Union, Optional, Dict, Tuple, TYPE_CHECKING
import  numpy as np

# Runtime import – LatticeDirection is used in actual code paths, not just annotations.
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
    >>> flux.is_trivial
    False
    
    For non-abelian gauge fields, more complex structures are needed.
    """

    values : Mapping['LatticeDirection', float]

    # ------------------------------------------------------------------
    #  Phase retrieval
    # ------------------------------------------------------------------

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

    def phase_product(self, wx: int = 0, wy: int = 0, wz: int = 0) -> complex:
        """
        Return total phase from combined winding numbers in all directions.
        
        Returns :math:`\\exp(i (w_x \\phi_x + w_y \\phi_y + w_z \\phi_z))`.
        """
        phi_total  = wx * float(self.values.get(LatticeDirection.X, 0.0))
        phi_total += wy * float(self.values.get(LatticeDirection.Y, 0.0))
        phi_total += wz * float(self.values.get(LatticeDirection.Z, 0.0))
        return np.exp(1j * phi_total)

    def get(self, direction: 'LatticeDirection') -> float:
        """Return raw flux (radians) for *direction*, defaulting to 0."""
        return float(self.values.get(direction, 0.0))

    # ------------------------------------------------------------------
    #  Topological / triviality queries
    # ------------------------------------------------------------------

    @property
    def is_trivial(self) -> bool:
        """
        ``True`` when all fluxes are effectively zero (mod 2π).
        
        A flux of exactly :math:`2\\pi n` (integer multiples) is considered
        trivial because the hopping phase reduces to unity.
        """
        tol = 1e-12
        for phi in self.values.values():
            # reduce modulo 2π into (-π, π]
            reduced = (float(phi) + np.pi) % (2 * np.pi) - np.pi
            if abs(reduced) > tol:
                return False
        return True

    @property
    def is_nontrivial(self) -> bool:
        """``True`` when any direction carries a non-zero flux (mod 2π)."""
        return not self.is_trivial

    @property
    def total_flux(self) -> float:
        """Sum of all flux values (in radians)."""
        return sum(float(v) for v in self.values.values())

    # ------------------------------------------------------------------
    #  k-space shift
    # ------------------------------------------------------------------

    def k_shift_fractions(self,
                          Lx: int = 1,
                          Ly: int = 1,
                          Lz: int = 1) -> Tuple[float, float, float]:
        r"""
        Return the *fractional* k-grid offset induced by the boundary fluxes.

        With flux :math:`\phi_\mu` in direction :math:`\mu` of length
        :math:`L_\mu`, the allowed Bloch momenta shift from
        :math:`f_\mu = n_\mu / L_\mu` to
        :math:`f_\mu + \phi_\mu / (2\pi L_\mu)`.

        Returns
        -------
        (delta_fx, delta_fy, delta_fz) : tuple[float, float, float]
            Fractional coordinate shifts (add to the standard grid).
        """
        twopi = 2.0 * np.pi
        dx = float(self.values.get(LatticeDirection.X, 0.0)) / (twopi * Lx) if Lx > 0 else 0.0
        dy = float(self.values.get(LatticeDirection.Y, 0.0)) / (twopi * Ly) if Ly > 0 else 0.0
        dz = float(self.values.get(LatticeDirection.Z, 0.0)) / (twopi * Lz) if Lz > 0 else 0.0
        return (dx, dy, dz)

    # ------------------------------------------------------------------
    #  Conversion helpers
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, float]:
        """Return a plain ``{direction_name: flux}`` dictionary."""
        return {d.name.lower(): float(v) for d, v in self.values.items()}

    def as_array(self) -> np.ndarray:
        """Return fluxes as ``[phi_x, phi_y, phi_z]`` array (radians)."""
        return np.array([
            float(self.values.get(LatticeDirection.X, 0.0)),
            float(self.values.get(LatticeDirection.Y, 0.0)),
            float(self.values.get(LatticeDirection.Z, 0.0)),
        ])

    # ------------------------------------------------------------------
    #  Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def zero(cls) -> 'BoundaryFlux':
        """Create a ``BoundaryFlux`` with zero flux in all directions."""
        return cls({d: 0.0 for d in LatticeDirection})

    @classmethod
    def uniform(cls, phi: float) -> 'BoundaryFlux':
        """Create a ``BoundaryFlux`` with *phi* (rad) in every direction."""
        return cls({d: float(phi) for d in LatticeDirection})

    # ------------------------------------------------------------------
    #  Dunder helpers
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        """``True`` when the flux is non-trivial (has physical effect)."""
        return self.is_nontrivial

    def __repr__(self):
        parts = []
        for d in LatticeDirection:
            phi = float(self.values.get(d, 0.0))
            parts.append(f'{phi:.4f}')
        return f"({','.join(parts)})"
    
    def __str__(self):
        return self.__repr__()

def _normalize_flux_dict(flux: Optional[Union[float, complex, 'BoundaryFlux', Mapping[Union[str, 'LatticeDirection'], float]]]) -> Optional[BoundaryFlux]:
    """
    Normalize flux input into a :class:`BoundaryFlux` instance.
    
    This function handles various input formats for flux specification, including:
    - A single float value representing uniform flux in all directions.
    - A single complex number representing a uniform complex phase in all directions.
    - A mapping from direction (either as a string or a LatticeDirection) to float values.
    - An existing :class:`BoundaryFlux` instance (returned as-is).
    
    Allows to treat the value as a complex phase if a complex number is provided,
    which can be useful for certain applications where the flux is naturally expressed
    as a phase factor.
    
    Parameters
    ----------
    flux : float or complex or BoundaryFlux or Mapping or None
        - If ``None``, return ``None`` (no twist).
        - If a ``BoundaryFlux``, returned unchanged.
        - If a float, interpreted as uniform flux (radians) in all directions.
        - If a complex number, interpreted as a uniform complex phase
          (``np.angle(flux)`` is used for all directions).
        - If a mapping, keys can be :class:`LatticeDirection` members
          or their string names (case-insensitive).  Values are fluxes in radians.
          
    Returns
    -------
    BoundaryFlux or None
    """
    if flux is None:
        return None     # No flux specified, return None to indicate no twist
    
    # Already a BoundaryFlux — pass through
    if isinstance(flux, BoundaryFlux):
        return flux
    
    if isinstance(flux, (int, float)):
        phi = float(flux)
        return BoundaryFlux({direction: phi for direction in LatticeDirection})
    
    if isinstance(flux, complex):
        # Complex number → extract angle and apply uniformly
        phase = np.angle(flux)
        return BoundaryFlux({direction: phase for direction in LatticeDirection})
    
    if isinstance(flux, Mapping):
        out: Dict[LatticeDirection, float] = {}
        
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
            
            if isinstance(value, (int, float)):
                out[direction] = float(value)
            elif isinstance(value, complex):
                out[direction] = np.angle(value)
            else:
                raise TypeError(f"Unsupported flux value type: {type(value)!r}")
            
        return BoundaryFlux(out)
    raise TypeError(f"Unsupported flux specification of type {type(flux)!r}.")

# -----------------------------------------------------------------------------------------------------------
#! END OF FILE
# ----------------------------------------------------------------------------------------------------------