"""
Triangular Lattice Class
Implements a 2D triangular lattice for general_python.

-----------------------
File        : general_python/lattices/triangular.py
Author      : Maksymilian Kliczkowski
Date        : 2025-12-22
-----------------------
"""

import  numpy as np
from    typing import Optional

try:
    from .                      import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType
    from .tools.lattice_kspace  import HighSymmetryPoints
except ImportError:
    raise ImportError("Could not import Lattice base classes. Ensure the module is in the PYTHONPATH.")

#######################################################################################

class TriangularLattice(Lattice):
    """
    Implementation of the Triangular Lattice (2D).
    The triangular lattice is a 2D Bravais lattice with each site having 6 nearest neighbors.
    """
    def __init__(self, *, dim=2, lx=3, ly=3, lz=1, bc='pbc', **kwargs):
        '''
        Initialize a Triangular Lattice.
        '''
        
        super().__init__(dim, lx, ly, lz, bc, **kwargs)
        self._type      = LatticeType.TRIANGULAR
        self._ns        = self.Lx * self.Ly * self.Lz

        # Primitive lattice vectors for 2D triangular lattice
        self._a1        = np.array([self.a, 0, 0])
        self._a2        = np.array([self.a/2, np.sqrt(3)*self.a/2, 0])
        self._a3        = np.array([0, 0, self.c])

        self._basis     = np.array([[0.0, 0.0, 0.0]])

        self._delta_x   = self._a1
        self._delta_y   = self._a2
        self._delta_z   = self._a3
        self.init(**kwargs)

    def __str__(self):
        return f"TRI,{self.bc},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"

    def __repr__(self):
        return self.__str__()

    # -----------------------------------------------------------------------

    def high_symmetry_points(self) -> Optional[HighSymmetryPoints]:
        """
        Return high-symmetry points for the triangular lattice Brillouin zone.
        """
        return HighSymmetryPoints.triangular_2d()

    # -----------------------------------------------------------------------

    def get_real_vec(self, x: int, y: int, z: int):
        """
        Returns the real-space vector for a given (x, y, z) coordinate.
        """
        base = x * self._a1 + y * self._a2
        return base + self._basis[0] + z * self._a3

    def get_norm(self, x: int, y: int, z: int):
        return np.sqrt(x**2 + y**2 + z**2)

    def get_nn_direction(self, site, direction):
        # For triangular lattice, directions can be mapped to 0-5 for 6 neighbors
        mapping = {LatticeDirection.X: 0, LatticeDirection.Y: 1, LatticeDirection.Z: 2}
        idx     = mapping.get(direction, -1)
        return self._nn[site][idx] if idx >= 0 and idx < len(self._nn[site]) else -1

    def calculate_nn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculates the nearest neighbors (NN) for the triangular lattice.
        Each site has 6 nearest neighbors in 2D.
        """
        self._nn            = [[] for _ in range(self.Ns)]
        self._nn_forward    = [[] for _ in range(self.Ns)]

    def calculate_nnn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculates the next-nearest neighbors (NNN) for the triangular lattice.
        """
        self._nnn           = [[] for _ in range(self.Ns)]

    def site_index(self, x, y, z):
        return z * (self.Lx * self.Ly) + y * self.Lx + x

# ---------------------------------------------------------------------------
#! EOF
# -------------------------------------------------------------------------
