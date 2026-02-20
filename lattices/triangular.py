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
        return f"TRI,{self.bc},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}{self._flux_suffix}"

    def __repr__(self):
        return self.__str__()

    # -----------------------------------------------------------------------

    def high_symmetry_points(self) -> Optional[HighSymmetryPoints]:
        """
        Return high-symmetry points for the triangular lattice Brillouin zone.
        """
        return HighSymmetryPoints.triangular_2d()

    def contains_special_point(self, point, *, tol: float = 1e-12) -> bool:
        """Check if a triangular special point is present in the current k-grid."""
        return super().contains_special_point(point, tol=tol)

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

        Each site has 6 nearest neighbors in 2D corresponding to the six
        lattice-vector displacements::

            +a1, -a1, +a2, -a2, +(a1-a2), -(a1-a2)

        i.e. cell-coordinate offsets:
            (+1,0), (-1,0), (0,+1), (0,-1), (+1,-1), (-1,+1)

        Forward bonds are those connecting to a site with *strictly higher*
        index so that each bond is counted exactly once.

        Parameters
        ----------
        pbcx, pbcy, pbcz : bool
            Whether periodic boundary conditions apply along each direction.
        """
        Lx, Ly, Lz = self._lx, self._ly, self._lz
        Ns          = self._ns

        self._nn            = [[] for _ in range(Ns)]
        self._nn_forward    = [[] for _ in range(Ns)]

        def _bc(val, L, pbc):
            if pbc:
                return val % L
            return val if 0 <= val < L else -1

        def _idx(cx, cy, cz):
            return (cz * Ly + cy) * Lx + cx

        # Six NN offsets (dx, dy) in cell coordinates
        nn_offsets = [
            (+1,  0),   # +a1
            (-1,  0),   # -a1
            ( 0, +1),   # +a2
            ( 0, -1),   # -a2
            (+1, -1),   # +(a1 - a2)
            (-1, +1),   # -(a1 - a2)
        ]

        for i in range(Ns):
            cx =  i               % Lx
            cy = (i // Lx)        % Ly
            cz =  i // (Lx * Ly)

            for dx, dy in nn_offsets:
                nx = _bc(cx + dx, Lx, pbcx)
                ny = _bc(cy + dy, Ly, pbcy)
                if nx == -1 or ny == -1:
                    j = -1
                else:
                    j = _idx(nx, ny, cz)
                self._nn[i].append(j)

            # Forward: only neighbours with strictly higher index
            self._nn_forward[i] = [j for j in self._nn[i] if j > i]

        return self._nn

    def calculate_nnn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculates the next-nearest neighbors (NNN) for the triangular lattice.

        NNN are at cell-coordinate offsets::

            (+2,0), (-2,0), (0,+2), (0,-2), (+2,-2), (-2,+2),
            (+1,+1), (-1,-1), (+2,-1), (-2,+1), (+1,-2), (-1,+2)
        """
        Lx, Ly, Lz = self._lx, self._ly, self._lz
        Ns          = self._ns

        self._nnn           = [[] for _ in range(Ns)]
        self._nnn_forward   = [[] for _ in range(Ns)]

        def _bc(val, L, pbc):
            if pbc:
                return val % L
            return val if 0 <= val < L else -1

        def _idx(cx, cy, cz):
            return (cz * Ly + cy) * Lx + cx

        nnn_offsets = [
            (+2,  0), (-2,  0),
            ( 0, +2), ( 0, -2),
            (+2, -2), (-2, +2),
            (+1, +1), (-1, -1),
            (+2, -1), (-2, +1),
            (+1, -2), (-1, +2),
        ]

        for i in range(Ns):
            cx =  i               % Lx
            cy = (i // Lx)        % Ly
            cz =  i // (Lx * Ly)

            for dx, dy in nnn_offsets:
                nx = _bc(cx + dx, Lx, pbcx)
                ny = _bc(cy + dy, Ly, pbcy)
                if nx == -1 or ny == -1:
                    self._nnn[i].append(-1)
                else:
                    self._nnn[i].append(_idx(nx, ny, cz))

            self._nnn_forward[i] = [j for j in self._nnn[i] if j > i]

        return self._nnn

    def site_index(self, x, y, z):
        return z * (self.Lx * self.Ly) + y * self.Lx + x

    @staticmethod
    def dispersion(k, a=1.0):
        """
        Simple triangular-lattice dispersion approximation:
        ω(k) = 2J * [3 - cos(k·a1) - cos(k·a2) - cos(k·(a1 - a2))]
        where a1=(a,0), a2=(a/2, √3 a/2).
        Accepts k as (2,) or (...,2).
        """
        k   = np.asarray(k)
        a1  = np.array([a, 0.0])
        a2  = np.array([a/2.0, np.sqrt(3.0) * a / 2.0])
        a3  = a1 - a2
        def _omega(kx, ky):
            return 2.0 * (3.0 - np.cos(kx * a1[0] + ky * a1[1]) - np.cos(kx * a2[0] + ky * a2[1]) - np.cos(kx * a3[0] + ky * a3[1]))
        if k.ndim == 1:
            kx, ky = k[0], k[1]
            return _omega(kx, ky)
        else:
            kx = k[..., 0]
            ky = k[..., 1]
            return _omega(kx, ky)

# ---------------------------------------------------------------------------
#! EOF
# -------------------------------------------------------------------------
