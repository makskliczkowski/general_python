'''
Contains the Honeycomb lattice implementation.
This module defines the HoneycombLattice class, which extends the base Lattice class
to represent a 2D honeycomb lattice structure. It includes methods for calculating
nearest and next-nearest neighbors, as well as lattice vectors and coordinates.

---------------------------------
File        : general_python/lattices/honeycomb.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-01
License     : MIT
---------------------------------
'''

import numpy as np
from typing import Optional

try:
    from . import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType
    from ..maths.math_utils import mod_euc
    from .tools.lattice_kspace import HighSymmetryPoints
except ImportError:
    raise ImportError("Could not import Lattice base classes. Ensure the module is in the PYTHONPATH.")

################################### LATTICE IMPLEMENTATION #######################################

X_BOND_NEI = 2
Y_BOND_NEI = 1
Z_BOND_NEI = 0
# X_BOND_NEI = 0
# Y_BOND_NEI = 1
# Z_BOND_NEI = 2

class HoneycombLattice(Lattice):
    """
    Implementation of the Honeycomb Lattice.
    
    The honeycomb lattice is a 2D lattice with a hexagonal structure. The lattice consists of
    two sublattices (A and B) arranged in a hexagonal pattern. Nearest and next-nearest neighbors
    are computed based on a hexagonal unit cell.
    
    High-symmetry points in the Brillouin zone:
    - Gamma: 
        Zone center at (0, 0)
    - K:
        Dirac point at (2/3, 1/3) - hosts linear band crossings in graphene
    - K': 
        Other Dirac point at (1/3, 2/3)
    - M:    
        Edge midpoint at (1/2, 0)
    
    Default path: Γ -> K -> M -> Γ
    
    References:
        - Phys. Rev. Research 3, 013160 (2021)
        - Fig. 2, https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.013160
    
    Attributes:
        Lx, Ly, Lz: Number of lattice sites in x, y, and z directions.
        bc        : Boundary condition (e.g. PBC or OBC).
        a, c      : Lattice parameters.
        vectors   : Primitive lattice vectors.
        kvectors  : Reciprocal lattice vectors.
        rvectors  : Real-space vectors.
    """

    def __init__(self, *, dim=2, lx=3, ly=1, lz=1, bc='pbc', **kwargs):
        """
        Initialize a honeycomb lattice.
        
        Args:
            dim (int)           : Lattice dimension (1, 2, or 3)
            lx, ly, lz (int)    : Lattice sizes in x, y, z directions.
            bc                  : Boundary condition (e.g. LatticeBC.PBC or LatticeBC.OBC)
        """
        super().__init__(dim, lx, ly, lz, bc, **kwargs)

        self._type = LatticeType.HONEYCOMB  # Lattice type

        # For the honeycomb lattice there are two sites per unit cell.
        self._ns        = 2 * self.Lx * self.Ly * self.Lz

        # Define lattice parameters
        # self._a1        = np.array([np.sqrt(3) * self.a, 0, 0])
        # self._a2        = np.array([np.sqrt(3) * self.a / 2.0, 3 * self.a / 2.0, 0])
        # self._a3        = np.array([0, 0, self.c])
        self._a1        = np.array([np.sqrt(3) * self.a / 2.0, 3 * self.a / 2.0, 0])
        self._a2        = np.array([-np.sqrt(3) * self.a / 2.0, 3 * self.a / 2.0, 0])
        self._a3        = np.array([0, 0, self.c])

        self._basis     = np.array([
                            [0.0, 0.0, 0.0],      # A sublattice
                            [0.0, self.a, 0.0]    # B sublattice
                        ])

        self._delta_x   = np.array([0.0, self.a, 0.0])
        self._delta_y   = np.array([-np.sqrt(3)*self.a/2.0, -self.a/2.0, 0.0])
        self._delta_z   = np.array([ np.sqrt(3)*self.a/2.0, -self.a/2.0, 0.0])
        self.init(**kwargs)
        
    def __str__(self):
        return f"HON,{self.bc},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"

    def __repr__(self):
        return self.__str__()

    # ---------------------------------------------------------------------------------
    #! High-symmetry points
    # ---------------------------------------------------------------------------------

    def high_symmetry_points(self) -> Optional[HighSymmetryPoints]:
        """
        Return high-symmetry points for the honeycomb lattice.
        
        Returns
        -------
        HighSymmetryPoints
            High-symmetry points for the hexagonal Brillouin zone:
            - Γ (Gamma): Zone center (0, 0)
            - K: Dirac point at (2/3, 1/3) - hosts linear band crossings
            - K': Other Dirac point at (1/3, 2/3)
            - M: Edge midpoint at (1/2, 0)
            
            Default path: Γ -> K -> M -> Γ
        """
        return HighSymmetryPoints.honeycomb_2d()

    ################################### 
    
    def sublattice(self, site: int) -> int:
        """
        Return the sublattice index for a given site.
        By default, returns 0 for all sites (single sublattice).
        Override in subclasses for multi-sublattice lattices.
        """
        return site % self.multipartity

    ###################################

    def get_real_vec(self, x: int, y: int, z: int):
        """
        Returns the real-space vector for a given (x, y, z) coordinate.
        """
        cell_x = x
        # coordinates are stored as (x, 2*y + sublattice, z)
        cell_y = y // 2
        sub = y % 2
        base = cell_x * self._a1 + cell_y * self._a2
        return base + self._basis[sub] + z * self._a3

    def get_norm(self, x: int, y: int, z: int):
        """
        Returns the Euclidean norm of the real-space vector.
        """
        return np.sqrt(x**2 + y**2 + z**2)

    def get_nn_direction(self, site, direction):
        """
        Returns the nearest neighbor in the specified direction.
        
        For the honeycomb lattice, we choose a mapping:
            LatticeDirection.X -> neighbor at index 0 of _nn[site]
            LatticeDirection.Y -> neighbor at index 1 of _nn[site]
            LatticeDirection.Z -> neighbor at index 2 of _nn[site]
        """
        mapping = {LatticeDirection.X: 0, LatticeDirection.Y: 1, LatticeDirection.Z: 2}
        idx     = mapping.get(direction, -1)
        return self._nn[site][idx] if idx >= 0 and idx < len(self._nn[site]) else -1

    def get_nn_forward(self, site : int, num : int = -1):
        """
        Returns the forward nearest neighbor for the given site.
        
        (For honeycomb, this could be defined as the first neighbor in a chosen ordering.)
        """
        if hasattr(self, '_nn_forward') and self._nn_forward:
            if num < 0:
                return self._nn_forward[site]
            return self._nn_forward[site][num] if num < len(self._nn_forward[site]) else -1
        return -1

    def get_nnn_forward(self, site : int, num : int = -1):
        """
        Returns the forward next-nearest neighbor for the given site.
        """
        if hasattr(self, '_nnn_forward') and self._nnn_forward:
            if num < 0:
                return self._nnn_forward[site]
            return self._nnn_forward[site][num] if num < len(self._nnn_forward[site]) else -1
        return -1
    
    ################################### NEIGHBORHOOD CALCULATORS #######################################

    def calculate_nn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculates the nearest neighbors (NN) using boundary conditions.
        
        The implementation uses a helper function to apply periodic or open boundary conditions.
        For 2D, for example, we use a different treatment on even and odd indices.
        """
        self._nn            = [[] for _ in range(self.Ns)]
        self._nn_forward    = [[] for _ in range(self.Ns)]
        
        # Helper function to apply periodic boundary conditions.
        def _bcfun(_i, _l, _pbc):
            if _pbc:
                return mod_euc(_i, _l)
            return _i if 0 <= _i < _l else -1
        
        # 1D: Each site has two neighbors.
        if self.dim == 1:
            for i in range(self.Ns):
                self._nn[i] = [
                    _bcfun(i + 1, self.Lx, pbcx),
                    _bcfun(i - 1, self.Lx, pbcx)
                ]
                self._nn_forward[i] = [_bcfun(i + 1, self.Lx, pbcx)]
            # (Optionally, you might also set forward neighbors here.)
        
        # 2D: Map honeycomb sites onto an underlying square lattice.
        elif self.dim == 2:
            for i in range(self.Ns):
                n       = i // 2        # n: site index on the square lattice.
                r       = i % 2         # r: sublattice index (0 for first node, 1 for second) - idx in the elementary cell.
                X       = n % self.Lx   # X: x coordinate on the square lattice.
                Y       = n // self.Lx  # Y: y coordinate on the square lattice.
                _even   = (r == 0)      # _even: whether the site is on the even sublattice.
                
                # z bond: for even sites
                YP      = _bcfun(Y - 1, self.Ly, pbcy) if _even else _bcfun(Y + 1, self.Ly, pbcy)
                XP      = X
                if YP == -1:
                    self._nn[i].append(-1)
                else:
                    self._nn[i].append((YP * self.Lx + XP) * 2 + int(_even))
                self._nn_forward[i].append((-1 if _even else self._nn[i][0]))
                
                # y bond: for even sites, use X - 1; for odd, use X + 1.
                XP      = _bcfun(X - 1, self.Lx, pbcx) if _even else _bcfun(X + 1, self.Lx, pbcx)
                YP      = Y
                if XP == -1:
                    self._nn[i].append(-1)
                else:
                    self._nn[i].append((YP * self.Lx + XP) * 2 + int(_even))
                self._nn_forward[i].append(-1 if _even else self._nn[i][1])
                
                # x bond: always within the same square cell.
                self._nn[i].append(i + 1 if _even else i - 1)
                self._nn_forward[i].append(-1 if not _even else self._nn[i][2])

        elif self.dim == 3:
            # 3D: Similar to 2D but with additional bonds in the z direction.
            for i in range(self.Ns):
                n       = i // 2
                r       = i % 2
                X       = n % self.Lx
                Y       = n // self.Lx
                Z       = n // (self.Lx * self.Ly)
                _even   = (r == 0)
                
                # z bond (in the xy plane): for even, Y - 1; odd, Y + 1.
                Yprime  = _bcfun(Y - 1, self.Ly, pbcy) if _even else _bcfun(Y + 1, self.Ly, pbcy)
                bond0   = (Yprime * self.Lx + X) * 2 + (1 if _even else 0)
                # y bond: for even, X - 1; odd, X + 1.
                Xprime  = _bcfun(X - 1, self.Lx, pbcx) if _even else _bcfun(X + 1, self.Lx, pbcx)
                bond1   = (Yprime * self.Lx + Xprime) * 2 + (1 if _even else 0)
                # x bond: within the same cell.
                bond2   = i + 1 if _even else i - 1
                # z top bond:
                Zprime_top = _bcfun(Z + 1, self.Lz, pbcz)
                bond3   = _bcfun(Zprime_top * self.Lx * self.Ly + Y * self.Lx + X, self.Ns, True)
                # z bottom bond:
                Zprime_bot = _bcfun(Z - 1, self.Lz, pbcz)
                bond4   = _bcfun(Zprime_bot * self.Lx * self.Ly + Y * self.Lx + X, self.Ns, True)
                self._nn[i] = [bond0, bond1, bond2, bond3, bond4]
                self._nn_forward[i] = [bond0, bond1, bond2, bond3, bond4]
        else:
            raise ValueError("Only dimensions 1, 2, and 3 are supported for nearest neighbor calculation.")
        
    def calculate_nnn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculates the next-nearest neighbors (NNN) of the honeycomb lattice.
        
        NNN are defined as the second-nearest (diagonal) neighbors within the same sublattice.
        """
        def _bcfun(_i, _L, _pbc):
            if _pbc:
                return mod_euc(_i, _L)
            return _i if 0 <= _i < _L else -1

        self._nnn = [[] for _ in range(self.Ns)]
        match self.dim:
            case 1:
                for i in range(self.Ns):
                    self._nnn[i] = [
                        _bcfun(i + 2, self.Lx, self.bc == LatticeBC.PBC),
                        _bcfun(i - 2, self.Lx, self.bc == LatticeBC.PBC)
                    ]
            case 2:
                for i in range(self.Ns):
                    x, y, z = self.get_coordinates(i)
                    even = (i % 2 == 0)
                    y1 = _bcfun(y - 2 if even else y + 2, self.Ly, self.bc == LatticeBC.PBC)
                    y2 = _bcfun(y, self.Ly, self.bc == LatticeBC.PBC)
                    x1 = _bcfun(x - 1, self.Lx, self.bc == LatticeBC.PBC)
                    x2 = _bcfun(x + 1, self.Lx, self.bc == LatticeBC.PBC)
                    self._nnn[i] = [
                        (y1 * self.Lx + x) * 2 + (0 if even else 1),
                        (y2 * self.Lx + x1) * 2 + (0 if even else 1),
                        (y2 * self.Lx + x2) * 2 + (0 if even else 1)
                    ]
            case 3:
                for i in range(self.Ns):
                    x, y, z = self.get_coordinates(i)
                    even = (i % 2 == 0)
                    y1 = _bcfun(y - 2 if even else y + 2, self.Ly, self.bc == LatticeBC.PBC)
                    y2 = _bcfun(y, self.Ly, self.bc == LatticeBC.PBC)
                    x1 = _bcfun(x - 1, self.Lx, self.bc == LatticeBC.PBC)
                    x2 = _bcfun(x + 1, self.Lx, self.bc == LatticeBC.PBC)
                    z1 = _bcfun(z + 1, self.Lz, self.bc == LatticeBC.PBC)
                    z2 = _bcfun(z - 1, self.Lz, self.bc == LatticeBC.PBC)
                    self._nnn[i] = [
                        (y1 * self.Lx + x) * 2 + (0 if even else 1),
                        (y2 * self.Lx + x1) * 2 + (0 if even else 1),
                        (y2 * self.Lx + x2) * 2 + (0 if even else 1),
                        z1 * self.Lx * self.Ly + y * self.Lx + x,
                        z2 * self.Lx * self.Ly + y * self.Lx + x
                    ]

    def calculate_norm_sym(self):
        """
        Calculates a symmetry normalization for each site.
        
        Here we simply use the Euclidean norm of the coordinate as a symmetry measure.
        In a more advanced implementation, this might account for sublattice or other symmetries.
        """
        self.norm_sym = {i: np.linalg.norm(self.rvectors[i]) for i in range(self.Ns)}

    ################################### SYMMETRY & INDEXING #######################################

    def site_index(self, x, y, z):
        """
        Convert (x, y, z) coordinates to a unique site index.
        """
        return z * (self.Lx * self.Ly) + y * self.Lx + x

    def get_sym_pos(self, x, y, z):
        """
        Returns the symmetry-transformed position.
        """
        return (x + self.Lx - 1, y + 2 * self.Ly - 1, z + self.Lz - 1)

    def get_sym_pos_inv(self, x, y, z):
        """
        Returns the inverse symmetry-transformed position.
        """
        return (x - (self.Lx - 1), y - (2 * self.Ly - 1), z - (self.Lz - 1))

    def symmetry_checker(self, x, y, z):
        """
        Placeholder for symmetry checking.
        """
        return True
    
    def get_bond_type(self, site1: int, site2: int):
        """
        Determines the bond type between two sites.
        
        Returns:
            int: 0 for x-bond, 1 for y-bond, 2 for z-bond, -1 if no bond exists.
        """
        if site2 in self._nn[site1]:
            idx = self._nn[site1].index(site2)
            if idx == 2:
                return X_BOND_NEI
            elif idx == 1:
                return Y_BOND_NEI
            elif idx == 0:
                return Z_BOND_NEI
        return -1

    ###############################################################################################
    #! Plaquettes
    ###############################################################################################
    
    def calculate_plaquettes(self):
        """
        Compute all hexagonal Kitaev plaquettes, matching the exact geometry
        and numbering convention in the hand-drawn lattice.

        Each plaquette is returned as 6 site indices in CCW order starting from
        the bottom-left A-sublattice site of the hexagon.

        Bond sequence:
            1 --Z--> 2 --X--> 3 --Y--> 4 --Z--> 5 --X--> 6 --Y--> 1
        """

        Z           = Z_BOND_NEI
        Y           = Y_BOND_NEI
        X           = X_BOND_NEI

        bond_cycle  = [Z, X, Y, Z, X, Y]

        plaquettes  = []
        seen        = set()

        for i in range(self._ns):

            # Only A sites (even indices) can be bottom-left of a plaquette
            if self.sublattice(i) != 1:
                continue

            # Must have a valid Z-neighbor above (this ensures bottom-left anchor)
            z1 = self.get_nn(i, Z)
            if z1 < 0:
                continue # boundary or missing hexagon here

            # Now walk the cycle
            loop    = [i]
            cur     = i
            valid   = True

            for b in bond_cycle:
                nxt = self.get_nn(cur, b)
                if self.wrong_nei(nxt):
                    valid = False
                    break
                loop.append(nxt)
                cur = nxt

            # Should return to starting site
            if not valid or loop[-1] != i:
                continue

            # Keep exactly the 6 unique sites
            hex_sites   = tuple(loop[:-1])

            # Deduplicate by sorted site set
            key         = tuple(sorted(hex_sites))
            if key not in seen:
                seen.add(key)
                plaquettes.append(list(hex_sites))

        self._plaquettes = plaquettes
        return plaquettes

# ---------------------------------------------------------------------------
#! EOF
# -------------------------------------------------------------------------