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

import  numpy   as np
from    typing  import Optional

try:
    from .                      import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType
    from ..maths.math_utils     import mod_euc
    from .tools.lattice_kspace  import HighSymmetryPoints
except ImportError:
    raise ImportError("Could not import Lattice base classes. Ensure the module is in the PYTHONPATH.")

################################### LATTICE IMPLEMENTATION #######################################

# X_BOND_NEI = 2
# Y_BOND_NEI = 1
# Z_BOND_NEI = 0
X_BOND_NEI = 0
Y_BOND_NEI = 1
Z_BOND_NEI = 2

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
        # self._a1        = np.array([np.sqrt(3) * self.a / 2.0, 3 * self.a / 2.0, 0])
        # self._a2        = np.array([-np.sqrt(3) * self.a / 2.0, 3 * self.a / 2.0, 0])
        # self._a3        = np.array([0, 0, self.c])
        self._a1        = np.array([3 * self.a / 2.0,  +np.sqrt(3) * self.a / 2.0, 0])
        self._a2        = np.array([0,  np.sqrt(3) * self.a, 0])
        self._a3        = np.array([0, 0, self.c])
        
        self._basis     = np.array([
                            [0.0, 0.0, 0.0],                            # A sublattice  - first node in the unit cell
                            [self.a / 2.0, np.sqrt(3)*self.a/2.0, 0.0]  # B sublattice  - second node in the unit cell
                        ])

        # self._delta_x   = np.array([0.0, self.a, 0.0])
        # self._delta_y   = np.array([-np.sqrt(3)*self.a/2.0, -self.a/2.0, 0.0])
        # self._delta_z   = np.array([ np.sqrt(3)*self.a/2.0, -self.a/2.0, 0.0])
        self._delta_x   = np.array([self.a / 2.0, -np.sqrt(3)*self.a/2.0, 0.0])
        self._delta_y   = np.array([self.a / 2.0,  np.sqrt(3)*self.a/2.0, 0.0])
        self._delta_z   = np.array([-self.a, 0.0, 0.0])
        
        self.init(**kwargs)
        
    def __str__(self):
        return f"HON,{self.bc},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}{self._flux_suffix}"

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

    def get_real_vec(self, x: int, y: int, z: int = 0):
        """
        Returns the real-space vector for a given (x, y, z) coordinate.
        """
        cell_x  = x                                     # cell x index
        # coordinates are stored as (x, 2*y + sublattice, z)
        cell_y  = y // 2                                # cell y index
        sub     = y % 2                                 # sublattice index (0 or 1)
        base    = cell_x * self._a1 + cell_y * self._a2 # base vector for the unit cell
        return base + self._basis[sub] + z * self._a3   # add z component

    def get_norm(self, x: int, y: int, z: int):
        """
        Returns the Euclidean norm of the real-space vector.
        """
        return np.linalg.norm(self.get_real_vec(x, y, z))

    def get_nn_direction(self, site, direction):
        """
        Returns the nearest neighbor in the specified direction.
        
        For the honeycomb lattice, we choose a mapping:
            LatticeDirection.X -> neighbor at index 0 of _nn[site]
            LatticeDirection.Y -> neighbor at index 1 of _nn[site]
            LatticeDirection.Z -> neighbor at index 2 of _nn[site]
        """
        mapping = { LatticeDirection.X: X_BOND_NEI, LatticeDirection.Y: Y_BOND_NEI, LatticeDirection.Z: Z_BOND_NEI }
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
                n           = i // 2        # n: site index on the square lattice.
                r           = i % 2         # r: sublattice index (0 for first node, 1 for second) - idx in the elementary cell.
                X           = n % self.Lx   # X: x coordinate on the corresponding square lattice.
                Y           = n // self.Lx  # Y: y coordinate on the corresponding square lattice.
                _even       = (r == 0)      # _even: whether the site is on the even sublattice.
                
                # Initialize bond indices
                self._nn[i]         = [-1, -1, -1]
                self._nn_forward[i] = [-1, -1, -1]
                
                # z bond: for even sites, we take z bond to be the one at X - 1 for even, X + 1 for odd. Same Y.
                XP          = _bcfun(X - 1, self.Lx, pbcx) if _even else _bcfun(X + 1, self.Lx, pbcx)
                YP          = Y
                if XP == -1:
                    self._nn[i][Z_BOND_NEI]     = -1
                else:
                    self._nn[i][Z_BOND_NEI]     = (YP * self.Lx + XP) * 2 + int(_even)          # changes sublattice
                self._nn_forward[i][Z_BOND_NEI] = (-1 if _even else self._nn[i][Z_BOND_NEI])    # z bond forward, we take only +1 direction for odd sites.
                
                # y bond: for even sites, it is in the same cell always but either we go to even->odd or odd->even in Y direction.
                self._nn[i][Y_BOND_NEI]         = (i + 1 if _even else i - 1)
                self._nn_forward[i][Y_BOND_NEI] = (self._nn[i][Y_BOND_NEI] if _even else -1)    # y bond forward, we take only +1 direction for even sites.
                
                # x bond: we need to go -1 for even sites, +1 for odd sites in Y direction.
                YP          = _bcfun(Y - 1, self.Ly, pbcy) if _even else _bcfun(Y + 1, self.Ly, pbcy)
                XP          = X
                if YP == -1:
                    self._nn[i][X_BOND_NEI]     = -1
                else:
                    self._nn[i][X_BOND_NEI]     = (YP * self.Lx + XP) * 2 + int(_even)          # changes sublattice
                self._nn_forward[i][X_BOND_NEI] = (self._nn[i][X_BOND_NEI] if _even else -1)    # x bond forward, we take only +1 direction for even sites.

        elif self.dim == 3:
            # 3D: Extend the 2D honeycomb logic to 3D, with clear sublattice and bond assignments. 
            for i in range(self.Ns):
                n       = i // 2
                r       = i % 2
                X       = n % self.Lx
                Y       = (n // self.Lx) % self.Ly
                Z       = n // (self.Lx * self.Ly)
                _even   = (r == 0)

                # Initialize bond indices
                self._nn[i] = [-1, -1, -1, -1]

                # z bond (in the xy plane): for even sites, X+1; for odd sites, X-1 (same Y, Z)
                XP = _bcfun(X + 1, self.Lx, pbcx) if _even else _bcfun(X - 1, self.Lx, pbcx)
                YP = Y
                ZP = Z
                if XP == -1:
                    self._nn[i][Z_BOND_NEI] = -1
                else:
                    self._nn[i][Z_BOND_NEI] = (ZP * self.Ly * self.Lx + YP * self.Lx + XP) * 2 + int(_even)
                self._nn_forward[i].append((-1 if _even else self._nn[i][Z_BOND_NEI]))

                # y bond: for even sites, it is in the same cell always but either we go to even->odd or odd->even in Y direction.
                self._nn[i][Y_BOND_NEI] = (i + 1 if _even else i - 1)

                # x bond: for even sites, Y-1; for odd sites, Y+1 (same X, Z)
                YP = _bcfun(Y - 1, self.Ly, pbcy) if _even else _bcfun(Y + 1, self.Ly, pbcy)
                XP = X
                ZP = Z
                if YP == -1:
                    self._nn[i][X_BOND_NEI] = -1
                else:
                    self._nn[i][X_BOND_NEI] = (ZP * self.Ly * self.Lx + YP * self.Lx + XP) * 2 + int(_even)
                self._nn_forward[i].append((self._nn[i][X_BOND_NEI] if _even else -1))
                
                # Go to next layer in Z direction
                ZP = _bcfun(Z + 1, self.Lz, pbcz)
                if ZP == -1:
                    self._nn[i][3] = -1
                else:
                    self._nn[i][3] = (ZP * self.Ly * self.Lx + Y * self.Lx + X) * 2 + int(_even)
                self._nn_forward[i].append((self._nn[i][3] if _even else -1))
                
        else:
            raise ValueError("Only dimensions 1, 2, and 3 are supported for nearest neighbor calculation.")
        
    def calculate_nnn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculates the next-nearest neighbors (NNN) of the honeycomb lattice.
        
        NNN are second-nearest neighbors, connecting sites on the *same* sublattice.
        For sublattice A (even sites), the three NNN directions are obtained by
        composing two consecutive NN hops:
            NNN_1: Y-bond then X-bond^{-1}  → cell shift  (0, -1)   [down in y]
            NNN_2: Z-bond then Y-bond^{-1}  → cell shift  (-1, 0)   [left in x]
            NNN_3: X-bond then Z-bond^{-1}  → cell shift  (+1, +1)  [diagonal]
        For sublattice B (odd sites), the shifts are inverted.
        """
        def _bcfun(_i, _L, _pbc):
            if _pbc:
                return mod_euc(_i, _L)
            return _i if 0 <= _i < _L else -1

        self._nnn           = [[] for _ in range(self.Ns)]
        self._nnn_forward   = [[] for _ in range(self.Ns)]
        
        match self.dim:
            case 1:
                for i in range(self.Ns):
                    self._nnn[i] = [
                        _bcfun(i + 2, self.Ns, pbcx),
                        _bcfun(i - 2, self.Ns, pbcx)
                    ]
                    self._nnn_forward[i] = [_bcfun(i + 2, self.Ns, pbcx)]
            case 2:
                # NNN shifts (dX, dY) for sublattice A (even sites)
                # These connect A→A via two NN hops
                nnn_shifts_A = [
                    ( 0, -1),   # down   : Y→X^{-1}
                    (-1,  0),   # left   : Z→Y^{-1}
                    ( 1,  1),   # diag   : X→Z^{-1}
                ]
                for i in range(self.Ns):
                    n       = i // 2
                    r       = i % 2
                    X       = n % self.Lx
                    Y       = n // self.Lx
                    _even   = (r == 0)
                    
                    nnn_list = []
                    for dX, dY in nnn_shifts_A:
                        if not _even:
                            dX, dY = -dX, -dY       # invert for B sublattice
                        Xn = _bcfun(X + dX, self.Lx, pbcx)
                        Yn = _bcfun(Y + dY, self.Ly, pbcy)
                        if Xn == -1 or Yn == -1:
                            nnn_list.append(-1)
                        else:
                            nnn_list.append((Yn * self.Lx + Xn) * 2 + r)
                    self._nnn[i] = nnn_list
                    # Forward: only the positive-direction hops (for even: all three, for odd: reversed)
                    self._nnn_forward[i] = [n for n in nnn_list if n >= 0] if _even else []

            case 3:
                nnn_shifts_A = [
                    ( 0, -1, 0),
                    (-1,  0, 0),
                    ( 1,  1, 0),
                ]
                for i in range(self.Ns):
                    n       = i // 2
                    r       = i % 2
                    X       = n % self.Lx
                    Y       = (n // self.Lx) % self.Ly
                    Z       = n // (self.Lx * self.Ly)
                    _even   = (r == 0)
                    
                    nnn_list = []
                    for dX, dY, dZ in nnn_shifts_A:
                        if not _even:
                            dX, dY, dZ = -dX, -dY, -dZ
                        Xn = _bcfun(X + dX, self.Lx, pbcx)
                        Yn = _bcfun(Y + dY, self.Ly, pbcy)
                        Zn = _bcfun(Z + dZ, self.Lz, pbcz)
                        if Xn == -1 or Yn == -1 or Zn == -1:
                            nnn_list.append(-1)
                        else:
                            nnn_list.append((Zn * self.Ly * self.Lx + Yn * self.Lx + Xn) * 2 + r)
                    # Add z-direction NNN (same sublattice, next layer)
                    Zp = _bcfun(Z + 1, self.Lz, pbcz)
                    Zm = _bcfun(Z - 1, self.Lz, pbcz)
                    nnn_list.append((Zp * self.Ly * self.Lx + Y * self.Lx + X) * 2 + r if Zp != -1 else -1)
                    nnn_list.append((Zm * self.Ly * self.Lx + Y * self.Lx + X) * 2 + r if Zm != -1 else -1)
                    self._nnn[i]         = nnn_list
                    self._nnn_forward[i] = [n for n in nnn_list if n >= 0] if _even else []

    def calculate_norm_sym(self):
        """Uses base implementation."""
        self._spatial_norm = { i: np.linalg.norm(self.rvectors[i]) for i in range(self.Ns) }

    ################################### SYMMETRY & INDEXING #######################################

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

    @staticmethod
    def dispersion(k, a=1.0):
        """
        Honeycomb (graphene-like) nearest-neighbour dispersion magnitude.
        Computes |f(k)| where f = sum_{δ} exp(-i k·δ) for the three A->B vectors
        used by this lattice implementation.
        """
        k   = np.asarray(k)
        s3  = np.sqrt(3.0)
        d1  = np.array([a/2.0, -s3 * a / 2.0])
        d2  = np.array([a/2.0,  s3 * a / 2.0])
        d3  = np.array([-a, 0.0])
        def _f(kx, ky):
            z1 = np.exp(-1j * (kx * d1[0] + ky * d1[1]))
            z2 = np.exp(-1j * (kx * d2[0] + ky * d2[1]))
            z3 = np.exp(-1j * (kx * d3[0] + ky * d3[1]))
            return np.abs(z1 + z2 + z3)
        if k.ndim == 1:
            kx, ky = k[0], k[1]
            return _f(kx, ky)
        else:
            kx = k[..., 0]
            ky = k[..., 1]
            return _f(kx, ky)

    def bond_type(self, s1: int, s2: int) -> int:
        """Return directional bond type (X_BOND_NEI, Y_BOND_NEI, Z_BOND_NEI) or -1."""
        if s2 == self._nn[s1][X_BOND_NEI]: return X_BOND_NEI
        if s2 == self._nn[s1][Y_BOND_NEI]: return Y_BOND_NEI
        if s2 == self._nn[s1][Z_BOND_NEI]: return Z_BOND_NEI
        return -1

    ###############################################################################################
    #! Plaquettes
    ###############################################################################################
    
    def calculate_plaquettes(self, open_bc: bool = True):
        X = X_BOND_NEI
        Y = Y_BOND_NEI
        Z = Z_BOND_NEI

        # For your NN convention, an A-anchored CCW walk is:
        bond_cycle = [Y, X, Z, Y, X, Z]

        plaquettes = []
        seen       = set()

        for i in range(self.Ns):

            # Anchor on A sites (r=0)
            if self.sublattice(i) != 0:
                continue

            loop  = [i]
            cur   = i
            cur_y = cur // (2 * self.Lx)    # y coordinate on square lattice
            cur_x = (cur // 2) % self.Lx    # x coordinate on square lattice
            valid = True

            for ib, b in enumerate(bond_cycle):
                nxt = self.get_nn(cur, b)
                if self.wrong_nei(nxt):
                    valid = False
                    break
                
                if open_bc:
                    # Hexagon can only have x+1 or y+1 steps, not x-1 or y-1
                    # but it can also be current
                    nxt_y = nxt // (2 * self.Lx)
                    nxt_x = (nxt // 2) % self.Lx
                    if (nxt_x not in (cur_x - 1, cur_x)) or (nxt_y not in (cur_y, cur_y + 1)):
                        valid = False
                        break
                
                loop.append(nxt)
                cur = nxt

            # loop has 7 entries, last must equal start
            if not valid or loop[-1] != i:
                continue

            hex_sites = tuple(loop[:-1])          # 6 unique sites in order
            key       = tuple(sorted(hex_sites))  # dedup independent of rotation

            if key not in seen:
                seen.add(key)
                plaquettes.append(list(hex_sites))

        self._plaquettes = plaquettes
        return plaquettes

# ---------------------------------------------------------------------------
#! EOF
# -------------------------------------------------------------------------