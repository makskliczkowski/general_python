import numpy as np
from . import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType

import numpy as np
from . import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType

import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from maths import MathMod

################################### LATTICE IMPLEMENTATION #######################################

class HoneycombLattice(Lattice):
    """
    Implementation of the Honeycomb Lattice.
    
    The honeycomb lattice is a 2D lattice with a hexagonal structure. The lattice consists of
    two sublattices (A and B) arranged in a hexagonal pattern. Nearest and next-nearest neighbors
    are computed based on a hexagonal unit cell.
    
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

        # Adjust lattice properties based on dimension
        if dim == 1:
            self._ly = 1
            self._lz = 1
        elif dim == 2:
            self._lz = 1
        else:
            self._ly = ly
            self._lz = lz

        # For the honeycomb lattice there are two sites per unit cell.
        self._ns = 2 * self.Lx * self.Ly * self.Lz

        # Initialize the primitive vectors
        self._a1 = np.array([np.sqrt(3) * self.a / 2.0, 3 * self.a / 2.0, 0])
        self._a2 = np.array([-np.sqrt(3) * self.a / 2.0, 3 * self.a / 2.0, 0])
        self._a3 = np.array([0, 0, self.c])
        self._basis = np.array([
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, self.a, 0.0]),
        ])

        # Compute lattice properties.
        self.calculate_coordinates()
        self.calculate_r_vectors()
        self.calculate_k_vectors()

        if self._ns < 100:
            self.calculate_dft_matrix()

        # Calculate neighbors using internal methods.
        self.calculate_nn()
        self.calculate_nnn()
        self.calculate_norm_sym()
        
        # Initialize the k vectors
        self._k1 = np.array([2 * np.pi / self.a, 0, 0])
        self._k2 = np.array([-np.pi / self.a, np.sqrt(3) * np.pi / self.a, 0])
        self._k3 = np.array([-np.pi / self.a, -np.sqrt(3) * np.pi / self.a, 0])
        

    def __str__(self):
        return f"HON,{self.bc},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"

    def __repr__(self):
        return self.__str__()

    ################################### GETTERS #######################################

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

    ################################### COORDINATE SYSTEM #######################################

    def calculate_coordinates(self):
        """
        Calculates the coordinates for each lattice site.
        
        Here we use the common honeycomb convention:
            For site index i, we use:
              x = (i // 2) mod Lx,
              y = ((i // 2) // Lx) mod Ly, then mapped to (2*y + (i mod 2))
              z = ((i // 2) // (Lx * Ly)) mod Lz.
        """
        self.coordinates = []
        for i in range(self.Ns):
            x = (i // 2) % self.Lx
            y = ((i // 2) // self.Lx) % self.Ly
            z = ((i // 2) // (self.Lx * self.Ly)) % self.Lz
            self.coordinates.append((x, 2 * y + (i % 2), z))

    def calculate_k_vectors(self):
        """
        Calculates the reciprocal lattice vectors.
        """
        two_pi_over_Lx = 2 * np.pi / self.Lx
        two_pi_over_Ly = 2 * np.pi / self.Ly
        two_pi_over_Lz = 2 * np.pi / self.Lz

        self.kvectors = np.array([
            [-np.pi + two_pi_over_Lx * qx,
             -np.pi + two_pi_over_Ly * qy,
             -np.pi + two_pi_over_Lz * qz]
            for qx in range(self.Lx) for qy in range(self.Ly) for qz in range(self.Lz)
        ])

    def calculate_r_vectors(self):
        """
        Calculates the real-space vectors for each site.
        """
        rv = np.zeros((self.Ns, 3))
        idx = 0
        for z in range(self.Lz):
            for y in range(self.Ly):
                for x in range(self.Lx):
                    cell_offset = x * self._a1 + y * self._a2 + z * self._a3
                    for sub in range(2):
                        rv[idx] = cell_offset + self._basis[sub]
                        idx += 1
        self.rvectors = rv

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
                return MathMod.mod_euc(_i, _l)
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
                return MathMod.mod_euc(_i, _L)
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
