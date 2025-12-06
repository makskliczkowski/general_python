import numpy as np
from typing import Optional
from . import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType
from .tools.lattice_kspace import HighSymmetryPoints

class HexagonalLattice(Lattice):
    """ 
    General Hexagonal Lattice type up to three dimensions.
    
    High-symmetry points in the Brillouin zone (2D):
    - Γ (Gamma): Zone center (0, 0)
    - K: Corner of hexagonal BZ (2/3, 1/3)
    - K': Inequivalent corner (1/3, 2/3)
    - M: Edge midpoint (1/2, 0)
    
    Default path: Γ → K → M → Γ
    """

    def __init__(self, dim, lx, ly, lz, bc, *args, **kwargs):
        """
        Initialize a hexagonal lattice.
        
        Args:
            - dim: Lattice dimension (1D, 2D, 3D)
            - lx, ly, lz: Lattice sizes in x, y, z
            - bc: Boundary conditions (PBC, OBC, etc.)
        """
        super().__init__(dim, lx, ly, lz, bc, *args, **kwargs)

        self._type = LatticeType.HEXAGONAL  # Lattice type

        # Define primitive lattice vectors
        self.vectors = LatticeBackend.array([
            [LatticeBackend.sqrt(3) / 2, 3 / 2, 0],  # a1
            [-LatticeBackend.sqrt(3) / 2, 3 / 2, 0],  # a2
            [0, 0, 1]  # a3 (for 3D)
        ])

        # Initialize coordinate storage
        self.kvectors = LatticeBackend.zeros((self.Ns, 3))
        self.rvectors = LatticeBackend.zeros((self.Ns, 3))

        # Adjust lattice properties based on dimension
        match dim:
            case 1:
                self.Ly = 1
                self.Lz = 1
            case 2:
                self.Lz = 1

        # Two atoms per elementary cell
        self._ns = 2 * self.Lx * self.Ly * self.Lz

        # Compute lattice properties
        self.calculate_coordinates()
        self.calculate_r_vectors()
        self.calculate_k_vectors()

        if self.Ns < 100:
            self.calculate_dft_matrix()

        self.calculate_nn()
        self.calculate_nnn()
        self.calculate_norm_sym()

    # -------------------------------------------------------------------------------------------

    def __str__(self):
        return f"HEX,{self.bc},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"

    def __repr__(self):
        return self.__str__()

    # -------------------------------------------------------------------------------------------
    #! High-symmetry points
    # -------------------------------------------------------------------------------------------
    
    def high_symmetry_points(self) -> Optional[HighSymmetryPoints]:
        """
        Return high-symmetry points for the hexagonal lattice.
        
        Returns
        -------
        HighSymmetryPoints
            High-symmetry points for the hexagonal Brillouin zone.
            Uses same points as honeycomb (hexagonal BZ structure).
        """
        return HighSymmetryPoints.hexagonal_2d()

    ######################################### GETTERS ###########################################

    def get_real_vec(self, x: int, y: int, z: int):
        """
        Returns the real vector for a given (x, y, z) coordinate.
        """
        y_floor = LatticeBackend.floor(y / 2)
        y_move = LatticeBackend.floor(y_floor / 2)
        tmp = y_move * (self.vectors[0] + self.vectors[1]) + (z * self.vectors[2])
        tmp += (y % 2) * self.vectors[0]
        return tmp + x * (self.vectors[0] - self.vectors[1])

    def get_norm(self, x: int, y: int, z: int):
        """
        Returns the Euclidean norm of a real-space vector.
        """
        return LatticeBackend.sqrt(x**2 + y**2 + z**2)

    # -------------------------------------------------------------------------------------------
    
    def get_nn_direction(self, site, direction):
        return super().get_nn_direction(site, direction)
    
    def get_nn_forward(self, site, num):
        return super().get_nn_forward(site, num)
    
    def get_nnn_forward(self, site, num):
        return super().get_nnn_forward(site, num)
    
    ################################### COORDINATE SYSTEM #######################################

    def calculate_coordinates(self):
        """
        Calculates the real lattice coordinates for each site.
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
        two_pi_over_Lx = 2 * LatticeBackend.pi / self.Lx
        two_pi_over_Ly = 2 * LatticeBackend.pi / self.Ly
        two_pi_over_Lz = 2 * LatticeBackend.pi / self.Lz

        b1 = LatticeBackend.array([1. / LatticeBackend.sqrt(3), 1. / 3., 0])
        b2 = LatticeBackend.array([-1. / LatticeBackend.sqrt(3), 1. / 3., 0])
        b3 = LatticeBackend.array([0, 0, 1])

        self.kvectors = LatticeBackend.array([
            two_pi_over_Lx * qx * b1 +
            two_pi_over_Ly * qy * b2 +
            two_pi_over_Lz * qz * b3
            for qx in range(self.Lx) for qy in range(self.Ly) for qz in range(self.Lz)
        ])

    def calculate_r_vectors(self):
        """
        Calculates the real-space vectors for each site.
        """
        self.rvectors = LatticeBackend.array([
            self.get_real_vec(x, y, z)
            for x in range(self.Lx) for y in range(self.Ly) for z in range(self.Lz)
        ])

    def calculate_norm_sym(self):
        """
        Calculate the symmetry of the lattice.
        """
        self.norm_sym = [self.get_norm(*coord) for coord in self.coordinates]
    
    #############################################################################################
    
    # Calculators
    
    #############################################################################################
    
    def calculate_nn_in(self):
        """
        Calculates the nearest neighbors of the hexagonal lattice.
        """
        def _bcfun(_i, _L, _pbc):
            if _pbc:
                return _i % _L
            else:
                return -1 if _i >= _L or _i < 0 else _i

        self._nn = [[] for _ in range(self.Ns)]

        match self.dim:
            case 1:
                for i in range(self.Ns):
                    self._nn[i] = [
                        _bcfun(i + 1, self.Lx, self.bc == LatticeBC.PBC),
                        _bcfun(i - 1, self.Lx, self.bc == LatticeBC.PBC)
                    ]
            case 2:
                for i in range(self.Ns):
                    x, y, z = self.get_coordinates(i)
                    even = (i % 2 == 0)

                    y_prime = _bcfun(y - 1 if even else y + 1, self.Ly, self.bc == LatticeBC.PBC)
                    x_prime = _bcfun(x - 1 if even else x + 1, self.Lx, self.bc == LatticeBC.PBC)

                    self._nn[i] = [
                        (y_prime * self.Lx + x) * 2 + (0 if even else 1),
                        (y * self.Lx + x_prime) * 2 + (0 if even else 1),
                        i + 1 if even else i - 1
                    ]
            case 3:
                for i in range(self.Ns):
                    x, y, z = self.get_coordinates(i)
                    even = (i % 2 == 0)

                    y_prime = _bcfun(y - 1 if even else y + 1, self.Ly, self.bc == LatticeBC.PBC)
                    x_prime = _bcfun(x - 1 if even else x + 1, self.Lx, self.bc == LatticeBC.PBC)
                    z_prime_top = _bcfun(z + 1, self.Lz, self.bc == LatticeBC.PBC)
                    z_prime_bottom = _bcfun(z - 1, self.Lz, self.bc == LatticeBC.PBC)

                    self._nn[i] = [
                        (y_prime * self.Lx + x) * 2 + (0 if even else 1),
                        (y * self.Lx + x_prime) * 2 + (0 if even else 1),
                        i + 1 if even else i - 1,
                        z_prime_top * self.Lx * self.Ly + y * self.Lx + x,
                        z_prime_bottom * self.Lx * self.Ly + y * self.Lx + x
                    ]

    def calculate_nnn_in(self):
        """
        Calculates the next-nearest neighbors (NNN) of the hexagonal lattice.
        
        NNN are second-nearest neighbors in a honeycomb structure.
        These are diagonal connections within the same sublattice.
        """

        def _bcfun(_i, _L, _pbc):
            """
            Helper function to apply periodic (PBC) or open (OBC) boundary conditions.
            
            Args:
                - _i (int): Index to be transformed
                - _L (int): Lattice size along a dimension
                - _pbc (bool): Flag for periodic boundary conditions
            Returns:
                - (int): New index after applying boundary conditions
            """
            if _pbc:
                return _i % _L
            else:
                return -1 if _i >= _L or _i < 0 else _i

        self._nnn = [[] for _ in range(self.Ns)]

        match self.dim:
            case 1:
                # In 1D, NNN are simply two steps away in both directions
                for i in range(self.Ns):
                    self._nnn[i] = [
                        _bcfun(i + 2, self.Lx, self.bc == LatticeBC.PBC),
                        _bcfun(i - 2, self.Lx, self.bc == LatticeBC.PBC)
                    ]

            case 2:
                # 2D Hexagonal lattice - Next-nearest neighbors (diagonal within sublattice)
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
                # 3D Hexagonal Lattice: Adds NNN along z-direction
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

    # -------------------------------------------------------------------------------------------
    
    def site_index(self, x, y, z):
        """
        Convert (x, y, z) coordinates to a site index.
        """
        return z * (self.Lx * self.Ly) + y * self.Lx + x

    # -------------------------------------------------------------------------------------------

    def get_sym_pos(self, x, y, z):
        """
        Returns symmetry position.
        """
        return (x + self.Lx - 1, y + 2 * self.Ly - 1, z + self.Lz - 1)

    def get_sym_pos_inv(self, x, y, z):
        """
        Returns inverse symmetry position.
        """
        return (x - (self.Lx - 1), y - (2 * self.Ly - 1), z - (self.Lz - 1))

    def symmetry_checker(self, x, y, z):
        """
        Always returns True (placeholder for future symmetry calculations).
        """
        return True
    
    ######################################### PLOTTING #########################################