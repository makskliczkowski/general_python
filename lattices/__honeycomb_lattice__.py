import numpy as np
from . import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType

class HoneycombLattice(Lattice):
    """
    Implementation of the Honeycomb Lattice.
    
    The honeycomb lattice is a 2D lattice with a hexagonal structure. The lattice consists of
    two sublattices, A and B, arranged in a hexagonal pattern. The nearest neighbors and
    next-nearest neighbors are calculated based on a hexagonal unit cell.

    Reference:
    - Phys. Rev. Research 3, 013160 (2021)
    - Fig. 2 of https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.013160
    
    Attributes:
        - Lx, Ly, Lz: Number of lattice sites in x, y, and z directions
        - bc: Boundary condition (PBC, OBC, etc.)
        - a, c: Lattice parameters
        - vectors: Lattice basis vectors
        - kvectors: Reciprocal lattice vectors
        - rvectors: Real-space vectors
    """

    def __init__(self, dim, lx, ly, lz, bc, *args, **kwargs):
        """
        Initialize a honeycomb lattice.
        
        Args:
            - dim: Lattice dimension (1D, 2D, 3D)
            - lx, ly, lz: Lattice sizes in x, y, z
            - bc: Boundary conditions (PBC, OBC, etc.)
        """
        super().__init__(dim, lx, ly, lz, bc, *args, **kwargs)

        self._type = LatticeType.HONEYCOMB  # Lattice type

        # Define primitive lattice vectors
        self.vectors = LatticeBackend.array([
            [1, 0, 0],  # a1
            [0.5, np.sqrt(3) / 2, 0],  # a2
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
        self.Ns = 2 * self.Lx * self.Ly * self.Lz

        # Compute lattice properties
        self.calculate_coordinates()
        self.calculate_r_vectors()
        self.calculate_k_vectors()

        if self.Ns < 100:
            self.calculate_dft_matrix()

        self.calculate_nn()
        self.calculate_nnn()
        self.calculate_norm_sym()

    def __str__(self):
        return f"HONEY,{self.bc},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"

    def __repr__(self):
        return self.__str__()

    ################################### GETTERS #######################################

    def get_real_vec(self, x: int, y: int, z: int):
        """
        Returns the real-space vector for a given (x, y, z) coordinate.
        """
        return x * self.vectors[0] + y * self.vectors[1] + z * self.vectors[2]

    def get_norm(self, x: int, y: int, z: int):
        """
        Returns the Euclidean norm of a real-space vector.
        """
        return np.sqrt(x**2 + y**2 + z**2)

    ################################### COORDINATE SYSTEM #######################################

    def calculate_coordinates(self):
        """
        Calculates the real-space coordinates for each lattice site.
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
        self.rvectors = np.array([
            self.get_real_vec(x, y, z)
            for x in range(self.Lx) for y in range(self.Ly) for z in range(self.Lz)
        ])

    ################################### NEIGHBORHOOD #######################################

    def calculate_nn(self):
        """
        Calculates the nearest neighbors of the honeycomb lattice.
        """
        self._nn = [[] for _ in range(self.Ns)]

        for i in range(self.Ns):
            x, y, z = self.get_coordinates(i)

            # Z-bond (always valid)
            z_bond = (i + 1) if (i % 2 == 0) else (i - 1)

            # X and Y bonds (depends on even/odd)
            if y % 2 == 0:
                x_bond = self.site_index((x - 1) % self.Lx, y, z)
                y_bond = self.site_index(x, (y + 1) % self.Ly, z)
            else:
                x_bond = self.site_index(x, y, z)
                y_bond = self.site_index((x + 1) % self.Lx, (y - 1) % self.Ly, z)

            self._nn[i] = [z_bond, x_bond, y_bond]

    def calculate_nnn(self):
        """
        Calculates the next-nearest neighbors for the honeycomb lattice.
        """
        self._nnn = [[] for _ in range(self.Ns)]

        for i in range(self.Ns):
            x, y, z = self.get_coordinates(i)

            # Second-nearest neighbors (diagonals)
            nnn_x = self.site_index((x + 1) % self.Lx, y, z)
            nnn_y = self.site_index(x, (y + 2) % self.Ly, z)

            self._nnn[i] = [nnn_x, nnn_y]

    def site_index(self, x, y, z):
        """
        Convert (x, y, z) coordinates to a site index.
        """
        return z * (self.Lx * self.Ly) + y * self.Lx + x

    ################################### SYMMETRIES #######################################

    def symmetry_checker(self, x, y, z):
        """
        Placeholder for symmetry calculations.
        """
        return True
