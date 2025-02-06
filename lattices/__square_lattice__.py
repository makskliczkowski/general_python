"""
Square Lattice Class...
@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""

from . import Lattice, LatticeBackend, LatticeBC, LatticeDirection, LatticeType

class SquareLattice(Lattice):
    """ 
    General Square Lattice type up to three dimensions!
    """
    
    def __init__(self, dim, lx, ly, lz, bc, *args, **kwargs):
        '''
        Initializer of the square lattice
        '''
        super().__init__(dim, lx, ly, lx, bc, *args, **kwargs)

        self._type      = LatticeType.SQUARE                            # Lattice type
        self.vectors    = LatticeBackend.array([[SquareLattice.a, 0, 0],
                                                [0, SquareLattice.b, 0],
                                                [0, 0, SquareLattice.c]])

        self._kvectors  = LatticeBackend.zeros((self.Ns, 3))
        self._rvectors  = LatticeBackend.zeros((self.Ns, 3))
        
        match(dim):
            case 1:
                self._lx = lx
                self._ly = 1
                self._lz = 1
                self._nn_forward    = [0]
                self._nnn_forward   = [0]
            case 2:
                self._lx = lx
                self._ly = ly
                self._lz = 1
                self._nn_forward    = [0, 1]
                self._nnn_forward   = [0, 1]
            case 3:
                self._lx = lx
                self._ly = ly
                self._lz = lz
                self._nn_forward    = [0, 1, 2]
                self._nnn_forward   = [0, 1, 2]
            case _:
                raise ValueError("Only 1D, 2D, and 3D lattices are supported.")
        self._ns        = self.lx * self.ly * self.lz                   # Total sites
        self._dim       = dim                                           # Dimension of the lattice
        
        # Compute lattice properties
        self.calculate_coordinates()
        self.calculate_r_vectors()
        self.calculate_k_vectors()

        if self.Ns < 100:
            self.calculate_dft_matrix()

        self.calculate_nn()
        self.calculate_nnn()
        self.calculate_norm_sym()
    
    # ---------------------------------------------------------------------------------
    
    def __str__(self):
        return f"SQ,{self.bc},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"

    def __repr__(self):
        return self.__str__()

    ################################### GETTERS #######################################

    def get_k_vec_idx(self, sym=False):
        """
        Returns the indices of kvectors, considering symmetry reduction.
        """
        all_momenta = []

        if self.dim == 1:
            all_momenta = [qx for qx in range(self.Lx)]
            return LatticeBackend.arange(self.Ns).tolist()
        
        elif self.dim == 2:
            all_momenta = [(qx, qy) for qx in range(self.Lx) for qy in range(self.Ly)]
            if sym:
                momenta = [(i, j) for i in range(self.Lx // 2 + 1) for j in range(i, self.Ly // 2 + 1)]
                return [i for i, mom in enumerate(all_momenta) if mom in momenta]
            else:
                return LatticeBackend.arange(self.Ns).tolist()

        elif self.dim == 3:
            all_momenta = [(qx, qy, qz) for qx in range(self.Lx) for qy in range(self.Ly) for qz in range(self.Lz)]
            return LatticeBackend.arange(self.Ns).tolist()

    ################################### CALCULATORS ###################################

    def calculate_coordinates(self):
        """
        Calculates the real lattice coordinates for each site.
        """
        self.coordinates = [
            (i % self.Lx, (i // self.Lx) % self.Ly, (i // (self.Ly * self.Lx)) % self.Lz) 
            for i in range(self.Ns)
        ]

    def calculate_k_vectors(self):
        """
        Calculates the inverse space (reciprocal lattice) vectors.
        """
        two_pi_over_lx = 2 * LatticeBackend.pi / SquareLattice.a / self.lx
        two_pi_over_ly = 2 * LatticeBackend.pi / SquareLattice.b / self.ly
        two_pi_over_lz = 2 * LatticeBackend.pi / SquareLattice.c / self.lz

        self.kvectors = LatticeBackend.array([
            [-LatticeBackend.pi + two_pi_over_lx * qx,
             -LatticeBackend.pi + two_pi_over_ly * qy,
             -LatticeBackend.pi + two_pi_over_lz * qz]
            for qx in range(self.lx) for qy in range(self.ly) for qz in range(self.lz)
        ])

    def calculate_r_vectors(self):
        """
        Calculates all possible real space vectors in the lattice.
        """
        self.rvectors = LatticeBackend.array([
            self.vectors[:, 0] * x + self.vectors[:, 1] * y + self.vectors[:, 2] * z
            for x in range(self.Lx) for y in range(self.Ly) for z in range(self.Lz)
        ])

    def calculate_norm_sym(self):
        """
        Calculates the normalization factors considering symmetric momenta.
        """
        try:
            self.sym_map = {}
            if self.dim == 2:
                for i in range(self.Lx // 2 + 1):
                    for j in range(i, self.Ly // 2 + 1):
                        self.sym_norm[(i, j)] = 0

                for y in range(self.Ly):
                    ky = y if y <= self.Ly // 2 else (-(y - self.Ly // 2)) % (self.Ly // 2)
                    for x in range(self.Lx):
                        kx = x if x <= self.Lx // 2 else (-(x - self.Lx // 2)) % (self.Lx // 2)
                        mom = (kx, ky)
                        if kx > ky:
                            mom = (ky, kx)
                        self.sym_norm[mom]      += 1
                        self.sym_map[(x, y)]    = mom
        except Exception as e:
            print(f"Error in calculate_norm_sym: {e}")
            pass

    ################################### NEIGHBORS ###################################

    # ---------------------------------------------------------------------------------
    
    def site_index(self, x, y, z):
        """
        Convert (x, y, z) coordinates to a site index.
        Args:
            x (int): x-coordinate
            y (int): y-coordinate
            z (int): z-coordinate
        """
        return z * self.Lx * self.Ly + y * self.Lx + x
    
    # ---------------------------------------------------------------------------------
    
    def get_real_vec(self, x: int, y: int, z: int):
        """
        Returns the real vector for a given (x, y, z) coordinate.
        """
        return self.vectors[:, 0] * x + self.vectors[:, 1] * y + self.vectors[:, 2] * z

    # ---------------------------------------------------------------------------------

    def get_norm(self, x: int, y: int, z: int):
        """
        Returns the Euclidean norm of a real-space vector.
        """
        return LatticeBackend.sqrt(x**2 + y**2 + z**2)

    # ---------------------------------------------------------------------------------
    
    def get_nn_direction(self, site: int, direction: LatticeDirection):
        """
        Returns nearest neighbors in a given direction (X, Y, Z).
        Args:
            site (int): Site index
            direction (LatticeDirection): Direction to get the nearest neighbors
        """
        switcher = {
            LatticeDirection.X: self._nn[site][0],
            LatticeDirection.Y: self._nn[site][1] if self.dim >= 2 else self._nn[site][0],
            LatticeDirection.Z: self._nn[site][2] if self.dim == 3 else self._nn[site][0]
        }
        return switcher.get(direction, self._nn[site][0])
    
    # ---------------------------------------------------------------------------------
    
    def get_nn_forward(self, site: int, num: int = -1):
        """ Returns the forward nearest neighbors of a given site. """
        if num < 0:
            return self._nn_forward[site]
        return self._nn_forward[site][num] if num < len(self._nn_forward[site]) else None

    def get_nnn_forward(self, site, num : int = -1):
        ''' Returns the forward next-nearest neighbors of a given site '''
        if num < 0:
            return self._nnn_forward[site]
        return self._nnn_forward[site][num] if num < len(self._nnn_forward[site]) else None

    # ---------------------------------------------------------------------------------
    
    # Calculate the nearest neighbors
    
    # ---------------------------------------------------------------------------------
    
    def calculate_nn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculates the nearest neighbors (NN) for 1D, 2D, and 3D square lattices.
        Also calculates the forward nearest neighbors (NNF).
        
        Args:
            - pbcx: Periodic boundary condition in x direction
            - pbcy: Periodic boundary condition in y direction
            - pbcz: Periodic boundary condition in z direction
        """

        def boundary_check(index, limit, pbc):
            """
            Helper function to handle periodic and open boundary conditions.
            Args:
                - index: Index to check
                - limit: Maximum index value
                - pbc: Periodic boundary condition flag
            Returns:
                - int: Index after applying boundary conditions (PBC or OBC)            
            """
            if pbc:
                return index % limit  # Apply periodic boundary condition
            return index if 0 <= index < limit else -1  # Apply open boundary condition

        self._nn            = [[] for _ in range(self.Ns)]
        self._nn_forward    = [[] for _ in range(self.Ns)]

        if self.dim == 1:
            # 1D Lattice: Each site has 2 neighbors (left and right)
            for i in range(self.Ns):
                right       = boundary_check(i + 1, self.Lx, pbcx)
                left        = boundary_check(i - 1, self.Lx, pbcx)

                self._nn[i]         = [right, left]
                self._nn_forward[i] = [right]       # Forward neighbor
        elif self.dim == 2:
            # 2D Lattice: Each site has 4 neighbors (right, top, left, bottom)
            for i in range(self.Ns):
                x, y, _ = self.get_coordinates(i)

                right   = boundary_check(i + 1, self.Lx, pbcx)
                top     = boundary_check(i + self.Lx, self.Ns, pbcy)
                left    = boundary_check(i - 1, self.Lx, pbcx)
                bottom  = boundary_check(i - self.Lx, self.Ns, pbcy)

                self._nn[i]         = [right, top, left, bottom]
                self._nn_forward[i] = [right, top] # Only forward in x and y
        elif self.dim == 3:
            # 3D Lattice: Each site has 6 neighbors (right, top, up, left, bottom, down)
            for i in range(self.Ns):
                x, y, z = self.get_coordinates(i)

                right   = boundary_check(i + 1, self.Lx, pbcx)
                top     = boundary_check(i + self.Lx, self.Ns, pbcy)
                up      = boundary_check(i + self.Lx * self.Ly, self.Ns, pbcz)
                left    = boundary_check(i - 1, self.Lx, pbcx)
                bottom  = boundary_check(i - self.Lx, self.Ns, pbcy)
                down    = boundary_check(i - self.Lx * self.Ly, self.Ns, pbcz)

                self._nn[i]         = [right, top, up, left, bottom, down]
                self._nn_forward[i] = [right, top, up]  # Forward in x, y, z only

        else:
            raise ValueError("Only 1D, 2D, and 3D lattices are supported.")

    # ---------------------------------------------------------------------------------
    
    # Calculate the next-nearest neighbors
    
    # ---------------------------------------------------------------------------------
    
    def calculate_nnn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        """
        Calculates the next-nearest neighbors (NNN) for 1D, 2D, and 3D square lattices.
        Also calculates the forward next-nearest neighbors (NNNF).
        
        Args:
            - pbcx: Periodic boundary condition in x direction
            - pbcy: Periodic boundary condition in y direction
            - pbcz: Periodic boundary condition in z direction
        """

        def boundary_check(index, limit, pbc):
            """
            Helper function to handle periodic and open boundary conditions.
            Args:
                - index: Index to check
                - limit: Maximum index value
                - pbc: Periodic boundary condition flag
            Returns:
                - int: Index after applying boundary conditions (PBC or OBC)            
            """
            if pbc:
                return index % limit
            return index if 0 <= index < limit else -1

        self._nnn           = [[] for _ in range(self.Ns)]
        self._nnn_forward   = [[] for _ in range(self.Ns)]
        
        if self.dim == 1:
            # 1D Lattice: Each site has 2 next-nearest neighbors
            for i in range(self.Ns):
                right   = boundary_check(i + 2, self.Lx, pbcx)
                left    = boundary_check(i - 2, self.Lx, pbcx)

                self._nnn[i]            = [right, left]
                self._nnn_forward[i]    = [right]
        elif self.dim == 2:
            # 2D Lattice: Each site has 4 next-nearest neighbors
            for i in range(self.Ns):
                x, y, _ = self.get_coordinates(i)

                right   = boundary_check(i + 2, self.Lx, pbcx)
                top     = boundary_check(i + 2 * self.Lx, self.Ns, pbcy)
                left    = boundary_check(i - 2, self.Lx, pbcx)
                bottom  = boundary_check(i - 2 * self.Lx, self.Ns, pbcy)

                self._nnn[i]            = [right, top, left, bottom]
                self._nnn_forward[i]    = [right, top]
        elif self.dim == 3:
            # 3D Lattice: Each site has 6 next-nearest neighbors
            for i in range(self.Ns):
                x, y, z = self.get_coordinates(i)

                right   = boundary_check(i + 2, self.Lx, pbcx)
                top     = boundary_check(i + 2 * self.Lx, self.Ns, pbcy)
                up      = boundary_check(i + 2 * self.lxly, self.Ns, pbcz)
                left    = boundary_check(i - 2, self.Lx, pbcx)
                bottom  = boundary_check(i - 2 * self.Lx, self.Ns, pbcy)
                down    = boundary_check(i - 2 * self.lxly, self.Ns, pbcz)

                self._nnn[i] = [right, top, up, left, bottom, down]
                self._nnn_forward[i] = [right, top, up]
        else:
            raise ValueError("Only 1D, 2D, and 3D lattices are supported.")
        
    # ---------------------------------------------------------------------------------