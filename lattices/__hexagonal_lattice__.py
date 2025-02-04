from .__lattice__ import Lattice, LAT_BC
from .__square_lattice__ import SquareLattice
import numpy as np

class HexagonalLattice(Lattice):
    """ 
    General Hexagonal Lattice type up to three dimensions!
    """
    
    # Lattice constants
    a = 1.
    b = 1.
    c = 1.
    def __init__(self, dim, Lx, Ly, Lz, _BC, *args, **kwargs):
        '''
        Initializer of the square lattice
        '''
        super().__init__(dim, Lx, Ly, Lz, _BC, *args, **kwargs)
        self.Ns         = Lx * Ly * Lz * (2 if dim > 1 else 1)
        self.vectors    = np.array([[SquareLattice.a,0,0],[0,SquareLattice.b,0],[0,0,SquareLattice.c]])
        self.kvectors   = np.zeros((self.Ns, 3))
        self.rvectors   = np.zeros((self.Ns, 3))
        #self.kvectors = np.zeros((self.Lx, self.Ly))
        self.name       = "Hexagonal"
        
    
    def __str__(self):
        '''
        Set the name
        '''
        return f"{self.name},{LAT_BC[self._BC]},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"
    
    def __repr__(self):
        '''
        Set the representation of the lattice
        '''
        return f"HEX,{'PBC' if self._BC == 0 else 'OBC'},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"

    def get_Ns(self):
        return self.Ns
    
    ################################### GETTERS #######################################
    
    def get_k_vec_idx(self, sym = False):
        '''
        Returns the indices of kvectors if symmetries are to be concerned
        '''
        all_momenta = []
        
        # two dimensions
        if self.dim == 2:   
            for qx in range(self.Lx):
                for qy in range(self.Ly):
                    all_momenta.append((qx,qy))

            # calculate the triangle symmetry
            if sym:
                momenta = []
                # calculate symmetric momenta
                for i in range(self.Lx//2 + 1):
                    for j in range(i, self.Ly//2 + 1):
                        momenta.append((i,j))
                return [i for i in np.arange(self.Ns) if all_momenta[i] in momenta]
            else:
                return [i for i in np.arange(self.Ns)]
        # three dimensions
        elif self.dim == 3:
            for qx in range(self.Lx):
                for qy in range(self.Ly):
                    for qz in range(self.Lz):
                        all_momenta.append((qx,qy,qz))
                        
            return [i for i in np.arange(self.Ns)]

    
    ################################### CALCULATORS ###################################

    def calculate_coordinates(self):
        '''
        Calculates the real lattice coordinates based on a lattice site
        '''
        self.coordinates = []
        for i in range(self.Ns):
            x = i % self.Lx
            y = (i // self.Lx) % self.Ly
            z = (i // (self.Ly * self.Lx)) % self.Lz
            self.coordinates.append((x,y,z))
        
    
    '''
    Calculates the inverse space vectors
    '''
    def calculate_k_vectors(self):
        two_pi_over_Lx = 2 * np.pi / SquareLattice.a / self.Lx;
        two_pi_over_Ly = 2 * np.pi / SquareLattice.b / self.Ly;
        two_pi_over_Lz = 2 * np.pi / SquareLattice.c / self.Lz;
    
        for qx in range(self.Lx):
            # kx = -np.pi + two_pi_over_Lx * qx
            kx = two_pi_over_Lx * qx
            for qy in range(self.Ly):
                # ky = -np.pi + two_pi_over_Ly * qy
                ky = two_pi_over_Ly * qy
                for qz in range(self.Lz):
                    # kz = -np.pi + two_pi_over_Lz * qz
                    kz = two_pi_over_Lz * qz
                    iter = qz * self.Lx * self.Ly + qy * self.Ly + qx
                    self.kvectors[iter,:] = np.array([kx, ky, kz])
                    
    '''
    Calculates all possible real space vectors on a lattice
    '''                
    def calculate_r_vectors(self):
        for x in range(self.Lx):
            for y in range(self.Ly):
                for z in range(self.Lz):
                    iter                = self.Lx * self.Ly * z + self.Lx * y + x
                    self.rvectors[iter] = self.vectors[:, 0] * x + self.vectors[:, 1] * y + self.vectors[:, 2] * z
    
    '''
    Calculates the symmetric normalization for different momenta considered
    '''
    def calculate_norm_sym(self):
        pass