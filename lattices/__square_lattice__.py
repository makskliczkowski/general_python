"""
Square Lattice Class...
@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""


from .__lattice__ import Lattice, LAT_BC
import numpy as np

class SquareLattice(Lattice):
    """ 
    General Square Lattice type up to three dimensions!
    """
    
    # Lattice constants
Hexa
    def __init__(self, dim, Lx, Ly, Lz, _BC, *args, **kwargs):
        '''
        Initializer of the square lattice
        '''
        super().__init__(dim, Lx, Ly, Lz, _BC, *args, **kwargs)

        self.Ns         = Lx * Ly * Lz
        self.vectors    = np.array([[SquareLattice.a,0,0],[0,SquareLattice.b,0],[0,0,SquareLattice.c]])
        self.kvectors   = np.zeros((self.Ns, 3))
        self.rvectors   = np.zeros((self.Ns, 3))
        #self.kvectors = np.zeros((self.Lx, self.Ly))
        
        self.calculate_coordinates()
        self.calculate_r_vectors()
        self.calculate_k_vectors()
        if self.Ns < 100:
            self.calculate_dft_matrix()
        self.calculate_norm_sym()
    
    '''
    Set the name
    '''
    def __str__(self):
        return f"SQ,{'PBC' if self._BC == 0 else 'OBC'},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"
    
    '''
    Set the representation of the lattice
    '''
    def __repr__(self):
        return f"SQ,{'PBC' if self._BC == 0 else 'OBC'},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"

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
            kx = -np.pi + two_pi_over_Lx * qx
            # kx = two_pi_over_Lx * qx
            for qy in range(self.Ly):
                ky = -np.pi + two_pi_over_Ly * qy
                # ky = two_pi_over_Ly * qy
                for qz in range(self.Lz):
                    kz = -np.pi + two_pi_over_Lz * qz
                    # kz = two_pi_over_Lz * qz
                    iters = qz * self.Lx * self.Ly + qy * self.Lx + qx
                    self.kvectors[iters,:] = np.array([kx, ky, kz])
                    
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
        # set the norm dictionary
        self.sym_norm   = super().calculate_norm_sym()
        self.sym_map    = {}
        try:
            if self.dim == 2:
                # iterate to set all possible momenta norm to 0
                for i in range(self.Lx//2 + 1):
                    for j in range(i, self.Ly//2 + 1):
                        mom                 = (i, j)
                        self.sym_norm[mom]  = 0
                
                # go through all momenta
                for y in range(self.Ly):
                    ky      = y
                    if y > int(self.Ly/2): ky = (-(ky - (self.Ly//2))) % (self.Ly//2)
                    for x in range(self.Lx):
                        kx  = x
                        # change from -
                        if x > int(self.Lx/2): 
                            kx = (-(kx - (self.Lx//2))) % (self.Lx//2)
                        mom = (kx, ky)    
                        # reverse the order if necessary
                        if kx > ky: 
                            mom     =   (ky, kx)
                        
                        # set the mapping
                        self.sym_norm[mom]      +=  1
                        self.sym_map[(x, y)]    =   mom
        except Exception as e:
            print(f"Error: {e}")