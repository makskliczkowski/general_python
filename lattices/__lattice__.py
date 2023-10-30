import numpy as np

class Lattice(object):
    a = 1
    b = 1
    c = 1
    
    '''
    Initialize the general lattice model
    '''
    def __init__(self, dim, Lx, Ly, Lz, _BC, *args, **kwargs):
        self.dim    = dim
        self._BC    = _BC
        self.Lx     = Lx
        self.Ly     = Ly
        self.Lz     = Lz
        self.Ns     = Lx * Ly * Lz
        
        # helping lists
        self.coordinates    = []
        
        # matrices for real space and inverse space vectors
        self.vectors        = np.zeros((3,3))
        self.rvectors       = np.zeros((self.Ns,3))
        self.kvectors       = np.zeros((self.Ns,3))
        # initialize dft matrix
        self.dft            = np.zeros((self.Ns, self.Ns))
        
        # symmetries for momenta (if one uses the symmetry, 
        # returning to original one may result in using normalization)
        self.sym_norm       = self.calculate_norm_sym()
        self.sym_map        = {}
        
    def __str__(self):
        return "General Lattice"   
    
    ################################### GETTERS ###################################   
    def get_Lx(self):
        return self.Lx
    
    def get_Ly(self):
        return self.Ly
    
    def get_Lz(self):
        return self.Lz
    
    '''
    For a given site return the real space coordinates
    '''
    def get_coordinates(self, *args):
        if len(args) == 0:
            return self.coordinates
        else:
            return self.coordinates[args[0]]
    
    '''
    Returns the k-vectors
    '''
    def get_k_vectors(self, *args):
        if len(args) == 0:
            return self.kvectors
        else:
            return self.kvectors[args[0]]
    
    '''
    Returns the site difference between sites
    - i - i'th lattice site
    - j - j'th lattice site
    '''
    def get_site_diff(self, i, j):
        return self.get_coordinates(j) - self.get_coordinates(i)
    
    '''
    Returns the real vector at lattice site i or all of them
    '''
    def get_r_vectors(self,*args):
        if len(args) == 0:
            return self.rvectors
        else:
            return self.rvectors[args[0]]
        
    '''
    Returns the DFT matrix
    '''
    def get_dft(self, *args):
        if len(args) == 0:
            return self.dft
        elif len(args) == 1:
            # return row
            return self.dft[args[0]]
        else:
            # return element
            return self.dft[args[0], args[1]]
    
    '''
    Returns the indices of kvectors if symmetries are to be concerned
    '''
    def get_k_vec_idx(self, sym = False):
        pass
    
    ################################### ABSTRACT CALCULATORS ###################################
    
    '''
    Calculates the real lattice coordinates based on a lattice site
    '''
    def calculate_coordinates(self):
        pass
    
    '''
    Calculates all possible vectors in real space
    '''
    def calculate_r_vectors(self):
        pass
    
    '''
    Calculates the inverse space vectors
    '''
    def calculate_k_vectors(self):
        pass
    
    '''
    Calculates the DFT matrix. Can be faster with using FFT -> to think about
    @url https://en.wikipedia.org/wiki/DFT_matrix
    - phase - shall add a complex phase to the k-vectors?
    '''
    def calculate_dft_matrix(self, phase = False):
        self.dft = np.zeros((self.Ns, self.Ns), dtype = np.complex128)
        
        omega_x = np.exp(-np.complex(0,1.0) * 2.0 * np.pi * self.a / self.Lx)
        omega_y = np.exp(-np.complex(0,1.0) * 2.0 * np.pi * self.b / self.Ly)
        omega_z = np.exp(-np.complex(0,1.0) * 2.0 * np.pi * self.c / self.Lz)
        
        e_min_pi = np.exp(np.complex(0.0,1.0) * np.pi)
        # do double loop - not perfect solution
        
        for row in np.arange(self.Ns):
            x_row, y_row, z_row     = self.get_coordinates(row)
            for col in np.arange(self.Ns):
                x_col, y_col, z_col = self.get_coordinates(col)
                # to shift by -PI
                phase_x             = (e_min_pi if not x_col % 2 == 0 else 1) if phase else 1
                phase_y             = (e_min_pi if not y_col % 2 == 0 else 1) if phase else 1
                phase_z             = (e_min_pi if not z_col % 2 == 0 else 1) if phase else 1
                # set the omegas not optimal powers, but is calculated once
                self.dft[row, col]  = np.power(omega_x, x_row * x_col) * np.power(omega_y, y_row * y_col) * np.power(omega_z, z_row * z_col) * phase_x * phase_y * phase_z
    
    '''
    Calculates the norm for a given symmetry
    '''
    def calculate_norm_sym(self):
        return {}
    
############################################## SQUARE LATTICE ##############################################

class Square(Lattice):
    # Lattice constants
    a = 1.
    b = 1.
    c = 1.

    '''
    Initializer of the square lattice
    '''
    def __init__(self, dim, Lx, Ly, Lz, _BC, *args, **kwargs):
        super().__init__(dim, Lx, Ly, Lz, _BC, *args, **kwargs)

        self.Ns         = Lx * Ly * Lz
        self.vectors    = np.array([[Square.a,0,0],[0,Square.b,0],[0,0,Square.c]])
        self.kvectors   = np.zeros((self.Ns, 3))
        self.rvectors   = np.zeros((self.Ns, 3))
        #self.kvectors = np.zeros((self.Lx, self.Ly))
        
        self.calculate_coordinates()
        self.calculate_r_vectors()
        self.calculate_k_vectors()
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
    
    '''
    Returns the indices of kvectors if symmetries are to be concerned
    '''
    def get_k_vec_idx(self, sym = False):
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

    '''
    Calculates the real lattice coordinates based on a lattice site
    '''
    def calculate_coordinates(self):
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
        two_pi_over_Lx = 2 * np.pi / Square.a / self.Lx;
        two_pi_over_Ly = 2 * np.pi / Square.b / self.Ly;
        two_pi_over_Lz = 2 * np.pi / Square.c / self.Lz;
    
        for qx in range(self.Lx):
            kx = -np.pi + two_pi_over_Lx * qx
            for qy in range(self.Ly):
                ky = -np.pi + two_pi_over_Ly * qy
                for qz in range(self.Lz):
                    kz = -np.pi + two_pi_over_Lz * qz
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
        # set the norm dictionary
        self.sym_norm   = super().calculate_norm_sym()
        self.sym_map    = {}
        
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