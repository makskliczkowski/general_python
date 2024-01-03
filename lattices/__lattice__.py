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
        super(*args, **kwargs)
        
    def __str__(self):
        return "General Lattice"  
    
    ################################### GETTERS ###################################   
    def get_Lx(self):
        return self.Lx
    
    def get_Ly(self):
        return self.Ly
    
    def get_Lz(self):
        return self.Lz
    
    def get_Ns(self):
        return self.Lx * self.Ly * self.Lz
    
    def get_Ls(self):
        return self.Lx, self.Ly, self.Lz
    
    def get_coordinates(self, *args):
        '''
        For a given site return the real space coordinates
        '''
        if len(args) == 0:
            return self.coordinates
        else:
            return self.coordinates[args[0]]
        
    def get_k_vectors(self, *args):
        '''
        Returns the k-vectors
        '''
        if len(args) == 0:
            return self.kvectors
        else:
            return self.kvectors[args[0]]
    
    def get_site_diff(self, i, j):
        '''
        Returns the site difference between sites
        - i - i'th lattice site
        - j - j'th lattice site
        '''
        return self.get_coordinates(j) - self.get_coordinates(i)
    
    def get_r_vectors(self,*args):
        '''
        Returns the real vector at lattice site i or all of them
        '''
        if len(args) == 0:
            return self.rvectors
        else:
            return self.rvectors[args[0]]
        
    def get_dft(self, *args):
        '''
        Returns the DFT matrix
        '''
        if len(args) == 0:
            return self.dft
        elif len(args) == 1:
            # return row
            return self.dft[args[0]]
        else:
            # return element
            return self.dft[args[0], args[1]]
    
    def get_k_vec_idx(self, sym = False):
        '''
        Returns the indices of kvectors if symmetries are to be concerned
        '''
        pass

    def get_difference_idx_matrix(self, cut = True) -> list:
        '''
        Returns the matrix with indcies corresponding to a slice from the QMC. 
        A usefull function for reading the position Green's function saved from:
        @url https://github.com/makskliczkowski/DQMC
        The Green's functions are saved in the following manner. If cut is True, data 
        has (2L_i - 1) possible position differences, otherwise we skip the negative ones and use L_i.
        For 1D simulation: 1 column and (2 * Lx - 1) rows for possition differences (-Lx, -Lx + 1, ..., 0, ..., Lx)
        For 2D simulation: (2 * Lx - 1) rows for possition differences (-Lx, -Lx + 1, ..., 0, ..., Lx) and (2 * Ly - 1) columns for possition differences (-Ly, -Ly + 1, ..., 0, ..., Ly)
        For 3D simulation: Same as in 2D but after (2 * Lx - 1) x (2 * Ly - 1) matrix has finished, a new slice for Lz appears for next columns Lz * (2*Ly - 1)
        - cut : if true (2L_i - 1) possible position differences, otherwise we skip the negative ones and use L_i.
        '''
        Lx, Ly, Lz  = self.get_Ls()
        xnum        = 2 * Lx - 1 if cut else Lx
        ynum        = 2 * Ly - 1 if cut else Ly
        znum        = 2 * Lz - 1 if cut else Lz
        
        _slice  = [[[0, 0, 0] for _ in range(ynum * znum)] for _ in range(xnum)]
        for k in range(znum):
            z   = k - (Lz if cut else 0)
            for i in range(xnum):
                x = i - (Lx if cut else 0)
                for j in range(ynum):
                    y = j - (Ly if cut else 0)
                    # x's are the rows and y's (*z's) are the columns 
                    _slice[i][j + k * ynum][0] = x
                    _slice[i][j + k * ynum][1] = y
                    _slice[i][j + k * ynum][2] = z                    
        return [[tuple(_slice[i][j]) for j in range(ynum * znum)] for i in range(xnum)]
    
    ############################ ABSTRACT CALCULATORS #############################
    
    def calculate_coordinates(self):
        '''
        Calculates the real lattice coordinates based on a lattice site
        '''
        pass
    
    def calculate_r_vectors(self):
        '''
        Calculates all possible vectors in real space
        '''
        pass
    
    def calculate_k_vectors(self):
        '''
        Calculates the inverse space vectors
        '''
        pass
    
    def calculate_dft_matrix(self, phase = False):
        '''
        Calculates the DFT matrix. Can be faster with using FFT -> to think about
        @url https://en.wikipedia.org/wiki/DFT_matrix
        - phase - shall add a complex phase to the k-vectors?
        '''
        self.dft = np.zeros((self.Ns, self.Ns), dtype = complex)
        
        omega_x = np.exp(-complex(0,1.0) * 2.0 * np.pi * self.a / self.Lx)
        omega_y = np.exp(-complex(0,1.0) * 2.0 * np.pi * self.b / self.Ly)
        omega_z = np.exp(-complex(0,1.0) * 2.0 * np.pi * self.c / self.Lz)
        
        e_min_pi = np.exp(complex(0.0,1.0) * np.pi)
        # do double loop - not perfect solution
        
        # rvectors
        for row in np.arange(self.Ns):
            x_row, y_row, z_row     = self.get_coordinates(row)
            # kvectors
            for col in np.arange(self.Ns):
                x_col, y_col, z_col = self.get_coordinates(col)
                # to shift by -PI
                phase_x             = (e_min_pi if not x_col % 2 == 0 else 1) if phase else 1
                phase_y             = (e_min_pi if not y_col % 2 == 0 else 1) if phase else 1
                phase_z             = (e_min_pi if not z_col % 2 == 0 else 1) if phase else 1
                # set the omegas not optimal powers, but is calculated once
                self.dft[row, col]  = np.power(omega_x, x_row * x_col) * np.power(omega_y, y_row * y_col) * np.power(omega_z, z_row * z_col) * phase_x * phase_y * phase_z      

    def calculate_norm_sym(self):
        '''
        Calculates the norm for a given symmetry
        '''
        return {}
    
############################################## SQUARE LATTICE ##############################################

class Square(Lattice):
    """ 
    General Square Lattice type up to three dimensions!
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
                    
                    
class Hexagonal(Lattice):
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
        self.vectors    = np.array([[Square.a,0,0],[0,Square.b,0],[0,0,Square.c]])
        self.kvectors   = np.zeros((self.Ns, 3))
        self.rvectors   = np.zeros((self.Ns, 3))
        #self.kvectors = np.zeros((self.Lx, self.Ly))
        
    
    def __str__(self):
        '''
        Set the name
        '''
        return f"HEX,{'PBC' if self._BC == 0 else 'OBC'},d={self.dim},Ns={self.Ns},Lx={self.Lx},Ly={self.Ly},Lz={self.Lz}"
    
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
        pass