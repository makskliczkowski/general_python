"""
Contains the general lattice class and its children for different lattice types. 
@Author: Maksymilian Kliczkowski
@Email: maksymilian.kliczkowski@pwr.edu.pl
@Date: 2025-02-01
"""
# import HDF5 handling
import os
import sys
sys.path.append('..')

# Import the necessary modules
from common import HDF5Mod
from common import DirectoriesMod

# Import the necessary modules
from abc import ABC, abstractmethod

import numpy as np
Backend = np
# try out with JAX library
try:
    import jax.numpy as jnp
    Backend = jnp
    print("JAX library has been imported.")
except ImportError:
    print("JAX library has not been imported. Using NumPy as a backend.")

# Import the necessary modules
from enum import Enum, auto, unique                 # for enumerations
from typing import Union, Callable, Tuple, List     # type hints for the functions and methods

#############################################################################################################
@unique
class LatticeDirection(Enum):
    '''
    Enumeration for the lattice directions
    '''
    X = auto()
    Y = auto()
    Z = auto()

# -----------------------------------------------------------------------------------------------------------
@unique
class LatticeBC(Enum):
    '''
    Enumeration for the boundary conditions in the lattice model.
    '''
    PBC = auto()    # Periodic Boundary Conditions
    OBC = auto()    # Open Boundary Conditions      
    MBC = auto()    # Mixed Boundary Conditions     - periodic in X direction, open in Y direction
    SBC = auto()    # Special Boundary Conditions   - periodic in Y direction, open in X direction
    
# -----------------------------------------------------------------------------------------------------------
@unique
class LatticeType(Enum):
    '''
    Contains all the implemented lattice types for the lattice model. 
    '''
    SQUARE      = auto()    # Square lattice
    HEXAGONAL   = auto()    # Hexagonal lattice
    HONEYCOMB   = auto()    # Honeycomb lattice

############################################## GENERAL LATTICE ##############################################

class Lattice(ABC):
    '''
    General Lattice class. This class contains the general lattice model.
    It is an abstract class and is not meant to be instantiated. It is meant to be inherited by other classes.
                
    check the boundary conditions
    the lattice sites, no matter the lattice type are indexed from 0 to Ns - 1
    Example:
    1D:
        - 1D lattice with 10 sites, site 0 has nearest neighbors 1 and 9 (PBC)
        - 1D lattice with 10 sites, site 0 has nearest neighbors 1 (OBC)
        - 1D lattice with 10 sites, site 0 has nearest neighbors 1 and 9 (MBC)
        - 1D lattice with 10 sites, site 0 has nearest neighbors 1 (SBC)
    2D:
        - 2D lattice with 9 sites, site 0 has nearest neighbors 1, 3, 8, 6 (PBC)
        - 2D lattice with 9 sites, site 0 has nearest neighbors 1, 3 (OBC)
        - 2D lattice with 9 sites, site 0 has nearest neighbors 1, 3, 8, 6 (MBC)
    - 2D lattice with 9 sites, site 0 has nearest neighbors 1, 3 (SBC)
    This means that no matter of the lattice types, the sites are counted from the left
    to the right and from the bottom to the top. The nearest neighbors are calculated like a 
    snake (reversed in reality, bigger numbers are on the top)
    - 1D: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
    - 2D: 0 -> 1 -> 2
            |    |    |
            3 -> 4 -> 5
            |    |    |
            6 -> 7 -> 8
    - 3D: 0 -> 1 -> 2 ---- 9  -> 10 -> 11
            |    |    |      |     |     |
            3 -> 4 -> 5 ---- 12 -> 13 -> 14
            |    |    |      |     |     |
            6 -> 7 -> 8 ---- 15 -> 16 -> 17    
            
    Example: 
        1) Square lattice with 3x3 sites and PBC:
            0 -> 1 -> 2
            |    |    |
            3 -> 4 -> 5
            |    |    |
            6 -> 7 -> 8
    '''
    
    _BAD_LATTICE_SITE = -1
    @property
    def bad_lattice_site(self):
        ''' Bad lattice site '''
        return self._BAD_LATTICE_SITE
    
    # Lattice constants
    a = 1
    b = 1
    c = 1
    
    def __init__(self, dim, lx, ly, lz, bc, *args, **kwargs):
        '''
        Initialize the general lattice model with the following parameters:
        - dim   : dimension of the lattice
        - lx    : number of sites in the x direction
        - ly    : number of sites in the y direction
        - lz    : number of sites in the z direction
        - bc    : boundary conditions
        '''
        
        
        self._dim   = dim
        self._bc    = bc            # Boundary conditions
        self._lx    = lx
        self._ly    = ly
        self._lz    = lz
        self._lxly  = lx * ly
        self._lxlylz= lx * ly * lz
        self._ns    = lx * ly * lz  # Number of sites - set only initially as it is implemented in the children
        self._type  = LatticeType.SQUARE
        
        # neighbors
        self._nn            = [[]]
        self._nn_forward    = [[]]
        self._nnn           = [[]]
        self._nnn_forward   = [[]]
        
        # helping lists
        self._coords        = []
        self._spatial_norm  = [[[]]]                        # three dimensional array for the spatial norm
        
        # matrices for real space and inverse space vectors
        self._a1            = Backend.zeros((dim, dim))     # real space vectors - base vectors of the lattice
        self._a2            = Backend.zeros((dim, dim))
        self._a3            = Backend.zeros((dim, dim))
        self._k1            = Backend.zeros((dim, dim))     # inverse space vectors - reciprocal lattice vectors
        self._k2            = Backend.zeros((dim, dim))
        self._k3            = Backend.zeros((dim, dim))
        self._rvectors      = Backend.zeros((self._ns, 3))  # allowed values of the real space vectors
        self._kvectors      = Backend.zeros((self._ns, 3))  # allowed values of the inverse space vectors
        # initialize dft matrix
        self._dft           = Backend.zeros((self._ns, self._ns), dtype = complex) # Discrete Fourier Transform matrix for the lattice model
        
        # symmetries for momenta (if one uses the symmetry,
        # returning to original one may result in using normalization)
        self.sym_norm       = {}
        self.sym_map        = {}

    # -----------------------------------------------------------------------------
            
    def __str__(self):
        ''' String representation of the lattice '''
        return "General Lattice"
    
    # -----------------------------------------------------------------------------
    
    def __repr__(self):
        ''' Representation of the lattice '''
        return f"{self._type.name},{self._bc.name},d={self._dim},Ns={self._ns},Lx={self._lx},Ly={self._ly},Lz={self._lz}"
    
    ################################### GETTERS ###################################
    
    @property
    def lx(self):
        ''' Number of sites in the x direction '''
        return self._lx
    
    @lx.setter
    def lx(self, value):
        self._lx = value
    
    @property
    def Lx(self):
        ''' Number of sites in the x direction '''
        return self._lx
    
    @Lx.setter
    def Lx(self, value):
        self._lx = value
    
    @property
    def ly(self):
        ''' Number of sites in the y direction '''
        return self._ly
    
    @ly.setter
    def ly(self, value):
        self._ly = value
    
    @property
    def Ly(self):
        ''' Number of sites in the y direction '''
        return self._ly
    
    @Ly.setter
    def Ly(self, value):
        self._ly = value
    
    @property
    def lz(self):
        ''' Number of sites in the z direction '''
        return self._lz
    
    @lz.setter
    def lz(self, value):
        self._lz = value
    
    @property
    def Lz(self):
        ''' Number of sites in the z direction '''
        return self._lz
    
    @Lz.setter
    def Lz(self, value):
        self._lz = value
    
    @property
    def lxly(self):
        ''' Number of sites in the x and y directions '''
        return self._lxly
    
    @lxly.setter
    def lxly(self, value):
        self._lxly = value
    
    @property
    def lxlylz(self):
        ''' Number of sites in the x, y and z directions '''
        return self._lxlylz
    
    @lxlylz.setter
    def lxlylz(self, value):
        self._lxlylz = value
    
    @property
    def dim(self):
        ''' Dimension of the lattice '''
        return self._dim
    
    @dim.setter
    def dim(self, value):
        self._dim = value
    
    @property
    def ns(self):
        ''' Number of sites '''
        return self._ns
    
    @ns.setter
    def ns(self, value):
        self._ns = value
    
    @property
    def Ns(self):
        ''' Number of sites '''
        return self._ns
    
    @Ns.setter
    def Ns(self, value):
        self._ns = value
    
    @property
    def a1(self):
        ''' Real space vector a1 '''
        return self._a1
    
    @a1.setter
    def a1(self, value):
        self._a1 = value
    
    @property
    def a2(self):
        ''' Real space vector a2 '''
        return self._a2
    
    @a2.setter
    def a2(self, value):
        self._a2 = value
    
    @property
    def a3(self):
        ''' Real space vector a3 '''
        return self._a3
    
    @a3.setter
    def a3(self, value):
        self._a3 = value
    
    @property
    def k1(self):
        ''' Inverse space vector k1 '''
        return self._k1
    
    @k1.setter
    def k1(self, value):
        self._k1 = value
    
    @property
    def k2(self):
        ''' Inverse space vector k2 '''
        return self._k2
    
    @k2.setter
    def k2(self, value):
        self._k2 = value
    
    @property
    def k3(self):
        ''' Inverse space vector k3 '''
        return self._k3
    
    @k3.setter
    def k3(self, value):
        self._k3 = value
    
    @property
    def dft(self):
        ''' Discrete Fourier Transform matrix '''
        return self._dft
    
    @dft.setter
    def dft(self, value):
        self._dft = value
    
    @property
    def nn(self):
        ''' Nearest neighbors '''
        return self._nn
    
    @nn.setter
    def nn(self, value):
        self._nn = value
    
    @property
    def nn_forward(self):
        ''' Nearest neighbors forward '''
        return self._nn_forward
    
    @nn_forward.setter
    def nn_forward(self, value):
        self._nn_forward = value
    
    @property
    def nnn(self):
        ''' Next nearest neighbors '''
        return self._nnn
    
    @nnn.setter
    def nnn(self, value):
        self._nnn = value
    
    @property
    def nnn_forward(self):
        ''' Next nearest neighbors forward '''
        return self._nnn_forward
    
    @nnn_forward.setter
    def nnn_forward(self, value):
        self._nnn_forward = value
    
    @property
    def coordinates(self):
        ''' Real space coordinates '''
        return self._coords
    
    @coordinates.setter
    def coordinates(self, value):
        self._coords = value
    
    @property
    def kvectors(self):
        ''' Inverse space vectors '''
        return self._kvectors
    
    @kvectors.setter
    def kvectors(self, value):
        self._kvectors = value
    
    @property
    def rvectors(self):
        ''' Real space vectors '''
        return self._rvectors
    
    @rvectors.setter
    def rvectors(self, value):
        self._rvectors = value
    
    @property
    def bc(self):
        ''' Boundary conditions '''
        return self._bc
    
    @bc.setter
    def bc(self, value):
        self._bc = value
    
    @property
    def type(self):
        ''' Lattice type '''
        return self._type
    
    @type.setter
    def type(self, value):
        self._type = value
    
    @property
    def spatial_norm(self):
        ''' Spatial norm '''
        return self._spatial_norm
    
    @spatial_norm.setter
    def spatial_norm(self, value):
        self._spatial_norm = value
    
    # -----------------------------------------------------------------------------
    
    @abstractmethod
    def site_index(self, x : int, y : int, z : int):
        '''
        Returns the site index given the coordinates
        '''
        pass
    
    # -----------------------------------------------------------------------------
    
    def site_diff(self, i : Union[int, tuple], j : Union[int, tuple]):
        '''
        Returns the site difference between sites
        - i - i'th lattice site or tuple of coordinates
        - j - j'th lattice site or tuple of coordinates
        Returns:
        - tuple of differences (x, y, z)
        '''
        x1, y1, z1 = self.get_coordinates(i) if isinstance(i, int) else i 
        x2, y2, z2 = self.get_coordinates(j) if isinstance(j, int) else j
        return (x2 - x1, y2 - y1, z2 - z1)
    
    # -----------------------------------------------------------------------------
    
    def site_distance(self, i : Union[int, tuple], j : Union[int, tuple]):
        '''
        Returns the site distance between sites
        - i - i'th lattice site or tuple of coordinates
        - j - j'th lattice site or tuple of coordinates
        Returns:
        - distance between sites
        '''
        x, y, z = self.site_diff(i, j)
        return np.sqrt(x**2 + y**2 + z**2)
    
    # -----------------------------------------------------------------------------
    
    def calculate_dft_matrix(self, phase = False):
        '''
        Calculates the Discrete Fourier Transform (DFT) matrix for the lattice.
        This method can be optimized using FFT (Fast Fourier Transform) in the future.
        Reference: https://en.wikipedia.org/wiki/DFT_matrix
        
        Args:
        - phase (bool): If True, adds a complex phase to the k-vectors.
        
        Returns:
        - DFT matrix (ndarray): The calculated DFT matrix.
        '''
        
        # initialize the DFT matrix
        self._dft   = Backend.zeros((self.ns, self.ns), dtype = complex)
        
        # Create omega values
        omega_x     = Backend.exp(-1j * 2.0 * Backend.pi * self.a / self._lx)
        omega_y     = Backend.exp(-1j * 2.0 * Backend.pi * self.b / self._ly)
        omega_z     = Backend.exp(-1j * 2.0 * Backend.pi * self.c / self._lz)

        e_min_pi    = Backend.exp(1j * Backend.pi)

        # Precompute coordinate arrays
        indices     = Backend.arange(self._ns)
        coords      = Backend.array([self.get_coordinates(i) for i in indices])  # Shape (Ns, 3)

        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]

        # Compute phase factors
        phase_x     = Backend.where(x_coords % 2 == 0, 1, e_min_pi) if phase else Backend.ones(self._ns)
        phase_y     = Backend.where(y_coords % 2 == 0, 1, e_min_pi) if phase else Backend.ones(self._ns)
        phase_z     = Backend.where(z_coords % 2 == 0, 1, e_min_pi) if phase else Backend.ones(self._ns)

        # Compute DFT matrix using broadcasting
        self.dft = (Backend.power(omega_x, x_coords[:, None] * x_coords[None, :]) *
                    Backend.power(omega_y, y_coords[:, None] * y_coords[None, :]) *
                    Backend.power(omega_z, z_coords[:, None] * z_coords[None, :]) *
                    phase_x[:, None] * phase_y[:, None] * phase_z[:, None])
        return self._dft
    
    # -----------------------------------------------------------------------------
    
    def get_nei(self, site: int, **kwargs):
        '''
        Returns the nearest neighbors of a given site. 
        
        Args:
            - direction : direction of the lattice (can be X, Y, Z - default is X)        
        '''
        if 'corr_len' in kwargs and 'direction' in kwargs:
            direction   = kwargs.get('direction', LatticeDirection.X)
            corr_len    = kwargs.get('corr_len', 1)
            direction   = self._adjust_direction(direction)
            return self._get_neighbor_with_corr_len(site, direction, corr_len)
        elif 'direction' in kwargs:
            direction   = kwargs.get('direction', LatticeDirection.X)
            direction   = self._adjust_direction(direction)
            return self.get_nn_direction(site, direction)
        else:
            return self._nn[site]

    def get_nei_forward(self, site: int, num : int = -1):
        '''
        Returns the forward nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of nearest neighbors
        Returns:
            - list of nearest neighbors        
        '''
        if num < 0:
            return self._nn_forward[site]
        return self._nn_forward[site][num]
    
    def _adjust_direction(self, direction : Union[LatticeDirection, int]):
        '''
        Adjust the direction to the lattice dimension
        Args:
            - direction : direction of the lattice (can be X, Y, Z - default is X)
        '''
        if self.dim == 1:
            return LatticeDirection.X
        elif self.dim == 2 and LatticeDirection(direction) > LatticeDirection.Y:
            return LatticeDirection.Y
        elif self.dim == 3 and LatticeDirection(direction) > LatticeDirection.Z:
            return LatticeDirection.Z
        return direction

    def _get_neighbor_with_corr_len(self, site : int, direction : Union[LatticeDirection, int], corr_len : int = 1):
        '''
        Returns the neighbor with a correlation length
        - site      : lattice site
        - direction : direction of the lattice
        - corr_len  : correlation length
        Returns:
        - neighbor site (int)
        '''
        
        if self.bc == LatticeBC.PBC:
            return self._pbc_neighbor(site, direction, corr_len)
        elif self.bc == LatticeBC.OBC:
            return self._obc_neighbor(site, direction, corr_len)
        elif self.bc == LatticeBC.MBC:
            return self._mbc_neighbor(site, direction, corr_len)
        elif self.bc == LatticeBC.SBC:
            return self._sbc_neighbor(site, direction, corr_len)
        return (site + corr_len) % self._lxlylz

    def _pbc_neighbor(self, site, direction, corr_len):
        if direction == LatticeDirection.X:
            return (site + corr_len) % self._lx
        elif direction == LatticeDirection.Y:
            return (site + corr_len * self._lx) % self._lxly
        elif direction == LatticeDirection.Z:
            return (site + corr_len * self._lxly) % self._lxlylz

    def _obc_neighbor(self, site, direction, corr_len):
        if direction == LatticeDirection.X:
            return (site + corr_len) if (site + corr_len) < self._lx else self.bad_lattice_site
        elif direction == LatticeDirection.Y:
            return (site + corr_len * self._lx) if (site + corr_len * self._lx) < self._lxly else self.bad_lattice_site
        elif direction == LatticeDirection.Z:
            return (site + corr_len * self._lxly) if (site + corr_len * self._lxly) < self._lxlylz else self.bad_lattice_site

    def _mbc_neighbor(self, site, direction, corr_len):
        if direction == LatticeDirection.X:
            return (site + corr_len) % self._lx
        elif direction == LatticeDirection.Y:
            return (site + corr_len * self._lx) if (site + corr_len * self._lx) < self._lxly else self.bad_lattice_site
        elif direction == LatticeDirection.Z:
            return (site + corr_len * self._lxly) if (site + corr_len * self._lxly) < self._lxlylz else self.bad_lattice_site

    def _sbc_neighbor(self, site, direction, corr_len):
        if direction == LatticeDirection.Y:
            return (site + corr_len * self._lx) % self._lxly
        elif direction == LatticeDirection.X:
            return (site + corr_len) if (site + corr_len) < self._lx else self.bad_lattice_site
        elif direction == LatticeDirection.Z:
            return (site + corr_len * self._lxly) if (site + corr_len * self._lxly) < self._lxlylz else self.bad_lattice_site
        
    # -----------------------------------------------------------------------------
    
    # Virtual methods
    
    # -----------------------------------------------------------------------------
    
    @abstractmethod
    def get_real_vec(self, x : int, y : int, z : int):
        '''
        Returns the real vector given the coordinates. Uses the lattice constants.
        '''
        pass
    
    @abstractmethod
    def get_norm(self, x : int, y : int, z : int):
        '''
        Returns the norm of the vector given the coordinates.
        '''
        pass
    
    @abstractmethod
    def get_nn_direction(self, site : int, direction : LatticeDirection):
        '''
        Returns the nearest neighbors in a given direction.
        '''
        pass
    
    # -----------------------------------------------------------------------------
    
    # Virtual methods for forward neighbors
    
    # -----------------------------------------------------------------------------
    
    def get_nn_forward_num(self, site : int):
        '''
        Returns the number of forward nearest neighbors of a given site.
        
        Args:
        - site : lattice site
        Returns:
        - number of nearest neighbors
        '''
        return len(self.nn_forward[site])
    
    @abstractmethod
    def get_nn_forward(self, site : int, num : int = -1):
        '''
        Returns the forward nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of nearest neighbors
        Returns:
            - list of nearest neighbors        
        '''
        pass
    
    # -----------------------------------------------------------------------------
    
    def get_nnn_forward_num(self, site : int):
        '''
        Returns the number of forward next nearest neighbors of a given site.
        
        Args:
        - site : lattice site
        Returns:
        - number of next nearest neighbors
        '''
        return len(self.nnn_forward[site])
    
    @abstractmethod
    def get_nnn_forward(self, site : int, num : int = -1):
        '''
        Returns the forward next nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of next nearest neighbors
        Returns:
            - list of next nearest neighbors        
        '''
        pass
    
    # -----------------------------------------------------------------------------
    
    # standard getters

    # -----------------------------------------------------------------------------

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
        Returns the k-vectors at lattice site i or all of them
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

    # -----------------------------------------------------------------------------
    
    # Spatial information
    
    # -----------------------------------------------------------------------------
    
    def get_spatial_norm(self, *args):
        '''
        Returns the spatial norm at lattice site i or all of them
        '''
        if len(args) == 0:
            return self.spatial_norm
        elif len(args) == 1:
            return self.spatial_norm[args[0]]
        elif len(args) == 2:
            return self.spatial_norm[args[0]][args[1]]
        else:
            return self.spatial_norm[args[0]][args[1]][args[2]]

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
        Lx, Ly, Lz  = self.Lx, self.Ly, self.Lz
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
    
    @abstractmethod
    def calculate_coordinates(self):
        '''
        Calculates the real lattice coordinates based on a lattice site
        '''
        pass
    
    @abstractmethod
    def calculate_r_vectors(self):
        '''
        Calculates all possible vectors in real space
        '''
        pass
    
    @abstractmethod
    def calculate_k_vectors(self):
        '''
        Calculates the inverse space vectors
        '''
        pass
    
    @abstractmethod
    def calculate_norm_sym(self):
        '''
        Calculates the norm for a given symmetry
        '''
        pass
    
    # -----------------------------------------------------------------------------
    
    # Nearest neighbors
    
    # -----------------------------------------------------------------------------
    
    def _calculate_nn_pbc(self):
        '''
        Calculates the nearest neighbors for periodic boundary conditions
        '''
        return self.calculate_nn_in(True, True, True)
    
    def _calculate_nn_obc(self):
        '''
        Calculates the nearest neighbors for open boundary conditions
        '''
        return self.calculate_nn_in(False, False, False)
    
    def _calculate_nn_mbc(self):
        '''
        Calculates the nearest neighbors for mixed boundary conditions
        '''
        return self.calculate_nn_in(True, False, False)
    
    def _calculate_nn_sbc(self):
        '''
        Calculates the nearest neighbors for special boundary conditions
        '''
        return self.calculate_nn_in(False, True, False)
    
    @abstractmethod
    def calculate_nn_in(self, pbcx : bool, pbcy : bool, pbcz : bool):
        '''
        Calculates the nearest neighbors based on the boundary conditions
        '''
        pass

    def calculate_nn(self):
        '''
        Calculates the nearest neighbors
        
        Args:
        - pbcx : periodic boundary conditions in the x direction
        - pbcy : periodic boundary conditions in the y direction
        - pbcz : periodic boundary conditions in the z direction
        '''
        
        match (self._bc):
            case LatticeBC.PBC:
                self._calculate_nn_pbc()
            case LatticeBC.OBC:
                self._calculate_nn_obc()
            case LatticeBC.MBC:
                self._calculate_nn_mbc()
            case LatticeBC.SBC:
                self._calculate_nn_sbc()
            case _:
                raise ValueError("The boundary conditions are not implemented.")

    # -----------------------------------------------------------------------------
    
    # Next nearest neighbors
    
    # -----------------------------------------------------------------------------
    
    def _calculate_nnn_pbc(self):
        '''
        Calculates the next nearest neighbors for periodic boundary conditions
        '''
        return self.calculate_nnn_in(True, True, True)
    
    def _calculate_nnn_obc(self):
        '''
        Calculates the next nearest neighbors for open boundary conditions
        '''
        return self.calculate_nnn_in(False, False, False)
        
    def _calculate_nnn_mbc(self):
        '''
        Calculates the next nearest neighbors for mixed boundary conditions
        '''
        return self.calculate_nnn_in(True, False, False)
    
    def _calculate_nnn_sbc(self):
        '''
        Calculates the next nearest neighbors for special boundary conditions
        '''
        return self.calculate_nnn_in(False, True, False)
    
    @abstractmethod
    def calculate_nnn_in(self, pbcx : bool, pbcy : bool, pbcz : bool):
        '''
        Calculates the next nearest neighbors based on the boundary conditions
        '''
        pass
    
    def calculate_nnn(self):
        '''
        Calculates the next nearest neighbors
        
        Args:
        - pbcx : periodic boundary conditions in the x direction
        - pbcy : periodic boundary conditions in the y direction
        - pbcz : periodic boundary conditions in the z direction
        '''
        
        match (self._bc):
            case LatticeBC.PBC:
                self._nnn = self._calculate_nnn_pbc()
            case LatticeBC.OBC:
                self._nnn = self._calculate_nnn_obc()
            case LatticeBC.MBC:
                self._nnn = self._calculate_nnn_mbc()
            case LatticeBC.SBC:
                self._nnn = self._calculate_nnn_sbc()
            case _:
                raise ValueError("The boundary conditions are not implemented.")

    # -----------------------------------------------------------------------------
    
    # Saving the lattice
    
    # -----------------------------------------------------------------------------
    
#############################################################################################################

def save_bonds(lattice : Lattice, directory : Union[DirectoriesMod, str], filename : str):
    '''
    Saves the bonds of the lattice to a file
    Args:
    - lattice   : lattice model
    - directory : directory to save the file
    - filename  : filename
    
    Returns:
    - True if the file has been saved, False otherwise
    '''
    if lattice.type == LatticeType.HONEYCOMB:
        
        # get the bonds
        bonds   =   -Backend.ones((lattice.ns, 3))
        
        # go through all
        for i in range(lattice.ns):
            
            num_of_nn = len(lattice.get_nn_forward_num(i))
            
            for nn in range(num_of_nn):
                nei = lattice.get_nei_forward(i, nn)
                if nei >= 0:
                    bonds[i, nn] = nei
        # save the bonds
        try:
            HDF5Mod.save_hdf5(directory, filename, bonds)
        except Exception as e:
            print(f"An error has occured while saving the bonds: {e}")
            return False
        return True
    return False
    
#############################################################################################################