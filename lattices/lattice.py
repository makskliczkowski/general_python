"""
Contains the general lattice class hierarchy and helpers.

This module defines the base :class:`Lattice` API used across QES, together with
utility routines for boundary handling and symmetry metadata.

Currently, up to 3-spatial dimensions are supported...

File    : QES/general_python/lattices/lattice.py
Author  : Maksymilian Kliczkowski
Date    : 2025-02-01
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum, auto, unique                 # for enumerations
from typing import Dict, List, Mapping, Optional, Tuple, Union, Any

import numpy as np
import scipy.sparse as sp

try:
    from .tools.lattice_tools import LatticeDirection, LatticeBC, LatticeType, handle_boundary_conditions, handle_dim
    from .tools.lattice_flux import BoundaryFlux, _normalize_flux_dict
    from .tools.lattice_kspace import ( extract_bz_path_data, StandardBZPath, PathTypes, brillouin_zone_path, 
                                    reciprocal_from_real, extract_momentum, reconstruct_k_grid_from_blocks,
                                    build_translation_operators
                                    )
    from ..common import hdf5_lib as HDF5Mod
    from ..common import directories as DirectoriesMod
except ImportError:
    raise ImportError("Failed to import modules from parent package. Ensure proper package structure.")

############################################## GENERAL LATTICE ##############################################

Backend = np

class Lattice(ABC):
    '''
    General Lattice class. This class contains the general lattice model.
    It is an abstract class and is not meant to be instantiated. It is meant to be inherited by other classes.

    The lattice sites, no matter the lattice type are indexed from 0 to Ns - 1. Importantly,
    it can include multiple top
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
    - 1D:   
            0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
    
    - 2D:   
            0 -> 1 -> 2
            |    |    |
            3 -> 4 -> 5
            |    |    |
            6 -> 7 -> 8
            
    - 3D:   
            0 -> 1 -> 2 ---- 9  -> 10 -> 11
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
    
    _BAD_LATTICE_SITE   = None
    _DFT_LIMIT_SITES    = 100 
    
    # ---------------------------------------------
    #! INITIALIZATION
    # ---------------------------------------------
    
    @property
    def bad_lattice_site(self):
        ''' Bad lattice site '''
        return self._BAD_LATTICE_SITE
    
    # Lattice constants - physical units where applicable
    a           = 1
    b           = 1
    c           = 1
    unit_length = 1     # unit length in Angstroms - helper for physical calculations

    def __init__(self, 
                dim     : int           = None,
                lx      : int           = 1,
                ly      : int           = 1,
                lz      : int           = 1,
                bc      : str           = None,             # boundary conditions
                adj_mat : np.ndarray    = None,             # can be controlled by the user for generic graphs
                flux    : np.ndarray    = None,             # flux piercing the boundaries
                *args,
                **kwargs):
        '''
        General Lattice class. This class contains the general lattice model.
        
        Parameters
        ----------
        dim : int, optional
            Dimension of the lattice (1, 2, or 3). If None, inferred from lx, ly, lz.
        lx : int, optional
            Length of the lattice in the x-direction.
        ly : int, optional
            Length of the lattice in the y-direction.
        lz : int, optional
            Length of the lattice in the z-direction.
        bc : str, optional
            Boundary conditions (e.g., 'PBC', 'OBC').
        adj_mat : np.ndarray, optional
            Adjacency matrix for the lattice.
        flux : np.ndarray, optional
            Flux piercing the boundaries.
        '''
        
        self._dim           = handle_dim(lx, ly, lz)[0] if dim is None else dim
        self._bc            = handle_boundary_conditions(bc)
        self._lx            = lx
        self._ly            = ly
        self._lz            = lz
        self._lxly          = lx * ly
        self._lxlz          = lx * lz
        self._lylz          = ly * lz
        self._lxlylz        = lx * ly * lz
        self._ns            = lx * ly * lz                  # Number of sites - set only initially as it is implemented in the children
        self._type          = LatticeType.SQUARE
        self._adj_mat       = adj_mat
        
        # flux piercing the boundaries - for topological models
        self._flux          = _normalize_flux_dict(flux)
        super().__init__(*args, **kwargs)
        
        # neighbors
        self._nn            = [[]]
        self._nn_forward    = [[]]
        self._nn_max_num    = 0
        self._nnn           = [[]]
        self._nnn_forward   = [[]]
        
        # helping lists
        self._coords        = []
        self._cells         = []
        self._fracs         = []
        self._subs          = []
        self._spatial_norm  = [[[]]]                                # three dimensional array for the spatial norm
        
        # matrices for real space and inverse space vectors
        vec_size            = max(3, self._dim)
        self._vectors       = Backend.zeros((vec_size, vec_size))   # real space vectors - base vectors of the lattice
        self._a1            = Backend.zeros(vec_size)               # real space vectors - base vectors of the lattice
        self._a2            = Backend.zeros(vec_size)
        self._a3            = Backend.zeros(vec_size)
        self._basis         = Backend.zeros((0, vec_size))          # basis vectors within the unit cell
        # inverse space vectors - ALWAYS 3D vectors for consistency
        self._k1            = Backend.zeros(3)                      # inverse space vectors - reciprocal lattice vectors (3D)
        self._k2            = Backend.zeros(3)
        self._k3            = Backend.zeros(3)
        # normal vectors (along the bonds - if required)
        self._n1            = Backend.zeros((self._dim, self._dim ))# normal vectors along the bonds
        self._n2            = Backend.zeros((self._dim, self._dim ))
        self._n3            = Backend.zeros((self._dim, self._dim ))
        # nearest neighbors vectors of the cells
        self._delta_z       = np.array([0.0, 0.0, self.a])          # UP
        self._delta_x       = np.array([self.a, 0.0, 0.0])          # RIGHT
        self._delta_y       = np.array([0.0, self.a, 0.0])          # FRONT
        
        # bonds
        self._bonds         = []                                    # empty if not calculated...
        
        self._rvectors      = Backend.zeros((self._ns, 3))          # allowed values of the real space vectors
        self._kvectors      = Backend.zeros((self._ns, 3))          # allowed values of the inverse space vectors
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
        
    def __repr__(self):
        ''' Representation of the lattice '''
        return f"{self._type.name},{self._bc.name},d={self._dim},Ns={self._ns},Lx={self._lx},Ly={self._ly},Lz={self._lz}"
    
    # -----------------------------------------------------------------------------
    
    def init(self, verbose: bool = False, *, force_dft: bool = False):
        """
        Initializes the lattice object by calculating coordinates, reciprocal vectors, and neighbor lists.
        
        This method performs the following steps:
        1. Calculates the real-space coordinates, r-vectors, and k-vectors of the lattice.
        2. If the number of sites (`self.Ns`) is less than 100, computes the discrete Fourier transform (DFT) matrix.
        3. If an adjacency matrix (`self._adj_mat`) is provided:
            - Determines the number of sites (`Ns`) from the adjacency matrix.
            - For each site, identifies nearest neighbors (nn) as those connected by the highest weight in the adjacency matrix, and next-nearest neighbors (nnn) as those connected by the next highest distinct weight.
            - Stores forward neighbors (indices greater than the current site) for both nn and nnn.
        4. If no adjacency matrix is provided, calculates nearest and next-nearest neighbors using default methods.
        5. Calculates normalization or symmetry properties of the lattice.
        This method sets up all necessary neighbor lists and lattice properties required for further computations.
        """

        self.calculate_reciprocal_vectors()
        self.calculate_coordinates()
        if verbose: print(" Lattice: Calculated coordinates.")
        
        self.calculate_r_vectors()
        if verbose: print(" Lattice: Calculated r-vectors.")

        self.calculate_k_vectors()
        if verbose: print(" Lattice: Calculated k-vectors.")

        if self.Ns < Lattice._DFT_LIMIT_SITES or force_dft:
            self.calculate_dft_matrix()
            if verbose: print(" Lattice: Calculated DFT matrix.")

        if self._adj_mat is not None:
            Ns                  = self._adj_mat.shape[0]
            self._ns            = Ns
            W                   = self._adj_mat
            nn_list             = []
            nnn_list            = []
            for i in range(Ns):
                #! get sorted neighbors by weight desc (exclude self-loop)
                js              = [j for j in range(Ns) if j != i and W[i, j] != 0]
                sorted_js       = sorted(js, key=lambda j: W[i, j], reverse=True)
                if not sorted_js:
                    nn_list.append([])
                    nnn_list.append([])
                    continue
                
                #! highest weight defines nn
                max_w           = W[i, sorted_js[0]]
                nn_js           = [j for j in sorted_js if W[i, j] == max_w]
                nn_list.append(nn_js)
                
                if len(sorted_js) > len(nn_js):
                    #! find next distinct weight
                    remaining   = [W[i, j] for j in sorted_js if W[i, j] != max_w]
                    if remaining:
                        second_w    = max(remaining)
                        nnn_js      = [j for j in sorted_js if W[i, j] == second_w]
                    else:
                        nnn_js      = []
                else:
                    nnn_js = []
                nnn_list.append(nnn_js)
            self._nn            = nn_list
            self._nn_forward    = [[j for j in nn_list[i] if j>i] for i in range(Ns)]
            self._nnn           = nnn_list
            self._nnn_forward   = [[j for j in nnn_list[i] if j>i] for i in range(Ns)]
            if verbose: print(" Lattice: Calculated neighbors from adjacency matrix.")
            if verbose: print(" Lattice: Calculated forward neighbors from adjacency matrix.")
            if verbose: print(" Lattice: Calculated next-nearest neighbors from adjacency matrix.")
            if verbose: print(" Lattice: Calculated forward next-nearest neighbors from adjacency matrix.")
        else:
            self.calculate_nn()
            if verbose: print(" Lattice: Calculated nearest neighbors.")
            self.calculate_nnn()
            if verbose: print(" Lattice: Calculated next-nearest neighbors.")
        
        self.calculate_norm_sym()
        if verbose: print(" Lattice: Calculated normalization/symmetry.")
    
        # Initialize the normal vectors along the bonds
        self._n1    = self._delta_x / np.linalg.norm(self._delta_x)
        self._n2    = self._delta_y / np.linalg.norm(self._delta_y)
        self._n3    = self._delta_z / np.linalg.norm(self._delta_z)
    
    ################################### GETTERS ###################################
    
    @property
    def lx(self):           return self._lx
    @property
    def Lx(self):           return self._lx
    @lx.setter
    def lx(self, value):    self._lx = value; self._lxly = self._lx * self._ly; self._lxlylz = self._lxly * self._lz; self._lxlz = self._lx * self._lz
    @Lx.setter
    def Lx(self, value):    self._lx = value; self._lxly = self._lx * self._ly; self._lxlylz = self._lxly * self._lz; self._lxlz = self._lx * self._lz
    
    @property
    def ly(self):           return self._ly
    @property
    def Ly(self):           return self._ly
    @ly.setter
    def ly(self, value):    self._ly = value; self._lxly = self._lx * self._ly; self._lxlylz = self._lxly * self._lz; self._lylz = self._ly * self._lz
    @Ly.setter
    def Ly(self, value):    self._ly = value; self._lxly = self._lx * self._ly; self._lxlylz = self._lxly * self._lz; self._lylz = self._ly * self._lz

    @property
    def lz(self):           return self._lz
    @property
    def Lz(self):           return self._lz
    @lz.setter
    def lz(self, value):    self._lz = value; self._lxlylz = self._lxly * self._lz; self._lylz = self._ly * self._lz; self._lxlz = self._lx * self._lz
    @Lz.setter
    def Lz(self, value):    self._lz = value; self._lxlylz = self._lxly * self._lz; self._lylz = self._ly * self._lz; self._lxlz = self._lx * self._lz

    @property
    def area(self):         return self._lxly
    @property
    def volume(self):       return self._lxlylz
    @property
    def lxly(self):         return self._lxly
    @property
    def lxlz(self):         return self._lxlz
    @property
    def lylz(self):         return self._lylz
    @property
    def lxlylz(self):       return self._lxlylz
    
    @property
    def dim(self):          return self._dim
    @dim.setter
    def dim(self, value):   self._dim = value
    
    @property
    def ns(self):           return self._ns
    @property
    def Ns(self):           return self._ns
    @property
    def sites(self):        return self._ns
    @property
    def size(self):         return self._ns
    @property
    def nsites(self):       return self._ns

    @ns.setter
    def ns(self, value):    self._ns = value
    @Ns.setter
    def Ns(self, value):    self._ns = value

    # -----------------------------------------------------------------------------
    #! Physical 
    # -----------------------------------------------------------------------------
    
    @property
    def a1(self):           return self._a1
    @a1.setter
    def a1(self, value):    self._a1 = value
    
    @property
    def a2(self):           return self._a2
    @a2.setter
    def a2(self, value):    self._a2 = value

    @property
    def a3(self):           return self._a3
    @a3.setter
    def a3(self, value):    self._a3 = value

    # -----------------------------------------------------------------------------
    # Inverse space vectors
    # -----------------------------------------------------------------------------

    @property
    def k1(self):           return self._k1
    @k1.setter
    def k1(self, value):    self._k1 = value
    @property
    def b1(self):           return self._k1
    @b1.setter
    def b1(self, value):    self._k1 = value
    
    @property
    def k2(self):           return self._k2
    @k2.setter
    def k2(self, value):    self._k2 = value
    @property
    def b2(self):           return self._k2
    @b2.setter
    def b2(self, value):    self._k2 = value
    
    @property
    def k3(self):           return self._k3
    @k3.setter
    def k3(self, value):    self._k3 = value
    @property
    def b3(self):           return self._k3
    @b3.setter
    def b3(self, value):    self._k3 = value
    
    # ------------------------------------------------------------------
    
    @property
    def n1(self):           return self._n1
    @n1.setter
    def n1(self, value):    self._n1 = value

    @property
    def n2(self):           return self._n2
    @n2.setter
    def n2(self, value):    self._n2 = value
    
    @property
    def n3(self):           return self._n3
    @n3.setter
    def n3(self, value):    self._n3 = value
    
    @property
    def basis(self):        return self._basis
    @basis.setter
    def basis(self, value): self._basis = value
    @property
    def multipartity(self): return self._basis.shape[0]
    
    @property
    def vectors(self):      return self._vectors
    @vectors.setter
    def vectors(self, value): self._vectors = value
    
    @property
    def avec(self):         return np.stack((self._a1, self._a2, self._a3), axis=0)
    @avec.setter
    def avec(self, value):  self._a1 = value[0]; self._a2 = value[1]; self._a3 = value[2]
    
    @property
    def bvec(self):         return np.stack((self._k1, self._k2, self._k3), axis=0)
    @bvec.setter
    def bvec(self, value):  self._k1 = value[0]; self._k2 = value[1]; self._k3 = value[2]

    # ------------------------------------------------------------------
    #! DFT Matrix
    # ------------------------------------------------------------------
    
    @property
    def dft(self):                  return self._dft
    @dft.setter
    def dft(self, value):           self._dft = value
    
    @property
    def nn(self):                   return self._nn
    @nn.setter
    def nn(self, value):            self._nn = value

    @property
    def bonds(self):                return self._bonds
    @bonds.setter
    def bonds(self, value):         self._bonds = value

    @property
    def nn_forward(self):           return self._nn_forward
    @nn_forward.setter
    def nn_forward(self, value):    self._nn_forward = value

    @property
    def nnn(self):                  return self._nnn
    @nnn.setter
    def nnn(self, value):           self._nnn = value
    
    @property
    def nnn_forward(self):          return self._nnn_forward
    @nnn_forward.setter
    def nnn_forward(self, value):   self._nnn_forward = value
    
    @property
    def coordinates(self):          return self._coords
    @coordinates.setter
    def coordinates(self, value):   self._coords = value
    
    @property
    def subs(self):                 return self._subs
    @subs.setter
    def subs(self, value):          self._subs = value
    
    @property
    def cells(self):                return self._cells
    @cells.setter
    def cells(self, value):         self._cells = value
    
    @property
    def fracs(self):                return self._fracs
    @fracs.setter
    def fracs(self, value):         self._fracs = value

    @property
    def kvectors(self):             return self._kvectors
    @kvectors.setter
    def kvectors(self, value):      self._kvectors = value
    
    @property
    def rvectors(self):             return self._rvectors
    @rvectors.setter
    def rvectors(self, value):      self._rvectors = value
    
    @property
    def bc(self):                   return self._bc
    @bc.setter
    def bc(self, value):            self._bc = value
    
    @property
    def cardinality(self):          return self.get_nn_forward_num_max()
    @cardinality.setter
    def cardinality(self, value):   self._nn_max_num = value
    
    # ------------------------------------------------------------------
    #! Sublattice
    # ------------------------------------------------------------------
    
    def sublattice(self, site: int) -> int:
        """
        Return the sublattice index for a given site.
        By default, returns 0 for all sites (single sublattice).
        Override in subclasses for multi-sublattice lattices.
        """
        return site % self.multipartity
    
    # ------------------------------------------------------------------
    #! K-space
    # ------------------------------------------------------------------
    
    def k_vector(self, qx, qy=0.0, qz=0.0) -> np.ndarray:
        """
        Return the k-vector in Cartesian coordinates for given (qx, qy, qz)
        in reciprocal lattice units.
        """
        if self.k1 == None or self.k2 == None or self.k3 == None:
            self.k1, self.k2, self.k3 = reciprocal_from_real(self.a1, self.a2, self.a3)
        
        kvec = qx * self.k1[0,:]
        if self.dim > 1:
            kvec += qy * self.k2[0,:]
        if self.dim > 2:
            kvec += qz * self.k3[0,:]
        return kvec
    
    def extract_momentum(self, 
                        eigvecs    : np.ndarray, 
                        *,
                        eigvals     : np.ndarray = None,
                        tol         : float      = 1e-10,
                        ) -> np.ndarray:
        """
        Extract crystal momentum vectors k from real-space eigenvectors.
        """
        return extract_momentum(eigvecs, self, eigvals=eigvals, tol=tol)

    def extract_bz_path_data(self, 
                            k_vectors           : np.ndarray,
                            k_vectors_frac      : np.ndarray,
                            values              : np.ndarray,
                            path                : Union[PathTypes, str, StandardBZPath]  = StandardBZPath.HONEYCOMB_2D,
                            ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], np.ndarray]:
        """
        Extract k-point path data in the Brillouin zone for band structure calculations.
        Parameters:
        -----------
        k_vectors : np.ndarray
            Array of k-vectors in Cartesian coordinates.
        values : np.ndarray
            Corresponding values (e.g., energies) at each k-vector.
        path : Union[PathTypes, str, StandardBZPath], optional
            Predefined path type or custom path specification. Default is StandardBZPath.HONEYCOMB_2D.
        mode : str, optional
            Mode of extraction: 'discrete' for discrete points, 'interpolated' for interpolated path. Default is 'discrete'.
        points_p_segment : int, optional
            Number of points per segment for interpolation. Default is 10.
        tol : float, optional
            Tolerance for numerical comparisons. Default is 1e-8.
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], np.ndarray]
            Tuple containing:
            - k_path : np.ndarray
                Array of k-vectors along the specified path.
            - values_path : np.ndarray
                Corresponding values along the k-path.
            - labels : List[Tuple[int, str]]
                List of tuples with indices and labels of high-symmetry points.
            - distances : np.ndarray
                Cumulative distances along the k-path.
        """
        return extract_bz_path_data(self, k_vectors, k_vectors_frac, values, path)
    
    # ------------------------------------------------------------------
    #! Boundary fluxes
    # ------------------------------------------------------------------
    
    @property
    def flux(self) -> BoundaryFlux: 
        return self._flux
    
    @flux.setter
    def flux(self, value: Optional[Union[float, Mapping[Union[str, LatticeDirection], float]]]):
        self._flux = _normalize_flux_dict(value)

    def boundary_phase(self, direction: LatticeDirection, winding: int = 1) -> complex:
        """
        Return the complex phase accumulated after crossing the boundary along ``direction``.
        """
        return self._flux.phase(direction, winding=winding)

    def boundary_phase_from_winding(self, wx: int, wy: int, wz: int) -> complex:
        """
        Return total complex boundary phase accumulated from winding numbers.
        If no winding (all zero), returns real 1.0.
        """
        if wx == 0 and wy == 0 and wz == 0:
            return 1.0
        phase = 1.0
        if wx != 0:
            phase *= self.boundary_phase(LatticeDirection.X, winding=wx)
        if wy != 0:
            phase *= self.boundary_phase(LatticeDirection.Y, winding=wy)
        if wz != 0:
            phase *= self.boundary_phase(LatticeDirection.Z, winding=wz)
        return phase if np.iscomplexobj(phase) and not np.isreal(phase) else float(np.real(phase))

    def bond_winding(self, i: int, j: int) -> tuple[int, int, int]:
        """
        Compute how many times a bond (i -> j) crosses the periodic boundary
        in each lattice direction.
        
        Returns (wx, wy, wz), where each entry is 0 if no crossing,
        +1 if wrapped positively, -1 if wrapped negatively.
        
        Parameters:
        -----------
        i : int
            Index of the starting lattice site.
        j : int
            Index of the ending lattice site.
        Returns:
        --------
        tuple[int, int, int]
            A tuple indicating the winding numbers (wx, wy, wz) for the bond from site i to site j.
        """
        x1, y1, z1  = self.get_coordinates(i)
        x2, y2, z2  = self.get_coordinates(j)
        wx          = 0
        wy          = 0
        wz          = 0

        # detect wrapping based on system size
        if abs(x2 - x1) > self.Lx // 2:                     # assume even sizes, we wrap when crossing half the system
            wx = -1 if x2 > x1 else +1
        if self.dim > 1 and abs(y2 - y1) > self.Ly // 2:    # assume even sizes, we wrap when crossing half the system
            wy = -1 if y2 > y1 else +1
        if self.dim > 2 and abs(z2 - z1) > self.Lz // 2:    # assume even sizes, we wrap when crossing half the system
            wz = -1 if z2 > z1 else +1

        return (wx, wy, wz)

    # ------------------------------------------------------------------
    #! Chirality helpers
    # ------------------------------------------------------------------
    
    def get_nnn_middle_sites(self, i: int, j: int,
                             orientation: Optional[str] = None) -> list[int]:
        """
        Return the list of 'middle' sites l that are nearest neighbors
        of both i and j - i.e., sites forming two-step NNN paths i-l-j.

        Works for any lattice that implements get_nn(site, idx)
        and get_nn_num(site).

        Parameters
        ----------
        i, j : int
            Site indices.
        orientation : {'anticlockwise', 'clockwise', None}, optional
            If provided, will sort/choose based on geometric angle.
            Default: None (return all middle sites).

        Returns
        -------
        list[int]
            List of middle-site indices (can be 0, 1, or 2 elements).
        """
        nn_i = [self.get_nn(i, k) for k in range(self.get_nn_num(i))]
        nn_j = [self.get_nn(j, k) for k in range(self.get_nn_num(j))]
        mids = list(set(nn_i).intersection(nn_j))
        if not mids or orientation is None:
            return mids

        # Optional: choose one by local geometry
        if len(mids) > 1:
            ri      = np.array(self.get_coordinates(i))
            rj      = np.array(self.get_coordinates(j))
            centers = []
            for l in mids:
                rl      = np.array(self.get_coordinates(l))
                cross_z = np.cross(rl - ri, rj - rl)[-1]
                centers.append((l, cross_z))
            if orientation.lower().startswith("anti"):
                mids    = [l for (l, cz) in centers if cz > 0]
            elif orientation.lower().startswith("clock"):
                mids    = [l for (l, cz) in centers if cz < 0]
        return mids

    def get_chirality_sign(self, i: int, j: int, normal: Optional[np.ndarray] = None, orientation: Optional[str] = None) -> int:
        r"""
        Compute the local orientation (chirality) sign \nu_{ij} = \pm 1 for a NNN pair (i,j),
        defined by the cross product of the two bond vectors i-l and l-j.

        Works for any 2D or quasi-2D lattice with known site coordinates.

        Parameters
        ----------
        i, j : int
            Site indices (next-nearest neighbors).
        normal : np.ndarray, optional
            Orientation of the lattice plane (default: +z for 2D).

        Returns
        -------
        int
            +1 for anticlockwise, -1 for clockwise, 0 if not a valid NNN pair.
        """
        if self.dim < 2:
            raise ValueError("Chirality sign is only defined for 2D or higher-dimensional lattices.")
        
        # Default normal: +z
        if normal is None:
            normal = np.array([0, 0, 1.0])

        # find common neighbor(s)
        mids = self.get_nnn_middle_sites(i, j, orientation=orientation)
        if not mids:
            return 0

        # Choose one middle site (if multiple, pick the first or average)
        l       = mids[0]
        ri      = np.array(self.get_coordinates(i), dtype=float)
        rj      = np.array(self.get_coordinates(j), dtype=float)
        rl      = np.array(self.get_coordinates(l), dtype=float)

        d1      = rl - ri
        d2      = rj - rl
        cross   = np.cross(d1, d2)
        sign    = np.sign(np.dot(cross, normal))

        return int(sign)

    def bond_type(self, i: int, j: int) -> str:
        """
        Determine the bond type between sites i and j.

        Parameters
        ----------
        i, j : int
            Site indices.

        Returns
        -------
        str
            'nn' for nearest neighbor, 'nnn' for next-nearest neighbor, 'none' otherwise.
        """
        if j in self.nn[i]:
            return 'nn'
        elif j in self.nnn[i]:
            return 'nnn'
        else:
            return 'none'

    # ------------------------------------------------------------------
    #! Boundary helpers
    # ------------------------------------------------------------------

    def periodic_flags(self) -> Tuple[bool, bool, bool]:
        """
        Return booleans indicating whether (x, y, z) directions are periodic.
        """
        match self._bc:
            case LatticeBC.PBC:
                return True, True, True
            case LatticeBC.OBC:
                return False, False, False
            case LatticeBC.MBC:
                return True, False, False
            case LatticeBC.SBC:
                return False, True, False
            case _:
                raise ValueError(f"Unsupported boundary condition {self._bc!r}")

    def is_periodic(self, direction: LatticeDirection) -> bool:
        """
        Check if a given direction has periodic boundary conditions.
        """
        flags = self.periodic_flags()
        index = {LatticeDirection.X: 0, LatticeDirection.Y: 1, LatticeDirection.Z: 2}[direction]
        return bool(flags[index])
    
    @property
    def typek(self):                return self._type
    @typek.setter
    def typek(self, value):         self._type = value
    
    @property
    def spatial_norm(self):         return self._spatial_norm
    @spatial_norm.setter
    def spatial_norm(self, value):  self._spatial_norm = value

    # -----------------------------------------------------------------------------
    
    @abstractmethod
    def site_index(self, x : int, y : int, z : int): pass

    # -----------------------------------------------------------------------------
    #! SITE HELPERS
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
    #! DFT MATRIX
    # -----------------------------------------------------------------------------

    def calculate_reciprocal_vectors(self):
        '''
        Calculates the reciprocal lattice vectors based on the primitive vectors.
        Always returns 3D vectors (padding with zeros for lower dimensions).
        
        Returns:
        - k1, k2, k3 : Reciprocal lattice vectors (always 3D)
        '''
        self._k1, self._k2, self._k3 = reciprocal_from_real(self.a1, self.a2, self.a3)
        
        # Ensure 3D form (pad zeros if 1D or 2D)
        self._k1    = np.pad(self._k1, (0, 3 - len(self._k1)))
        self._k2    = np.pad(self._k2, (0, 3 - len(self._k2)))
        self._k3    = np.pad(self._k3, (0, 3 - len(self._k3)))
        return self._k1, self._k2, self._k3
    
    def calculate_dft_matrix(self, phase = False, use_fft: bool = False) -> np.ndarray:
        r'''
        Bloch-type DFT matrix on the site basis.
        
        Indices:
            i  = (R, beta)   real-space cell R and sublattice beta
            n  = (k, alpha)  k-point k and sublattice alpha

        Elements:
        $$
            F_{(k,alpha),(R,beta)} =
                1/sqrt(Nc) * delta_{alpha,beta} * exp(-i k . R).
        $$
        This is unitary:
        $$
            F^\dagger F = I_{Ns},   F F^\dagger = I_{Ns},
        $$
        where Ns = Nc * Nb is the total number of sites, Nc is the number of
        unit cells, and Nb is the number of sublattices.
        
        Note that this DFT matrix does not include basis-dependent phases
        (i.e., exp(-i k . r_basis)).
        
        Calculates the Discrete Fourier Transform (DFT) matrix for the lattice.
        This method can be optimized using FFT (Fast Fourier Transform) in the future.
        Reference: https://en.wikipedia.org/wiki/DFT_matrix
        
        Args:
        - phase (bool): If True, adds a complex phase to the k-vectors.
        
        Returns:
        - DFT matrix (ndarray): The calculated DFT matrix.
        '''
        Ns              = self.Ns
        Lx, Ly, Lz      = self._lx, max(self._ly, 1), max(self._lz, 1)
        Nc              = Lx * Ly * Lz
        Nb              = len(self._basis) if (self._basis is not None and len(self._basis) > 0) else 1  # Avoid division by zero
        
        # Get site coordinates
        # r_vectors       = np.asarray(self.coordinates, dtype=float)  # (Ns, 3)
        r_vectors       = np.asarray(self.cells, dtype=float)  # (Ns, 3)
        if not r_vectors.shape[0] == Ns:
            raise ValueError("Mismatch in number of sites and coordinates.")

        sub_idx         = self.subs # (Ns,)

        # Generate k-vectors
        frac_x          = np.linspace(0, 1, Lx, endpoint=False)
        frac_y          = np.linspace(0, 1, Ly, endpoint=False)
        frac_z          = np.linspace(0, 1, Lz, endpoint=False)
        
        kx_frac, ky_frac, kz_frac = np.meshgrid(frac_x, frac_y, frac_z, indexing='ij')
        
        b1              = np.asarray(self._k1, float).reshape(3)
        b2              = np.asarray(self._k2, float).reshape(3)
        b3              = np.asarray(self._k3, float).reshape(3)
        
        kgrid           = (kx_frac[..., None] * b1 + 
                           ky_frac[..., None] * b2 + 
                           kz_frac[..., None] * b3)
        k_vectors       = kgrid.reshape(-1, 3) # (Nc, 3)
        
        # Build block DFT matrix
        # Row index: ik*Nb + \alpha (k-point ik, sublattice \alpha)
        # Col index: i (site i in real space)
        F_block         = np.zeros((Nc * Nb, Ns), dtype=complex)    # DFT matrix
        norm            = np.sqrt(Nc)                               # Bloch normalization factor

        # numpy path version (not used in loop)
        phase_matrix    = np.exp(-1j * (k_vectors @ r_vectors.T)) / norm    # (Nc, Ns)
        selector        = (sub_idx[None, :] == np.arange(Nb)[:, None])      # (Nb, Ns)
        F               = phase_matrix[:, None, :] * selector[None, :, :]   # (Nc,Nb,Ns)
        F_block         = F.reshape(Nc * Nb, Ns)
        
        self._dft       = F_block
        return F_block

        # leave loop path
        # for ik in range(Nc):
        #     # Phases for all sites at this k
        #     k       = k_vectors[ik]
        #     phases  = np.exp(-1j * (k @ r_vectors.T)) / norm  # (Ns,)
            
        #     # Fill rows for this k-point (one row per sublattice)
        #     for alpha in range(Nb):
        #         row_idx = ik * Nb + alpha
                
        #         # Only connect to sites of sublattice \alpha
        #         for i in range(Ns):
        #             if sub_idx[i] == alpha:
        #                 F_block[row_idx, i] = phases[i]
        
        # return F_block
    
    # -----------------------------------------------------------------------------
    #! NEAREST NEIGHBORS
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

    # -----------------------------------------------------------------------------
    #! BOUNDARY CONDITIONS HELPERS
    # -----------------------------------------------------------------------------

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
    #! Virtual methods
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
    
    def get_nnn_direction(self, site : int, direction : LatticeDirection):
        '''
        Returns the next nearest neighbors in a given direction.
        '''
        pass
    
    # -----------------------------------------------------------------------------
    #! NEAREST NEIGHBORS HELPERS
    # -----------------------------------------------------------------------------
    
    def wrong_nei(self, nei):
        """
        Check if a given neighbor index is invalid.

        A neighbor is considered invalid if it is:
            - None
            - Equal to self.bad_lattice_site
            - NaN (not a number)
            - Less than 0

        Parameters
        ----------
        nei : Any
            The neighbor index to check.

        Returns
        -------
        bool
            True if the neighbor index is invalid, False otherwise.
        """
        return  nei is None or                  \
                nei == self.bad_lattice_site or \
                np.isnan(nei) or                \
                nei < 0
    
    def get_nn_num(self, site : int):
        '''
        Returns the number of nearest neighbors of a given site.
        
        Args:
        - site : lattice site
        Returns:
        - number of nearest neighbors
        '''
        if self._nn is None:
            return 0
        return len(self.nn[site])
    
    def get_nn(self, site, num : int = -1):
        '''
        Returns the nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of nearest neighbors
        Returns:
            - list of nearest neighbors        
        '''
        if num < 0:
            return self._nn[site]
        if self._nn is None:
            return []
        return self._nn[site][num]
    
    def get_nnn_num(self, site : int):
        '''
        Returns the number of next nearest neighbors of a given site.
        
        Args:
        - site : lattice site
        Returns:
        - number of next nearest neighbors
        '''
        if self._nnn is None:
            return 0
        return len(self._nnn[site])
    
    def get_nnn(self, site, num : int = -1):
        '''
        Returns the next nearest neighbors of a given site.
        
        Args:
            - site : lattice site
            - num  : number of next nearest neighbors
        Returns:
            - list of next nearest neighbors        
        '''
        if num < 0:
            return self._nnn[site]
        if self._nnn is None:
            return []
        return self._nnn[site][num]
    
    # -----------------------------------------------------------------------------
    #! FORWARD NEAREST NEIGHBORS HELPERS
    # -----------------------------------------------------------------------------
    
    def get_nn_forward_num_max(self):
        '''
        Returns the maximum number of forward nearest neighbors in the lattice.
        
        Returns:
        - maximum number of nearest neighbors
        '''
        if (self._nn_max_num is None or self._nn_max_num == 0) and self.nn_forward is not None:
            max_nn = 0
            for site in range(self.ns):
                nn_num = len(self.nn_forward[site])
                if nn_num > max_nn:
                    max_nn = nn_num
            self._nn_max_num = max_nn
        return self._nn_max_num
    
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
    #! FORWARD NEXT NEAREST NEIGHBORS HELPERS
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
    #! GENERAL NEIGHBORS HELPERS
    # -----------------------------------------------------------------------------
    
    def neighbors(self, site: int, order=1):
        '''Return neighbors of a site: 1 for nn (all with highest weight), 2 for nnn (all with second-highest), 'all' for both.'''
        if order == 1:
            return self._nn[site]
        elif order == 2:
            return self._nnn[site]
        elif order == 'all':
            if self._adj_mat is not None:
                # return all neighbors from adjacency matrix
                non_zero_indices = np.nonzero(self._adj_mat[site])[0]
                return [i for i in non_zero_indices if i != site]
            else:
                return list(set(self._nn[site]) | set(self._nnn[site]))
        else:
            raise ValueError(f"Invalid neighbor order: {order}")
        
    def neighbors_forward(self, site: int, order=1):
        '''Return forward neighbors of a site: 1 for nn (all with highest weight), 2 for nnn (all with second-highest), 'all' for both.'''
        if order == 1:
            return self._nn_forward[site]
        elif order == 2:
            return self._nnn_forward[site]
        elif order == 'all':
            if self._adj_mat is not None:
                # return all neighbors from adjacency matrix
                non_zero_indices = np.nonzero(self._adj_mat[site])[0]
                return [i for i in non_zero_indices if i != site and i > site]
            else:
                return list(set(self._nn_forward[site]) | set(self._nnn_forward[site]))
        else:
            raise ValueError(f"Invalid neighbor order: {order}")
    
    def any_neighbor(self, site: int, order=1):
        '''Return any neighbor (first) of given order or None.'''
        neigh = self.neighbors(site, order)
        return neigh[0] if neigh else Lattice._BAD_LATTICE_SITE
    
    def any_neighbor_forward(self, site: int, order=1):
        '''Return any forward neighbor (first) of given order or None.'''
        neigh = self.neighbors_forward(site, order)
        return neigh[0] if neigh else Lattice._BAD_LATTICE_SITE
    
    # -----------------------------------------------------------------------------
    #! Standard getters
    # -----------------------------------------------------------------------------

    def get_coordinates(self, *args):           return self.coordinates if len(args) == 0 else self.coordinates[args[0]]
    def get_r_vectors(self,*args):              return self.rvectors if len(args) == 0 else self.rvectors[args[0]]
    def get_k_vectors(self, *args):             return self.kvectors if len(args) == 0 else self.kvectors[args[0]]
    def get_site_diff(self, i: int, j: int):    return self.get_coordinates(j) - self.get_coordinates(i)
    def get_k_vec_idx(self, sym = False):       pass

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
    
    # -----------------------------------------------------------------------------
    #! Spatial information
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
    
    def calculate_bonds(self):
        '''
        Calculates the bonds for the lattice using forward nn.
        '''
        self._bonds = []
        for i in range(self.Ns):
            for idx in range(self.get_nn_forward_num(i)):
                j = self.get_nn_forward(i, idx)
                if self.wrong_nei(j):
                    continue
                self._bonds.append((i, j))
        return self._bonds
    
    def calculate_coordinates(self):
        """
        Calculates the coordinates for each lattice site in up to 3D.

        Each site index i corresponds to:
            cell = i // n_basis
            sub  = i % n_basis
        where n_basis = len(self._basis) (e.g., 2 for honeycomb).
        
        Works for any lattice with defined self._a1, _a2, _a3 and self._basis list.
        """
        n_basis             = len(self._basis)
        self.coordinates    = []
        self.cells          = []
        self.fracs          = []
        self.subs           = []

        for i in range(self.Ns):
            cell    = i // n_basis          # integer division
            sub     = i % n_basis           # remainder

            nx      =  cell              % self.Lx
            ny      = (cell // self.Lx)  % self.Ly if self._dim >= 2 else 0
            nz      = (cell // (self.Lx  * self.Ly)) % self.Lz if self._dim >= 3 else 0

            R       = nx * self._a1 + ny * self._a2 + nz * self._a3     # lattice vector
            r       = R + self._basis[sub]                              # add basis vector
            self.coordinates.append(r)
            self.cells.append(R)
            self.fracs.append((nx, ny, nz))
            self.subs.append(sub)
            

        self.coordinates    = np.array(self.coordinates)
        self.cells          = np.array(self.cells)
        self.fracs          = np.array(self.fracs)
        self.subs           = np.array(self.subs)
        return self.coordinates
        
    def calculate_r_vectors(self):
        """
        Calculates the real-space vectors (r) for each site.
        Must match the ordering in calculate_coordinates().
        """
        n_basis = len(self._basis)
        rv = np.zeros((self.Ns, 3))
        
        for i in range(self.Ns):
            cell    = i // n_basis
            sub     = i % n_basis
            
            nx      = cell % self.Lx
            ny      = (cell // self.Lx) % self.Ly if self._dim >= 2 else 0
            nz      = (cell // (self.Lx * self.Ly)) % self.Lz if self._dim >= 3 else 0
            
            rv[i]   = nx * self._a1 + ny * self._a2 + nz * self._a3 + self._basis[sub]
        
        self.rvectors = rv
        return self.rvectors

    def calculate_k_vectors(self):
        """
        Calculates the allowed reciprocal-space k-vectors (momentum grid)
        consistent with the lattice size and primitive reciprocal vectors.

        The sampling follows the same fftfreq ordering used by the Bloch
        transform (Γ at index [0,0,0], followed by positive frequencies and
        finally the negative branch).  This keeps the analytic grids aligned
        with the numerically constructed H(k) blocks.
        """
        Lx      = self.Lx
        Ly      = self.Ly if self._dim >= 2 else 1
        Lz      = self.Lz if self._dim >= 3 else 1

        frac_x  = np.fft.fftfreq(Lx)
        frac_y  = np.fft.fftfreq(Ly)
        frac_z  = np.fft.fftfreq(Lz)

        kx_frac, ky_frac, kz_frac = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")

        k_grid = (
              kx_frac[..., None] * self._k1
            + ky_frac[..., None] * self._k2
            + kz_frac[..., None] * self._k3
        )

        self.kvectors       = k_grid.reshape(-1, 3)
        self.kvectors_frac  = np.stack([kx_frac, ky_frac, kz_frac], axis=-1).reshape(-1, 3)
        return self.kvectors

    def translation_operators(self):
        """Return translation matrices T1, T2, T3 on the one-hot basis."""
        self._T1, self._T2, self._T3 = build_translation_operators(self)
        return self._T1, self._T2, self._T3

    # -----------------------------------------------------------------------------
    #! Spatial norm calculators
    # -----------------------------------------------------------------------------

    @abstractmethod
    def calculate_norm_sym(self):       pass
    
    # -----------------------------------------------------------------------------
    #! Nearest neighbors
    # -----------------------------------------------------------------------------
    
    def _calculate_nn_pbc(self):        return self.calculate_nn_in(True, True, True)
    def _calculate_nn_obc(self):        return self.calculate_nn_in(False, False, False)
    def _calculate_nn_mbc(self):        return self.calculate_nn_in(True, False, False)
    def _calculate_nn_sbc(self):        return self.calculate_nn_in(False, True, False)
    
    @abstractmethod
    def calculate_nn_in(self, pbcx : bool, pbcy : bool, pbcz : bool): pass

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

    def calculate_plaquettes(self):     raise NotImplementedError("Plaquette calculation not implemented for this lattice.")

    # -----------------------------------------------------------------------------
    #! Next nearest neighbors
    # -----------------------------------------------------------------------------
    
    def _calculate_nnn_pbc(self):       return self.calculate_nnn_in(True, True, True)
    def _calculate_nnn_obc(self):       return self.calculate_nnn_in(False, False, False)
    def _calculate_nnn_mbc(self):       return self.calculate_nnn_in(True, False, False)
    def _calculate_nnn_sbc(self):       return self.calculate_nnn_in(False, True, False)
    
    @abstractmethod
    def calculate_nnn_in(self, pbcx : bool, pbcy : bool, pbcz : bool): pass
    
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
    #! Saving the lattice
    # -----------------------------------------------------------------------------
    
    def adjacency_matrix(self, sparse: bool = False, save: bool = True) -> np.ndarray:
        r"""
        Construct adjacency matrix A_ij = 1 if i and j are neighbors.

        Args:
            sparse      (bool):
                return a scipy.sparse CSR matrix if True.

        Returns:
            A (ndarray or sparse CSR): adjacency matrix of size (Ns, Ns).
        """
        if self._adj_mat is None:
            # fallback to binary from lists
            Ns      = self.ns
            rows    = []
            cols    = []
            data    = []
            
            for i in range(Ns):
                if self._nn and self._nn[i]:
                    for jj, j in enumerate(self._nn[i]):
                        if self._nn[i][jj] and not np.isnan(self._nn[i][jj]) and self._nn[i][jj] > 0:
                            rows.append(i)
                            cols.append(j)
                            data.append(3)
                if self._nnn and self._nnn[i]:
                    for jj, j in enumerate(self._nnn[i]):
                        if self._nnn[i][jj] and not np.isnan(self._nnn[i][jj]) and self._nnn[i][jj] > 0:
                            rows.append(i)
                            cols.append(j)
                            data.append(2)
                # diagonal
                rows.append(i)
                cols.append(i)
                data.append(1)
                
            # remove duplicates
            rows, cols, data    = np.unique((rows, cols, data), axis=1)
            # # remove nans
            # rows, cols, data    = rows[~np.isnan(data)], cols[~np.isnan(data)], data[~np.isnan(data)]
            # as integers
            rows, cols, data    = rows.astype(int), cols.astype(int), data
            
            if sparse:
                A = sp.csr_matrix((data, (rows, cols)), shape=(Ns, Ns))
            else:
                A               = np.zeros((Ns, Ns))
                A[rows, cols]   = data
                A[cols, rows]   = data
        else:
            if sparse:
                A = self._adj_mat if isinstance(self._adj_mat, sp.csr_matrix) else sp.csr_matrix(self._adj_mat)
            else:
                A = self._adj_mat.toarray() if isinstance(self._adj_mat, sp.csr_matrix) else self._adj_mat
        if save:
            self._adj_mat = A
        return A
    
    def print_neighbors(self, logger : 'Logger'):
        """
        Logs the neighbors of each site in the lattice using the provided logger.

        For each site in the lattice, this method retrieves its nearest neighbors and logs their indices.
        Additionally, for each neighbor, it logs detailed information using a higher verbosity level.

        Args:
            logger: An object with an `info` method for logging messages. The `info` method should accept
                    parameters `lvl` (int) for verbosity level and `color` (str) for message color.

        """
        def print_nei(msg, lvl = 1, color = 'green'):
            if logger is not None:
                logger.info(msg, lvl = lvl, color = color)
            else:
                print(msg)
        
        for i in range(self.ns):
            neighbors = self.get_nn(i)
            print_nei(f"Neighbors of site {i}: {neighbors}", lvl = 1, color = 'green')
            for j in range(len(neighbors)):
                nei_in = self.get_nei(i, j)
                print_nei(f"Neighbor {j} of site {i}: {nei_in}", lvl = 2, color = 'blue')

    def print_forward(self, logger : 'Logger'):
        """
        Logs the forward nearest neighbors for each site in the lattice.

        For each site in the lattice, this method retrieves the number of forward nearest neighbors
        and logs their indices using the provided logger. The method outputs two levels of information:
        - Level 1 (green): Lists the neighbors of each site.
        - Level 2 (blue): Details each neighbor's index for the site.

        Args:
            logger: A logging object with an `info` method that accepts a message, 
                    a logging level (`lvl`), and a color (`color`).
        """
        
        def print_nei(msg, lvl = 1, color = 'green'):
            if logger is not None:
                logger.info(msg, lvl = lvl, color = color)
            else:
                print(msg)
        
        for i in range(self.ns):
            neighbors = self.get_nn_forward_num(i)
            print_nei(f"Neighbors of site {i}: {neighbors}", lvl = 1, color = 'green')
            for j in range(neighbors):
                nei_in = self.get_nn_forward(i, j)
                print_nei(f"Neighbor {j} of site {i}: {nei_in}", lvl = 2, color = 'blue')

    # -------------------------------------------------------------------------
    #! Bloch Transform & Basis Operations
    # -------------------------------------------------------------------------

    def get_geometric_encoding(self, *, tol=1e-6):
        """
        Map each site i to (cell_idx, sub_idx) purely from geometry.

        Returns
        -------
        cell_idx : (Ns,) int array in [0, Nc-1]
        sub_idx  : (Ns,) int array in [0, Nb-1]
        """

        coords      = np.asarray(self.coordinates, float)          # (Ns,3)
        a1          = np.asarray(self._a1, float).reshape(3)
        a2          = np.asarray(self._a2, float).reshape(3)
        a3          = np.asarray(self._a3, float).reshape(3)
        A           = np.column_stack([a1, a2, a3])                # (3,3)
        Ainv        = np.linalg.inv(A)

        Nb          = len(self._basis)
        basis       = np.zeros((Nb,3), float)
        basis[:, :self._basis.shape[1]] = np.asarray(self._basis, float)

        Lx, Ly, Lz  = self._lx, max(self._ly,1), max(self._lz,1)
        Nc          = Lx*Ly*Lz
        Ns          = coords.shape[0]

        # fractional cell coords (may be non-integers due to numeric noise)
        frac        = (Ainv @ coords.T).T                           # (Ns,3)
        # wrap to [0,L) and round to nearest cell
        cx          = np.mod(np.rint(frac[:,0]), Lx).astype(int)
        cy          = np.mod(np.rint(frac[:,1]), max(Ly,1)).astype(int) if self._dim >= 2 else np.zeros(Ns, int)
        cz          = np.mod(np.rint(frac[:,2]), max(Lz,1)).astype(int) if self._dim >= 3 else np.zeros(Ns, int)

        # residual within unit cell
        Rrec        = (cx[:,None]*a1[None,:] + cy[:,None]*a2[None,:] + cz[:,None]*a3[None,:])   # (Ns,3)
        r_in        = coords - Rrec

        # assign sublattice by nearest basis vector
        d2          = ((r_in[:,None,:] - basis[None,:,:])**2).sum(axis=2)   # (Ns,Nb)
        sub         = np.argmin(d2, axis=1)
        if not np.all(np.take_along_axis(d2, sub[:,None], axis=1)[:,0] < tol):
            # If this trips, increase tol or check a1,a2,a3/basis consistency
            raise ValueError("Some sites could not be matched to a basis position; adjust tolerance or geometry.")

        cell        = ((cz*Ly + cy)*Lx + cx).astype(int)            # (Ns,)
        return cell, sub

    # ============================================================
    #  INVERSE BLOCH TRANSFORM & K-SPACE OPERATIONS
    # ============================================================

    def realspace_from_kspace(
        self,
        H_k                     : np.ndarray,
        *,
        block_diag              : bool = True,
        kgrid                   : Optional[np.ndarray] = None):
        """
        Inverse Bloch transform: H(k) blocks -> H_real (Ns x Ns).
        
        See lattice_kspace.realspace_from_kspace for full documentation.
        """
        from .tools.lattice_kspace import realspace_from_kspace, full_k_space_transform
        if block_diag is False:
            return full_k_space_transform(lattice=self, mat_k=H_k, inverse=True)
        return realspace_from_kspace(
            lattice = self,
            H_k     = H_k,
            kgrid   = kgrid)

    def kspace_from_realspace(self, mat: np.ndarray, block_diag: bool = False):
        """
        Transform real-space Hamiltonian to k-space.
        
        Parameters
        ----------
        mat : np.ndarray
            Real-space matrix (Ns x Ns)
        block_diag : bool
            If True, return k-space blocks (Lx, Ly, Lz, Nb, Nb)
            If False, return full transformed matrix (Ns x Ns)
            
        Returns
        -------
        If block_diag=True:
            H_k, k_grid, k_frac : k-space blocks and grid
        If block_diag=False:
            H_k_full : full transformed matrix (Ns x Ns)
        """
        from .tools.lattice_kspace import full_k_space_transform, kspace_from_realspace
        if block_diag:
            return kspace_from_realspace(lattice=self, H_real=mat)
        return full_k_space_transform(lattice=self, mat=mat)

    # -------------------------------------------------------------------------
    #! Presentation helpers (text / plots)
    # -----------------------------------------------------------------------------

    def summary_string(self, *, precision: int = 3) -> str:
        """
        Return a textual summary of lattice metadata.
        """
        from .visualization import format_lattice_summary

        return format_lattice_summary(self, precision=precision)

    def real_space_table(self, *, max_rows: int = 10, precision: int = 3) -> str:
        """
        Return a formatted table of real-space vectors.
        """
        from .visualization import format_real_space_vectors

        return format_real_space_vectors(self, max_rows=max_rows, precision=precision)

    def reciprocal_space_table(self, *, max_rows: int = 10, precision: int = 3) -> str:
        """
        Return a formatted table of reciprocal-space vectors.
        """
        from .visualization import format_reciprocal_space_vectors

        return format_reciprocal_space_vectors(self, max_rows=max_rows, precision=precision)

    def brillouin_zone_overview(self, *, precision: int = 3) -> str:
        """
        Return a textual overview of the sampled Brillouin zone.
        """
        from .visualization import format_brillouin_zone_overview

        return format_brillouin_zone_overview(self, precision=precision)

    def describe(self, *, 
                precision               : int = 3,
                max_rows                : int = 10,
                include_vectors         : bool = True,
                include_reciprocal      : bool = True,
                include_brillouin_zone  : bool = True) -> str:
        """
        Combine multiple presentation helpers into a single multi-section string.
        """
        sections: list[str] = [self.summary_string(precision=precision)]

        if include_vectors:
            sections.append("Real-space vectors:\n" + self.real_space_table(max_rows=max_rows, precision=precision))

        if include_reciprocal:
            sections.append(
                "Reciprocal-space vectors:\n" + self.reciprocal_space_table(max_rows=max_rows, precision=precision)
            )

        if include_brillouin_zone:
            sections.append("Brillouin zone:\n" + self.brillouin_zone_overview(precision=precision))

        return "\n\n".join(sections)

    # -----------------------------------------------------------------------------
    #! PLOTTING HELPERS
    # -----------------------------------------------------------------------------

    def plot_real_space(self, **kwargs):
        """
        Convenience wrapper returning the matplotlib figure and axes for a real-space scatter plot.
        """
        from .visualization import plot_real_space
        return plot_real_space(self, **kwargs)

    def plot_reciprocal_space(self, **kwargs):
        """
        Convenience wrapper returning the matplotlib figure and axes for a reciprocal-space scatter plot.
        """
        from .visualization import plot_reciprocal_space

        return plot_reciprocal_space(self, **kwargs)

    def plot_brillouin_zone(self, **kwargs):
        """
        Convenience wrapper returning the matplotlib figure and axes for a Brillouin zone plot.
        """
        from .visualization import plot_brillouin_zone

        return plot_brillouin_zone(self, **kwargs)

    def plot_structure(self, **kwargs):
        """
        Visualise lattice connectivity with boundary condition annotations.
        """
        from .visualization import plot_lattice_structure

        return plot_lattice_structure(self, **kwargs)

#############################################################################################################
#! SAVE LATTICE HELPERS
#############################################################################################################

def save_bonds(lattice : Lattice, directory : Union[str], filename : str):
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
#! END OF FILE
#############################################################################################################
