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
from typing import Dict, Mapping, Optional, Tuple, Union, Any

import numpy as np
import scipy.sparse as sp

from ..common import hdf5_lib as HDF5Mod
from ..common import directories as DirectoriesMod

# -----------------------------------------------------------------------------------------------------------
# LATTICE ENUMERATIONS
# -----------------------------------------------------------------------------------------------------------

class LatticeDirection(Enum):
    '''
    Enumeration for the lattice directions
    '''
    X = auto()
    Y = auto()
    Z = auto()
    
    def __str__(self):      return str(self.name).lower()
    def __repr__(self):     return f"LatticeDirection.{self.name}"

# -----------------------------------------------------------------------------------------------------------

class LatticeBC(Enum):
    '''
    Enumeration for the boundary conditions in the lattice model.
    '''
    PBC         = auto()    # Periodic Boundary Conditions
    OBC         = auto()    # Open Boundary Conditions      
    MBC         = auto()    # Mixed Boundary Conditions     - periodic in X direction, open in Y direction
    SBC         = auto()    # Special Boundary Conditions   - periodic in Y direction, open in X direction

# -----------------------------------------------------------------------------------------------------------

class LatticeType(Enum):
    '''
    Contains all the implemented lattice types for the lattice model. 
    '''
    SQUARE      = auto()    # Square lattice
    HEXAGONAL   = auto()    # Hexagonal lattice
    HONEYCOMB   = auto()    # Honeycomb lattice
    GRAPH       = auto()    # Generic graph lattice (adjacency-defined)

############################################## GENERAL LATTICE ##############################################

@dataclass(frozen=True)
class BoundaryFlux:
    """
    Collection of magnetic fluxes piercing lattice boundary loops.

    The value associated with a direction is interpreted as the phase 
    ``phi``
    (in radians) acquired upon wrapping around the boundary once along that
    direction. The corresponding hopping phase factor is ``exp(1j * phi)``.
    
    The fluxes are stored as a mapping from :class:`LatticeDirection` to corresponding
    complex phase values.
    
    Options for specifying fluxes include:
    - Uniform flux in all directions (single float value).
    - Direction-specific fluxes (mapping from direction to float).
    - Zero flux (empty mapping).
    
    Physically, these fluxes correspond to magnetic fluxes threading
    the holes of a torus formed by periodic boundary conditions.
    
    Example:
    >>> flux = BoundaryFlux({LatticeDirection.X: np.pi/2, LatticeDirection.Y: np.pi})
    >>> flux.phase(LatticeDirection.X)
    (6.123233995736766e-17+1j)
    >>> flux.phase(LatticeDirection.Y)
    (-1+0j)
    
    For non-abelian gauge fields, more complex structures are needed.
    """
    
    values  : Mapping[LatticeDirection, float]

    def phase(self, direction: LatticeDirection, winding: int = 1) -> complex:
        """
        Return ``exp(1j * winding * phi_direction)``.
        
        Parameters:
        -----------
        direction : LatticeDirection
            The lattice direction for which to get the phase factor.
        winding : int, optional
            The winding number for the phase factor. Defaults to 1.
        """
        phi = float(self.values.get(direction, 0.0))
        return np.exp(1j * winding * phi)

def _normalize_flux_dict(flux: Optional[Union[float, Mapping[Union[str, LatticeDirection], float]]]) -> BoundaryFlux:
    """
    Normalize flux input into a :class:`BoundaryFlux` instance.
    
    Parameters:
    -----------
    flux : float or Mapping[Union[str, LatticeDirection], float] or None
        If a float, interpreted as uniform flux in all directions.
        If a mapping, keys can be either :class:`LatticeDirection` members
        or their string names (case-insensitive).  Values are fluxes in radians.
        If None, interpreted as zero flux in all directions.
    """
    if flux is None:
        return BoundaryFlux({})
    
    if isinstance(flux, (int, float)):
        phi = float(flux)
        return BoundaryFlux({direction: phi for direction in LatticeDirection})
    
    if isinstance(flux, Mapping):
        out: Dict[LatticeDirection, float] = {}
        
        # parse mapping
        for key, value in flux.items():
            
            if isinstance(key, LatticeDirection):
                direction = key
                
            elif isinstance(key, str):
                try:
                    direction = LatticeDirection[key.upper()]
                except KeyError as exc:
                    raise ValueError(f"Unknown lattice direction '{key}' for flux specification.") from exc
            else:
                raise TypeError(f"Unsupported flux key type: {type(key)!r}")
            out[direction] = float(value)
        return BoundaryFlux(out)
    raise TypeError(f"Unsupported flux specification of type {type(flux)!r}.")

# -----------------------------------------------------------------------------------------------------------
#! HELPER FUNCTIONS
# -----------------------------------------------------------------------------------------------------------

def handle_boundary_conditions(bc: Any):
    """
    Handles and normalizes the input for boundary conditions.
    Parameters:
    -----------
        bc (str, LatticeBC, or None):
            The boundary condition to handle. Can be a string
            ("pbc", "obc", "mbc", "sbc"), an instance of LatticeBC, or None.
    Returns:
    --------
        LatticeBC: The corresponding LatticeBC enum value for the given boundary condition.
    Raises:
        ValueError: If the provided boundary condition is not recognized.
    """

    if bc is None:
        bc = LatticeBC.PBC
    elif isinstance(bc, str):
        if bc.lower() == "pbc":
            bc = LatticeBC.PBC
        elif bc.lower() == "obc":
            bc = LatticeBC.OBC
        elif bc.lower() == "mbc":
            bc = LatticeBC.MBC
        elif bc.lower() == "sbc":
            bc = LatticeBC.SBC
        else:
            raise ValueError(f"Unknown boundary condition: {bc}")
    elif not isinstance(bc, LatticeBC):
        raise ValueError(f"Unknown boundary condition: {bc}")
    return bc

def handle_dim(lx, ly, lz):
    """
    Handles and normalizes the input for lattice dimensions.
    Parameters:
        lx (int):
            Number of sites in the x-direction.
        ly (int):
            Number of sites in the y-direction.
        lz (int):
            Number of sites in the z-direction.
    Returns:
        tuple: A tuple containing the dimensions (lx, ly, lz).
    """
    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError("Lattice dimensions must be positive integers.")
    
    dim = 1
    if ly > 1:
        dim += 1
    if lz > 1:
        dim += 1
    return dim, lx, ly, lz

Backend = np

# -----------------------------------------------------------------------------------------------------------

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
        self._spatial_norm  = [[[]]]                        # three dimensional array for the spatial norm
        
        # matrices for real space and inverse space vectors
        self._vectors       = Backend.zeros((3, 3))         # real space vectors - base vectors of the lattice
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
    def Ns(self):            return self._ns
    @property
    def sites(self):        return self._ns
    
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
    def k2(self):           return self._k2
    @k2.setter
    def k2(self, value):    self._k2 = value
    
    @property
    def k3(self):           return self._k3
    @k3.setter
    def k3(self, value):    self._k3 = value
    
    @property
    def dft(self):                  return self._dft
    @dft.setter
    def dft(self, value):           self._dft = value
    
    @property
    def nn(self):                   return self._nn
    @nn.setter
    def nn(self, value):            self._nn = value

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
        of both i and j — i.e., sites forming two-step NNN paths i-l-j.

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
    
    @abstractmethod
    def calculate_coordinates(self):    pass
    @abstractmethod
    def calculate_r_vectors(self):      pass
    @abstractmethod
    def calculate_k_vectors(self):      pass
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

    # -----------------------------------------------------------------------------
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
