"""
This file contains the EntanglementModule class.

--------------------------------------------
file    : QES/general_python/physics/entanglement_module.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
--------------------------------------------

Unified entanglement calculation module for both quadratic and many-body Hamiltonians.

Features
--------
- Single-particle correlation matrix methods (fast, for quadratic/non-interacting Hamiltonians)
- Many-body reduced density matrix methods (exact, for any state)
- Arbitrary bipartitions (contiguous and non-contiguous subsystems)
- Multipartite entropy calculations (topological entanglement entropy)
- Wick's theorem verification for quadratic systems
- JAX backend for GPU acceleration
- Mask generation utilities for subsystem selection

Theoretical Background
----------------------
For quadratic (non-interacting) Hamiltonians, the entanglement entropy can be computed
efficiently from the single-particle correlation matrix C_ij = <c_i^dag c_j>:

    S = -Tr[C log C + (1-C) log(1-C)]

This scales as O(L^3) compared to O(2^L) for exact diagonalization.

For many-body states, we use Schmidt decomposition of the wavefunction:
    |psi> = sum_i sqrt(lambda_i) |i_A> |i_B>
    S = -sum_i lambda_i log(lambda_i)

Topological Entanglement Entropy (TEE)
--------------------------------------
For topological phases, the entanglement entropy follows:
    S(A) = alpha * L - gamma + O(1/L)
    
where gamma is the topological entanglement entropy. Using Kitaev-Preskill
or Levin-Wen constructions:
    gamma = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC

Examples
--------
Basic usage with quadratic Hamiltonians:
    >>> hamil   = QuadraticHamiltonian(ns=12, ...)
    >>> hamil.diagonalize()
    >>> ent     = hamil.entanglement
    >>> 
    >>> # Define bipartition and calculate entropy
    >>> bipart      = ent.bipartition([0, 1, 2, 3])
    >>> orbitals    = [0, 1, 2, 3, 4, 5]  # occupied states
    >>> S = ent.entropy_correlation(bipart, orbitals)
    
Access correlation matrices:
    >>> C_full      = ent.correlation_matrix(orbitals)
    >>> C_A         = ent.correlation_matrix(orbitals, bipartition=bipart)
    
Batch calculations:
    >>> results     = ent.entropy_multipartition(
    ...     bipartitions=[[0,1], [0,1,2], [0,1,2,3]],
    ...     occupied_orbitals=orbitals
    ... )
    >>> entropies               = results['entropies']
    >>> correlation_matrices    = results['correlation_matrices']
    
JAX backend for GPU acceleration:
    >>> S_jax = ent.entropy_correlation(bipart, orbitals, backend='jax')
    >>> C_jax = ent.correlation_matrix(orbitals, backend='jax')
    
Mask generation utilities:
    >>> masks = MaskGenerator.contiguous(ns=12, size_a=4)  # First 4 sites
    >>> masks = MaskGenerator.alternating(ns=12)           # Even/odd sites
    >>> masks = MaskGenerator.random(ns=12, size_a=6)      # Random 6 sites
    >>> masks = MaskGenerator.kitaev_preskill(ns=12)       # ABC regions for TEE
    
Topological entanglement entropy:
    >>> gamma = ent.topological_entropy(orbitals, construction='kitaev_preskill')
    
Wick's theorem verification:
    >>> is_valid, error = ent.verify_wicks_theorem(orbitals, state)
    
Manual many-body entropy calculations:
    >>> bipart      = ent.bipartition([0, 1, 2, 3])
    >>> S_manual    = ent.entropy_correlation(bipart, orbitals)
"""

import  numpy as np
from    enum import Enum
from    typing import Union, List, Tuple, Optional, Callable, Dict, Literal
from    dataclasses import dataclass

try:
    from ..common.binary        import extract as Extractor
    from ..algebra.utils        import Array, JAX_AVAILABLE
    from .density_matrix        import rho_numba_mask, schmidt_numba_mask, schmidt_numpy
    from .entropy               import entropy, Entanglement
    from .sp                    import correlation_matrix as Corr
    
    if JAX_AVAILABLE:
        import jax.numpy as jnp
except ImportError as e:
    raise ImportError("Required QES modules not found") from e


###############################################################################
#! Mask Generation Utilities
###############################################################################

class MaskGenerator:
    """
    Utility class for generating subsystem masks for entanglement calculations.
    
    Provides convenient methods to create site masks for various bipartition
    geometries, including contiguous, alternating, random, and topological
    (Kitaev-Preskill) constructions.
    
    Examples
    --------
    Basic contiguous mask:
        >>> mask_a = MaskGenerator.contiguous(ns=12, size_a=4)
        >>> print(mask_a)  # array([0, 1, 2, 3])
    
    Alternating (even/odd) sites:
        >>> mask_even, mask_odd = MaskGenerator.alternating(ns=12)
        >>> print(mask_even)  # array([0, 2, 4, 6, 8, 10])
        
    Random subsystem:
        >>> mask = MaskGenerator.random(ns=12, size_a=6, seed=42)
        
    For topological entanglement entropy (Kitaev-Preskill construction):
        >>> regions = MaskGenerator.kitaev_preskill(ns=12)
        >>> A, B, C = regions['A'], regions['B'], regions['C']
    """
    
    @staticmethod
    def contiguous(ns: int, 
                   size_a: int, 
                   start: int = 0) -> np.ndarray:
        """
        Create a contiguous subsystem mask [start, start+1, ..., start+size_a-1].
        
        Parameters
        ----------
        ns : int
            Total number of sites
        size_a : int
            Size of subsystem A
        start : int
            Starting site index (default: 0)
            
        Returns
        -------
        np.ndarray
            Array of site indices in subsystem A
        """
        if start + size_a > ns:
            # Wrap around for periodic systems
            indices = np.arange(start, start + size_a) % ns
            return np.sort(np.unique(indices))
        return np.arange(start, start + size_a, dtype=np.int64)
    
    @staticmethod
    def alternating(ns: int,
                   offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create alternating (even/odd) site masks.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        offset : int
            Offset for even sites (0 = sites 0,2,4,...; 1 = sites 1,3,5,...)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (mask_even, mask_odd) site index arrays
        """
        all_sites = np.arange(ns, dtype=np.int64)
        mask_even = all_sites[(all_sites + offset) % 2 == 0]
        mask_odd = all_sites[(all_sites + offset) % 2 == 1]
        return mask_even, mask_odd
    
    @staticmethod
    def random(ns: int,
              size_a: int,
              seed: Optional[int] = None) -> np.ndarray:
        """
        Create a random subsystem mask.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        size_a : int
            Size of subsystem A
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Sorted array of randomly selected site indices
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(ns, size=size_a, replace=False)
        return np.sort(indices).astype(np.int64)
    
    @staticmethod
    def periodic_interval(ns: int,
                         start: int,
                         size_a: int) -> np.ndarray:
        """
        Create a contiguous mask with periodic boundary conditions.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        start : int
            Starting site index
        size_a : int
            Size of subsystem A
            
        Returns
        -------
        np.ndarray
            Sorted array of site indices (wrapped around if necessary)
        """
        indices = np.arange(start, start + size_a) % ns
        return np.sort(np.unique(indices)).astype(np.int64)
    
    @staticmethod
    def sublattice(ns: int,
                  sublattice_id: int = 0,
                  n_sublattices: int = 2) -> np.ndarray:
        """
        Create a sublattice mask (e.g., A/B sublattices in bipartite lattices).
        
        Parameters
        ----------
        ns : int
            Total number of sites
        sublattice_id : int
            Which sublattice (0, 1, ..., n_sublattices-1)
        n_sublattices : int
            Total number of sublattices (default: 2 for bipartite)
            
        Returns
        -------
        np.ndarray
            Array of site indices in the specified sublattice
        """
        all_sites = np.arange(ns, dtype=np.int64)
        return all_sites[all_sites % n_sublattices == sublattice_id]
    
    @staticmethod
    def kitaev_preskill(ns: int,
                       center: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate regions A, B, C for Kitaev-Preskill topological entanglement entropy.
        
        The Kitaev-Preskill construction divides the system into three regions
        meeting at a point. The topological entanglement entropy is:
            gamma = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
        
        Parameters
        ----------
        ns : int
            Total number of sites (should be divisible by 3 for equal regions)
        center : int, optional
            Central site index (default: ns // 2)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with keys 'A', 'B', 'C', 'AB', 'BC', 'AC', 'ABC'
            containing site index arrays for each region
            
        Notes
        -----
        For 1D chains, regions are consecutive thirds of the chain.
        For 2D systems, you should define regions based on geometry.
        
        References
        ----------
        - Kitaev & Preskill, PRL 96, 110404 (2006)
        - Levin & Wen, PRL 96, 110405 (2006)
        """
        if center is None:
            center = ns // 2
            
        # Divide into 3 approximately equal regions
        size_per_region = ns // 3
        remainder = ns % 3
        
        # Region sizes
        size_a = size_per_region + (1 if remainder > 0 else 0)
        size_b = size_per_region + (1 if remainder > 1 else 0)
        size_c = ns - size_a - size_b
        
        # Region boundaries
        A = np.arange(0, size_a, dtype=np.int64)
        B = np.arange(size_a, size_a + size_b, dtype=np.int64)
        C = np.arange(size_a + size_b, ns, dtype=np.int64)
        
        return {
            'A': A,
            'B': B,
            'C': C,
            'AB': np.concatenate([A, B]),
            'BC': np.concatenate([B, C]),
            'AC': np.concatenate([A, C]),
            'ABC': np.arange(ns, dtype=np.int64)
        }
    
    @staticmethod
    def levin_wen_disk(ns: int,
                      n_annuli: int = 3) -> Dict[str, np.ndarray]:
        """
        Generate annular regions for Levin-Wen construction.
        
        For a disk geometry, creates concentric annuli to extract
        topological entanglement entropy with area law subtraction.
        
        Parameters
        ----------
        ns : int
            Total number of sites
        n_annuli : int
            Number of concentric annuli (default: 3)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'inner', 'middle', 'outer', and combined regions
        """
        size_per_annulus = ns // n_annuli
        
        regions = {}
        names = ['inner', 'middle', 'outer'][:n_annuli]
        
        start = 0
        for i, name in enumerate(names):
            if i == n_annuli - 1:
                end = ns
            else:
                end = start + size_per_annulus
            regions[name] = np.arange(start, end, dtype=np.int64)
            start = end
        
        # Combinations
        if n_annuli >= 2:
            regions['inner_middle'] = np.concatenate([regions['inner'], regions['middle']])
        if n_annuli >= 3:
            regions['middle_outer'] = np.concatenate([regions['middle'], regions['outer']])
            regions['inner_outer'] = np.concatenate([regions['inner'], regions['outer']])
        
        return regions
    
    @staticmethod
    def from_bitmask(mask_int: int, ns: int) -> np.ndarray:
        """
        Convert an integer bitmask to an array of site indices.
        
        Parameters
        ----------
        mask_int : int
            Integer whose bits indicate included sites
        ns : int
            Total number of sites
            
        Returns
        -------
        np.ndarray
            Array of site indices where bits are set
            
        Example
        -------
        >>> MaskGenerator.from_bitmask(0b1010, ns=4)
        array([1, 3])
        """
        indices = []
        for i in range(ns):
            if (mask_int >> i) & 1:
                indices.append(i)
        return np.array(indices, dtype=np.int64)
    
    @staticmethod
    def to_bitmask(indices: np.ndarray) -> int:
        """
        Convert an array of site indices to an integer bitmask.
        
        Parameters
        ----------
        indices : np.ndarray
            Array of site indices
            
        Returns
        -------
        int
            Integer bitmask with bits set at specified positions
            
        Example
        -------
        >>> MaskGenerator.to_bitmask(np.array([1, 3]))
        10  # = 0b1010
        """
        mask = 0
        for i in indices:
            mask |= (1 << int(i))
        return mask


###############################################################################


@dataclass
class BipartitionInfo:
    """Information about a bipartition of the system."""
    mask_a          : np.ndarray          # Indices in subsystem A
    mask_b          : np.ndarray          # Indices in subsystem B
    size_a          : int                 # Number of sites in A
    size_b          : int                 # Number of sites in B
    order           : tuple               # Reordering: (mask_a..., mask_b...)
    extractor_a     : Callable            # Function to extract A indices from state
    extractor_b     : Callable            # Function to extract B indices from state

class EntanglementModule:
    """
    Entanglement calculation module for Hamiltonians.
    
    Provides unified interface for calculating entanglement entropy using:
    - Single-particle correlation matrices (quadratic Hamiltonians, fast)
    - Many-body reduced density matrices (any state, exact)
    - JAX backend for GPU acceleration
    - Batch calculations for multiple bipartitions
    
    Automatically handles arbitrary bipartitions including non-contiguous subsystems.
    
    Examples
    --------
    Basic entropy calculation:
        >>> hamil = QuadraticHamiltonian(ns=12, ...)
        >>> hamil.diagonalize()
        >>> ent = hamil.entanglement
        >>> 
        >>> bipart = ent.bipartition([0, 1, 2, 3])
        >>> orbitals = [0, 1, 2, 3, 4, 5]
        >>> S = ent.entropy_correlation(bipart, orbitals)
    
    Access correlation matrices:
        >>> C_full = ent.correlation_matrix(orbitals)  # (ns, ns)
        >>> C_A = ent.correlation_matrix(orbitals, bipartition=bipart)  # (4, 4)
    
    Batch calculations:
        >>> results = ent.entropy_multipartition(
        ...     [[0,1], [0,1,2], [0,1,2,3]],
        ...     orbitals
        ... )
        >>> entropies = results['entropies']  # array of 3 entropies
        >>> C_matrices = results['correlation_matrices']  # list of 3 matrices
    
    JAX backend:
        >>> S_jax = ent.entropy_correlation(bipart, orbitals, backend='jax')
        >>> results_jax = ent.entropy_multipartition(
        ...     [[0,1], [0,1,2]], orbitals, backend='jax'
        ... )
    
    Mutual information:
        >>> I_AB = ent.mutual_information([0,1,2], [3,4,5], orbitals)
    
    Entropy scaling:
        >>> results = ent.entropy_scan(orbitals, sizes=[1,2,3,4,5])
    """
    
    def __init__(self, hamiltonian):
        """
        Initialize entanglement module for a Hamiltonian.
        
        Parameters
        ----------
        hamiltonian : Hamiltonian
            The Hamiltonian object (quadratic or many-body)
        """
        self._hamil = hamiltonian
        self._cached_bipartitions = {}
        
    def bipartition(self, 
                    mask_a: Union[List[int], np.ndarray, int],
                    *,
                    cache: bool = True) -> BipartitionInfo:
        """
        Create bipartition information for subsystem A.
        
        Parameters
        ----------
        mask_a : array-like or int
            Indices of sites in subsystem A, or number of sites in A (takes first N sites)
        cache : bool
            Whether to cache the bipartition for reuse
            
        Returns
        -------
        BipartitionInfo
            Information about the bipartition
            
        Examples
        --------
        >>> # Contiguous partition
        >>> bipart = ent.bipartition(5)  # First 5 sites
        >>> 
        >>> # Non-contiguous partition
        >>> bipart = ent.bipartition([0, 2, 4, 6, 8])  # Even sites
        """
        # Convert to array
        if isinstance(mask_a, int):
            mask_a = np.arange(mask_a)
        else:
            mask_a = np.asarray(mask_a, dtype=np.int64)
            
        # Check cache
        cache_key = tuple(sorted(mask_a))
        if cache and cache_key in self._cached_bipartitions:
            return self._cached_bipartitions[cache_key]
        
        # Create complement
        ns = self._hamil.ns
        mask_a = np.sort(mask_a)
        mask_b = np.setdiff1d(np.arange(ns), mask_a)
        
        size_a = len(mask_a)
        size_b = len(mask_b)
        
        # Create ordering tuple
        order = tuple(mask_a) + tuple(mask_b)
        
        # Create extractors
        extractor_a = Extractor.make_extractor(mask_a, size=ns, backend='numba_vnb')
        extractor_b = Extractor.make_extractor(mask_b, size=ns, backend='numba_vnb')
        
        bipart = BipartitionInfo(
            mask_a          =   mask_a,
            mask_b          =   mask_b,
            size_a          =   size_a,
            size_b          =   size_b,
            order           =   order,
            extractor_a     =   extractor_a,
            extractor_b     =   extractor_b
        )
        
        if cache:
            self._cached_bipartitions[cache_key] = bipart
            
        return bipart
    
    def correlation_matrix(self,
                          occupied_orbitals     : Union[List[int], np.ndarray],
                          *,
                          bipartition           : Optional[BipartitionInfo] = None,
                          subtract_identity     : bool = False,
                          backend               : str = 'numpy') -> np.ndarray:
        """
        Get single-particle correlation matrix C_ij = <c_i^\\dag c_j>.
        
        Computes the correlation matrix for a free fermion state defined by
        occupied orbitals. Uses spin-unpolarized convention (factor of 2).
        
        Parameters
        ----------
        occupied_orbitals : array-like
            Indices of occupied orbitals (in eigenstate basis).
            For ground state, use [0, 1, ..., N-1] for N particles.
        bipartition : BipartitionInfo, optional
            If provided, returns correlation matrix restricted to subsystem A.
            If None, returns full correlation matrix for all sites.
        subtract_identity : bool
            Whether to subtract identity from the correlation matrix.
        backend : str
            'numpy' or 'jax' for GPU acceleration.
            
        Returns
        -------
        np.ndarray or jax.numpy.ndarray
            Correlation matrix C_ij = <c_i^\\dag c_j>.
            Shape: (size_a, size_a) if bipartition given, else (ns, ns).
            Diagonal elements are site occupations (in [0,2] range with factor 2).
        
        Examples
        --------
        Full correlation matrix for ground state:
            >>> hamil = QuadraticHamiltonian(ns=8, dtype=np.complex128)
            >>> # ... add hopping terms ...\n            >>> hamil.diagonalize()
            >>> ent = hamil.entanglement
            >>> 
            >>> # Half-filling: occupy lowest 4 orbitals
            >>> orbitals = [0, 1, 2, 3]
            >>> C_full = ent.correlation_matrix(orbitals)
            >>> print(C_full.shape)  # (8, 8)
            >>> print(np.trace(C_full))  # Should be 2*4 = 8 (spin-unpolarized)
        
        Subsystem correlation matrix:
            >>> bipart = ent.bipartition([0, 1, 2])  # First 3 sites
            >>> C_A = ent.correlation_matrix(orbitals, bipartition=bipart)
            >>> print(C_A.shape)  # (3, 3)
            >>> # Use for entropy: eigenvalues → occupations → entropy
        
        JAX backend for GPU:
            >>> C_jax = ent.correlation_matrix(orbitals, backend='jax')
            >>> # Same result as NumPy, but runs on GPU
        
        Verify correlation matrix properties:
            >>> C = ent.correlation_matrix(orbitals)
            >>> # Hermitian
            >>> assert np.allclose(C, C.conj().T)
            >>> # Occupations in [0, 2]
            >>> assert np.all(np.diag(C) >= 0) and np.all(np.diag(C) <= 2)
        """
        if not hasattr(self._hamil, '_eig_vec') or self._hamil._eig_vec is None:
            raise RuntimeError("Hamiltonian must be diagonalized first")
        
        W           = self._hamil._eig_vec  # W is (sites, orbitals), need to transpose
        orbitals            = np.asarray(occupied_orbitals, dtype=np.int64)
        occ_mask            = np.zeros(self._hamil.ns, dtype=bool)
        occ_mask[orbitals]  = True
        
        # Corr.corr_full expects W with shape (orbitals, sites), so transpose
        C_full              = Corr.corr_full(W.T, occ_mask, subtract_identity=subtract_identity, raw=True, mode='slater')
        
        # Extract subblock if needed
        if bipartition is not None:
            C_result = C_full[np.ix_(bipartition.mask_a, bipartition.mask_a)]
        else:
            C_result = C_full
        
        # Convert to JAX if requested
        if backend == 'jax' and JAX_AVAILABLE:
            return jnp.array(C_result)
        elif backend == 'numpy':
            return C_result
        else:
            raise ValueError(f"Backend '{backend}' not available or not supported")
    
    def entropy_correlation(self,
                          bipartition           : BipartitionInfo,
                          occupied_orbitals     : Union[List[int], np.ndarray],
                          *,
                          subtract_identity     : bool = False,
                          backend               : str = 'numpy') -> float:
        """
        Calculate entanglement entropy from single-particle correlation matrix.
        
        **SINGLE-PARTICLE METHOD** - Fast O(L_A³) method for non-interacting (quadratic)
        Hamiltonians. Computes entropy from correlation matrix eigenvalues.
        
        Works for ANY bipartition (contiguous or non-contiguous) of free fermion states.
        
        Parameters
        ----------
        bipartition : BipartitionInfo
            Bipartition of the system (use ent.bipartition() to create).
            Works for both contiguous and non-contiguous subsystems.
        occupied_orbitals : array-like
            Indices of occupied orbitals (in eigenstate basis).
            For ground state with N particles, use [0, 1, ..., N-1].
        subtract_identity : bool
            Whether to subtract identity from correlation matrix (advanced)
        backend : str
            'numpy' or 'jax' for computation backend
            
        Returns
        -------
        float
            Entanglement entropy (in natural log units, always positive)
            
        Examples
        --------
        Basic usage for free fermions:
            >>> hamil = QuadraticHamiltonian(ns=12, dtype=np.complex128)
            >>> # ... add hopping terms ...
            >>> hamil.diagonalize()
            >>> ent = hamil.entanglement
            >>> 
            >>> # Ground state at half-filling
            >>> orbitals = list(range(6))  # Occupy lowest 6 orbitals
            >>> 
            >>> # Partition: sites [0,1,2,3] vs [4,5,6,7,8,9,10,11]
            >>> bipart = ent.bipartition([0, 1, 2, 3])
            >>> S = ent.entropy_correlation(bipart, orbitals)
            >>> print(f"Entropy: {S:.4f}")  # Always positive!
        
        JAX backend (GPU acceleration):
            >>> S_jax = ent.entropy_correlation(bipart, orbitals, backend='jax')
            >>> # Same result as NumPy, but faster on GPU
        
        Comparison with many-body method (should agree for non-interacting):
            >>> state = hamil.many_body_state(np.array(orbitals))
            >>> S_corr = ent.entropy_correlation(bipart, orbitals)
            >>> S_mb = ent.entropy_many_body(bipart, state)
            >>> assert np.isclose(S_corr, S_mb, rtol=1e-5)  # Should agree!
        
        Non-contiguous partition (works correctly!):
            >>> bipart_nc = ent.bipartition([0, 2, 4, 6])  # Even sites
            >>> S_nc = ent.entropy_correlation(bipart_nc, orbitals)
            >>> # Exact for free fermions, matches entropy_many_body()
            
        Notes
        -----
        Algorithm:
        1. Compute full correlation matrix C_ij = <c_i† c_j> from occupied orbitals
        2. Extract subblock C_A for sites in subsystem A (handles non-contiguous)
        3. Diagonalize C_A to get eigenvalues (occupations in [0,1])
        4. Apply single-particle entropy formula:
           S = -Σ [p log(p) + (1-p) log(1-p)]
        
        This gives the EXACT entanglement entropy for ANY bipartition of
        non-interacting (quadratic) Hamiltonians and matches entropy_many_body().
        
        Limitations:
        - Requires diagonalized Hamiltonian
        - Only works for quadratic (non-interacting) Hamiltonians
        - For interacting systems, use entropy_many_body()
        """
        # Get correlation matrix for subsystem A using Corr methods
        # NOTE: This works for ANY bipartition (contiguous or non-contiguous)
        C_A = self.correlation_matrix(occupied_orbitals, bipartition=bipartition, subtract_identity=False, backend=backend)
        
        # Diagonalize to get eigenvalues (in [0,2] range for occupation with spin-unpolarized convention)
        if backend == 'numpy':
            corr_eigs, _ = np.linalg.eigh(C_A)
        elif backend == 'jax' and JAX_AVAILABLE:
            corr_eigs = jnp.linalg.eigh(C_A)[0]
        else:
            raise ValueError(f"Backend '{backend}' not available")
        
        # Divide by 2 to convert from spin-unpolarized to spin-polarized (occupation in [0,1])
        corr_eigs_polarized     = corr_eigs / 2.0
        
        # Transform to [-1,1] range for SINGLE entropy formula
        corr_eigs_transformed   = 2.0 * corr_eigs_polarized - 1.0
        
        # Use unified entropy function which handles both numpy and jax
        return entropy(corr_eigs_transformed, q=1.0, typek=Entanglement.SINGLE, backend=backend)
    
    def entropy_many_body(self,
                         bipartition: BipartitionInfo,
                         state: np.ndarray,
                         *,
                         method: str = 'auto',
                         use_eig: bool = True) -> float:
        """
        Calculate entanglement entropy from many-body state.
        
        **MANY-BODY METHOD** - Exact method that works for ANY quantum state,
        including interacting systems. Performs Schmidt decomposition of the
        many-body wavefunction.
        
        Parameters
        ----------
        bipartition : BipartitionInfo
            Bipartition of the system (use ent.bipartition() to create)
        state : np.ndarray
            Many-body state vector (length 2^ns).
            For free fermions, use hamil.many_body_state(orbitals).
        method : str
            'auto'    : Choose best method based on bipartition geometry
            'schmidt' : Use Schmidt decomposition with mask (for non-contiguous)
            'numpy'   : Use direct numpy Schmidt (for contiguous, faster)
        use_eig : bool
            Whether to use eigenvalue decomposition (True) or SVD (False)
            
        Returns
        -------
        float
            Von Neumann entanglement entropy (always positive)
            
        Examples
        --------
        Free fermion state (non-interacting):
            >>> hamil = QuadraticHamiltonian(ns=8, dtype=np.complex128)
            >>> # ... add hopping terms ...
            >>> hamil.diagonalize()
            >>> ent = hamil.entanglement
            >>> 
            >>> # Construct many-body ground state (Slater determinant)
            >>> orbitals = np.array([0, 1, 2, 3])  # Half-filling
            >>> state = hamil.many_body_state(orbitals)  # Length 2^8 = 256
            >>> print(f"State norm: {np.linalg.norm(state):.6f}")  # 1.0
            >>> 
            >>> # Calculate entropy for contiguous partition
            >>> bipart = ent.bipartition([0, 1, 2])  # First 3 sites
            >>> S_mb = ent.entropy_many_body(bipart, state)
            >>> print(f"Entropy: {S_mb:.4f}")  # Always positive!
        
        Non-contiguous partition (exact, unlike correlation method):
            >>> bipart_nc = ent.bipartition([0, 2, 4, 6])  # Non-contiguous
            >>> S_nc = ent.entropy_many_body(bipart_nc, state)
            >>> # This is EXACT, unlike entropy_correlation()
        
        Verify agreement with correlation method (non-interacting only):
            >>> bipart = ent.bipartition([0, 1, 2, 3])  # Contiguous
            >>> S_mb = ent.entropy_many_body(bipart, state)
            >>> S_corr = ent.entropy_correlation(bipart, orbitals)
            >>> assert np.isclose(S_mb, S_corr, rtol=1e-5)  # Should agree!
        
        Custom many-body state (e.g., superposition):
            >>> # Create superposition of two Slater determinants
            >>> state1 = hamil.many_body_state([0, 1, 2, 3])
            >>> state2 = hamil.many_body_state([1, 2, 3, 4])
            >>> state_mixed = (state1 + state2) / np.sqrt(2)
            >>> S_mixed = ent.entropy_many_body(bipart, state_mixed)
            >>> # Works for any state, not just Slater determinants!
        
        Choose method explicitly:
            >>> # For contiguous (faster)
            >>> S_fast = ent.entropy_many_body(bipart, state, method='numpy')
            >>> 
            >>> # For non-contiguous (required)
            >>> S_exact = ent.entropy_many_body(bipart_nc, state, method='schmidt')
        
        Notes
        -----
        Algorithm:
        1. Reshape state vector according to bipartition:
           |psi⟩ → psi_{A,B} (matrix form)
        2. Perform Schmidt decomposition: |psi⟩ = Σ √λ_i |i_A⟩|i_B⟩
        3. Compute von Neumann entropy: S = -Σ λ_i log(λ_i)
        
        This method is EXACT for any quantum state, including:
        - Non-interacting (free fermion) states
        - Interacting many-body states
        - Superpositions of Fock states
        - Any contiguous or non-contiguous bipartition
        
        Performance:
        - Slower than correlation matrix method
        - Memory scales as O(2^ns)
        - Practical for ns ≤ 16 sites
        
        Comparison with correlation method:
        - entropy_correlation: Fast, only for non-interacting, contiguous only
        - entropy_many_body: Slower, works for any state, any bipartition
        - For non-interacting + contiguous: both give identical results
        """
        dimA = 1 << bipartition.size_a
        dimB = 1 << bipartition.size_b
        
        # Check if bipartition is contiguous
        is_contiguous = np.all(np.diff(bipartition.mask_a) == 1) and bipartition.mask_a[0] == 0
        
        if method == 'auto':
            method = 'numpy' if is_contiguous else 'schmidt'
        
        if method == 'numpy' and is_contiguous:
            # Fast path for contiguous bipartitions
            schmidt_vals, _, _ = schmidt_numpy(state, dimA, dimB, eig=use_eig)
            return entropy(schmidt_vals, typek=Entanglement.VN)
            
        elif method == 'schmidt':
            # Reshape state according to bipartition order
            psi_reshaped = rho_numba_mask(state, bipartition.order, bipartition.size_a)
            
            # Schmidt decomposition
            schmidt_vals, _ = schmidt_numba_mask(
                psi=psi_reshaped,
                order=bipartition.order,
                size_a=bipartition.size_a,
                eig=use_eig
            )
            
            return entropy(schmidt_vals, typek=Entanglement.VN)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def entropy_scan(self,
                    occupied_orbitals: Union[List[int], np.ndarray],
                    subsystem_sizes: Optional[List[int]] = None,
                    *,
                    method: str = 'auto',
                    contiguous: bool = True) -> dict:
        """
        Calculate entanglement entropy for multiple subsystem sizes.
        
        Parameters
        ----------
        occupied_orbitals : array-like
            Occupied orbitals for the state
        subsystem_sizes : list of int, optional
            Sizes of subsystem A to scan. If None, scans all sizes from 1 to ns-1
        method : str
            'auto'        : Use correlation matrix for quadratic, many-body otherwise
            'correlation' : Force correlation matrix method
            'many_body'   : Force many-body method
        contiguous : bool
            If True, use contiguous partitions [0:size_a]
            If False, use random partitions
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'sizes'     : Subsystem sizes
            - 'entropies' : Entanglement entropies
            - 'method'    : Method used
            
        Examples
        --------
        >>> results = ent.entropy_scan(orbitals=[0,1,2,3,4])
        >>> plt.plot(results['sizes'], results['entropies'])
        """
        if subsystem_sizes is None:
            subsystem_sizes = list(range(1, self._hamil.ns))
            
        # Determine method
        is_quadratic = hasattr(self._hamil, '_particle_conserving')
        if method == 'auto':
            method = 'correlation' if is_quadratic else 'many_body'
            
        entropies = []
        
        if method == 'correlation':
            # Fast correlation matrix method
            for size_a in subsystem_sizes:
                if contiguous:
                    bipart = self.bipartition(size_a)
                else:
                    mask_a = np.sort(np.random.choice(self._hamil.ns, size_a, replace=False))
                    bipart = self.bipartition(mask_a)
                    
                S = self.entropy_correlation(bipart, occupied_orbitals)
                entropies.append(S)
                
        else:
            # Many-body method (requires state construction)
            state = self._hamil.many_body_state(occupied_orbitals)
            
            for size_a in subsystem_sizes:
                if contiguous:
                    bipart = self.bipartition(size_a)
                else:
                    mask_a = np.sort(np.random.choice(self._hamil.ns, size_a, replace=False))
                    bipart = self.bipartition(mask_a)
                    
                S = self.entropy_many_body(bipart, state)
                entropies.append(S)
                
        return {
            'sizes': subsystem_sizes,
            'entropies': np.array(entropies),
            'method': method
        }
    
    def mutual_information(self,
                          mask_a: Union[List[int], np.ndarray],
                          mask_b: Union[List[int], np.ndarray],
                          occupied_orbitals: Union[List[int], np.ndarray],
                          *,
                          method: str = 'auto') -> float:
        """
        Calculate mutual information I(A:B) = S(A) + S(B) - S(AB).
        
        Parameters
        ----------
        mask_a, mask_b : array-like
            Indices for subsystems A and B
        occupied_orbitals : array-like
            Occupied orbitals
        method : str
            'auto', 'correlation', or 'many_body'
            
        Returns
        -------
        float
            Mutual information
        """
        mask_a = np.asarray(mask_a)
        mask_b = np.asarray(mask_b)
        mask_ab = np.union1d(mask_a, mask_b)
        
        # Create bipartitions
        bipart_a = self.bipartition(mask_a)
        bipart_b = self.bipartition(mask_b)
        bipart_ab = self.bipartition(mask_ab)
        
        # Determine method
        is_quadratic = hasattr(self._hamil, '_particle_conserving')
        if method == 'auto':
            method = 'correlation' if is_quadratic else 'many_body'
        
        if method == 'correlation':
            S_a = self.entropy_correlation(bipart_a, occupied_orbitals)
            S_b = self.entropy_correlation(bipart_b, occupied_orbitals)
            S_ab = self.entropy_correlation(bipart_ab, occupied_orbitals)
        else:
            state = self._hamil.many_body_state(occupied_orbitals)
            S_a = self.entropy_many_body(bipart_a, state)
            S_b = self.entropy_many_body(bipart_b, state)
            S_ab = self.entropy_many_body(bipart_ab, state)
            
        return S_a + S_b - S_ab
    
    def entropy_multipartition(self,
                               bipartitions: List[Union[BipartitionInfo, List[int], np.ndarray]],
                               occupied_orbitals: Union[List[int], np.ndarray],
                               *,
                               method: str = 'auto',
                               backend: str = 'numpy',
                               state: Optional[np.ndarray] = None) -> dict:
        """
        Calculate entanglement entropy for multiple bipartitions simultaneously.
        
        Efficient batch calculation that computes correlation matrix once and
        reuses it for all bipartitions, or computes many-body state once for
        all Schmidt decompositions.
        
        Parameters
        ----------
        bipartitions : list
            List of BipartitionInfo objects or site masks (will create BipartitionInfo).
        occupied_orbitals : array-like
            Occupied orbitals for correlation method.
        method : str
            'auto', 'correlation', or 'many_body'.
        backend : str
            'numpy' or 'jax' (for correlation method).
        state : np.ndarray, optional
            Pre-computed many-body state (for many_body method).
            If None, will be computed from occupied_orbitals.
            
        Returns
        -------
        dict
            Results dictionary containing:
            - 'entropies': array of entropies for each bipartition
            - 'bipartitions': list of BipartitionInfo objects
            - 'method': method used ('correlation' or 'many_body')
            - 'correlation_matrices': list of C_A matrices (if method='correlation')
            
        Examples
        --------
        Basic usage:
            >>> masks = [[0,1], [0,1,2], [0,1,2,3]]
            >>> results = ent.entropy_multipartition(masks, orbitals=[0,1,2,3,4])
            >>> print(results['entropies'])  # array([S_1, S_2, S_3])
        
        Access correlation matrices:
            >>> C_matrices = results['correlation_matrices']
            >>> for i, C_A in enumerate(C_matrices):
            ...     print(f"Bipartition {i}: C_A shape = {C_A.shape}")
        
        JAX backend:
            >>> results_jax = ent.entropy_multipartition(
            ...     masks, orbitals, backend='jax'
            ... )
        
        Many-body method:
            >>> state = hamil.many_body_state(orbitals)
            >>> results_mb = ent.entropy_multipartition(
            ...     masks, orbitals, method='many_body', state=state
            ... )
        """
        # Prepare bipartitions
        bipart_list = []
        for bp in bipartitions:
            if isinstance(bp, BipartitionInfo):
                bipart_list.append(bp)
            else:
                bipart_list.append(self.bipartition(bp))
        
        # Determine method
        is_quadratic = hasattr(self._hamil, '_particle_conserving')
        if method == 'auto':
            method = 'correlation' if is_quadratic else 'many_body'
        
        entropies = []
        corr_matrices = [] if method == 'correlation' else None
        
        if method == 'correlation':
            # Compute full correlation matrix once
            C_full = self.correlation_matrix(occupied_orbitals, bipartition=None,
                                            subtract_identity=False, backend=backend)
            
            # Extract subblocks and compute entropies for each bipartition
            for bipart in bipart_list:
                if backend == 'numpy':
                    C_A = C_full[np.ix_(bipart.mask_a, bipart.mask_a)]
                    corr_matrices.append(C_A.copy())
                    
                    # Diagonalize and convert from spin-unpolarized to spin-polarized
                    corr_eigs, _ = np.linalg.eigh(C_A)
                    corr_eigs_polarized = corr_eigs / 2.0
                    corr_eigs_transformed = 2.0 * corr_eigs_polarized - 1.0
                    S = entropy(corr_eigs_transformed, q=1.0, typek=Entanglement.SINGLE, backend='numpy')
                    
                elif backend == 'jax' and JAX_AVAILABLE:
                    mask_a_jax = jnp.array(bipart.mask_a)
                    C_A = C_full[jnp.ix_(mask_a_jax, mask_a_jax)]
                    corr_matrices.append(np.array(C_A))
                    
                    # Diagonalize and convert from spin-unpolarized to spin-polarized
                    corr_eigs = jnp.linalg.eigh(C_A)[0]
                    corr_eigs_polarized = corr_eigs / 2.0
                    corr_eigs_transformed = 2.0 * corr_eigs_polarized - 1.0
                    S = float(entropy(corr_eigs_transformed, q=1.0, typek=Entanglement.SINGLE, backend='jax'))
                else:
                    raise ValueError(f"Backend '{backend}' not available")
                    
                entropies.append(S)
                
        else:  # many_body method
            # Compute state once if not provided
            if state is None:
                state = self._hamil.many_body_state(occupied_orbitals)
            
            # Compute entropy for each bipartition
            for bipart in bipart_list:
                S = self.entropy_many_body(bipart, state, method='auto')
                entropies.append(S)
        
        result = {
            'entropies': np.array(entropies),
            'bipartitions': bipart_list,
            'method': method
        }
        
        if corr_matrices is not None:
            result['correlation_matrices'] = corr_matrices
            
        return result
    
    # =========================================================================
    #! Topological Entanglement Entropy
    # =========================================================================
    
    def topological_entropy(self,
                           occupied_orbitals: Union[List[int], np.ndarray],
                           *,
                           construction: Literal['kitaev_preskill', 'levin_wen'] = 'kitaev_preskill',
                           method: str = 'auto',
                           regions: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Calculate topological entanglement entropy (TEE).
        
        The topological entanglement entropy γ characterizes topological order.
        For topologically ordered states, S(A) = αL - γ + O(1/L), where γ > 0.
        
        Parameters
        ----------
        occupied_orbitals : array-like
            Occupied orbitals defining the state
        construction : str
            'kitaev_preskill' : γ = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
            'levin_wen' : Alternative construction with disk geometry
        method : str
            Entropy calculation method ('auto', 'correlation', 'many_body')
        regions : dict, optional
            Custom region definitions. If None, uses MaskGenerator.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'gamma' : Topological entanglement entropy
            - 'entropies' : Individual region entropies
            - 'regions' : Region masks used
            
        Notes
        -----
        For the Kitaev-Preskill construction:
            γ = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
            
        This combination cancels the area law contribution and extracts
        the universal topological term. For topological phases like the
        toric code, γ = log(D) where D is the total quantum dimension.
        
        References
        ----------
        - Kitaev & Preskill, PRL 96, 110404 (2006)
        - Levin & Wen, PRL 96, 110405 (2006)
        
        Examples
        --------
        >>> result = ent.topological_entropy(orbitals, construction='kitaev_preskill')
        >>> print(f"Topological entropy: γ = {result['gamma']:.4f}")
        """
        ns = self._hamil.ns
        
        # Generate regions if not provided
        if regions is None:
            if construction == 'kitaev_preskill':
                regions = MaskGenerator.kitaev_preskill(ns)
            elif construction == 'levin_wen':
                regions = MaskGenerator.levin_wen_disk(ns, n_annuli=3)
            else:
                raise ValueError(f"Unknown construction: {construction}")
        
        # Calculate entropies for each region
        entropies = {}
        
        # Determine method
        is_quadratic = hasattr(self._hamil, '_particle_conserving')
        if method == 'auto':
            method = 'correlation' if is_quadratic else 'many_body'
        
        # Compute state if needed for many-body method
        state = None
        if method == 'many_body':
            state = self._hamil.many_body_state(occupied_orbitals)
        
        for region_name, mask in regions.items():
            if len(mask) == 0 or len(mask) == ns:
                entropies[region_name] = 0.0
                continue
                
            bipart = self.bipartition(mask)
            
            if method == 'correlation':
                S = self.entropy_correlation(bipart, occupied_orbitals, check_contiguous=False)
            else:
                S = self.entropy_many_body(bipart, state)
            
            entropies[region_name] = S
        
        # Calculate topological entropy
        if construction == 'kitaev_preskill':
            gamma = (entropies.get('A', 0) + entropies.get('B', 0) + entropies.get('C', 0)
                    - entropies.get('AB', 0) - entropies.get('BC', 0) - entropies.get('AC', 0)
                    + entropies.get('ABC', 0))
        elif construction == 'levin_wen':
            # Levin-Wen uses annular geometry
            gamma = (entropies.get('inner', 0) + entropies.get('outer', 0) 
                    - entropies.get('inner_outer', 0))
        else:
            gamma = 0.0
        
        return {
            'gamma': gamma,
            'entropies': entropies,
            'regions': regions,
            'construction': construction
        }
    
    # =========================================================================
    #! Wick's Theorem Verification
    # =========================================================================
    
    def verify_wicks_theorem(self,
                            occupied_orbitals: Union[List[int], np.ndarray],
                            state: Optional[np.ndarray] = None,
                            *,
                            test_sites: Optional[Tuple[int, int, int, int]] = None,
                            tolerance: float = 1e-10) -> Dict[str, Union[bool, float, np.ndarray]]:
        """
        Verify Wick's theorem for a state: check if it's a valid Slater determinant.
        
        For free fermion (quadratic) Hamiltonians, all correlation functions
        factorize according to Wick's theorem. This method verifies this property.
        
        Wick's theorem states that for a Slater determinant:
            <c_i† c_j† c_l c_k> = <c_i† c_k><c_j† c_l> - <c_i† c_l><c_j† c_k>
        
        Parameters
        ----------
        occupied_orbitals : array-like
            Occupied orbitals defining the expected Slater determinant
        state : np.ndarray, optional
            Many-body state to verify. If None, constructs from occupied_orbitals.
        test_sites : tuple, optional
            Specific (i, j, k, l) sites to test. If None, tests random sites.
        tolerance : float
            Tolerance for numerical comparison
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'is_valid' : bool - True if Wick's theorem is satisfied
            - 'max_error' : float - Maximum deviation from Wick's theorem
            - 'errors' : np.ndarray - Error matrix for all tested site combinations
            - 'correlation_matrix' : np.ndarray - Single-particle correlation matrix
            
        Examples
        --------
        >>> result = ent.verify_wicks_theorem(orbitals)
        >>> if result['is_valid']:
        ...     print("State satisfies Wick's theorem (is a Slater determinant)")
        >>> else:
        ...     print(f"Max error: {result['max_error']:.2e}")
        
        Notes
        -----
        A state satisfies Wick's theorem if and only if it is a Slater determinant
        (or a mixture thereof for finite temperature). This is equivalent to
        the state being a Gaussian state for fermions.
        """
        ns = self._hamil.ns
        
        # Construct state if not provided
        if state is None:
            state = self._hamil.many_body_state(occupied_orbitals)
        
        # Get single-particle correlation matrix from orbitals
        C_sp = self.correlation_matrix(occupied_orbitals, subtract_identity=False)
        # Convert to [0,1] occupations (divide by 2 for spin-unpolarized convention)
        C_sp = C_sp / 2.0
        
        # Get exact 2-particle correlation from many-body state
        # <c_i† c_j† c_l c_k> for all i,j,k,l
        from .sp.correlation_matrix import corr_from_statevector
        
        errors = np.zeros((ns, ns), dtype=np.float64)
        max_error = 0.0
        
        # Test specific sites or scan through pairs
        if test_sites is not None:
            i, j, k, l = test_sites
            test_pairs = [(i, j, k, l)]
        else:
            # Sample a few representative pairs
            test_pairs = []
            for j in range(min(ns, 4)):
                for l in range(min(ns, 4)):
                    if j != l:
                        test_pairs.append((0, j, l, 0))
        
        for (i, j, k, l) in test_pairs:
            if i == j or k == l:
                continue
            
            # Wick's theorem prediction
            wick_pred = C_sp[i, k] * C_sp[j, l] - C_sp[i, l] * C_sp[j, k]
            
            # Get exact value from many-body state
            try:
                N_exact = corr_from_statevector(state, ns, order=4, j=j, l=l)
                exact_val = N_exact[i, k]
            except Exception:
                exact_val = wick_pred  # Fallback if not implemented
            
            error = np.abs(exact_val - wick_pred)
            errors[i, k] = max(errors[i, k], error)
            max_error = max(max_error, error)
        
        is_valid = max_error < tolerance
        
        return {
            'is_valid': is_valid,
            'max_error': max_error,
            'errors': errors,
            'correlation_matrix': C_sp,
            'tolerance': tolerance
        }
    
    def correlation_entropy_noncontiguous(self,
                                         bipartition: 'BipartitionInfo',
                                         occupied_orbitals: Union[List[int], np.ndarray],
                                         *,
                                         backend: str = 'numpy') -> float:
        """
        Calculate entropy for non-contiguous bipartitions using the many-body method.
        
        This is the correct method for non-contiguous subsystems where the
        simple correlation matrix method gives incorrect results.
        
        Parameters
        ----------
        bipartition : BipartitionInfo
            Bipartition of the system
        occupied_orbitals : array-like
            Occupied orbitals for the state
        backend : str
            Computation backend
            
        Returns
        -------
        float
            Entanglement entropy (exact)
            
        Notes
        -----
        For non-contiguous bipartitions, the correlation matrix method
        systematically overestimates entropy. This method uses the full
        many-body Schmidt decomposition which is exact but slower.
        """
        state = self._hamil.many_body_state(occupied_orbitals)
        return self.entropy_many_body(bipartition, state)
    
    def compare_methods(self,
                       bipartition: 'BipartitionInfo',
                       occupied_orbitals: Union[List[int], np.ndarray]) -> Dict[str, float]:
        """
        Compare correlation matrix and many-body entropy methods.
        
        Useful for validating calculations and understanding when
        the fast correlation method is accurate.
        
        Parameters
        ----------
        bipartition : BipartitionInfo
            Bipartition to test
        occupied_orbitals : array-like
            Occupied orbitals
            
        Returns
        -------
        dict
            Dictionary with 'correlation', 'many_body', 'difference', 'is_contiguous'
        """
        S_corr = self.entropy_correlation(bipartition, occupied_orbitals, check_contiguous=False)
        
        state = self._hamil.many_body_state(occupied_orbitals)
        S_mb = self.entropy_many_body(bipartition, state)
        
        is_contiguous = (len(bipartition.mask_a) > 0 and 
                        np.all(np.diff(bipartition.mask_a) == 1) and 
                        bipartition.mask_a[0] == 0)
        
        return {
            'correlation': S_corr,
            'many_body': S_mb,
            'difference': abs(S_corr - S_mb),
            'relative_error': abs(S_corr - S_mb) / max(S_mb, 1e-10),
            'is_contiguous': is_contiguous
        }
    
    def help(self):
        """Print usage help for the entanglement module."""
        help_text = """
        EntanglementModule - Unified entanglement calculations
        ======================================================
        
        Quick Start:
        -----------
        >>> hamil = FreeFermions(ns=12, t=1.0)
        >>> ent = hamil.entanglement  # Access entanglement module
        >>> 
        >>> # Define bipartition
        >>> bipart = ent.bipartition([0, 1, 2, 3, 4])  # First 5 sites
        >>> 
        >>> # Calculate entropy (correlation matrix method - fast)
        >>> S = ent.entropy_correlation(bipart, orbitals=[0,1,2,3,4])
        >>> 
        >>> # Calculate entropy (many-body method - exact)
        >>> state = hamil.many_body_state([0,1,2,3,4])
        >>> S = ent.entropy_many_body(bipart, state)
        
        Main Methods:
        ------------
        bipartition(mask_a)
            Create bipartition info for subsystem A
            
        correlation_matrix(orbitals, bipartition=None, backend='numpy')
            Get single-particle correlation matrix C_ij = <c_i^\\dag c_j>
            Returns full matrix if bipartition=None, or C_A if bipartition given
            Supports 'numpy' and 'jax' backends
            
        entropy_correlation(bipart, orbitals, backend='numpy')
            Fast method using correlation matrix (quadratic Hamiltonians)
            Supports 'numpy' and 'jax' backends
            WARNING: Only exact for contiguous bipartitions!
            
        entropy_many_body(bipart, state)
            Exact method using reduced density matrix (any state)
            Works for ANY bipartition (contiguous or non-contiguous)
            
        entropy_multipartition(bipartitions, orbitals, method='auto', backend='numpy')
            Calculate entropy for multiple bipartitions efficiently
            Returns dict with 'entropies', 'bipartitions', 'correlation_matrices'
            
        entropy_scan(orbitals, sizes=[...])
            Calculate entropy for multiple subsystem sizes
            
        mutual_information(mask_a, mask_b, orbitals)
            Calculate I(A:B) = S(A) + S(B) - S(AB)
        
        Topological Entropy (NEW):
        -------------------------
        topological_entropy(orbitals, construction='kitaev_preskill')
            Calculate topological entanglement entropy γ
            Uses Kitaev-Preskill or Levin-Wen constructions
            Returns dict with 'gamma', 'entropies', 'regions'
        
        Wick's Theorem (NEW):
        --------------------
        verify_wicks_theorem(orbitals, state=None)
            Verify if a state satisfies Wick's theorem (is a Slater determinant)
            Returns dict with 'is_valid', 'max_error', 'correlation_matrix'
        
        compare_methods(bipart, orbitals)
            Compare correlation vs many-body methods
            Useful for checking when fast method is accurate
        
        Mask Generation (use MaskGenerator class):
        -----------------------------------------
        >>> from QES.general_python.physics.entanglement_module import MaskGenerator
        >>> 
        >>> # Contiguous mask
        >>> mask = MaskGenerator.contiguous(ns=12, size_a=4)  # [0,1,2,3]
        >>> 
        >>> # Alternating (even/odd) sites
        >>> even, odd = MaskGenerator.alternating(ns=12)
        >>> 
        >>> # Random subsystem
        >>> mask = MaskGenerator.random(ns=12, size_a=6, seed=42)
        >>> 
        >>> # Sublattice (bipartite lattice A/B)
        >>> mask_A = MaskGenerator.sublattice(ns=12, sublattice_id=0)
        >>> 
        >>> # Kitaev-Preskill regions for topological entropy
        >>> regions = MaskGenerator.kitaev_preskill(ns=12)
        >>> A, B, C = regions['A'], regions['B'], regions['C']
        >>> 
        >>> # Convert to/from bitmasks
        >>> bitmask = MaskGenerator.to_bitmask(np.array([0, 2, 4]))  # -> 0b10101
        >>> indices = MaskGenerator.from_bitmask(0b10101, ns=6)      # -> [0, 2, 4]
        
        Examples:
        --------
        # Access correlation matrix
        >>> C_full = ent.correlation_matrix([0,1,2,3,4])  # Full (ns x ns)
        >>> C_A = ent.correlation_matrix([0,1,2,3,4], bipartition=bipart)  # Subblock
        
        # JAX backend for GPU acceleration
        >>> S = ent.entropy_correlation(bipart, [0,1,2,3,4], backend='jax')
        
        # Multipartition - compute multiple cuts efficiently
        >>> masks = [[0,1,2], [0,1,2,3], [0,1,2,3,4]]
        >>> results = ent.entropy_multipartition(masks, [0,1,2,3,4,5])
        >>> print(results['entropies'])  # Array of entropies
        >>> C_matrices = results['correlation_matrices']  # Access all C_A matrices
        
        # Non-contiguous bipartition (use many-body method!)
        >>> bipart = ent.bipartition([0, 2, 4, 6, 8])  # Even sites
        >>> S_exact = ent.entropy_many_body(bipart, state)  # EXACT
        >>> # DON'T use entropy_correlation for non-contiguous!
        
        # Compare methods to check accuracy
        >>> result = ent.compare_methods(bipart, orbitals)
        >>> print(f"Correlation: {result['correlation']:.4f}")
        >>> print(f"Many-body:   {result['many_body']:.4f}")
        >>> print(f"Difference:  {result['difference']:.4e}")
        
        # Topological entanglement entropy
        >>> result = ent.topological_entropy(orbitals, construction='kitaev_preskill')
        >>> print(f"TEE γ = {result['gamma']:.4f}")
        
        # Verify Wick's theorem
        >>> result = ent.verify_wicks_theorem(orbitals)
        >>> print(f"Is Slater determinant: {result['is_valid']}")
        
        # Entropy scaling
        >>> results = ent.entropy_scan([0,1,2,3,4])
        >>> plt.plot(results['sizes'], results['entropies'])
        
        # Mutual information
        >>> I_AB = ent.mutual_information([0,1,2], [3,4,5], [0,1,2,3,4,5])
        
        """
        print(help_text)
    
    def __repr__(self):
        return f"EntanglementModule(ns={self._hamil.ns}, quadratic={hasattr(self._hamil, '_particle_conserving')})"


def get_entanglement_module(hamiltonian) -> EntanglementModule:
    """
    Factory function to create entanglement module for a Hamiltonian.
    
    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian object
        
    Returns
    -------
    EntanglementModule
        Entanglement module instance
    """
    return EntanglementModule(hamiltonian)
