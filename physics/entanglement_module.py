"""
This file contains the EntanglementModule class.

--------------------------------------------
file    : QES/general_python/physics/entanglement_module.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
--------------------------------------------

Unified entanglement calculation module for both quadratic and many-body Hamiltonians.

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
    
Manual many-body entropy calculations:
    >>> bipart      = ent.bipartition([0, 1, 2, 3])
    >>> S_manual    = ent.entropy_correlation(bipart, orbitals)
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Callable
from dataclasses import dataclass

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
                          check_contiguous      : bool = True,
                          backend               : str = 'numpy') -> float:
        """
        Calculate entanglement entropy from single-particle correlation matrix.
        
        **SINGLE-PARTICLE METHOD** - Fast method for non-interacting (quadratic)
        Hamiltonians. Computes entropy from correlation matrix eigenvalues.
        
        **IMPORTANT**: This method is only exact for CONTIGUOUS bipartitions!
        For non-contiguous bipartitions, use entropy_many_body() instead.
        
        Parameters
        ----------
        bipartition : BipartitionInfo
            Bipartition of the system (use ent.bipartition() to create)
        occupied_orbitals : array-like
            Indices of occupied orbitals (in eigenstate basis).
            For ground state with N particles, use [0, 1, ..., N-1].
        subtract_identity : bool
            Whether to subtract identity from correlation matrix (advanced)
        check_contiguous : bool
            If True, warns when used with non-contiguous bipartitions
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
        
        Non-contiguous partition (NOT exact, will warn):
            >>> bipart_nc = ent.bipartition([0, 2, 4, 6])  # Non-contiguous
            >>> S_nc = ent.entropy_correlation(bipart_nc, orbitals)
            >>> # WARNING: Will overestimate! Use entropy_many_body() instead.
            
        Notes
        -----
        Algorithm:
        1. Compute full correlation matrix C from occupied orbitals
        2. Extract subblock C_A for subsystem A  
        3. Diagonalize C_A to get eigenvalues (occupations)
        4. Apply single-particle entropy formula:
           S = -Σ [p log(p) + (1-p) log(1-p)] where p = λ/2
        
        For non-interacting systems with contiguous bipartitions, this gives
        the exact entanglement entropy and matches entropy_many_body().
        
        Limitations:
        - Only exact for contiguous bipartitions (subsystem A is a single interval)
        - For non-contiguous bipartitions, overestimates entropy by ~30-40%
        - Requires diagonalized Hamiltonian
        - Only works for quadratic (non-interacting) Hamiltonians
        """
        # Check if bipartition is contiguous
        is_contiguous = (len(bipartition.mask_a) > 0 and 
                        np.all(np.diff(bipartition.mask_a) == 1) and 
                        bipartition.mask_a[0] == 0)
        
        if check_contiguous and not is_contiguous:
            import warnings
            warnings.warn(
                "Using correlation matrix method for non-contiguous bipartition. "
                "This systematically overestimates entropy by ~30-40%. "
                "Consider using entropy_many_body() for accurate results.",
                UserWarning
            )
        
        # Get correlation matrix for subsystem A using Corr methods
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
           |ψ⟩ → ψ_{A,B} (matrix form)
        2. Perform Schmidt decomposition: |ψ⟩ = Σ √λ_i |i_A⟩|i_B⟩
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
            
        entropy_many_body(bipart, state)
            Exact method using reduced density matrix (any state)
            
        entropy_multipartition(bipartitions, orbitals, method='auto', backend='numpy')
            Calculate entropy for multiple bipartitions efficiently
            Returns dict with 'entropies', 'bipartitions', 'correlation_matrices'
            
        entropy_scan(orbitals, sizes=[...])
            Calculate entropy for multiple subsystem sizes
            
        mutual_information(mask_a, mask_b, orbitals)
            Calculate I(A:B) = S(A) + S(B) - S(AB)
        
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
        
        # Non-contiguous bipartition
        >>> bipart = ent.bipartition([0, 2, 4, 6, 8])  # Even sites
        
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
