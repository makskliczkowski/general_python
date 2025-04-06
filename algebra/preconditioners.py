'''
file:       general_python/algebra/preconditioners.py
author:     Maksymilian Kliczkowski

This module contains the implementation of preconditioners for iterative solvers
of linear systems Ax = b. Preconditioners transform the system into an
equivalent one,

M^{-1}Ax = M^{-1}b (left preconditioning) 

or

A M^{-1} y = b, x = M^{-1}y (right preconditioning),

where M is the preconditioner matrix.

The goal is for the transformed system to have more
favorable spectral properties (e.g., eigenvalues clustered around 1, lower
condition number), leading to faster convergence of iterative methods like CG,
MINRES, GMRES. The matrix M should approximate A in some sense, while the
operation M^{-1}r should be computationally inexpensive.
'''

# Import the required modules
from abc import ABC, abstractmethod
from typing import Union, Callable, Optional, Any, Type
import inspect
import numpy as np
import numpy.random as nrn
import scipy as sp
# Add sparse imports at the top of the file
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from .utils import _JAX_AVAILABLE, get_backend as get_backend, maybe_jit
from enum import Enum, auto, unique

# ---------------------------------------------------------------------

try:
    if _JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
        Array = Union[np.ndarray, jnp.ndarray]
    else:
        jnp   = None
        jax   = None # Define jax as None if not available
        Array = np.ndarray
except ImportError:
    jnp   = None
    jax   = None
    Array = np.ndarray

# ---------------------------------------------------------------------

# Interface for Preconditioner Apply Function (Static version)
# Takes residual, sigma, backend module, and precomputed data
PreconitionerApplyFun = Callable[[Array, float], Array]

# just the example
def preconditioner_idn(r: Array, sigma: Optional[float] = None) -> Array:
    """
    Identity function for preconditioner apply.
    
    Parameters:
        r (Array): The input array.
        
    Returns:
        Array: The same input array.
    """
    return r

@unique
class PreconditionersType(Enum):
    '''
    Enumeration of the symmetry type of preconditioners.
    '''
    SYMMETRIC           = auto()
    NONSYMMETRIC        = auto()

@unique
class PreconditionersTypeSym(Enum):
    """
    Enumeration of specific symmetric preconditioner types.
    """
    IDENTITY            = auto()
    JACOBI              = auto()
    INCOMPLETE_CHOLESKY = auto()
    COMPLETE_CHOLESKY   = auto()
    SSOR                = auto()
@unique
class PreconditionersTypeNoSym(Enum):
    """
    Enumeration of specific potentially non-symmetric preconditioner types.
    """
    IDENTITY            = auto() # Identity is also symmetric
    INCOMPLETE_LU       = auto()
    # Add others like Gauss-Seidel etc.
    
# ---------------------------------------------------------------------
#! Preconditioners
# ---------------------------------------------------------------------

class Preconditioner(ABC):
    """
    Abstract base class for preconditioners M used in iterative solvers.

    Provides a framework for setting up the preconditioner based on a matrix A
    (or its factors S, Sp for Gram matrices) and applying the inverse operation
    M^{-1}r efficiently. Supports different computational backends (NumPy, JAX).

    Attributes:
        is_positive_semidefinite (bool):
            Indicates if the original matrix A (and potentially M) is
            assumed to be positive semi-definite. Important for methods like Cholesky.
        is_gram (bool):
            True if the preconditioner setup uses factors S, Sp such that A = Sp @ S / N.
        sigma (float):
            Regularization parameter sigma added during setup, effectively forming M
            based on A + sigma*I.
        type (Enum):
            The specific type of the preconditioner (e.g., JACOBI, ILU). Set by subclass.
        stype (PreconditionersType):
            Symmetry type (SYMMETRIC/NONSYMMETRIC).
        backend_str (str): The name of the current backend ('numpy', 'jax').
    """
    
    _type : Optional[Union[PreconditionersTypeNoSym, PreconditionersTypeSym]] = None
    _name : str = "General Preconditioner"
    
    def __init__(self,
                is_positive_semidefinite                        = False,
                is_gram                                         = False,
                backend                                         = 'default',
                apply_func: Optional[PreconitionerApplyFun]     = None):
        """
        Initialize the preconditioner.
        
        Parameters:
            is_positive_semidefinite (bool):
                True if the matrix is positive semidefinite.
            is_gram (bool):
                True if the preconditioner setup uses factors S, Sp such that A = Sp @ S / N.
            backend (optional):
                The computational backend to be used by the preconditioner.
            apply_func (PreconitionerApplyFun, optional):
                The apply function for the preconditioner.
        """
        self._backend_str                : str
        self._backend                    : Any  # The numpy-like module (np or jnp)
        self._backends                   : Any  # The scipy-like module (sp or jax.scipy)
        self._isjax                      : bool # True if using JAX backend
        self.reset_backend(backend) # Sets backend attributes
        self._tol_small                  = 1e-10
        self._tol_big                    = 1e10
        self._zero                       = 1e10

        self._is_positive_semidefinite   = is_positive_semidefinite     # Positive semidefinite flag
        self._is_gram                    = is_gram                      # Gram matrix flag

        if self._is_positive_semidefinite:
            self._stype                  = PreconditionersType.SYMMETRIC
        else:
            # May still be symmetric, but we don't assume based on this flag alone
            self._stype                  = PreconditionersType.NONSYMMETRIC

        self._sigma                      = 0.0                          # Regularization parameter

        # Store reference to the static apply logic of the concrete class
        self._base_apply_logic : Callable[..., Array] = self.__class__.apply if not apply_func else apply_func

        # Compiled/wrapped apply function for instance usage
        self._apply_func : PreconitionerApplyFun = None
        
        # Trigger initial creation of the wrapped/compiled function
        self._update_apply_func()
    # -----------------------------------------------------------------
    
    def reset_backend(self, backend: str):
        '''
        Resets the backend and recompiles the internal apply function.

        Parameters:
            backend (str): The name of the new backend ('numpy', 'jax').
        '''
        new_backend_str = backend if backend != 'default' else 'numpy' # Resolve default
        if not hasattr(self, '_backend_str') or self._backend_str != new_backend_str:
            print(f"({self._name}) Resetting backend to: {new_backend_str}")
            self._backend_str               = new_backend_str
            self._backend, self._backends   = get_backend(self._backend_str, scipy=True)
            self._isjax                     = _JAX_AVAILABLE and self._backend is not np
            # Re-create the wrapped/compiled apply function for the new backend
            self._update_apply_func()
        
    # -----------------------------------------------------------------
    #! Closure for the apply function
    # -----------------------------------------------------------------
    
    def _update_apply_func(self):
        """
        Creates the instance's apply function, potentially JIT-compiled.

        This internal method wraps the static `apply` logic of the concrete class,
        injecting `self` (for precomputed data) and `self._sigma`, and handles
        JIT compilation if the backend is JAX.
        """
        if not hasattr(self, '_base_apply_logic'):
            # Should not happen if __init__ runs correctly
            print("Warning: _base_apply_logic not set during _update_apply_func. Setting to None.")
            self._apply_func = self.__class__.apply
            return

        base_apply      = self._base_apply_logic
        backend_mod     = self._backend

        # Define the function to be potentially compiled.
        # It calls the static method, passing necessary instance data.
        def wrapped_apply_instance(r: Array, sig: float) -> Array:
            # Get data computed during set() specific to the instance
            precomputed_data    = self._get_precomputed_data()
            # Call the static apply logic of the concrete class
            # Pass instance data as well
            return base_apply(  r               =   r,
                                sigma           =   sig,
                                backend_mod     =   backend_mod,
                                **precomputed_data)

        # Apply JIT compilation if using JAX backend
        if self._isjax and jax is not None:
            print(f"({self._name}) JIT compiling apply function...")
            self._apply_func = jax.jit(wrapped_apply_instance)
        else:
            # !TODO: Add numba?
            self._apply_func = wrapped_apply_instance
    
    # -----------------------------------------------------------------
    
    def get_apply(self) -> Callable[[Array], Array]:
        '''
        Returns the compiled/wrapped version of the apply function for this instance.

        The returned function takes only the residual vector `r` as input.
        '''
        if self._apply_func is None:
            # Should be created by __init__ or reset_backend
            raise RuntimeError("Preconditioner apply function not initialized.")
        return self._apply_func
    
    # -----------------------------------------------------------------
    
    @property
    def is_positive_semidefinite(self) -> bool:
        '''True if the matrix A (and potentially M) is positive semidefinite.'''
        return self._is_positive_semidefinite

    @property
    def stype(self) -> PreconditionersType:
        '''Symmetry type (SYMMETRIC/NONSYMMETRIC).'''
        return self._stype

    @property
    def is_gram(self) -> bool:
        '''True if the preconditioner is set up from Gram matrix factors S, Sp.'''
        return self._is_gram

    @is_gram.setter
    def is_gram(self, value: bool):
        '''Set the is_gram flag. Requires re-running set().'''
        if self._is_gram != value:
            print(f"({self._name}) Changed is_gram to {value}. Remember to call set() again.")
            self._is_gram = value

    @property
    def type(self) -> Optional[Union[PreconditionersTypeNoSym, PreconditionersTypeSym]]:
        """Specific preconditioner type (e.g., JACOBI, ILU)."""
        return self._type

    @property
    def backend_str(self) -> str:
        """Name of the current backend ('numpy', 'jax')."""
        return self._backend_str
    
    # -----------------------------------------------------------------
    #! Reg
    # -----------------------------------------------------------------
    
    @property
    def sigma(self):
        """Regularization parameter."""
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
    
    # -----------------------------------------------------------------
    #! Tolerances
    # -----------------------------------------------------------------
    
    @property
    def tol_big(self):
        '''Tolerance for big values.'''
        return self._tol_big

    @tol_big.setter
    def tol_big(self, value):
        self._tol_big = value
        
    @property
    def tol_small(self):
        '''Tolerance for small values.'''
        return self._tol_small
    
    @tol_small.setter
    def tol_small(self, value):
        self._tol_small = value
    
    @property
    def zero(self):
        '''Value treated as zero.'''
        return self._zero

    @zero.setter
    def zero(self, value):
        self._zero = value
    
    # -----------------------------------------------------------------
    

    @abstractmethod
    def _set_gram(self, s: Array, sp: Array, sigma: float):
        """
        Abstract method to set up the preconditioner using Gram matrix factors S, Sp.
        Stores necessary precomputed data within the instance.

        Effectively based on A = (1/N) Sp @ S + sigma * I (N often inferred or provided).

        Args:
            s (Array)      : Matrix S.
            sp (Array)     : Matrix Sp (transpose or conjugate transpose of S).
            sigma (float)  : Regularization parameter.
        """
        # Implementation in subclass extracts/computes necessary data (e.g., diagonal)
        # and stores it in instance attributes (e.g., self._inv_diag).
        # This method should use self._backend for array operations.
        raise NotImplementedError

    @abstractmethod
    def _set_standard(self, a: Array, sigma: float):
        """
        Abstract method to set up the preconditioner using the standard matrix A.
        Stores necessary precomputed data within the instance.

        Effectively based on A + sigma * I.

        Args:
            a (Array)     : The matrix A.
            sigma (float) : Regularization parameter.
        """
        # Implementation in subclass extracts/computes necessary data (e.g., diagonal, ILU factors)
        # and stores it in instance attributes (e.g., self._inv_diag, self._L, self._U).
        # This method should use self._backend and self._backends (for scipy functions).
        raise NotImplementedError
    
    @abstractmethod
    def _get_precomputed_data(self) -> dict:
        """
        Abstract method to retrieve the precomputed data needed by the static apply method.

        Returns:
            dict: A dictionary containing necessary data (e.g., {'inv_diag': self._inv_diag}).
                  Keys must match the expected **kwargs in the static apply method.
        """
        # Example: return {'inv_diag': self._inv_diag} for Jacobi
        raise NotImplementedError
    
    # -----------------------------------------------------------------
    
    def set(self, a: Array, sigma: float = 0.0, ap: Optional[Array] = None, backend: Optional[str] = None):
        """
        Set up the preconditioner from matrix A or factors S (a) and Sp (ap).

        Determines whether to use standard (A) or Gram (S, Sp) setup based on
        `self.is_gram` and provided arguments. Resets backend if specified.

        Args:
            a (Array):
                Matrix A, or matrix S if `is_gram` is True.
            sigma (float):
                Regularization parameter.
            ap (Optional[Array]):
                Matrix Sp if `is_gram` is True. If None and `is_gram`,
                Sp is computed as conjugate(a).T.
            backend (Optional[str]):
                Backend ('numpy', 'jax') to switch to before setting up.
        """
        if backend is not None and backend != self.backend_str:
            self.reset_backend(backend)
        else:
            a       = self._backend.asarray(a)
            if ap is not None:
                ap  = self._backend.asarray(ap)

        actual_sigma    = sigma if sigma is not None else self._sigma
        self._sigma     = actual_sigma
        
        # set it all up
        print(f"({self._name}) Setting up preconditioner with sigma={self.sigma} using backend='{self.backend_str}'...")

        # is a Gram matrix
        if self.is_gram:
            s_mat           = a
            sp_mat: Array   = ap
            if sp_mat is None:
                print(f"({self._name}) Computing Sp = conjugate(S).T")
                sp_mat      = self._backend.conjugate(s_mat).T
                
            # Validate shapes
            if s_mat.shape[1] != sp_mat.shape[0] or s_mat.shape[0] != sp_mat.shape[1]:
                raise ValueError(f"Shape mismatch for Gram setup: S={s_mat.shape}, Sp={sp_mat.shape}")
            self._set_gram(s_mat, sp_mat, self.sigma)
        else:
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise ValueError(f"Standard setup requires a square 2D matrix A, got shape {a.shape}")
            self._set_standard(a, self.sigma)

    @staticmethod
    @abstractmethod
    def apply(r: Array, sigma: float, backend_mod: Any, **precomputed_data: Any) -> Array:
        '''
        Applies the core preconditioner logic M^{-1}r statically.

        This method contains the mathematical algorithm using the provided backend
        and precomputed data.

        Args:
            r (Array):
                    The residual vector to precondition.
            sigma (float):
                    The regularization parameter used in M.
            backend_mod (Any):
                    The backend module (e.g., np, jnp) to use for computations.
            **precomputed_data (Any):
                    Precomputed data stored by the instance during set()
                    (e.g., inv_diag for Jacobi, L/U factors for ILU).
                    Keys must match what the specific implementation expects.

        Returns:
            wrapper that accepts an array and sigma and returns the array after preconditioning

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        '''
        raise NotImplementedError("Static apply method must be implemented by subclasses.")
    
    def __call__(self, r: Array) -> Array:
        """
        Apply the configured preconditioner instance M^{-1} to a vector r.

        Uses the internally wrapped and potentially compiled apply function.

        Args:
            r (Array): The vector to precondition.

        Returns:
            Array: The preconditioned vector M^{-1}r.
        """
        if self._apply_func is None:
            raise RuntimeError(f"({self._name}) Preconditioner apply function not available. Was set() called?")
        # Ensure input array matches backend
        r_backend = self._backend.asarray(r)
        return self._apply_func(r_backend)

    # -----------------------------------------------------------------
    
    def __repr__(self) -> str:
        ''' Returns the name and configuration of the preconditioner. '''
        return f"{self._name}(sigma={self.sigma}, backend='{self.backend_str}', type={self.type})"

    def __str__(self) -> str:
        ''' Returns the name of the preconditioner. '''
        return self.__repr__()
    
    # -----------------------------------------------------------------

# =====================================================================
#! Identity preconditioner
# =====================================================================

class IdentityPreconditioner(Preconditioner):
    """
    Identity preconditioner M = I. Applying M^{-1} simply returns the input vector.

    This is the simplest preconditioner and has no effect on the system.
    It serves as a baseline or placeholder.

    Math:
        M = I
        M^{-1}r = I^{-1}r = r
    """

    _name = "Identity Preconditioner"
    _type = PreconditionersTypeSym.IDENTITY # Can be Sym or NoSym

    def __init__(self, backend: str = 'default'):
        """
        Initialize the Identity preconditioner.

        Args:
            backend (str): The computational backend ('numpy', 'jax', 'default').
        """
        # is_positive_semidefinite doesn't matter, is_gram=False
        super().__init__(is_positive_semidefinite=True, is_gram=False, backend=backend)
        # Identity is always symmetric
        self._stype = PreconditionersType.SYMMETRIC

    # -----------------------------------------------------------------
    #! Setup Methods (Do Nothing)
    # -----------------------------------------------------------------

    def _set_gram(self, s: Array, sp: Array, sigma: float):
        """ Setup for Gram matrix (no-op for Identity). """
        # No data needs to be computed or stored.
        pass

    def _set_standard(self, a: Array, sigma: float):
        """ Setup for standard matrix (no-op for Identity). """
        # No data needs to be computed or stored.
        pass

    def _get_precomputed_data(self) -> dict:
        """ Returns empty dict as no precomputed data is needed. """
        return {}

    # -----------------------------------------------------------------
    #! Static Apply Method
    # -----------------------------------------------------------------

    @staticmethod
    def apply(r: Array, sigma: float, backend_mod: Any, **precomputed_data: Any) -> Array:
        '''
        Static apply method for the Identity preconditioner. Returns the input vector `r`.

        Args:
            r (Array)             : The residual vector.
            sigma (float)         : Regularization (ignored).
            backend_mod (Any)     : The backend module (ignored).
            **precomputed_data (Any): Ignored.

        Returns:
            Array: The input vector `r`.
        '''
        # Ensure output has same type as input for consistency if needed by backend
        return backend_mod.asarray(r) if backend_mod is not None else r

# =====================================================================
#! Jacobi preconditioner
# =====================================================================

class JacobiPreconditioner(Preconditioner):
    """
    Jacobi (Diagonal) Preconditioner. M = diag(A + sigma*I).

    Uses the diagonal of the (potentially regularized) matrix A as the
    preconditioner M. Applying the inverse M^{-1}r involves element-wise
    division by the diagonal entries.

    Math:
        M       = diag(A) + sigma*I = D + sigma*I
        M^{-1}r = (D + sigma*I)^{-1} r = [1 / (A_ii + sigma)] * r_i

    References:
        - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM. Chapter 10.
    """
    _name = "Jacobi Preconditioner"
    _type = PreconditionersTypeSym.JACOBI

    def __init__(self,
                is_positive_semidefinite: bool = False,
                is_gram                 : bool = False,
                backend                 : str  = 'default',
                # Tolerances specific to Jacobi
                tol_small               : float = 1e-10,
                zero_replacement        : float = 1e10):
        """
        Initialize the Jacobi preconditioner.

        Args:
            is_positive_semidefinite (bool):
                If A is positive semi-definite.
            is_gram (bool):
                If setting up from Gram matrix factors.
            backend (str):
                The computational backend.
            tol_small (float):
                Values on diagonal smaller than this (in magnitude)
                after regularization are considered zero.
            zero_replacement (float):
                Value used to replace division by zero
                (effectively setting the result component to zero,
                1 / large_number -> 0).
        """
        super().__init__(is_positive_semidefinite   =   is_positive_semidefinite,
                        is_gram                     =   is_gram,
                        backend                     =   backend)
        # Tolerances / constants for safe division
        self._tol_small         = tol_small
        self._zero              = zero_replacement
        # Precomputed data storage
        self._inv_diag : Optional[Array] = None # Stores 1 / (diag(A) + sigma)

    # -----------------------------------------------------------------

    def _compute_inv_diag(self, diag_a: Array, sigma: float):
        """
        Internal helper to compute the inverse diagonal safely.
        
        Math:
            M = diag(A) + sigma*I
            M^{-1} = diag(1 / (A_ii + sigma))
        
        Sets self._inv_diag to the computed inverse diagonal of diagonal matrix A + sigma*I.
        """
        be              = self._backend
        if sigma is not None and sigma != 0.0:
            reg_diag    = diag_a + sigma
        else:
            reg_diag    = diag_a
        # Handle small or zero values after regularization
        is_small        = be.abs(reg_diag) < self._tol_small
        # Replace small values with a large number before inversion
        safe_diag       = be.where(is_small, be.sign(reg_diag) * self._zero, reg_diag)
        # Replace exact zeros with the replacement value directly
        safe_diag       = be.where(safe_diag == 0.0, self._zero, safe_diag)
        
        self._inv_diag  = 1.0 / safe_diag
        # Ensure components corresponding to initially small values are effectively zero
        self._inv_diag  = be.where(is_small, 0.0, self._inv_diag)
        print(f"({self._name}) Computed inverse diagonal. "
            f"Number of near-zero entries handled: {be.sum(is_small)}")

    # -----------------------------------------------------------------

    def _set_standard(self, a: Array, sigma: float):
        """
        Sets up Jacobi from matrix A.
        
        Computes diag(A + sigma*I) and stores the inverse diagonal.
        """
        diag_a      = self._backend.diag(a)
        self._compute_inv_diag(diag_a, sigma)

    def _set_gram(self, s: Array, sp: Array, sigma: float):
        """
        Sets up Jacobi from factors S, Sp. Computes diag(Sp @ S / N).
        
        This is especially usefull when one does not want to compute the full
        Gram matrix A = Sp @ S / N explicitly.
        
        Parameters:
            s (Array):
                Matrix S.
            sp (Array):
                Matrix Sp (conjugate transpose of S).
            sigma (float):
                Regularization parameter.
        """
        be          = self._backend
        # Estimate N if not provided? Assume N = s.shape[0] (num outputs)
        n           = float(s.shape[0])
        if n <= 0.0:
            n = 1.0
        
        # Efficiently compute diagonal of Sp @ S without forming the full matrix
        # diag(Sp @ S)_i = sum_k Sp_{ik} S_{ki} = sum_k (S^T)_{ik} S_{ki}
        # = sum_k S_{ki} * S_{ki} (if real)
        # = sum_k conjugate(S_{ki}) * S_{ki} (if complex) = sum_k |S_{ki}|^2
        # This uses element-wise multiplication and sum along axis 0 of Sp^T (or axis 1 of S)
        # diag_SpS = be.sum(be.conjugate(sp.T) * s, axis=1)
        diag_s_dag_s    = be.einsum('ij,ji->i', sp, s)
        diag_a_approx   = diag_s_dag_s / n
        self._compute_inv_diag(diag_a_approx, sigma)

    # -----------------------------------------------------------------

    def _get_precomputed_data(self) -> dict:
        """ Returns the computed inverse diagonal. """
        if self._inv_diag is None:
            raise RuntimeError(f"({self._name}) Preconditioner not set up. Call set() first.")
        return {'inv_diag': self._inv_diag}

    # -----------------------------------------------------------------

    @staticmethod
    def apply(r: Array, sigma: float, backend_mod: Any, inv_diag: Array) -> Array:
        '''
        Static apply method for Jacobi: element-wise multiplication by inverse diagonal.

        Args:
            r (Array):
                The residual vector.
            sigma (float):
                Regularization (already included in inv_diag).
            backend_mod (Any):
                The backend module (e.g., np, jnp).
            inv_diag (Array):
                The precomputed inverse diagonal (1 / (A_ii + sigma)).

        Returns:
            Array:
                The preconditioned vector inv_diag * r.
        '''
        if inv_diag is None:
            raise ValueError("Jacobi 'inv_diag' data not provided to static apply.")
        if r.shape != inv_diag.shape or r.ndim != 1:
            raise ValueError(f"Shape mismatch in Jacobi apply: r={r.shape}, inv_diag={inv_diag.shape}")
        
        # Element-wise multiplication
        return inv_diag * r

    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        ''' 
        Returns the name and configuration of the Jacobi preconditioner. 
        '''
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, tol_small={self._tol_small})"
    
    # -----------------------------------------------------------------

# =====================================================================
#! Complete Cholesky factorization
# =====================================================================

class CholeskyPreconditioner(Preconditioner):
    """
    Cholesky Preconditioner using complete Cholesky decomposition.

    Suitable for symmetric positive-definite matrices A. The preconditioner M
    is defined by the Cholesky factorization of the regularized matrix:
    M = L @ L.T (or L @ L.conj().T for complex), where L is the lower
    Cholesky factor of (A + sigma*I).

    Applying the inverse M^{-1}r involves solving two triangular systems:
    1. Solve L y = r for y (forward substitution)
    2. Solve L.T z = y for z (backward substitution)
        (or L.conj().T z = y for complex)

    Note: 
        This performs a *complete* Cholesky factorization. For *incomplete*
        Cholesky (suitable for large sparse matrices), specialized routines
        (often from sparse linear algebra libraries) are required.

    References:
        - Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). JHU Press. Chapter 4.
        - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM. Chapter 10.
    """
    _name = "Cholesky Preconditioner"
    _type = PreconditionersTypeSym.COMPLETE_CHOLESKY

    def __init__(self, backend: str = 'default'):
        """
        Initialize the Cholesky preconditioner.

        Args:
            backend (str): The computational backend ('numpy', 'jax', 'default').
        """
        
        # Always requires positive definite matrix (after regularization)
        # Set is_gram=False, as Cholesky typically applies directly to A.
        
        super().__init__(is_positive_semidefinite=True, is_gram=False, backend=backend)
        self._stype                 = PreconditionersType.SYMMETRIC
        self._l: Optional[Array]    = None # Stores the lower Cholesky factor        

    # -----------------------------------------------------------------
    #! Properties
    # -----------------------------------------------------------------
    
    @property
    def l(self) -> Optional[Array]:
        '''The computed lower Cholesky factor L.'''
        return self._l

    # -----------------------------------------------------------------
    #! Setup Methods
    # -----------------------------------------------------------------

    def _set_standard(self, a: Array, sigma: float):
        """ 
        Sets up Cholesky by factorizing A + sigma*I. 
        Computes the lower Cholesky factor L.
        Parameters:
            a (Array):
                The matrix A.
            sigma (float):
                Regularization parameter.
        Raises:
            ValueError: If the matrix is not square or not positive definite.
        """
        
        be              = self._backend
        print(f"({self._name}) Performing Cholesky decomposition...")
        try:
            # Regularize the matrix
            a_reg       = a + sigma * be.eye(a.shape[0], dtype=a.dtype)
            
            # Perform complete Cholesky decomposition
            self._l     = be.linalg.cholesky(a_reg, lower=True)
            print(f"({self._name}) Cholesky decomposition successful.")
            
        except Exception as e:
            print(f"({self._name}) Cholesky decomposition failed: {e}")
            print(f"({self._name}) Matrix might not be positive definite after regularization (sigma={sigma}).")
            self._l = None

    # -----------------------------------------------------------------

    def _set_gram(self, s: Array, sp: Array, sigma: float):
        """
        Sets up Cholesky by first forming A = Sp @ S / N and then factorizing.
        
        Warning:
            Forming A explicitly can be computationally expensive for large S.
        """
        be      = self._backend
        # Estimate N if not provided? Assume N = s.shape[0] (num outputs)
        n       = float(s.shape[0])
        if n <= 0.0:
            n = 1.0
        
        print(f"({self._name}) Warning: Forming explicit Gram matrix A = Sp @ S / N for Cholesky setup (N={n}).")
        a_gram  = (sp @ s) / n
        
        # Now call the standard setup
        self._set_standard(a_gram, sigma)

    # -----------------------------------------------------------------

    def _get_precomputed_data(self) -> dict:
        """ 
        Returns the computed Cholesky factor L.
        """
        
        # Do not raise error here, let apply handle None L
        # if self._l is None:
        #      raise RuntimeError(f"({self._name}) Preconditioner not set up or failed. Call set() first.")
        return {'l': self._l}

    # -----------------------------------------------------------------
    #! Static Apply Method
    # -----------------------------------------------------------------

    @staticmethod
    def apply(r: Array, sigma: float, backend_mod: Optional[Any], l: Optional[Array]) -> Array:
        '''
        Static apply method for Cholesky: solves L y = r and L.T z = y.

        Args:
            r (Array):
                The residual vector.
            sigma (float):
                Regularization (used during factorization, ignored here).
            backend_mod (Any):
                The backend numpy-like module (e.g., numpy or jax.numpy).
            l (Optional[Array]):
                The precomputed lower Cholesky factor.

        Returns:
            Array:
                The preconditioned vector M^{-1}r, or r if l is None.
        '''
        
        # Factorization failed or not performed, return original vector
        if l is None:
            print("Warning: Cholesky factor l is None in apply, returning original vector.")
            return r
        
        if backend_mod is None:
            raise ValueError(f"Which backend to use?")
        
        if r.shape[0] != l.shape[0]:
            raise ValueError(f"Shape mismatch in Cholesky apply: r ({r.shape[0]}) vs L ({l.shape[0]})")

        try:
            # Check if matrix is complex to use conjugate transpose
            use_conj_transpose = backend_mod.iscomplexobj(l)
            
            lh                  = backend_mod.conjugate(l).T if use_conj_transpose else l.T

            # Forward substitution: Solve L y = r
            y                   = backend_mod.linalg.solve_triangular(l, r, lower=True, check_finite=False)
            # Backward substitution: Solve L^H z = y
            z                   = backend_mod.linalg.solve_triangular(lh, y, lower=False, check_finite=False, trans='N')
            return z
        except Exception as e:
            print(f"Cholesky triangular solve failed during apply: {e}")
            return r            # Return original vector if solve fails


    # -----------------------------------------------------------------
    
    def __repr__(self) -> str:
        status      = "Factorized" if self._l is not None else "Not Factorized/Failed"
        base_repr   = super().__repr__()
        return f"{base_repr[:-1]}, status='{status}')"
    
    # -----------------------------------------------------------------

# =====================================================================
#! SSOR Preconditioner
# =====================================================================

class SSORPreconditioner(Preconditioner):
    """
    Symmetric Successive Over-Relaxation (SSOR) Preconditioner.

    Suitable for symmetric matrices A, particularly those that are positive definite.
    It involves forward and backward Gauss-Seidel-like sweeps with a relaxation parameter omega.

    Math:
        Let A = D + L + U, where D is diag(A), L is strictly lower part, U is strictly upper part.
        The SSOR preconditioner matrix M is defined implicitly by its application M^{-1}r:
        1. Solve (D/omega + L) y = r        (Forward sweep)
        2. Solve (D/omega + U) z = D y / omega  (Backward sweep - check formulation)

        A common formulation for the inverse application M^{-1}r is:
        M^{-1} = omega * (2 - omega) * (D/omega + U)^{-1} D (D/omega + L)^{-1}
        Applying M^{-1}r involves solving the two triangular systems mentioned above.

        The choice of omega (0 < omega < 2) is crucial for performance. omega=1 gives
        Symmetric Gauss-Seidel (SGS).

    References:
        - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM. Chapter 10.
        - Axelsson, O. (1996). Iterative Solution Methods. Cambridge University Press.
    """
    
    # -----------------------------------------------------------------
    
    _name = "SSOR Preconditioner"           # Although often used for SPD, the setup doesn't strictly require it
    _type = PreconditionersTypeSym.JACOBI   # No specific SSOR enum yet, Jacobi is placeholder

    def __init__(self,
                omega                   : float = 1.0,
                is_positive_semidefinite: bool = False,     # User hint
                is_gram                 : bool = False,     # Typically used with explicit A
                backend                 : str  = 'default', # Backend for computations
                tol_small               : float = 1e-10,    # For inverting diagonal
                zero_replacement        : float = 1e10):
        """
        Initialize the SSOR preconditioner.

        Args:
            omega (float):
                Relaxation parameter (0 < omega < 2). Default is 1.0 (SGS).
            is_positive_semidefinite (bool):
                If A is assumed positive semi-definite.
            is_gram (bool):
                If setting up from Gram matrix factors (less common for SSOR).
            backend (str):
                The computational backend.
            tol_small (float):
                Tolerance for safe diagonal inversion.
            zero_replacement (float):
                Value for safe diagonal inversion.
        """
        if not (0 < omega < 2):
            raise ValueError("SSOR relaxation parameter omega must be between 0 and 2.")

        super().__init__(is_positive_semidefinite=is_positive_semidefinite,
                        is_gram=is_gram,
                        backend=backend)
        # Assume symmetric application if A is symmetric
        self._stype             = PreconditionersType.SYMMETRIC

        self._omega             = omega
        self._tol_small         = tol_small
        self._zero              = zero_replacement  

        # Precomputed data storage
        self._inv_diag_scaled : Optional[Array] = None # Stores omega / (diag(A) + sigma)
        self._L_scaled        : Optional[Array] = None # Stores omega * L
        self._U_scaled        : Optional[Array] = None # Stores omega * U

    # -----------------------------------------------------------------
    #! Properties
    # -----------------------------------------------------------------
    
    @property
    def omega(self) -> float:
        ''' Relaxation parameter omega (0 < omega < 2). '''
        return self._omega
    
    @omega.setter
    def omega(self, value: float):
        if not (0 < value < 2):
            raise ValueError("omega must be in (0, 2)")
        self._omega = value
        # Note: Need to call set() again to recompute scaled factors

    # -----------------------------------------------------------------
    #! Setup Methods
    # -----------------------------------------------------------------
    
    def _set_standard(self, a: Array, sigma: float):
        """ 
        Sets up SSOR from matrix A = D + L + U.
        Computes the lower and upper triangular parts of A + sigma*I.
        Parameters:
            a (Array):
                The matrix A.
            sigma (float):
                Regularization parameter.
        Raises:
            ValueError: If the matrix is not square or not positive definite.
        """
        
        print(f"({self._name}) Setting up SSOR factors...")
        be          = self._backend
        a_reg       = a + sigma * be.eye(a.shape[0], dtype=a.dtype)

        # Extract parts
        diag_a_reg  = be.diag(a_reg)
        # Use backend's tril/triu. Ensure k=-1/+1 for strict parts.
        L           = be.tril(a_reg, k=-1)
        U           = be.triu(a_reg, k=1)

        # Compute scaled inverse diagonal safely
        diag_scaled             = diag_a_reg / self._omega # Use original omega
        is_small                = be.abs(diag_scaled) < self._tol_small
        safe_diag_scaled        = be.where(is_small, be.sign(diag_scaled) * self._zero, diag_scaled)
        safe_diag_scaled        = be.where(safe_diag_scaled == 0.0, self._zero, safe_diag_scaled)
        
        self._inv_diag_scaled   = 1.0 / safe_diag_scaled
        self._inv_diag_scaled   = be.where(is_small, 0.0, self._inv_diag_scaled)
        
        # Store D_inv_scaled, L and U (or scaled versions if preferred for apply)
        # Let's store the original L and U, apply will use omega
        self._L                 = L
        self._U                 = U 
        
        # We need D for the backward step, store its inverse
        is_small_d              = be.abs(diag_a_reg) < self._tol_small
        safe_d                  = be.where(is_small_d, be.sign(diag_a_reg) * self._zero, diag_a_reg)
        safe_d                  = be.where(safe_d == 0.0, self._zero, safe_d)
        self._inv_diag_unscaled = 1.0 / safe_d
        self._inv_diag_unscaled = be.where(is_small_d, 0.0, self._inv_diag_unscaled)
        
        print(f"({self._name}) SSOR setup complete.")

    def _set_gram(self, s: Array, sp: Array, sigma: float):
        """
        Set up SSOR by forming A = Sp @ S / n first.
        This is less common for SSOR, but can be useful in some cases.
        """
        
        # Similar warning as Cholesky
        be      = self._backend
        n       = float(s.shape[0])
        if n <= 0.0:
            n = 1.0
        
        print(f"({self._name}) Warning: Forming explicit Gram matrix A = Sp @ S / N for SSOR setup (N={N}).")
        a_gram  = (sp @ s) / n
        self._set_standard(a_gram, sigma)

    def _get_precomputed_data(self) -> dict:
        """ Returns the computed factors needed for SSOR apply. """
        if self._inv_diag_unscaled is None or self._L is None or self._U is None:
            raise RuntimeError(f"({self._name}) Preconditioner not set up. Call set() first.")
        
        return {
            'inv_diag_unscaled' : self._inv_diag_unscaled,
            'L'                 : self._L,
            'U'                 : self._U,
            'omega'             : self._omega
        }

    # --- Static Apply ---
    @staticmethod
    def apply(r                 : Array,
            sigma               : float,
            backend_mod         : Any,
            inv_diag_unscaled   : Array, L: Array, U: Array, omega: float) -> Array:
        '''
        Static apply method for SSOR: forward and backward triangular solves.
        Solves Mz = r where M = (D/w + L) D^{-1} (D/w + U) / (w(2-w))

        Args:
            r (Array):
                The residual vector.
            sigma (float):
                Regularization (used during setup).
            backend_mod (Any):
                The backend numpy-like module.
            inv_diag_unscaled (Array): Precomputed 1.0 / (diag(A)+sigma).
            L (Array):
                Precomputed strictly lower part of A+sigma*I.
            U (Array):
                Precomputed strictly upper part of A+sigma*I.
            omega (float):
                Relaxation parameter.

        Returns:
            Array: The preconditioned vector M^{-1}r.
        '''
        if inv_diag_unscaled is None or L is None or U is None:
            print("Warning: SSOR factors missing in apply, returning original vector.")
            return r

        be          = backend_module

        # Check for complex type
        use_conj    = be.iscomplexobj(L) # Assume L/U reflect complexness of A
        
        # Factor D/omega + L = D/omega * (I + omega * D^-1 * L)
        # Factor D/omega + U = D/omega * (I + omega * D^-1 * U)
        
        # Forward sweep: Solve (I + omega * D^-1 * L) y = r
        # This requires a triangular solve with matrix (I + omega*L*D_inv) -> Requires op @ vec
        # OR solve (D/omega + L) y = r directly
        d_over_omega_plus_l = be.diag(1.0 / (omega * inv_diag_unscaled)) + L
        y = be.linalg.solve_triangular(d_over_omega_plus_l, r, lower=True, check_finite=False)

        # Intermediate step (needed for common formulation):
        # y_intermediate = D y / omega (element-wise mult)
        y_intermediate = (1.0/omega * inv_diag_unscaled) * y # Check this scaling

        # Backward sweep: Solve (D/omega + U) z = y_intermediate
        # U_H = U.conj().T if use_conj else U.T # SSOR uses U, not U^H typically
        d_over_omega_plus_u = be.diag(1.0 / (omega * inv_diag_unscaled)) + U
        z = be.linalg.solve_triangular(d_over_omega_plus_u, y_intermediate, lower=False, check_finite=False)

        # No final scaling needed for this common form Mz=r solve
        
        # Alternative formulation M = (I + wL D^-1) D (I + wU D^-1) / (w(2-w))
        # Solve (I + w L D^-1) y = r (forward)
        # Solve (I + w U D^-1) z = D y (backward)
        # M^-1 r = w(2-w) z -> seems different from Saad? Check sources.
        
        # Let's stick to the solve Mz=r formulation -> z is the result
        return z

    # -----------------------------------------------------------------
    #! String Representation
    # -----------------------------------------------------------------
    
    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, omega={self.omega})"

    # -----------------------------------------------------------------
    
# =====================================================================
#! Incomplete Cholesky Preconditioner (using ILU Proxy)
# =====================================================================

class IncompleteCholeskyPreconditioner(Preconditioner):
    """
    Incomplete Cholesky Preconditioner (Approximation using ILU(0)).

    Suitable for large, sparse, symmetric positive-definite matrices A.
    This implementation uses SciPy's sparse ILU(0) decomposition as a proxy
    for IC(0), as a direct IC(0) is not available in SciPy/NumPy/JAX.
    ILU(0) computes factors L and U such that L @ U approximates A,
    maintaining the sparsity pattern of A (zero fill-in).

    Applying the inverse M^{-1}r involves solving L y = P r and U z = y, where P
    is a permutation matrix handled internally by the ILU object.

    **Note:**
    - Requires SciPy and the NumPy backend. Not JAX compatible.
    - Designed for sparse matrices. Dense input matrices will be converted.
    - Uses ILU(0) factorization as a proxy for IC(0).

    Args for `set`:
        fill_factor (float): See `scipy.sparse.linalg.spilu`. Default 1.
        drop_tol (float): See `scipy.sparse.linalg.spilu`. Default None.

    References:
        - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM. Chapter 10 (discusses both IC and ILU).
        - SciPy documentation for `scipy.sparse.linalg.spilu`.
    """
    _name = "Incomplete Cholesky Preconditioner (ILU Proxy)"
    _type = PreconditionersTypeSym.INCOMPLETE_CHOLESKY

    def __init__(self, backend: str = 'default'):
        """
        Initialize the Incomplete Cholesky (ILU Proxy) preconditioner.

        Args:
            backend (str): The computational backend. Must be 'numpy'.
        """
        
        # Requires positive definite matrix for true IC, ILU is more general but works best for SPD.
        super().__init__(is_positive_semidefinite=True, is_gram=False, backend=backend)
        self._stype = PreconditionersType.SYMMETRIC

        # Check for backend compatibility
        if self._backend != np:
            raise NotImplementedError(f"{self._name} requires the 'numpy' backend (uses SciPy sparse). "
                                f"Current backend: '{self.backend_str}'.")

        # Precomputed data storage
        self._ilu_obj : Optional[spsla.SuperLU] = None # Stores the SuperLU object from spilu
        self._fill_factor                       = 1.0
        self._drop_tol                          = None

    # -----------------------------------------------------------------
    #! Properties
    # -----------------------------------------------------------------
    
    @property
    def fill_factor(self) -> float:
        ''' The fill factor for ILU(0) (default 1.0). '''
        return self._fill_factor
    
    @fill_factor.setter
    def fill_factor(self, value: float): 
        if value <= 0:
            raise ValueError("fill_factor must be positive.")
        self._fill_factor = value

    @property
    def drop_tol(self) -> Optional[float]: 
        ''' The drop tolerance for ILU(0) (default None). '''
        return self._drop_tol
    
    @drop_tol.setter
    def drop_tol(self, value: Optional[float]): 
        if value is not None and value < 0:
            raise ValueError("drop_tol must be non-negative.")
        self._drop_tol = value

    # -----------------------------------------------------------------
    #! Setup Methods
    # -----------------------------------------------------------------

    def _set_standard(self, a: Array, sigma: float, **kwargs):
        """
        Sets up ILU factorization for A + sigma*I. 
        Converts dense input to sparse CSC format if needed.
        
        Parameters:
            a (Array):
                The matrix A.
            sigma (float):
                Regularization parameter.
            kwargs (dict):
                Additional parameters for ILU setup (e.g., fill_factor, drop_tol).
        Raises:
            RuntimeError: If the backend is not NumPy or if ILU factorization fails.
        """
        
        # Ensure NumPy backend
        if self._backend is not np:
            raise RuntimeError(f"{self._name} internal error: Backend is not NumPy despite check.")

        print(f"({self._name}) Setting up ILU factorization...")

        # Convert dense input to sparse CSC format (efficient for spilu)
        if not sps.issparse(a):
            print(f"({self._name}) Warning: Input matrix is dense. Converting to sparse CSC format.")
            a_sparse = sps.csc_matrix(a)
        else:
            # Ensure it's CSC format
            a_sparse = a.tocsc()

        # Apply regularization (sparse identity needed)
        if sigma != 0.0:
            print(f"({self._name}) Applying regularization sigma={sigma} to sparse matrix.")
            eye_sparse      = sps.identity(a.shape[0], dtype=a.dtype, format='csc')
            a_reg_sparse    = a_sparse + sigma * eye_sparse
        else:
            a_reg_sparse    = a_sparse

        # Get ILU parameters from kwargs or use defaults
        fill_factor         = kwargs.get('fill_factor', self._fill_factor)
        drop_tol            = kwargs.get('drop_tol', self._drop_tol)

        try:
            # Perform ILU decomposition using SuperLU through spilu
            self._ilu_obj   = spsla.spilu(a_reg_sparse,
                                        drop_tol            = drop_tol,
                                        fill_factor         = fill_factor,
                                        # panel_size        = options.panel_size,
                                        # relax             = options.relax,
                                        )
            print(f"({self._name}) ILU decomposition successful.")
        except RuntimeError as e:
            print(f"({self._name}) ILU decomposition failed: {e}")
            print(f"({self._name}) Matrix might be singular or factorization numerically difficult.")
            self._ilu_obj = None # Ensure object is None if decomposition fails

    # -----------------------------------------------------------------

    def _set_gram(self, s: Array, sp: Array, sigma: float, **kwargs):
        """ Set up ILU by forming A = Sp @ S / N first (sparse recommended). """
        # Ensure NumPy backend
        if self._backend is not np:
            raise RuntimeError(f"{self._name} internal error: Backend is not NumPy despite check.")

        be      = self._backend
        n       = float(s.shape[0])
        if n <= 0.0: 
            n = 1.0

        # Check if s/sp are sparse
        s_is_sparse     = sps.issparse(s)
        sp_is_sparse    = sps.issparse(sp)

        if not s_is_sparse or not sp_is_sparse:
            print(f"({self._name}) Warning: Forming explicit Gram matrix A = Sp @ S / N for ILU setup (N={n}). "
                "Input factors should ideally be sparse.")
            # Convert to sparse if necessary before matmul
            s_mat   = sps.csc_matrix(s) if not s_is_sparse else s.tocsc()
            sp_mat  = sps.csc_matrix(sp) if not sp_is_sparse else sp.tocsc()
            a_gram  = (sp_mat @ s_mat) / n
        else:
            # Perform sparse matrix multiplication
            a_gram  = (sp.tocsc() @ s.tocsc()) / n

        # Now call the standard setup
        self._set_standard(a_gram, sigma, **kwargs)

    # -----------------------------------------------------------------

    def _get_precomputed_data(self) -> dict:
        """ Returns the computed ILU object. """
        # Let apply handle None case
        return {'ilu_obj': self._ilu_obj}

    # -----------------------------------------------------------------
    #! Static Apply Method
    # -----------------------------------------------------------------

    @staticmethod
    def apply(r: Array, sigma: float, backend_module: Any, ilu_obj: Optional[spsla.SuperLU]) -> Array:
        '''
        Static apply method for ILU: solves Mz = r using the LU factors.

        Args:
            r (Array):
                The residual vector.
            sigma (float):
                Regularization (used during setup).
            backend_module (Any):
                The backend numpy module (must be numpy).
            ilu_obj (Optional[SuperLU]):
                The precomputed ILU object from spilu.

        Returns:
            Array: The preconditioned vector M^{-1}r, or r if ilu_obj is None.
        '''
        if backend_module is not np:
            # This check is important because spsla.SuperLU is SciPy/NumPy specific
            raise RuntimeError("IncompleteCholeskyPreconditioner.apply requires NumPy backend.")

        if ilu_obj is None:
            # Factorization failed or not performed, return original vector
            print("Warning: ILU object is None in apply, returning original vector.")
            return r

        try:
            # Use the solve method of the SuperLU object
            # This handles permutations and solves L(U(x)) = Pr
            return ilu_obj.solve(r) # Input r should be a NumPy array
        except Exception as e: # Catch specific linalg errors if possible
            print(f"ILU solve failed during apply: {e}")
            # Return original vector if solve fails
            return r

    # -----------------------------------------------------------------
    
    def __repr__(self) -> str:
        '''
        Returns the name and configuration of the Incomplete Cholesky preconditioner.
        '''
        
        # Check if the ILU object is None (not factorized) or not
        status      = "Factorized" if self._ilu_obj is not None else "Not Factorized/Failed"
        base_repr   = super().__repr__()
        
        # Add ILU specific params
        return f"{base_repr[:-1]}, ILU_status='{status}')"
    
    # -----------------------------------------------------------------

# =====================================================================
#! Choose wisely
# =====================================================================

def _resolve_precond_type(
    precond_id: Any
    ) -> Union[PreconditionersTypeSym, PreconditionersTypeNoSym]:
    """
    Helper to convert string/int/Enum id to a specific Enum member. 
    
    Args:
        precond_id (Any):
            Identifier (instance, Enum, str, int).
    Raises:
        ValueError:
            If the id is not recognized.
        TypeError:
            If the id is of an unsupported type.
    """
    
    # Check if precond_id is None
    precond_type = None
    if isinstance(precond_id, str):
        try:
            precond_type                = PreconditionersTypeSym[precond_id]
        except KeyError as e:
            try:
                precond_type            = PreconditionersTypeNoSym[precond_id]
            except KeyError as e:
                raise ValueError(f"Unknown preconditioner name: '{precond_id}'.") from e
    elif isinstance(precond_id, int):
        try:
            precond_type                = PreconditionersTypeSym(precond_id)
        except ValueError as e:
            try:
                precond_type            = PreconditionersTypeNoSym(precond_id)
            except ValueError as e:
                raise ValueError(f"Unknown preconditioner value: {precond_id}.") from e
    elif isinstance(precond_id, (PreconditionersTypeSym, PreconditionersTypeNoSym)):
        precond_type                    = precond_id
    else:
        raise TypeError(f"Unsupported type for precond_id: {type(precond_id)}. Expected Enum, str, or int.")
    return precond_type

# =====================================================================

def _get_precond_class_and_defaults(precond_type: Union[PreconditionersTypeSym, PreconditionersTypeNoSym]) -> Tuple[Type[Preconditioner], dict]:
    """
    Helper to map Enum type to class and set default kwargs.
    
    Returns:
        target_class (Type[Preconditioner]):
            The target preconditioner class.
        defaults (dict):
            Default arguments for the preconditioner constructor.
    Raises:
        ValueError:
            If the preconditioner type is not recognized.
        TypeError:
            If the preconditioner type is of an unsupported type.
    """
    target_class : Type[Preconditioner] = None
    defaults     : dict                 = {}

    if isinstance(precond_type, PreconditionersTypeSym):
        match precond_type:
            case PreconditionersTypeSym.IDENTITY:
                target_class            = IdentityPreconditioner
                defaults['is_positive_semidefinite'] = True
            case PreconditionersTypeSym.JACOBI:
                target_class            = JacobiPreconditioner
            case PreconditionersTypeSym.INCOMPLETE_CHOLESKY:
                target_class            = IncompleteCholeskyPreconditioner
            case PreconditionersTypeSym.COMPLETE_CHOLESKY:
                target_class            = CholeskyPreconditioner
            case PreconditionersTypeSym.SSOR:
                target_class            = SSORPreconditioner
            case _:
                raise ValueError(f"Symmetric type {precond_type} not handled.")
    elif isinstance(precond_type, PreconditionersTypeNoSym):
        match precond_type:
            case PreconditionersTypeNoSym.IDENTITY:
                target_class            = IdentityPreconditioner
                defaults['is_positive_semidefinite'] = False
            case PreconditionersTypeNoSym.INCOMPLETE_LU:
                # TODO: Implement ILUPreconditioner similar to IncompleteCholesky
                raise NotImplementedError("IncompleteLU preconditioner not implemented yet.")
            case _:
                raise ValueError(f"Non-Symmetric type {precond_type} not handled.")
    else:
        raise TypeError("Internal error: Invalid precond_type.")

    return target_class, defaults

# =====================================================================

def choose_precond(precond_id: Any, **kwargs) -> Preconditioner:
    """
    Factory function to select and instantiate a preconditioner.

    Accepts various identifiers (Enum, str, int, instance) and passes kwargs
    to the specific preconditioner's constructor.

    Args:
        precond_id (Any): Identifier (instance, Enum, str, int).
        **kwargs: Additional arguments for the constructor (e.g., backend='jax').

    Returns:
        Preconditioner: An instance of the selected preconditioner.
    """
    
    # 1. Handle Instance Passthrough
    if isinstance(precond_id, Preconditioner):
        if kwargs:
            print(f"Warning: Instance provided; ignoring kwargs: {kwargs}")
        return precond_id

    # 2. Resolve ID to Enum Type
    try:
        precond_type                    = _resolve_precond_type(precond_id)
    except (ValueError, TypeError) as e:
        # Re-raise with more context if needed, or just let the original error propagate
        raise e

    # 3. Get Target Class and Default Kwargs
    try:
        target_class, default_kwargs    = _get_precond_class_and_defaults(precond_type)
    except (ValueError, TypeError, NotImplementedError) as e:
        raise e

    # 4. Combine Defaults and User Kwargs (User kwargs override defaults)
    final_kwargs                        = default_kwargs.copy()
    final_kwargs.update(kwargs)

    # 5. Filter Kwargs for Constructor and Instantiate
    try:
        valid_args                      = inspect.signature(target_class.__init__).parameters
        filtered_kwargs                 = {k: v for k, v in final_kwargs.items() if k in valid_args}
        ignored_kwargs                  = {k: v for k, v in final_kwargs.items() if k not in valid_args and k != 'self'}
        if ignored_kwargs:
            print(f"Warning: Ignoring invalid kwargs for {target_class.__name__}: {ignored_kwargs}")

        return target_class(**filtered_kwargs)
    except Exception as e:
        print(f"Error instantiating {target_class.__name__} with kwargs {filtered_kwargs}: {e}")
        raise e
    return None # Fallback, should not reach here if everything is correct

# =====================================================================
#! End of File
# =====================================================================