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
from typing import Union, Callable, Optional, Any, Type, Tuple, Dict
from enum import Enum, auto, unique
import inspect
import numpy as np

# Add sparse imports at the top of the file
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import scipy.linalg as sla

from ..algebra.utils import JAX_AVAILABLE, get_backend, Array
from ..common.flog import get_global_logger, Logger

# ---------------------------------------------------------------------

try:
    if JAX_AVAILABLE:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
    else:
        jnp   = None
        jax   = None # Define jax as None if not available
except ImportError:
    jnp   = None
    jax   = None

# ---------------------------------------------------------------------

# Interface for Preconditioner Apply Function (Static version)
# Takes residual, and precomputed data...
PreconitionerApplyFun   = Callable[[Array], Array]
# Setup returns a dictionary of precomputed data
StaticSetupKernel       = Callable[..., Dict[str, Any]]
# can be:
#   - r, backend_mod, sigma, precomputed_data
#   - r, a, sigma, backend_mod, precomputed_data
#   - r, s, sp, sigma, backend_mod, precomputed_data
# Apply uses the precomputed data dictionary
StaticApplyKernel       = Callable[[Array, Any, float, Dict[str, Any]], Array] # r, backend_mod, sigma, precomputed_data

_TOLERANCE_SMALL        = 1e-13
_TOLERANCE_BIG          = 1e13

# just the example
def preconditioner_idn(r: Array) -> Array:
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
    IDENTITY            = 0
    JACOBI              = 1
    INCOMPLETE_CHOLESKY = 2
    COMPLETE_CHOLESKY   = 3
    SSOR                = 4

@unique
class PreconditionersTypeNoSym(Enum):
    """
    Enumeration of specific potentially non-symmetric preconditioner types.
    """
    IDENTITY            = 0
    INCOMPLETE_LU       = 1
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
    _dcol : str = "yellow"
    
    # -----------------------------------------------------------------
    
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
        self._logger: Logger             = get_global_logger()
        self._TOLERANCE_SMALL                  = _TOLERANCE_SMALL
        self._TOLERANCE_BIG                    = _TOLERANCE_BIG
        self._zero                       = _TOLERANCE_BIG
        self._sigma                      = 0.0                          # Regularization parameter
        self.reset_backend(backend) # Sets backend attributes

        self._is_positive_semidefinite   = is_positive_semidefinite     # Positive semidefinite flag
        self._is_gram                    = is_gram                      # Gram matrix flag

        if self._is_positive_semidefinite:
            self._stype                  = PreconditionersType.SYMMETRIC
        else:
            #! May still be symmetric, but we don't assume based on this flag alone
            self._stype                  = PreconditionersType.NONSYMMETRIC

        # Store reference to the static apply logic of the concrete class
        self._base_apply_logic : Callable[..., Array] = self.__class__._apply_kernel if not apply_func else apply_func
        
        # Preconditioner setup
        self._precomputed_data_instance  : Optional[Dict[str, Any]] = None
        
        # Compiled/wrapped apply function for instance usage
        self._apply_func_instance        : Optional[Callable[[Array], Array]] = None
        self._update_instance_apply_func() # Create initial apply(r)
        
    # -----------------------------------------------------------------
    #! Logging
    # -----------------------------------------------------------------
    
    def log(self, msg : str, log : Union[int, str] = Logger.LEVELS_R['info'],
        lvl : int = 0, color : str = "white", append_msg = True):
        """
        Log the message.
        
        Args:
            msg (str) :
                The message to log.
            log (Union[int, str]) :
                The flag to log the message (default is 'info').
            lvl (int) :
                The level of the message.
            color (str) :
                The color of the message.
            append_msg (bool) :
                Flag to append the message.
        """
        if isinstance(log, str):
            log = Logger.LEVELS_R[log]
        if append_msg:
            msg = f"[{self._name}] {msg}"
        msg = self._logger.colorize(msg, color)
        self._logger.say(msg, log=log, lvl=lvl)
    
    # -----------------------------------------------------------------
    #! Backend Management
    # -----------------------------------------------------------------
    
    def reset_backend(self, backend: str):
        '''
        Resets the backend and recompiles the internal apply function.

        Parameters:
            backend (str): The name of the new backend ('numpy', 'jax').
        '''
        new_backend_str = backend if backend != 'default' else 'numpy' # Resolve default
        if not hasattr(self, '_backend_str') or self._backend_str != new_backend_str:
            self.log(f"Resetting backend to: {new_backend_str}",
                    log=Logger.LEVELS_R['info'], lvl=1, color=self._dcol)
            self._backend_str               = new_backend_str
            self._backend, self._backends   = get_backend(self._backend_str, scipy=True)
            self._isjax                     = JAX_AVAILABLE and self._backend is not np
            # Re-create the wrapped/compiled apply function for the new backend
            self._update_instance_apply_func()
        
    # -----------------------------------------------------------------
    #! Closure for the apply function
    # -----------------------------------------------------------------
    
    def _update_instance_apply_func(self):
        """
        Creates/updates the instance's `apply(r)` func using stored data.
        """
        static_apply    = self.__class__._apply_kernel
        backend_mod     = self._backend
        # Capture current sigma for the closure
        current_sigma   = self._sigma
        instance_self   = self

        def wrapped_apply_instance(r: Array) -> Array:
            precomputed_data = instance_self._get_precomputed_data_instance()
            return static_apply(r               = r,
                                backend_mod     = backend_mod,
                                sigma           = current_sigma,
                                **precomputed_data)

        if self._isjax and jax is not None:
            self._apply_func_instance = jax.jit(wrapped_apply_instance)
        else:
            self._apply_func_instance = wrapped_apply_instance
    
    # -----------------------------------------------------------------
    #! Getters for apply functions
    # -----------------------------------------------------------------
    
    def get_apply(self) -> Callable[[Array], Array]:
        '''
        Returns the potentially JIT-compiled function `apply(r)`.
        Uses precomputed data stored by the last call to `set()`.
        Raises RuntimeError if called before `set()` to match test expectations.
        '''
        # Ensure precomputed data exists before returning apply function
        if self._precomputed_data_instance is None:
            raise RuntimeError("Preconditioner apply function could not be initialized before set().")
        
        if self._apply_func_instance is None:
            self._update_instance_apply_func()
            if self._apply_func_instance is None: # If still None, raise error
                raise RuntimeError("Preconditioner apply(r) function failed to initialize.")
            
        return self._apply_func_instance
    
    def get_apply_mat(self, **default_setup_kwargs) -> Callable[[Array, Array, float], Array]:
        '''
        Returns a potentially JIT-compiled function `apply_mat(r, A, sigma, **override_kwargs)`
        that computes preconditioner data from `A` and applies it on the fly.

        Params:
            **default_setup_kwargs:
                Default keyword arguments for the setup kernel
                (e.g., tol_small for Jacobi). These are fixed
                at compile time if using JIT.

        Returns:
            Callable:
                The compiled function.
        '''
        static_setup        = self.__class__._setup_standard_kernel
        static_apply        = self.__class__._apply_kernel
        backend_mod         = self._backend
        instance_defaults   = default_setup_kwargs
        sigma               = self._sigma # Capture current sigma for the closure

        # Define the function that performs setup + apply
        def wrapped_apply_mat(r: Array, a: Array, **call_time_kwargs) -> Array:
            # Merge default setup kwargs with call-time kwargs (call-time overrides)
            setup_kwargs        = {**instance_defaults, **call_time_kwargs}
            # Perform setup on the fly
            precomputed_data    = static_setup(a, sigma, backend_mod, **setup_kwargs)
            # Apply using the computed data
            return static_apply(r, backend_mod, sigma, **precomputed_data)

        if self._isjax and jax is not None:
            # JIT the wrapper. A, sigma, r are dynamic. Backend is fixed.
            # We can make setup_kwargs static *if needed* by requiring them
            # to be passed to get_apply_mat instead of the returned function.
            # For simplicity now, assume setup_kwargs are dynamic unless performance dictates otherwise.
            self.log("JIT compiling apply_mat(r, A, sigma, ...) function...", log='info', lvl=2, color=self._dcol)
            return jax.jit(wrapped_apply_mat)
        else:
            self.log("Using numpy backend for apply_mat(r, A, sigma, ...), no JIT.", log='info', lvl=2, color=self._dcol)
            return wrapped_apply_mat

    def get_apply_gram(self, **default_setup_kwargs) -> Callable[[Array, Array, Array, float], Array]:
        '''
        Returns a potentially JIT-compiled function `apply_gram(r, S, Sp, sigma, **override_kwargs)`
        that computes preconditioner data from `S`, `Sp` and applies it on the fly.

        Params:
            **default_setup_kwargs: Default keyword arguments for the setup kernel.

        Returns:
            Callable: The compiled function.
        '''
        static_setup        = self.__class__._setup_gram_kernel
        static_apply        = self.__class__._apply_kernel
        backend_mod         = self._backend
        instance_defaults   = default_setup_kwargs
        sigma               = self._sigma # Capture current sigma for the closure

        def wrapped_apply_gram(r: Array, s: Array, sp: Array, **call_time_kwargs) -> Array:
            setup_kwargs        = {**instance_defaults, **call_time_kwargs}
            # Perform setup on the fly
            precomputed_data    = static_setup(s, sp, sigma, backend_mod, **setup_kwargs)
            # Apply using the computed data
            return static_apply(r, backend_mod, sigma, **precomputed_data)

        if self._isjax and jax is not None:
            self.log("JIT compiling apply_gram(r, S, Sp, ...) function...", log='info', lvl=2, color=self._dcol)
            return jax.jit(wrapped_apply_gram)
        else:
            self.log("Using numpy backend for apply_gram(r, S, Sp, sigma, ...), no JIT.", log='info', lvl=2, color=self._dcol)
            return wrapped_apply_gram
    
    # -----------------------------------------------------------------
    #! Properties
    # -----------------------------------------------------------------
    
    # -----------------------------------------------------------------
    #! Properties: General Attributes
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        """Name of the preconditioner."""
        return self._name

    @property
    def dcol(self) -> str:
        """Color for logging messages."""
        return self._dcol

    @property
    def backend_str(self) -> str:
        """Name of the current backend ('numpy', 'jax')."""
        return self._backend_str

    @property
    def backend(self) -> Any:
        """The backend module (e.g., np, jnp)."""
        return self._backend

    @property
    def backends(self) -> Any:
        """The backend module for scipy-like operations."""
        return self._backends

    @property
    def isjax(self) -> bool:
        """True if using JAX backend."""
        return self._isjax

    @property
    def precomputed_data(self) -> dict:
        """
        Returns empty dict as no precomputed data is needed. 
        """
        return self._get_precomputed_data()

    # -----------------------------------------------------------------
    #! Properties: Preconditioner Type and Symmetry
    # -----------------------------------------------------------------

    @property
    def type(self) -> Optional[Union[PreconditionersTypeNoSym, PreconditionersTypeSym]]:
        """Specific preconditioner type (e.g., JACOBI, ILU)."""
        return self._type

    @property
    def stype(self) -> PreconditionersType:
        """Symmetry type (SYMMETRIC/NONSYMMETRIC)."""
        return self._stype

    @property
    def is_positive_semidefinite(self) -> bool:
        """True if the matrix A (and potentially M) is positive semidefinite."""
        return self._is_positive_semidefinite

    # -----------------------------------------------------------------
    #! Properties: Gram Matrix Setup
    # -----------------------------------------------------------------

    @property
    def is_gram(self) -> bool:
        """True if the preconditioner is set up from Gram matrix factors S, Sp."""
        return self._is_gram

    @is_gram.setter
    def is_gram(self, value: bool):
        """Set the is_gram flag. Requires re-running set()."""
        if self._is_gram != value:
            self.log(f"({self._name}) Changed is_gram to {value}. Remember to call set() again.",
                    log=Logger.LEVELS_R['warning'], lvl=1, color=self._dcol)
            self._is_gram = value
    
    # -----------------------------------------------------------------
    #! Regularization parameter
    # -----------------------------------------------------------------
    
    @property
    def sigma(self):
        """Regularization parameter."""
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self._update_instance_apply_func() # Recompile apply(r) if data changes
    
    # -----------------------------------------------------------------
    #! Tolerances
    # -----------------------------------------------------------------
    
    @property
    def tol_big(self):
        '''Tolerance for big values.'''
        return self._TOLERANCE_BIG

    @tol_big.setter
    def tol_big(self, value):
        self._TOLERANCE_BIG = value
        
    @property
    def tol_small(self):
        '''Tolerance for small values.'''
        return self._TOLERANCE_SMALL
    
    @tol_small.setter
    def tol_small(self, value):
        self._TOLERANCE_SMALL = value
    
    @property
    def zero(self):
        '''Value treated as zero.'''
        return self._zero

    @zero.setter
    def zero(self, value):
        self._zero = value
    
    # -----------------------------------------------------------------
    #! KERNELS
    # -----------------------------------------------------------------
    
    @staticmethod
    @abstractmethod
    def _setup_standard_kernel(a: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """Static Kernel: Computes precond data dict from matrix A."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _setup_gram_kernel(s: Array, sp: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """Static Kernel: Computes precond data dict from factors S, Sp."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _apply_kernel(r: Array, backend_mod: Any, sigma: float, **precomputed_data: Any) -> Array:
        """Static Kernel: Applies M^{-1}r using precomputed data."""
        raise NotImplementedError
    
    # -----------------------------------------------------------------
    #! Instance Setup Methods
    # -----------------------------------------------------------------
        
    def _set_standard(self, a: Array, sigma: float, **kwargs):
        """Instance: Calls static setup kernel and stores result."""
        self._precomputed_data_instance = self.__class__._setup_standard_kernel(
            a, sigma, self._backend, **kwargs
        )
        self._update_instance_apply_func() # Recompile apply(r) if data changes

    def _set_gram(self, s: Array, sp: Array, sigma: float, **kwargs):
        """Instance: Calls static setup kernel and stores result."""
        self._precomputed_data_instance = self.__class__._setup_gram_kernel(
            s, sp, sigma, self._backend, **kwargs
        )
        self._update_instance_apply_func() # Recompile apply(r) if data changes

    def _get_precomputed_data_instance(self) -> Dict[str, Any]:
        """Instance: Returns the stored precomputed data."""
        if self._precomputed_data_instance is None:
            # Include standardized substring expected by tests
            raise RuntimeError(f"Preconditioner data not available - ({self._name}) not set up. Call set() first.")
        return self._precomputed_data_instance
    
    # -----------------------------------------------------------------
    #! General Setup Method
    # -----------------------------------------------------------------
    
    def set(self, a: Array, sigma: float = 0.0, ap: Optional[Array] = None, backend: Optional[str] = None, **kwargs):
        '''
        Sets up the preconditioner using the provided matrix A and optional parameters.
        This method computes the preconditioner data and prepares the apply function.
        
        Params:
            a (Array):
                The matrix to be used for setting up the preconditioner.
            sigma (float, optional):
                The regularization parameter. Defaults to 0.0.
            ap (Optional[Array], optional):
                An optional second matrix for Gram setup. Defaults to None.
            backend (Optional[str], optional):
                The backend to use for computations. Defaults to None.
            **kwargs:
                Additional keyword arguments for specific implementations.
        '''
        
        if backend is not None and backend != self.backend_str:
            self.reset_backend(backend) # Will trigger _update_instance_apply_func

        # Ensure backend consistency for inputs
        a_be        = self._backend.asarray(a)
        ap_be       = self._backend.asarray(ap) if ap is not None else None

        # Use provided sigma or instance sigma, update instance sigma via setter
        self.sigma  = sigma if sigma is not None else self._sigma

        self.log(f"Setting up preconditioner state with sigma={self.sigma} using backend='{self.backend_str}'...", log='info', lvl=1, color=self._dcol)

        if self.is_gram:
            s_mat, sp_mat = a_be, ap_be
            if sp_mat is None:
                sp_mat = self._backend.conjugate(s_mat).T
            # Shape checks...
            if s_mat.shape[1] != sp_mat.shape[0] or s_mat.shape[0] != sp_mat.shape[1]: 
                raise ValueError("Shape mismatch")
            self._set_gram(s_mat, sp_mat, self.sigma, **kwargs) # Pass kwargs
        else:
            if a_be.ndim != 2 or a_be.shape[0] != a_be.shape[1]: 
                raise ValueError("Needs square matrix")
            self._set_standard(a_be, self.sigma, **kwargs)

    # -----------------------------------------------------------------
    #! Apply Method
    # -----------------------------------------------------------------

    def __call__(self, r: Array) -> Array:
        """
        Apply the configured preconditioner instance M^{-1} to vector r using precomputed data.
        """
        apply_func = self.get_apply()
        return apply_func(r)

    # -----------------------------------------------------------------
    #! String Representation
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

    @staticmethod
    def _setup_standard_kernel(a: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """Static Setup Kernel for Identity (no-op)."""
        return {} # No data needed

    @staticmethod
    def _setup_gram_kernel(s: Array, sp: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """Static Setup Kernel for Identity (no-op)."""
        return {} # No data needed

    @staticmethod
    def _apply_kernel(r: Array, backend_mod: Any, sigma: float, **precomputed_data: Any) -> Array:
        """Static Apply Kernel for Identity."""
        return backend_mod.asarray(r) # Ensure correct backend type

    # Convenience static/class apply used in tests
    @staticmethod
    def apply(r: Array, backend_mod: Any, sigma: float = 0.0, **precomputed_data: Any) -> Array:
        """
        Static apply convenience wrapper used by tests.
        Mirrors the signature expected in test files.
        """
        return IdentityPreconditioner._apply_kernel(r=r, backend_mod=backend_mod, sigma=sigma, **precomputed_data)

    def _set_standard(self, a: Array, sigma: float, **kwargs):
        self._precomputed_data_instance = self.__class__._setup_standard_kernel(a, sigma, self._backend, **kwargs)
        self._update_instance_apply_func()

    def _set_gram(self, s: Array, sp: Array, sigma: float, **kwargs):
        self._precomputed_data_instance = self.__class__._setup_gram_kernel(s, sp, sigma, self._backend, **kwargs)
        self._update_instance_apply_func()

    def _get_precomputed_data(self) -> dict:
        return self._get_precomputed_data_instance()
    
    def __repr__(self) -> str:
        ''' 
        Returns the name and configuration of the Identity preconditioner. 
        '''
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, type={self.type})"

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
                is_positive_semidefinite: bool  = False,
                is_gram                 : bool  = False,
                backend                 : str   = 'default',
                # Tolerances specific to Jacobi
                tol_small               : float = _TOLERANCE_SMALL,
                zero_replacement        : float = _TOLERANCE_BIG):
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
        self._TOLERANCE_SMALL         = tol_small
        self._zero              = zero_replacement
        # Precomputed data storage
        self._inv_diag : Optional[Array] = None # Stores 1 / (diag(A) + sigma)

    # -----------------------------------------------------------------

    @staticmethod
    def _static_compute_inv_diag(diag_a         : Array,
                                sigma           : float,
                                backend_mod     : Any, 
                                tol_small       : float, 
                                zero_replacement: float) -> Array:

        """
        Static helper to compute inverse diagonal safely.
        Handles small values and avoids division by zero.
        
        ':math:`M^{-1}r = [1 / (A_ii + sigma)] * r_i`
        This function computes the inverse diagonal of the matrix A.
        
        Parameters:
            diag_a (Array):
                Diagonal of the matrix A.
            sigma (float):
                Regularization parameter.
            backend_mod (Any):
                The backend module (e.g., np, jnp).
            tol_small (float):
                Tolerance for small values.
            zero_replacement (float):
                Value to replace small values with.
        Returns:
            Array:
                The inverse diagonal (1 / (A_ii + sigma)).
        """
        be              = backend_mod
        reg_diag        = diag_a + sigma
        abs_reg_diag    = be.abs(reg_diag)
        is_small        = abs_reg_diag < tol_small
        # Use where for conditional assignment (JAX compatible)
        # Assign large magnitude for small values before inversion
        safe_diag       = be.where(is_small, be.sign(reg_diag) * zero_replacement, reg_diag)
        # Avoid division by exact zero if somehow it occurred after clamping
        safe_diag       = be.where(safe_diag == 0.0, zero_replacement, safe_diag)
        inv_diag        = 1.0 / safe_diag
        # Ensure inverse is zero where original was small
        inv_diag        = be.where(is_small, 0.0, inv_diag)
        return inv_diag

    # -----------------------------------------------------------------

    @staticmethod
    def _setup_standard_kernel(a: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """
        Static Setup Kernel for Jacobi from matrix A.
        """
        tol_small           = kwargs.get('tol_small', _TOLERANCE_SMALL)        # Get tolerances from kwargs or default
        zero_replacement    = kwargs.get('zero_replacement', _TOLERANCE_BIG)  # Replacement for small values inverse
        diag_a              = backend_mod.diag(a)
        inv_diag            = JacobiPreconditioner._static_compute_inv_diag(
            diag_a, sigma, backend_mod, tol_small, zero_replacement
        )
        return {'inv_diag': inv_diag}

    @staticmethod
    def _setup_gram_kernel(s: Array, sp: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """
        Static Setup Kernel for Jacobi from Gram factors S, Sp.
        """
        tol_small           = kwargs.get('tol_small', _TOLERANCE_SMALL)
        zero_replacement    = kwargs.get('zero_replacement', _TOLERANCE_BIG)
        be                  = backend_mod
        n                   = float(s.shape[0]) if s.shape[0] > 0 else 1.0
        diag_s_dag_s        = be.einsum('ij,ji->i', sp, s)
        diag_a_approx       = diag_s_dag_s / n
        inv_diag            = JacobiPreconditioner._static_compute_inv_diag(
            diag_a_approx, sigma, backend_mod, tol_small, zero_replacement
        )
        return {'inv_diag': inv_diag}

    # Backwards-compat alias expected by tests (instance method using internal backend and tolerances)
    def _compute_inv_diag(self, diag_a: Array, sigma: float) -> Array:
        return JacobiPreconditioner._static_compute_inv_diag(
            diag_a, sigma, self._backend, self._TOLERANCE_SMALL, self._zero
        )

    # Convenience static apply used in tests
    @staticmethod
    def apply(r: Array, backend_mod: Any, sigma: float = 0.0, **precomputed_data: Any) -> Array:
        """Static apply wrapper matching test signature."""
        return JacobiPreconditioner._apply_kernel(r=r, backend_mod=backend_mod, sigma=sigma, **precomputed_data)

    @staticmethod
    def _apply_kernel(r: Array, backend_mod: Any, sigma: float, **precomputed_data: Any) -> Array:
        '''
        Static Apply Kernel for Jacobi using precomputed data.
        Applies the preconditioner M^{-1}r using the inverse diagonal.
        
        Parameters:
            r (Array):
                The residual vector to precondition.
            backend_mod (Any):
                The backend module (e.g., np, jnp).
            sigma (float):
                Regularization parameter (ignored here).
            **precomputed_data (Any):
                Precomputed data from setup_kernel().
                Must include 'inv_diag' key.
        Returns:
            Array:
                The preconditioned vector M^{-1}r.
        '''
        inv_diag    = precomputed_data.get('inv_diag', None)
        if inv_diag is None:
            raise ValueError("Jacobi apply kernel requires 'inv_diag' in precomputed_data.")
        if r.ndim != 1 or inv_diag.ndim != 1 or r.shape[0] != inv_diag.shape[0]:
            raise ValueError(f"Shape mismatch in Jacobi apply: r={r.shape}, inv_diag={inv_diag.shape}")
        return inv_diag * r

    # -----------------------------------------------------------------
    #! Instance Setup Methods
    # -----------------------------------------------------------------
    
    def _set_standard(self, a: Array, sigma: float, **kwargs):
        # Pass instance tolerances to static kernel via kwargs
        kwargs.setdefault('tol_small', self._TOLERANCE_SMALL)
        kwargs.setdefault('zero_replacement', self._zero)
        self._precomputed_data_instance = self.__class__._setup_standard_kernel(a, sigma, self._backend, **kwargs)
        self._update_instance_apply_func()

    def _set_gram(self, s: Array, sp: Array, sigma: float, **kwargs):
        kwargs.setdefault('tol_small', self._TOLERANCE_SMALL)
        kwargs.setdefault('zero_replacement', self._zero)
        self._precomputed_data_instance = self.__class__._setup_gram_kernel(s, sp, sigma, self._backend, **kwargs)
        self._update_instance_apply_func()

    def _get_precomputed_data(self) -> dict:
        return self._get_precomputed_data_instance()

    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        ''' 
        Returns the name and configuration of the Jacobi preconditioner. 
        '''
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, tol_small={self._TOLERANCE_SMALL})"

    # Expose zero_replacement for tests/consumers expecting that name
    @property
    def zero_replacement(self) -> float:
        return self._zero

    @zero_replacement.setter
    def zero_replacement(self, value: float):
        self._zero = value
    
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

    @staticmethod
    def _setup_standard_kernel(a: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """
        Static Setup Kernel: Computes the Cholesky factor L of the matrix A + sigma*I.

        Parameters:
            a (Array): 
                The input matrix A.
            sigma (float): 
                Regularization parameter. Adds sigma*I to A.
            backend_mod (Any): 
                The computational backend (e.g., numpy, jax).
            **kwargs: 
                Additional keyword arguments (not used here).

        Returns:
            Dict[str, Any]: 
                A dictionary containing the Cholesky factor 'l'. If decomposition fails, 'l' is set to None.
        """
        be = backend_mod
        l_factor = None  # Default to None if decomposition fails
        print(f"({CholeskyPreconditioner._name}) Performing Cholesky decomposition...")
        try:
            a_reg = a + sigma * be.eye(a.shape[0], dtype=a.dtype)
            # Use appropriate Cholesky per backend
            if be is np:
                l_factor = sla.cholesky(a_reg, lower=True, check_finite=False)
            elif jnp is not None and be is jnp:
                l_factor = jsp.linalg.cholesky(a_reg, lower=True)
            else:
                l_factor = be.linalg.cholesky(a_reg)
            print(f"({CholeskyPreconditioner._name}) Cholesky decomposition successful.")
        except Exception as e:
            print(f"({CholeskyPreconditioner._name}) Cholesky decomposition failed: {e}")
            print(f"({CholeskyPreconditioner._name}) Matrix might not be positive definite after regularization (sigma={sigma}).")
        return {'l': l_factor}

    @staticmethod
    def _setup_gram_kernel(s: Array, sp: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """
        Static Setup Kernel: Forms the Gram matrix A = Sp @ S / N and computes its Cholesky factor L.

        Parameters:
            s (Array): 
                The matrix S (factor of the Gram matrix).
            sp (Array): 
                The matrix Sp (conjugate transpose of S or another factor).
            sigma (float): 
                Regularization parameter. Adds sigma*I to A.
            backend_mod (Any): 
                The computational backend (e.g., numpy, jax).
            **kwargs: 
                Additional keyword arguments (not used here).

        Returns:
            Dict[str, Any]: 
                A dictionary containing the Cholesky factor 'l'. If decomposition fails, 'l' is set to None.
        """
        be = backend_mod
        n = float(s.shape[0]) if s.shape[0] > 0 else 1.0
        print(f"({CholeskyPreconditioner._name}) Warning: Forming explicit Gram matrix A = Sp @ S / N for Cholesky setup (N={n}).")
        a_gram = (sp @ s) / n
        # Call the standard setup kernel on the computed Gram matrix
        return CholeskyPreconditioner._setup_standard_kernel(a_gram, sigma, backend_mod, **kwargs)

    @staticmethod
    def _apply_kernel(r: Array, backend_mod: Any, sigma: float, **precomputed_data: Any) -> Array:
        """
        Static Apply Kernel: Solves the system M^{-1}r using the Cholesky factor L.

        The system is solved in two steps:
        1. Solve L y = r (forward substitution).
        2. Solve L^H z = y (backward substitution).

        Parameters:
            r (Array): 
                The input vector to precondition.
            backend_mod (Any): 
                The computational backend (e.g., numpy, jax).
            sigma (float): 
                Regularization parameter (not used here, as it is applied during setup).
            **precomputed_data (Any): 
                Precomputed data from the setup kernel. Must include the Cholesky factor 'l'.

        Returns:
            Array: 
                The preconditioned vector M^{-1}r. If the Cholesky factor is missing, returns the input vector r.
        """
        l = precomputed_data.get('l')
        if l is None:
            print(f"Warning: ({CholeskyPreconditioner._name}) Cholesky factor 'l' is None in apply, returning original vector.")
            return r
        be = backend_mod
        if r.shape[0] != l.shape[0]:
            raise ValueError(f"Shape mismatch in Cholesky apply: r ({r.shape[0]}) vs L ({l.shape[0]})")

        try:
            use_conj = be.iscomplexobj(l)
            lh = be.conjugate(l).T if use_conj else l.T
            # Forward substitution: Solve L y = r
            if be is np:
                y = sla.solve_triangular(l, r, lower=True, check_finite=False)
            else:
                y = jsp.linalg.solve_triangular(l, r, lower=True)
            # Backward substitution: Solve L^H z = y
            if be is np:
                z = sla.solve_triangular(lh, y, lower=False, check_finite=False)
            else:
                z = jsp.linalg.solve_triangular(lh, y, lower=False)
            return z
        except Exception as e:
            print(f"({CholeskyPreconditioner._name}) Cholesky triangular solve failed during apply: {e}")
            return r  # Return the original vector if the solve fails

    # -----------------------------------------------------------------
    
    def _set_standard(self, a: Array, sigma: float, **kwargs):
        """
        Instance method to set up the preconditioner using the matrix A.

        Parameters:
            a (Array): 
                The input matrix A.
            sigma (float): 
                Regularization parameter. Adds sigma*I to A.
            **kwargs: 
                Additional keyword arguments for the setup kernel.
        """
        self._precomputed_data_instance = self.__class__._setup_standard_kernel(a, sigma, self._backend, **kwargs)
        self._update_instance_apply_func()

    def _set_gram(self, s: Array, sp: Array, sigma: float, **kwargs):
        """
        Instance method to set up the preconditioner using Gram matrix factors S and Sp.

        Parameters:
            s (Array): 
                The matrix S (factor of the Gram matrix).
            sp (Array): 
                The matrix Sp (conjugate transpose of S or another factor).
            sigma (float): 
                Regularization parameter. Adds sigma*I to A.
            **kwargs: 
                Additional keyword arguments for the setup kernel.
        """
        self._precomputed_data_instance = self.__class__._setup_gram_kernel(s, sp, sigma, self._backend, **kwargs)
        self._update_instance_apply_func()

    def _get_precomputed_data(self) -> dict:
        """
        Retrieves the precomputed data for the preconditioner.

        Returns:
            dict: 
                A dictionary containing the precomputed Cholesky factor 'l'.
        """
        return self._get_precomputed_data_instance()

    def __repr__(self) -> str:
        """
        Returns a string representation of the Cholesky preconditioner, including its status.

        Returns:
            str: 
                A string indicating whether the preconditioner is factorized or not.
        """
        status = "Factorized" if self._precomputed_data_instance and self._precomputed_data_instance.get('l') is not None else "Not Factorized/Failed"
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, status='{status}')"

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
                tol_small               : float = _TOLERANCE_SMALL,    # For inverting diagonal
                zero_replacement        : float = _TOLERANCE_BIG):
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
        self._TOLERANCE_SMALL         = tol_small
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
    
    @staticmethod
    def _setup_standard_kernel(a: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """
        Static Setup for SSOR: compute D (diag of A+sigma I), and strict L, U parts.
        Returns dict with keys: d_diag, L, U, omega
        """
        be          = backend_mod
        omega       = kwargs.get('omega', 1.0)
        a_reg       = a + sigma * be.eye(a.shape[0], dtype=a.dtype)
        diag_a_reg  = be.diag(a_reg)
        L           = be.tril(a_reg, k=-1)
        U           = be.triu(a_reg, k=1)
        return {'d_diag': diag_a_reg, 'L': L, 'U': U, 'omega': omega}

    @staticmethod
    def _setup_gram_kernel(s: Array, sp: Array, sigma: float, backend_mod: Any, **kwargs) -> Dict[str, Any]:
        """
        Static Setup for SSOR from Gram factors by forming A = Sp @ S / n.
        """
        be          = backend_mod
        n           = float(s.shape[0]) if s.shape[0] > 0 else 1.0
        a_gram      = (sp @ s) / n
        return SSORPreconditioner._setup_standard_kernel(a_gram, sigma, be, **kwargs)
    
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
        is_small                = be.abs(diag_scaled) < self._TOLERANCE_SMALL
        safe_diag_scaled        = be.where(is_small, be.sign(diag_scaled) * self._zero, diag_scaled)
        safe_diag_scaled        = be.where(safe_diag_scaled == 0.0, self._zero, safe_diag_scaled)
        
        self._inv_diag_scaled   = 1.0 / safe_diag_scaled
        self._inv_diag_scaled   = be.where(is_small, 0.0, self._inv_diag_scaled)
        
        # Store D_inv_scaled, L and U (or scaled versions if preferred for apply)
        # Let's store the original L and U, apply will use omega
        self._L                 = L
        self._U                 = U 
        
        # We need D for the backward step, store its inverse
        is_small_d              = be.abs(diag_a_reg) < self._TOLERANCE_SMALL
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

        print(f"({self._name}) Warning: Forming explicit Gram matrix A = Sp @ S / N for SSOR setup (N={n}).")
        a_gram  = (sp @ s) / n
        self._set_standard(a_gram, sigma)

    def _get_precomputed_data(self) -> dict:
        """ Returns the computed factors needed for SSOR apply. """
        
        # Ensure D_inv_scaled is computed and stored during set if using the alternative form below
        # For the direct solve form, we need L, U, and D/w. Let's store D and use it.
        
        if self._L is None or self._U is None or self._inv_diag_unscaled is None:
            raise RuntimeError(f"({self._name}) Preconditioner not set up. Call set() first.")

        # Return D itself (or its diagonal) instead of its inverse if computing D/w directly in apply
        diag_a_reg = 1.0 / self._inv_diag_unscaled  # Reconstruct the regularized diagonal
        return {
            'd_diag': diag_a_reg,                   # Pass the diagonal D = diag(A_reg)
            'L': self._L,
            'U': self._U,
            'omega': self._omega
        }

    # --- Static Apply ---
    @staticmethod
    def _apply_kernel(r         : Array,
            backend_mod         : Any,
            sigma               : float,
            **precomputed_data) -> Array:
        """
        Static apply method for SSOR: forward and backward triangular solves.
        Solves Mz = r where M = (D/w + L) D^{-1} (D/w + U) / (w(2-w)).
        This implementation directly performs the forward and backward solves
        associated with Mz = r, which corresponds to:
            1. Solve (D/omega + L) y_temp = r          (Forward sweep)
            2. Solve (D/omega + U) z = D y_temp / omega (Backward sweep)

        Precomputed data: d_diag, L, U, omega

        Returns: The preconditioned vector M^{-1}r (z).
        """
        d_diag = precomputed_data.get('d_diag')
        L      = precomputed_data.get('L')
        U      = precomputed_data.get('U')
        omega  = precomputed_data.get('omega', 1.0)
        if d_diag is None or L is None or U is None:
            print("Warning: SSOR factors missing in apply, returning original vector.")
            return r

        be = backend_mod # Use the shorter alias

        # 1. Forward sweep: Solve (D/omega + L) y_temp = r
        try:
            # Ensure d_diag/omega doesn't contain zeros; setup already handles small diagonals
            diag_scaled         = d_diag / omega
            m_fwd               = be.diag(diag_scaled) + L
            if be is np:
                y_temp          = sla.solve_triangular(m_fwd, r, lower=True, check_finite=False)
            else:
                y_temp          = jsp.linalg.solve_triangular(m_fwd, r, lower=True)

            # 2. Backward sweep setup: rhs_bwd = (D/omega) * y_temp (element-wise via diag_scaled)
            rhs_bwd             = diag_scaled * y_temp

            # 3. Backward sweep: Solve (D/omega + U) z = rhs_bwd
            m_bwd               = be.diag(diag_scaled) + U
            if be is np:
                z               = sla.solve_triangular(m_bwd, rhs_bwd, lower=False, check_finite=False)
            else:
                z               = jsp.linalg.solve_triangular(m_bwd, rhs_bwd, lower=False)

            return z
        except Exception as e:
            # Catch potential LinAlgError if triangular matrices are singular
            print(f"SSOR triangular solve failed during apply: {e}")
            return r # Return original vector if solve fails
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

def _resolve_precond_type(precond_id: Any) -> Union[PreconditionersTypeSym, PreconditionersTypeNoSym]:
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
        name = precond_id.strip().replace('-', '_').replace(' ', '_').upper()
        try:
            precond_type                = PreconditionersTypeSym[name]
        except KeyError as e:
            try:
                precond_type            = PreconditionersTypeNoSym[name]
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
        # Unsupported identifier type
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
    elif isinstance(precond_type, (int)):
        # 0 or 1 only for now
        if precond_type == 0:
            target_class            = IdentityPreconditioner
            defaults['is_positive_semidefinite'] = True
        elif precond_type == 1:
            target_class            = JacobiPreconditioner
        else:
            raise ValueError(f"Unknown preconditioner integer value: {precond_type}.")
    elif precond_type is None:
        raise ValueError("Preconditioner type could not be resolved (None).")
    else:
        raise TypeError("Internal error: Invalid precond_type.")

    return target_class, defaults

# =====================================================================
#! Main Factory Function
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
    
    if precond_id is None:
        return None
    
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