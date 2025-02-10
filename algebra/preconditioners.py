'''
This module provides linear algebra functions and utilities.
'''

# Import the required modules
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import numpy.random as nrn
import scipy as sp

from . import _JAX_AVAILABLE, get_backend as __backend, maybe_jit, _KEY

from enum import Enum, auto, unique                 # for enumerations

# ---------------------------------------------------------------------

@unique
class PreconditionersType(Enum):
    '''
    Enumeration of the available preconditioners.
    '''
    SYMMETRIC           = auto()
    NONSYMMETRIC        = auto()
    
@unique
class PreconditionersTypeSym(Enum):
    """
    Enumeration of the available preconditioners - symmetric.
    """
    IDENTITY            = auto()
    JACOBI              = auto()
    INCOMPLETE_CHOLESKY = auto()
    INCOMPLETE_LU       = auto()

@unique
class PreconditionersTypeNoSym(Enum):
    """
    Enumeration of the available preconditioners - nonsymmetric.
    """
    IDENTITY            = auto()

# ---------------------------------------------------------------------

# PRECONDITIONERS

# ---------------------------------------------------------------------

class Preconditioner(ABC):
    """
    Preconditioner interface for use in iterative solvers.
    
    Attributes:
        is_positive_semidefinite (bool) : True if the matrix is positive semidefinite.
        is_gram (bool)                  : True if the preconditioner is based on a Gram matrix
                                        The Gram matrix is the product of two matrices (e.g. Sp and S).
        sigma (float)                   : Regularization parameter.
        type (int)                      : An integer flag for the preconditioner type.
    """
    def __init__(self, is_positive_semidefinite = False, backend = 'default'):
        """
        Initialize the preconditioner.
        
        Parameters:
            is_positive_semidefinite (bool) : True if the matrix is positive semidefinite.
            backend (optional)              : The computational backend to be used by the preconditioner.
        """
        self._backend_str               = backend
        self._backend, self._backends   = __backend(self._backend_str, scipy=True)
        self._is_positive_semidefinite  = is_positive_semidefinite
        
        if self._is_positive_semidefinite:
            self._stype                 = PreconditionersType.SYMMETRIC
        else:
            self._stype                 = PreconditionersType.NONSYMMETRIC
            
        self._is_gram                   = False # Gram matrix flag
        self._sigma                     = 0.0
        self._type                      = None  # Each concrete class sets its type
        # Tolerances
        self._tol_big                   = 1e10  # tolerance for big values
        self._tol_small                 = 1e-10 # tolerance for small values
        self._zero                      = 1e10  # treats 1/_zero as 0
        
    @property
    def is_positive_semidefinite(self):
        '''True if the matrix is positive semidefinite.'''
        return self._is_positive_semidefinite
    
    @property
    def stype(self):
        '''Symmetric?'''
        return self._stype

    @property
    def is_gram(self):
        '''True if the preconditioner is based on a Gram matrix.'''
        return self._is_gram
    
    @property
    def sigma(self):
        """Regularization parameter."""
        return self._sigma
    
    @property
    def type(self):
        """Preconditioner type."""
        return self._type
    
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
    
    # -----------------------------------------------------------------
    # Tolerances
    # -----------------------------------------------------------------
    
    @property
    def tol_big(self):
        return self._tol_big

    @tol_big.setter
    def tol_big(self, value):
        self._tol_big = value
        
    @property
    def tol_small(self):
        return self._tol_small
    
    @tol_small.setter
    def tol_small(self, value):
        self._tol_small = value
    
    @property
    def zero(self):
        return self._zero

    @zero.setter
    def zero(self, value):
        self._zero = value
    
    # -----------------------------------------------------------------
    
    @abstractmethod
    def _set_gram(self, sp : 'array-like', s : "array-like", sigma : float = 0.0, backend = 'default'):
        """
        Set the preconditioner using two matrices (e.g. Sp and S).
        
        Args:
            sp, s (np.ndarray): Matrices whose product is used.
            sigma (float): Regularization parameter.
        """
        pass

    @abstractmethod
    def _set_standard(self, A, sigma : float = 0.0, backend = 'default'):
        """
        Set the preconditioner using standard form.
        """
        pass
    
    # -----------------------------------------------------------------
    
    def set(self, A, sigma : float = 0.0, Ap = None, backend = 'default'):
        """
        Set the preconditioner from a matrix A.
        
        Args:
            A (np.ndarray)  : The matrix to decompose.
            sigma (float)   : Regularization parameter.
        """
        if self._backend_str != backend:
            self._backend, self._backends = __backend(backend, scipy=True)
            
        if self.is_gram:
            if Ap is not None:
                self._set_gram(A, Ap, sigma, backend = backend)
            else:
                self._set_gram(A, self.backend.conjugate(A).T, sigma, backend = backend)
        else:
            self._set_standard(A, sigma=sigma, backend = backend)

    @abstractmethod
    def apply(self, r, sigma : float = 0.0, backend = 'default') -> np.ndarray:
        """
        Apply the preconditioner to a vector.
        
        Args:
            r (np.ndarray): The vector to precondition.
            sigma (float): Optional regularization override.
            
        Returns:
            np.ndarray: The preconditioned vector.
        """
        if self._backend_str != backend:
            self._backend, self._backends = __backend(backend, scipy=True)
        pass

    def __call__(self, r: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        return self.apply(r, sigma)

# =====================================================================
# Identity preconditioner
# =====================================================================

class IdentityPreconditioner(Preconditioner):
    """
    Identity preconditioner.
    """
    def __init__(self, is_positive_semidefinite = False, backend = 'default'):
        super().__init__(is_positive_semidefinite = is_positive_semidefinite, backend = backend)
        self._type  = PreconditionersTypeSym.IDENTITY

    def _set_gram(self, Sp, S, sigma : float = 0.0, backend = 'default'):
        pass

    def _set_standard(self, A, sigma : float = 0.0, backend = 'default'):
        pass

    def apply(self, r, sigma : float = 0.0, backend = 'default'):
        return r
    
    def __repr__(self):
        return "Identity Preconditioner"
    
    def __str__(self):
        return "Identity Preconditioner"
    
# =====================================================================
# Jacobi preconditioner
# =====================================================================

class JacobiPreconditioner(Preconditioner):
    """
    Jacobi preconditioner. It is a simple diagonal preconditioner that 
    uses the inverse of the diagonal entries of the matrix.
    """
    def __init__(self, backend = 'default'):
        super().__init__(is_positive_semidefinite = True, backend = backend)
        self._type      = PreconditionersTypeSym.JACOBI
        self._diaginv   = None
        
    @property
    def diaginv(self):
        '''Inverse of the diagonal.'''
        return self._diaginv
    
    @diaginv.setter
    def diaginv(self,value):
        self._diaginv = value

    # -----------------------------------------------------------------
    
    @maybe_jit
    def _set_gram(self, Sp, S, sigma : float = 0.0, backend = 'default'):
        super()._set_gram(Sp, S, sigma, backend)
        self.sigma      = sigma
        self.diaginv    = 1.0 / self._backend.clip(self._backend.linalg.norm(S, axis=0) + sigma, self.tol_small, self.tol_big)

    @maybe_jit
    def _set_standard(self, A, sigma : float = 0.0, backend = 'default'):
        super()._set_standard(A, sigma, backend)
        self.sigma      = sigma
        self.diaginv    = 1.0 / self._backend.clip(self._backend.diag(A) + sigma, self.tol_small, self.tol_big)
        
    def apply(self, r, sigma : float = 0.0, backend = 'default'):
        return self.diaginv % r
    
    def __repr__(self):
        return f"Jacobi Preconditioner (s={self._sigma:.2e})"
    
    def __str__(self):
        return f"pre_jacobi_{self._sigma:.2e}"
    
# =====================================================================
# Incomplete Cholesky factorization
# =====================================================================

class IncompleteCholeskyPreconditioner(Preconditioner):
    """
    Incomplete Cholesky preconditioner.
    """
    def __init__(self, backend = 'default'):
        super().__init__(is_positive_semidefinite = True, backend = backend)
        self._type      = PreconditionersTypeSym.INCOMPLETE_CHOLESKY
        self._l         = None
        
    @property
    def l(self):
        '''Lower triangular matrix.'''
        return self._l

    # -----------------------------------------------------------------

    @maybe_jit
    def _set_gram(self, sp, s, sigma : float = 0.0, backend = 'default'):
        ''' Set the preconditioner using two matrices (e.g. Sp and S). '''
        super()._set_gram(sp, s, sigma, backend)
        return self._set_standard(sp @ s, sigma, backend)

    @maybe_jit
    def _set_standard(self, A, sigma : float = 0.0, backend = 'default'):
        super()._set_standard(A, sigma, backend)
        self.sigma = sigma
        try:
            self._l = self._backend.linalg.cholesky(A + sigma * self._backend.eye(A.shape[0]), lower = True)
        except Exception as e:
            print("Cholesky decomposition failed:", e)
            self._l = None

    @maybe_jit
    def _apply(self, r, backend = 'default'):
        ''' Apply the preconditioner to a vector. '''
        y = self._backend.linalg.solve_triangular(self._l, r, lower=True, check_finite=False)
        return self._backend.linalg.solve_triangular(self._l.T, y, lower=False, check_finite=False)

    def apply(self, r, sigma : float = 0.0, backend = 'default'):
        """
        Apply the preconditioner to a given vector.
        This method applies a Cholesky-based preconditioner to the input vector r.
        If the Cholesky decomposition (stored in self._l) is not available (i.e., None),
        the original vector is returned. Depending on the availability of JAX, the method
        uses either JAX's or SciPy's triangular solve functions to perform a forward solve
        on L * y = r followed by a backward solve on Lᵀ * z = y. In case any exception is raised
        during these operations, the exception is caught, an error message is printed, and the
        original vector r is returned.
        Parameters:
            r (array_like): The input vector to which the preconditioner is applied.
            sigma (float, optional): A scalar parameter for preconditioning (currently unused). Defaults to 0.0.
        Returns:
            array_like: The result of applying the preconditioner to the vector r,
            or the original vector if the Cholesky decomposition is unavailable or an error occurs.
        """
        
        # If the Cholesky decomposition failed, return the original vector
        if self._l is None:
            return r
        
        try:
            return self._apply(r, backend)
        except Exception as e:
            print("Cholesky solve failed:", e)
            return r
            
    def __repr__(self):
        return f"Incomplete Cholesky Preconditioner (s={self._sigma:.2e})"
    
    def __str__(self):
        return f"pre_ichol_{self._sigma:.2e}"
    
# =====================================================================
# Choose wisely
# =====================================================================

def choose_precond(precond_type  : Union[PreconditionersTypeSym, PreconditionersTypeNoSym],
                    backend      = 'default') -> Preconditioner:
    """
    Select and return the appropriate preconditioner instance based on the provided preconditioner type.
    Parameters:
        precond_type (Union[PreconditionersTypeSym, PreconditionersTypeNoSym]):
            The type of preconditioner requested. For symmetric problems, use PreconditionersTypeSym;
            for non-symmetric problems, use PreconditionersTypeNoSym.
        backend (optional):
            The computational backend to be used by the preconditioner. Defaults to 'default'.
    Returns:
        An instance of the selected preconditioner:
            - For PreconditionersTypeNoSym.IDENTITY:
                Returns an IdentityPreconditioner initialized for non-symmetric systems.
            - For PreconditionersTypeSym.IDENTITY:
                Returns an IdentityPreconditioner initialized for symmetric systems.
            - For PreconditionersTypeSym.JACOBI:
                Returns a JacobiPreconditioner instance.
            - For PreconditionersTypeSym.INCOMPLETE_CHOLESKY:
                Returns an IncompleteCholeskyPreconditioner instance.
    Raises:
        ValueError:
            If precond_type is not an instance of either PreconditionersTypeSym or PreconditionersTypeNoSym.
    """
    
    if isinstance(precond_type, PreconditionersTypeNoSym):
        match (precond_type):
            case PreconditionersTypeNoSym.IDENTITY:
                return IdentityPreconditioner(False, backend = backend)
    elif isinstance(precond_type, PreconditionersTypeSym):
        match (precond_type):
            case PreconditionersTypeSym.IDENTITY:
                return IdentityPreconditioner(True, backend=backend)
            case PreconditionersTypeSym.JACOBI:
                return JacobiPreconditioner(backend = backend)
            case PreconditionersTypeSym.INCOMPLETE_CHOLESKY:
                return IncompleteCholeskyPreconditioner(backend = backend)
    else:
        raise ValueError("Only PreconditionersTypeNoSym and PreconditionersTypeSym are supported.")
    
# =====================================================================

