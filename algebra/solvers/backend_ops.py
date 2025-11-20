'''
Provides a clean abstraction layer for NumPy and JAX operations used, for example, in
iterative solvers. This allows solvers to be backend-independent while
maintaining performance.

This module leverages the existing backend infrastructure from utils.py
while providing solver-specific operations like Givens rotations.

File:       general_python/algebra/solvers/backend_ops.py
Author:     Maksymilian Kliczkowski
Desc:       Backend-agnostic operations for linear algebra solvers.
'''

from typing import Tuple, Any, Callable, Union
import numpy as np

# Import from the existing backend infrastructure
try:
    from ..utils import (
        JAX_AVAILABLE,
        backend_mgr,
        get_backend,
        Array,
        is_jax_array
    )
except ImportError as e:
    raise ImportError("Failed to import utils from algebra module. Ensure QES package is correctly installed.") from e

# ---------------------------------------------------------------------------
# JAX imports if available
# ---------------------------------------------------------------------------

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        import jax.lax as lax
    except ImportError:
        JAX_AVAILABLE   = False
        jax             = None
        jnp             = np
        lax             = None
else:
    jax = None
    jnp = np
    lax = None

# ============================================================================
#! Type Aliases
# ============================================================================

# Use the Array type from utils.py for consistency
Backend = Union[type(np), type(jnp)]

# ============================================================================
#! Backend Operations Class
# ============================================================================

class BackendOps:
    """
    Backend-agnostic operations for linear algebra.
    
    Provides a unified interface for common operations that works with both
    NumPy and JAX backends. This class wraps the existing backend infrastructure
    from utils.py and adds solver-specific operations like Givens rotations.
    
    Example:
        >>> ops     = BackendOps.get_ops('jax')
        >>> x       = ops.zeros(10)
        >>> norm_x  = ops.norm(x)
    """
    
    def __init__(self, backend: Union[str, Backend]):
        """
        Initialize backend operations.
        
        Args:
            backend: Either 'numpy', 'jax', 'default', or a module (np/jnp)
        """
        if isinstance(backend, str):
            backend_lower = backend.lower()
            
            if backend_lower in ['numpy', 'np']:
                self.backend        = np
                self.backend_sp     = None  # Will use scipy
                self.is_jax         = False
                self.backend_name   = 'numpy'
            elif backend_lower in ['jax', 'jnp'] and JAX_AVAILABLE:
                self.backend        = jnp
                self.backend_sp     = jax.scipy if jax else None
                self.is_jax         = True
                self.backend_name   = 'jax'
            elif backend_lower in ['default', 'auto']:
                # Use the global backend manager's active backend
                self.backend_name   = backend_mgr.name
                self.backend        = backend_mgr.np
                self.backend_sp     = backend_mgr.scipy
                self.is_jax         = (self.backend_name == 'jax')
            else:
                raise ValueError(f"Unknown backend: {backend}")
        else:
            # Assume it's a module
            self.backend            = backend
            self.is_jax             = (backend is jnp) or (hasattr(backend, '__name__') and 'jax' in backend.__name__)
            self.backend_name       = 'jax' if self.is_jax else 'numpy'
            self.backend_sp         = jax.scipy if self.is_jax and jax else None
    
    @staticmethod
    def get_ops(backend: Union[str, Backend] = 'default') -> 'BackendOps':
        """
        Factory method to create BackendOps instance.
        
        Args:
            backend: Backend identifier ('numpy', 'jax', 'default', or module)
            
        Returns:
            BackendOps instance
        """
        return BackendOps(backend)
    
    @staticmethod
    def from_array(arr: Array) -> 'BackendOps':
        """
        Create BackendOps instance based on array type.
        
        This is useful when you have an array and want to get the appropriate
        backend operations for it.
        
        Args:
            arr: NumPy or JAX array
            
        Returns:
            BackendOps instance for the array's backend
            
        Example:
            >>> x = jnp.array([1, 2, 3])
            >>> ops = BackendOps.from_array(x)
            >>> assert ops.backend_name == 'jax'
        """
        if is_jax_array(arr):
            return BackendOps('jax')
        else:
            return BackendOps('numpy')
    
    # ------------------------------------------------------------------------
    #! Array Creation
    # ------------------------------------------------------------------------
    
    def zeros(self, shape, dtype=None) -> Array:
        """Create array of zeros."""
        return self.backend.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None) -> Array:
        """Create array of ones."""
        return self.backend.ones(shape, dtype=dtype)
    
    def array(self, data, dtype=None) -> Array:
        """Create array from data."""
        return self.backend.array(data, dtype=dtype)
    
    def copy(self, arr: Array) -> Array:
        """Create a copy of array."""
        if self.is_jax:
            return arr.copy() if hasattr(arr, 'copy') else self.backend.array(arr)
        return arr.copy()
    
    # ------------------------------------------------------------------------
    #! Linear Algebra Operations
    # ------------------------------------------------------------------------
    
    def dot(self, a: Array, b: Array) -> Union[float, complex, Array]:
        """
        Compute dot product of two arrays.
        
        For complex arrays, this is the conjugate dot product: conj(a) @ b
        """
        return self.backend.dot(self.backend.conj(a), b)
    
    def vdot(self, a: Array, b: Array) -> Union[float, complex]:
        """
        Compute conjugate dot product (always returns scalar).
        Equivalent to np.vdot or jnp.vdot.
        """
        return self.backend.vdot(a, b)
    
    def norm(self, x: Array, ord=None) -> float:
        """Compute vector/matrix norm."""
        return self.backend.linalg.norm(x, ord=ord)
    
    def solve(self, A: Array, b: Array) -> Array:
        """Solve linear system Ax = b."""
        return self.backend.linalg.solve(A, b)
    
    def pinv(self, A: Array, rcond=1e-15) -> Array:
        """Compute Moore-Penrose pseudo-inverse."""
        return self.backend.linalg.pinv(A, rcond=rcond)
    
    def cholesky(self, A: Array, lower: bool = True) -> Array:
        """
        Compute Cholesky decomposition.
        
        Returns lower triangular L such that A = L @ L.T
        """
        if self.is_jax:
            return jax.scipy.linalg.cholesky(A, lower=lower)
        else:
            import scipy.linalg
            return scipy.linalg.cholesky(A, lower=lower)
    
    def triangular_solve(self, A: Array, b: Array, lower: bool = True) -> Array:
        """
        Solve triangular system.
        
        Args:
            A: Triangular matrix
            b: Right-hand side
            lower: If True, A is lower triangular, else upper
        """
        if self.is_jax:
            return jax.scipy.linalg.solve_triangular(A, b, lower=lower)
        else:
            import scipy.linalg
            return scipy.linalg.solve_triangular(A, b, lower=lower)
    
    # ------------------------------------------------------------------------
    #! Element-wise Operations
    # ------------------------------------------------------------------------
    
    def sqrt(self, x: Union[float, Array]) -> Union[float, Array]:
        """Element-wise square root."""
        return self.backend.sqrt(x)
    
    def abs(self, x: Union[float, complex, Array]) -> Union[float, Array]:
        """Element-wise absolute value/magnitude."""
        return self.backend.abs(x)
    
    def real(self, x: Union[complex, Array]) -> Union[float, Array]:
        """Real part of complex array."""
        return self.backend.real(x)
    
    def imag(self, x: Union[complex, Array]) -> Union[float, Array]:
        """Imaginary part of complex array."""
        return self.backend.imag(x)
    
    def conj(self, x: Array) -> Array:
        """Complex conjugate."""
        return self.backend.conj(x)
    
    def maximum(self, a, b):
        """Element-wise maximum."""
        return self.backend.maximum(a, b)
    
    def minimum(self, a, b):
        """Element-wise minimum."""
        return self.backend.minimum(a, b)
    
    # ------------------------------------------------------------------------
    #! Utility Operations
    # ------------------------------------------------------------------------
    
    def where(self, condition, x, y):
        """Element-wise selection based on condition."""
        return self.backend.where(condition, x, y)
    
    def allclose(self, a: Array, b: Array, rtol=1e-5, atol=1e-8) -> bool:
        """Check if arrays are element-wise equal within tolerance."""
        return self.backend.allclose(a, b, rtol=rtol, atol=atol)
    
    def isnan(self, x: Array) -> Array:
        """Check for NaN values."""
        return self.backend.isnan(x)
    
    def isinf(self, x: Array) -> Array:
        """Check for infinite values."""
        return self.backend.isinf(x)
    
    def any(self, x: Array) -> bool:
        """Check if any element is True."""
        return self.backend.any(x)
    
    def all(self, x: Array) -> bool:
        """Check if all elements are True."""
        return self.backend.all(x)
    
    # ------------------------------------------------------------------------
    #! Special Solver Operations
    # ------------------------------------------------------------------------
    
    def sym_ortho(self, a: float, b: float) -> Tuple[float, float, float]:
        """
        Symmetric orthogonalization (Givens rotation).
        
        Computes c, s, r such that:
            [ c  s ] [ a ]   [ r ]
            [-s  c ] [ b ] = [ 0 ]
        
        where c^2 + s^2 = 1 and r = sqrt(a^2 + b^2).
        
        This is used in MINRES and MINRES-QLP for Givens rotations.
        
        Args:
            a, b: Scalars to orthogonalize
            
        Returns:
            c, s, r: Givens rotation parameters
        """
        if b == 0:
            if a == 0:
                c = 1.0
            else:
                c = np.sign(a) if not isinstance(a, complex) else a / np.abs(a)
            s = 0.0
            r = np.abs(a)
        elif a == 0:
            c = 0.0
            s = np.sign(b) if not isinstance(b, complex) else b / np.abs(b)
            r = np.abs(b)
        elif np.abs(b) > np.abs(a):
            tau = a / b
            s = np.sign(b) / np.sqrt(1.0 + tau * tau)
            c = s * tau
            r = b / s
        else:
            tau = b / a
            c = np.sign(a) / np.sqrt(1.0 + tau * tau)
            s = c * tau
            r = a / c
        
        return c, s, r
    
    def apply_givens_rotation(self, h1: float, h2: float, c: float, s: float) -> Tuple[float, float]:
        """
        Apply Givens rotation to a 2-vector.
        
        Computes [ c  s ] [ h1 ]
                 [-s  c ] [ h2 ]
        
        Args:
            h1, h2: Vector components
            c, s: Givens rotation parameters
            
        Returns:
            Rotated components (h1_new, h2_new)
        """
        h1_new = c * h1 + s * h2
        h2_new = -s * h1 + c * h2
        return h1_new, h2_new

# ============================================================================
#! Global Backend Operation Instances
# ============================================================================

# Pre-create instances for common backends
# These use the global backend manager's configuration
numpy_ops = BackendOps.get_ops('numpy')

if JAX_AVAILABLE:
    jax_ops = BackendOps.get_ops('jax')
else:
    jax_ops = numpy_ops  # Fallback

# Default ops uses the global backend manager's active backend
default_ops = BackendOps.get_ops('default')


# ============================================================================
#! Helper Functions
# ============================================================================

def get_backend_ops(backend: Union[str, Backend, None] = None) -> BackendOps:
    """
    Get backend operations instance.
    
    This is the main entry point for getting backend operations in solvers.
    It integrates with the global backend manager from utils.py.
    
    Args:
        backend: Backend identifier ('numpy', 'jax', 'default', None, or module)
                If None, uses the global backend manager's active backend.
        
    Returns:
        BackendOps instance
        
    Example:
        >>> ops = get_backend_ops('jax')
        >>> x = ops.zeros(10)
        
        >>> # Use global backend
        >>> ops = get_backend_ops()
        >>> y = ops.ones(5)
    """
    if backend is None or backend == 'default':
        return default_ops
    elif backend == 'numpy' or backend == 'np':
        return numpy_ops
    elif backend == 'jax' or backend == 'jnp':
        if JAX_AVAILABLE:
            return jax_ops
        else:
            import warnings
            warnings.warn("JAX not available, using NumPy backend", RuntimeWarning)
            return numpy_ops
    else:
        # Custom backend module or string
        return BackendOps.get_ops(backend)
