'''
Spectral backend for quadratic Hamiltonians.
Provides functions to compute Green's functions for quadratic Hamiltonians.

---------------------------------
File        : general_python/physics/spectral/quadratic/spectral_backend_quadratic.py
Author      : Maksymilian Kliczkowski
Date        : 2025-11-19
Description : Spectral functions for quadratic Hamiltonians.
---------------------------------
'''

from typing import Optional, Union, Tuple, Literal, Any, Callable
import numpy as np
import numba as nb

from numpy.typing import NDArray
from sympy import denom

# Backend imports
try:
    from ....algebra.utils import JAX_AVAILABLE, get_backend
except ImportError:
    JAX_AVAILABLE   = False
    get_backend     = lambda x="default": np

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = None

# Type alias
Array = Union[np.ndarray, Any]  # Any allows JAX arrays

# -----------------------------------------
# Quadratic Hamiltonian Green's functions
# -----------------------------------------

@nb.njit(cache=True)
def _greens_function_quadratic_scalar(ev, A, B, occ, omega_val, eta):
    r"""
    Numba JIT-compiled helper for many-body Green's function for quadratic Hamiltonians.
    Scalar version for single omega value.
    Optimized with early exits and reduced operations.
    """

    N = ev.shape[0]
    G_real = 0.0
    G_imag = 0.0

    # prebuild complex z = omega + i eta
    z_real = omega_val
    z_imag = eta

    for m in range(N):
        if not occ[m]:
            continue
        em = ev[m]

        for n in range(N):
            if occ[n]:
                continue

            # compute matrix element first - early exit if zero
            A_mn = A[m, n]
            B_nm = B[n, m]
            num = A_mn * B_nm
            
            # Skip if numerator is negligible
            if abs(num) < 1e-16:
                continue

            en = ev[n]
            deltaE = en - em

            # denominator d = (z - deltaE) = (omega - deltaE) + i eta
            d_real = z_real - deltaE
            d_imag = z_imag

            # complex division: num / d
            denom_norm = d_real*d_real + d_imag*d_imag

            t_real = (num.real * d_real + num.imag * d_imag) / denom_norm
            t_imag = (num.imag * d_real - num.real * d_imag) / denom_norm

            G_real += t_real
            G_imag += t_imag

    return G_real + 1j * G_imag

def _greens_function_quadratic(ev, A, B, occ, omega, eta):
    r"""
    Wrapper for _greens_function_quadratic_scalar that handles both scalar and array omega.
    """
    # Check if omega is scalar
    if np.ndim(omega) == 0:
        return _greens_function_quadratic_scalar(ev, A, B, occ, omega, eta)
    
    # Array case: vectorize over omega
    G = np.empty(omega.shape, dtype=np.complex128)
    for i in range(omega.size):
        G.flat[i] = _greens_function_quadratic_scalar(ev, A, B, occ, omega.flat[i], eta)
    
    return G

def greens_function_quadratic(
        omega           : float,
        eigenvalues     : Array,
        eigenvectors    : Optional[Array]   = None,   # c_i = \sum_\alpha U_{i\alpha} d_\alpha
        eta             : float             = 0.01,
        *,
        operator_a      : Optional[Array]   = None,   # matrix A in c-basis
        operator_b      : Optional[Array]   = None,   # matrix B in c-basis
        occupations     : Optional[Array]   = None,   # n_\alpha (0/1) in eigenbasis
        basis_transform : bool              = True,
        backend         : str               = "default"
    ) -> Array:
    r"""
    Correct many-body Green's function for quadratic Hamiltonians.

    If operator_a is None:
        Returns single-particle resolvent:
            G(\omega) = (\omega + i\eta - h)^(-1)  =  U diag(1 / (\omega + i\eta - ε_\alpha)) U^†

    If operator_a is provided (and operator_b optionally):
        Returns the full many-body zero-temperature Green's function:
            G_AB(\omega) = \Sum _{m,n} n_m (1 - n_n) A_{mn} B_{nm}
                                       / (\omega + i\eta - (ε_n - ε_m))
        where A_e = U^+ A U and similarly for B.

    Parameters
    ----------
    occupations:
        n_\alpha in eigenbasis (0 or 1).  If None -> half-filling by default.
        This is NOT a many-body state; it is the Slater determinant occupation mask.
    """

    be      = get_backend(backend)
    ev      = be.asarray(eigenvalues,   dtype=be.complex128)
    omega   = be.asarray(omega,         dtype=be.complex128)
    eta     = be.asarray(eta,           dtype=be.complex128)
    z       = omega + 1j * eta

    # -----------------------------------------
    # No eigenvectors -> diagonal resolvent
    # -----------------------------------------
    if eigenvectors is None:
        return 1.0 / (z - ev)

    U       = be.asarray(eigenvectors, dtype=be.complex128)
    N       = ev.shape[0]

    # -----------------------------------------
    # No operators -> single-particle resolvent
    # -----------------------------------------
    if operator_a is None:
        denom = 1.0 / (z - ev)
        return U @ (be.diag(denom)) @ U.T.conj()

    # -----------------------------------------
    # Many-body Green's function G_AB(\omega)
    # -----------------------------------------

    # Default occupations: half-filling
    if occupations is None:
        occ             = be.zeros(N, dtype=be.bool_)
        occ[:N//2]      = True
    else:
        occ             = be.asarray(occupations, dtype=be.bool_)

    # Transform operators to eigenbasis
    A = U.T.conj() @ be.asarray(operator_a, dtype=be.complex128) @ U if basis_transform else be.asarray(operator_a, dtype=be.complex128)
    if operator_b is None:
        B = A.T.conj()
    else:
        B = U.T.conj() @ be.asarray(operator_b, dtype=be.complex128) @ U if basis_transform else be.asarray(operator_b, dtype=be.complex128)

    if be.__name__ == 'numpy':
        omega_val   = omega.real    if hasattr(omega, 'real')   else omega
        eta_val     = eta.real      if hasattr(eta, 'real')     else eta
        return _greens_function_quadratic(ev, A, B, occ, omega_val, eta_val)

    # Scalar Green's function
    G = omega * 0.0 + 0.0j

    # Double sum over m,n with occupation factors
    # Computes the Green's function <A| 1 / (w + i eta - (H - E0)) |B>
    # where <A| = <c0| A and |B> = B |c0>
    for m in range(N):
        if not occ[m]:    # needs to be occupied
            continue
        for n in range(N):
            if occ[n]:    # needs to be empty
                continue

            deltaE = ev[n] - ev[m]
            G     += (A[m, n] * B[n, m]) / (z - deltaE)

    return G

@nb.njit(cache=True)
def _greens_function_quadratic_finite_T_scalar(ev, A, B, beta, mu, omega_val, eta):
    r"""
    Numba helper for finite-temperature many-body Green's function (scalar omega):

        G(w) = \Sum _{m,n} f_m (1 - f_n) A[m,n] B[n,m] 
                          / (w + in - (ε_n - ε_m))

    where f_m = 1 / (1 + exp[β(ε_m - μ)]).
    Optimized with early exits and cached Fermi factors.
    """

    N = ev.shape[0]
    G_real = 0.0
    G_imag = 0.0

    # -----------------------------------
    # Build Fermi occupations f_m
    # -----------------------------------
    f = np.empty(N, dtype=np.float64)
    for m in range(N):
        x = beta * (ev[m].real - mu)
        # avoid overflow with safe exp calculation
        if x > 50.0:
            f[m] = 0.0
        elif x < -50.0:
            f[m] = 1.0
        else:
            f[m] = 1.0 / (1.0 + np.exp(x))

    # -----------------------------------
    # Complex frequency z = w + in
    # -----------------------------------
    z_real = omega_val
    z_imag = eta

    # -----------------------------------
    # Double sum over (m,n) with optimizations
    # -----------------------------------
    for m in range(N):
        fm = f[m]
        if fm < 1e-15:  # Skip if occupation negligible
            continue
        em = ev[m]

        for n in range(N):
            fn = f[n]
            weight = fm * (1.0 - fn)
            if weight < 1e-15:  # Skip if weight negligible
                continue

            # Compute matrix elements early
            A_mn = A[m, n]
            B_nm = B[n, m]
            num_prefactor = A_mn * B_nm
            
            if abs(num_prefactor) < 1e-16:  # Skip if matrix element negligible
                continue

            en = ev[n]
            deltaE = en - em

            # d = z - deltaE
            d_real = z_real - deltaE.real
            d_imag = z_imag - deltaE.imag

            # numerator = weight * A[m,n] * B[n,m]
            num = weight * num_prefactor

            # Complex division num / d
            denom_norm = d_real*d_real + d_imag*d_imag
            t_real = (num.real * d_real + num.imag * d_imag) / denom_norm
            t_imag = (num.imag * d_real - num.real * d_imag) / denom_norm

            G_real += t_real
            G_imag += t_imag

    return G_real + 1j * G_imag

def _greens_function_quadratic_finite_T(ev, A, B, beta, mu, omega, eta):
    r"""
    Wrapper for _greens_function_quadratic_finite_T_scalar that handles both scalar and array omega.
    """
    # Check if omega is scalar
    if np.ndim(omega) == 0:
        return _greens_function_quadratic_finite_T_scalar(ev, A, B, beta, mu, omega, eta)
    
    # Array case: vectorize over omega
    G = np.empty(omega.shape, dtype=np.complex128)
    for i in range(omega.size):
        G.flat[i] = _greens_function_quadratic_finite_T_scalar(ev, A, B, beta, mu, omega.flat[i], eta)
    
    return G

def greens_function_quadratic_finite_T(
        omega           : float,
        eigenvalues     : Array,
        eigenvectors    : Optional[Array]   = None, 
        eta             : float             = 0.01,
        *,
        operator_a      : Optional[Array]   = None,
        operator_b      : Optional[Array]   = None,
        beta            : float             = 1.0,
        mu              : float             = 0.0,
        basis_transform : bool              = True,
        backend         : str               = "default"
    ) -> Array:
    r"""
    Finite-temperature many-body Green's function for a quadratic Hamiltonian.

    Zero-temperature limit is recovered when beta -> +∞.

    If operator_a is None:
        Return the single-particle finite-T resolvent:
            G(w) = U diag(1 / (w + in - ε_a)) U^†
        (finite T does not change the resolvent itself)

    If operator_a is provided:
        Return the finite-temperature many-body Green's function:
            G_AB(w) = \Sum _{m,n} f_m (1 - f_n) A_{mn} B_{nm}
                                   / (w + in - (ε_n - ε_m)),
        where f_m is the Fermi-Dirac factor.

    Parameters
    ----------
    beta : float
        Inverse temperature β = 1/T.
    mu : float
        Chemical potential μ.
    """

    be      = get_backend(backend)
    ev      = be.asarray(eigenvalues, dtype=be.complex128)

    omega_c = be.asarray(omega, dtype=be.complex128)
    eta_c   = be.asarray(eta,   dtype=be.complex128)
    z       = omega_c + 1j * eta_c

    # -----------------------------------------
    # 0. No eigenvectors -> diagonal resolvent
    # -----------------------------------------
    if eigenvectors is None:
        return 1.0 / (z - ev)

    U       = be.asarray(eigenvectors, dtype=be.complex128)
    N       = ev.shape[0]

    # -----------------------------------------
    # 1. No operators -> single-particle resolvent
    # -----------------------------------------
    if operator_a is None:
        denom = 1.0 / (z - ev)
        return U @ (be.diag(denom)) @ U.T.conj()

    # -----------------------------------------
    # 2. Many-body finite-temperature Green's function
    # -----------------------------------------
    A = U.T.conj() @ be.asarray(operator_a, dtype=be.complex128) @ U if basis_transform else be.asarray(operator_a, dtype=be.complex128)
    if operator_b is None:
        B = A.T.conj()
    else:
        B = U.T.conj() @ be.asarray(operator_b, dtype=be.complex128) @ U if basis_transform else be.asarray(operator_b, dtype=be.complex128)

    # Numba path for NumPy backend
    if be.__name__ == 'numpy':
        return _greens_function_quadratic_finite_T(
            np.asarray(ev),
            np.asarray(A),
            np.asarray(B),
            float(beta),
            float(mu),
            omega.real if hasattr(omega, 'real') else omega,
            eta.real if hasattr(eta, 'real') else eta,
        )

    # ----------------------------------------------------
    # JAX or other backend -> pure Python implementation
    # ----------------------------------------------------
    G = omega_c * 0.0 + 0.0j

    # build Fermi factor f_m
    f = be.asarray(1.0 / (1.0 + be.exp(beta * (ev.real - mu))))

    for m in range(N):
        em = ev[m]
        fm = f[m]

        for n in range(N):
            fn      = f[n]
            weight  = fm * (1.0 - fn)
            if weight == 0:
                continue

            deltaE = ev[n] - em
            G     += weight * (A[m,n] * B[n,m]) / (z - deltaE)
    return G

# -----------------------------------------
#! EOF
# -----------------------------------------