"""
QES/general_python/algebra/spectral_backend.py

Unified spectral function backend for computing spectral properties
of quantum systems from eigenvalues, eigenvectors, or Lanczos coefficients.

Features:
  - Spectral function from Green's function
  - Direct spectral function from eigenvalues (no full diagonalization)
  - Momentum-resolved spectral function for periodic systems
  - Green's function in eigenbasis or Lanczos basis
  - Fast spectral calculations from Lanczos coefficients (tridiagonal form)
  - Integrated spectral weight
  - Peak detection
  - Backend support: NumPy and JAX

Type Safety:
  - Explicit handling of complex spectral functions
  - Proper dtype preservation (complex stays complex, real stays real)
  - No implicit casting of complex to real

----------------------------------------------------------------------------
File        : general_python/algebra/spectral_backend.py
Author      : Maksymilian Kliczkowski
Email       : maksymilian.kliczkowski@pwr.edu.pl
Description : Spectral function calculations using various backends.
----------------------------------------------------------------------------
"""

from typing import Optional, Union, Tuple, Literal, Any, Callable
import numpy as np
import numba as nb

from numpy.typing import NDArray
from sympy import denom

# Backend imports
try:
    from ...algebra.utils import JAX_AVAILABLE, get_backend
except ImportError:
    JAX_AVAILABLE = False
    get_backend = lambda x="default": np

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = None

# Type alias
Array = Union[np.ndarray, Any]  # Any allows JAX arrays

# =============================================================================
# Green's Function Calculations - 0 TEMPERATURE
# =============================================================================

def greens_function_diagonal(omega: Union[float, Array], eigenvalues: Array, eta: float = 0.01, backend: str = "default") -> Array:
    r"""
    Compute diagonal Green's function directly from eigenvalues.
    
    G_nn(omega) = 1 / (omega + ieta  - E_n)
    
    Parameters
    ----------
    omega : float or array
        Frequency omega (can be complex). If array, broadcasts over eigenvalues.
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n.
    eta : float, optional
        Broadening parameter eta  (default: 0.01).
    backend : str, optional
        Numerical backend: 'numpy' or 'jax' (default: 'default').
        
    Returns
    -------
    Array
        Diagonal Green's function elements.
        - If omega is scalar: shape (N,)
        - If omega is array of shape (M,): shape (N, M)
        
    Notes
    -----
    Efficient for diagonal systems or when only diagonal elements are needed.
    """
    be              = get_backend(backend)
    eigenvalues     = be.asarray(eigenvalues, dtype=be.complex128)
    eta_complex     = be.asarray(eta, dtype=be.complex128)
    
    # Handle scalar or array omega with proper broadcasting
    if isinstance(omega, (int, float, complex)):
        omega_complex = be.asarray(omega, dtype=be.complex128)
        return 1.0 / (omega_complex + 1j * eta_complex - eigenvalues)
    else:
        omega_complex = be.asarray(omega, dtype=be.complex128)
        # Broadcasting: (N, M) = eigenvalues[:, None] and omega[None, :]
        return 1.0 / (omega_complex[None, :] + 1j * eta_complex - eigenvalues[:, None])
    
#? single particle Green's function calculations

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
            G_AB(\omega) = Σ_{m,n} n_m (1 - n_n) A_{mn} B_{nm}
                                       / (\omega + i\eta - (ε_n - ε_m))
        where A_e = U^+ A U and similarly for B.

    Parameters
    ----------
    occupations:
        n_\alpha in eigenbasis (0 or 1).  If None -> half-filling by default.
        This is NOT a many-body state; it is the Slater determinant occupation mask.
    """

    be      = get_backend(backend)
    ev      = be.asarray(eigenvalues, dtype=be.complex128)
    omega   = be.asarray(omega,         dtype=be.complex128)
    eta     = be.asarray(eta,           dtype=be.complex128)
    z       = omega + 1j * eta

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
    # 2. Many-body Green's function G_AB(\omega)
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
        omega_val = omega.real if hasattr(omega, 'real') else omega
        eta_val = eta.real if hasattr(eta, 'real') else eta
        return _greens_function_quadratic(ev, A, B, occ, omega_val, eta_val)

    # Scalar Green's function
    G = omega * 0.0 + 0.0j

    # Double sum over m,n with occupation factors
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

        G(ω) = Σ_{m,n} f_m (1 - f_n) A[m,n] B[n,m] 
                          / (ω + iη - (ε_n - ε_m))

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
    # Complex frequency z = ω + iη
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
            G(ω) = U diag(1 / (ω + iη - ε_a)) U^†
        (finite T does not change the resolvent itself)

    If operator_a is provided:
        Return the finite-temperature many-body Green's function:
            G_AB(ω) = Σ_{m,n} f_m (1 - f_n) A_{mn} B_{nm}
                                   / (ω + iη - (ε_n - ε_m)),
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

#? many-body Green's function calculations

def greens_function_manybody(
        omega           : Union[float, Array],
        eigenvalues     : Array,
        operator_a      : Array,
        eta             : float             = 0.01,
        *,
        mb_states       : Optional[Union[Array, int]] = None, # vector of indices is sufficient, as we have the matrix elements via operators
        operator_b      : Optional[Array]   = None,
        backend         : str               = "default",
        kind            : str               = "retarded") -> Array:
    r"""
    The Green's function in the many-body representation
    (for any two operators A and B):
    
    $$
    G_{AB}(\omega) = \sum_{m,n} \frac{<0|A|n><n|B|0>}{\omega + i\eta  - (E_n - E_0)} + \frac{<0|B|n><n|A|0>}{\omega + i\eta  + (E_n - E_0)}
    $$

    where |m> is the many-body state (usually the ground state), 
    |n> are excited states with energies E_n.
    
    Parameters
    ----------
    omega : float or array-like
        Frequency omega (can be array for multiple frequencies).
    eigenvalues : array-like, shape (N,)
        Many-body eigenvalues E_n.
    operator_a : array-like, shape (N, N)
        Many-body operator A in the eigenbasis.
    eta : float, optional
        Broadening parameter eta  (default: 0.01).
    mb_states : array-like or int, optional
        Many-body states |m> to consider (indices). If int, single state index.
        If None, defaults to ground state |0>.
    operator_b : array-like, shape (N, N), optional
        Many-body operator B in the eigenbasis. If None, uses A^†.
    backend : str, optional
        Numerical backend to use (default: "default").
    kind : str, optional
        Kind of Green's function to compute (default: "retarded").
        Supported kinds:
            - "retarded"        (default)
            - "advanced"
            - "greater"
            - "lesser"
            - "timeordered"
    """
    be  = get_backend(backend)
    E   = be.asarray(eigenvalues)

    # prepare omega array
    if isinstance(omega, (int, float, complex)):
        w = be.asarray([omega], dtype=be.complex128)
    else:
        w = be.asarray(omega, dtype=be.complex128)

    # reference states
    if mb_states is None:
        mb = be.asarray([0], dtype=int)
    elif isinstance(mb_states, int):
        mb = be.asarray([mb_states], dtype=int)
    else:
        mb = be.asarray(mb_states, dtype=int)

    # operator B defaults to A^dagger
    if operator_b is None:
        operator_b = operator_a.conj()

    A = be.asarray(operator_a)
    B = be.asarray(operator_b)

    # output
    G = be.zeros((len(mb), len(w)), dtype=be.complex128)

    # compute GF for each reference state
    for i, m in enumerate(mb):
        # energy differences DeltaE = E_n - E_m
        deltaE = E - E[m]

        # matrix elements:
        # A_mn = <m|A|n>, B_nm = <n|B|m>
        A_mn        = A[m, :]         # shape (Ns,)
        B_nm        = B[:, m]         # shape (Ns,)

        # denominators
        denom_pos   = w[None, :] + 1j * eta - deltaE[:, None]
        denom_neg   = w[None, :] + 1j * eta + deltaE[:, None]

        # left term: A_mn * B_nm / (w + i*eta - DeltaE)
        G[i, :]    += be.sum((A_mn[:, None] * B_nm[:, None]) / denom_pos, axis=0)

        # right term: B_nm * A_mn / (w + i*eta + DeltaE)
        G[i, :]    += be.sum((B_nm[:, None] * A_mn[:, None]) / denom_neg, axis=0)

    # Handle different Green's function types (kind)
    if kind == "advanced":
        G = G.conj()
    # For retarded, no modification needed
    # Note: "greater", "lesser", "timeordered" require full many-body treatment beyond standard Lehmann
    
    # return with collapsed shape if possible
    if len(mb) == 1 and len(w) == 1:
        return G[0, 0]
    if len(mb) == 1:
        return G[0, :]
    if len(w) == 1:
        return G[:, 0]

    return G

def greens_function_manybody_finite_T(
        omega           : Union[float, Array],
        eigenvalues     : Array,
        operator_a      : Array,
        eta             : float             = 0.01,
        *,
        beta            : float             = 1.0,
        operator_b      : Optional[Array]   = None,
        backend         : str               = "default",
        kind            : str               = "retarded",
        lehmann_full    : bool              = False) -> Array:
    r"""
    Finite-temperature many-body Green's function in the Lehmann representation.

    At finite temperature T > 0, the system is in a thermal mixed state:
        \rho = e^{-\beta H} / Z,  Z = Tr[e^{-\beta H}]

    Two conventions are supported:

    1. Full retarded GF (lehmann_full=True):
        G_{AB}^R(\omega) = (1/Z) \sum_{m,n} (e^{-\beta E_m} - e^{-\beta E_n})
                                    <m|A|n><n|B|m> / (\omega + i\eta - (E_n - E_m))

       This is the standard finite-T retarded Green's function.
       At T=0, reduces to sum over n from ground state m=0 (NOT the same as
       zero-T Lehmann which has both +/- frequency terms).

    2. Single-pole (lehmann_full=False, default):
        G_{AB}(\omega) = (1/Z) \sum_{m,n} e^{-\beta E_m} 
                                <m|A|n><n|B|m> / (\omega + i\eta - (E_n - E_m))

       Only positive-frequency poles. Useful for spectral functions where
       you want A(\omega) = -Im[G]/pi to have simple pole structure.

    Note: The zero-T greens_function_manybody() uses a DIFFERENT formula
    with both +/- frequency terms (particle addition + removal), which is
    appropriate for T=0 but does not directly match finite-T conventions.

    Parameters
    ----------
    omega : float or array-like
        Frequency \omega (can be array for multiple frequencies).
    eigenvalues : array-like, shape (N,)
        Many-body eigenvalues E_n.
    operator_a : array-like, shape (N, N)
        Many-body operator A in the eigenbasis.
    eta : float, optional
        Broadening parameter \eta (default: 0.01).
    beta : float, optional
        Inverse temperature \beta = 1/T (default: 1.0).
    operator_b : array-like, shape (N, N), optional
        Many-body operator B in the eigenbasis. If None, uses A^\dagger.
    backend : str, optional
        Numerical backend to use (default: "default").
    kind : str, optional
        Kind of Green's function (default: "retarded").
        Supported: "retarded", "advanced"
    lehmann_full : bool, optional
        If True (default), include (p_m - p_n) factor (standard retarded GF).
        If False, only p_m factor (single-pole, for spectral functions).
        
        **Important**: For computing spectral functions A(omega) = -Im[G]/pi,
        use lehmann_full=False to ensure A(omega) >= 0. The full Lehmann form
        with (p_m - p_n) can give negative spectral weights at finite T.

    Returns
    -------
    Array
        Finite-temperature Green's function G_{AB}(\omega).
        Shape: (len(omega),) if omega is array, scalar if omega is scalar.

    Notes
    -----
    This is the exact finite-T formula requiring full eigenspectrum.
    For large systems, use finite-temperature Lanczos methods (FTLM) instead.

    Physics:
    - All states contribute weighted by Boltzmann factor e^{-\beta E_m}
    - Full Lehmann (p_m - p_n): Standard retarded GF for response functions
    - Single-pole (p_m only): For spectral densities A(\omega) >= 0
    - Partition function Z normalizes the thermal ensemble
    
    Recommendation:
    - Use lehmann_full=True for dynamic susceptibilities, correlation functions
    - Use lehmann_full=False for spectral functions (ensures positivity)

    Examples
    --------
    >>> # For spectral functions (recommended: single-pole)
    >>> G_spec = greens_function_manybody_finite_T(
    ...     omega=omega_grid, eigenvalues=E_all, operator_a=A_op,
    ...     eta=0.05, beta=10.0, lehmann_full=False
    ... )
    >>> A_omega = spectral_function(greens_function=G_spec)  # A(omega) >= 0
    >>> 
    >>> # For response/correlation functions (full Lehmann)
    >>> G_resp = greens_function_manybody_finite_T(
    ...     omega=omega_grid, eigenvalues=E_all, operator_a=A_op,
    ...     eta=0.05, beta=10.0, lehmann_full=True
    ... )
    """
    be  = get_backend(backend)
    E   = be.asarray(eigenvalues, dtype=be.complex128)

    # prepare omega array
    if isinstance(omega, (int, float, complex)):
        w = be.asarray([omega], dtype=be.complex128)
        scalar_omega = True
    else:
        w = be.asarray(omega, dtype=be.complex128)
        scalar_omega = False

    # operator B defaults to A^\dagger
    if operator_b is None:
        operator_b = operator_a.conj().T

    A = be.asarray(operator_a, dtype=be.complex128)
    B = be.asarray(operator_b, dtype=be.complex128)

    # Compute Boltzmann weights e^{-\beta E_m}
    # Shift by minimum energy for numerical stability
    E_min = be.min(E.real)
    exp_factors = be.exp(-beta * (E.real - E_min))
    Z = be.sum(exp_factors)  # Partition function

    # Thermal weights
    p_m = exp_factors / Z  # p_m = e^{-\beta E_m} / Z

    # Output array
    G = be.zeros((len(w),), dtype=be.complex128)

    # Double sum over m, n with thermal weights
    for m in range(len(E)):
        # Skip negligible thermal populations
        if p_m[m] < 1e-15:
            continue

        # Energy differences \Delta E = E_n - E_m
        deltaE = E - E[m]

        # Matrix elements
        A_mn = A[m, :]  # <m|A|n>
        B_nm = B[:, m]  # <n|B|m>

        if lehmann_full:
            # Full Lehmann: (p_m - p_n) / (omega + ieta - deltaE)
            # This is the correct retarded GF at finite T
            for n in range(len(E)):
                if be.abs(deltaE[n]) < 1e-15:  # Skip m=n (gives 0/0)
                    continue
                    
                weight_diff = p_m[m] - p_m[n]  # e^{-beta E_m} - e^{-beta E_n}
                
                if be.abs(weight_diff) < 1e-15:
                    continue
                
                denom = w + 1j * eta - deltaE[n]
                G += weight_diff * (A_mn[n] * B_nm[n]) / denom
        else:
            # Single-pole: only p_m term (for spectral functions)
            denom_pos = w[None, :] + 1j * eta - deltaE[:, None]
            G += p_m[m] * be.sum((A_mn[:, None] * B_nm[:, None]) / denom_pos, axis=0)

    # Handle different Green's function types
    if kind == "advanced":
        G = G.conj()
    elif kind != "retarded":
        raise ValueError(f"Unsupported kind '{kind}'. Use 'retarded' or 'advanced'.")

    # Return scalar if input was scalar
    if scalar_omega:
        return G[0]
    return G

#? Lanczos Green's function calculations

def _greens_lanczos_single_chain(
        omega,
        lanczos_eigenvalues,
        lanczos_eigenvector,
        eta=0.01,
        *,
        mb_states=None,
        backend="default",
        kind="retarded"):
    """
    Zero-T retarded GF using Hermitian Lanczos.
    No operator info needed because Lanczos was started from |v0> = A|0>.
    
    Optimized for vectorization when possible.
    """
    be      = get_backend(backend)
    evals   = be.asarray(lanczos_eigenvalues, dtype=be.complex128)
    U_L     = be.asarray(lanczos_eigenvector, dtype=be.complex128)
    eta_c   = be.asarray(eta, dtype=be.complex128)
    M       = evals.shape[0]

    if mb_states is None:
        mb = be.asarray([0], dtype=int)
    elif isinstance(mb_states, int):
        mb = be.asarray([mb_states], dtype=int)
    else:
        mb = be.asarray(mb_states, dtype=int)

    # Vectorized calculation: shape (len(mb), M)
    weights = be.abs(U_L[mb, :])**2
    
    # Broadcast: (len(mb), M, len(omega))
    # denom[i,n,w] = omega[w] + i*eta - evals[n]
    denom = omega[None, None, :] + 1j*eta_c - evals[None, :, None]
    
    # Sum over eigenstates: (len(mb), len(omega))
    G = be.sum(weights[:, :, None] / denom, axis=1)

    if kind == "advanced":
        G = G.conj()
    elif kind != "retarded":
        raise ValueError("Single-chain supports 'retarded' or 'advanced' only.")

    # shape collapse
    if len(mb) == 1 and len(omega) == 1:
        return G[0, 0]
    if len(mb) == 1:
        return G[0, :]
    if len(omega) == 1:
        return G[:, 0]
    return G

def _greens_lanczos_bilanczos(
        omega,
        H,
        A_op,
        B_op,
        eta=0.01,
        *,
        mb_states=None,
        backend="default",
        kind="retarded",
        max_krylov=200):
    """
    General zero-T Green function G_AB using bi-Lanczos iteration.

    Builds left and right Krylov chains for each reference state m:
        |v0> = B|m>
        <w0| = <m|A^dagger
    Computes bi-orthonormal basis, tridiagonal T, and evaluates:
        G_m(z) = <w0| (z - T)^(-1) |v0>

    This supports any A,B (non-Hermitian, non-equal).
    Loops over all requested mb_states.
    """
    be = get_backend(backend)

    H = be.asarray(H, dtype=be.complex128)
    A = be.asarray(A_op, dtype=be.complex128)
    B = be.asarray(B_op, dtype=be.complex128)

    # Handle mb_states like single-chain version
    if mb_states is None:
        mb = be.asarray([0], dtype=int)
    elif isinstance(mb_states, int):
        mb = be.asarray([mb_states], dtype=int)
    else:
        mb = be.asarray(mb_states, dtype=int)

    # Result array: (len(mb), len(omega))
    G_all = be.zeros((len(mb), len(omega)), dtype=be.complex128)

    # Loop over each reference state
    for idx, m0 in enumerate(mb):
        # starting vectors for this reference state
        v0 = B[:, m0]            # |v0> = B|m0>
        w0 = A[m0, :].conj()     # <w0| = <m0|A^dagger

        # normalize and enforce <w0|v0> = 1
        s = be.vdot(w0, v0)
        if be.abs(s) < 1e-14:
            # Zero overlap - skip this state
            continue
        w0 = w0 / s

        # allocate arrays for this Lanczos chain
        alpha = []
        beta  = []
        gamma = []

        # first entries
        v_prev = be.zeros_like(v0)
        w_prev = be.zeros_like(w0)

        v = v0
        w = w0

        # bi-Lanczos iterations
        for j in range(max_krylov):
            Hv = H @ v
            Hw = H.conj().T @ w

            a = be.vdot(w, Hv)

            # residuals
            r = Hv - a * v - (gamma[-1] * v_prev if j > 0 else 0.0)
            l = Hw - a.conj() * w - (beta[-1] * w_prev if j > 0 else 0.0)

            # new coupling
            b2 = be.vdot(l, r)
            if be.abs(b2) < 1e-14:
                alpha.append(a)
                break

            b = be.sqrt(b2)
            alpha.append(a)
            beta.append(b)
            gamma.append(b)

            # normalize next vectors
            v_next = r / b
            w_next = l / b

            # enforce bi-orthonormality <w_next|v_next> = 1
            # Only rescale w_next; v_next is already normalized from r/b
            scale = be.vdot(w_next, v_next)
            if be.abs(scale) < 1e-14:
                # Loss of bi-orthogonality
                alpha.append(a)
                break
            w_next = w_next / scale

            # shift vectors
            v_prev, v = v, v_next
            w_prev, w = w, w_next

        # now build the continued fraction for e1^T (z-T)^(-1) e1
        def cf(z):
            g = 1.0 / (z - alpha[-1])
            # backward from last to first
            for j in range(len(alpha)-2, -1, -1):
                g = 1.0 / (z - alpha[j] - beta[j] * gamma[j] * g)
            return g

        # compute G(omega) for this reference state
        for i, wv in enumerate(omega):
            z = wv + 1j * eta
            G_all[idx, i] = cf(z)

    if kind == "advanced":
        G_all = G_all.conj()
    elif kind != "retarded":
        raise ValueError("kind must be retarded or advanced in bi-Lanczos.")

    # Shape collapse like single-chain
    if len(mb) == 1 and len(omega) == 1:
        return G_all[0, 0]
    if len(mb) == 1:
        return G_all[0, :]
    if len(omega) == 1:
        return G_all[:, 0]
    return G_all

def greens_function_lanczos(
        omega               : Union[float, Array],
        lanczos_eigenvalues : Array,                        # after diagonalization of tridiagonal matrix
        lanczos_vectors     : Array,                        # Lanczos vectors |v_0>, H|v_0>, ...
        lanczos_eigenvector : Array,                        # matrix to transform to tridiagonal basis
        eta                 : float = 0.01,
        *,
        mb_states           : Optional[Union[Array, int]]   = None,         # vector of indices is sufficient, as we have the matrix elements via operators
        hamiltonian         : Optional[Array]               = None,         # in original basis - needed to construct new lanczos vectors, different operators
        operator_a          : Optional[Array]               = None,         # in original basis
        operator_b          : Optional[Array]               = None,         # in original basis
        backend             : str                           = "default",    # numerical backend
        kind                : str                           = "retarded") -> Array:
        
    r"""
    Zero-temperature many-body Green function using a Lanczos-truncated eigenbasis.

    This uses the Lehmann representation restricted to the subspace spanned by
    the Lanczos vectors. We assume:

        - lanczos_eigenvalues: eigenvalues of the tridiagonal matrix T (size M)
        - lanczos_vectors    : columns are Lanczos basis vectors |v_j> in original basis
        - lanczos_eigenvector: unitary which diagonalizes T:
                                T = U_L diag(lanczos_eigenvalues) U_L^\dagger

    The approximate eigenvectors of H in the original basis are:

        Psi = lanczos_vectors @ lanczos_eigenvector

    In this truncated eigenbasis {|psi_n>}, the retarded Green function is

        G_AB^R(omega; m) =
            sum_n <m|A|n><n|B|m> / (omega + i*eta - (E_n - E_m))
          + sum_n <m|B|n><n|A|m> / (omega + i*eta + (E_n - E_m)),

    where E_n are lanczos_eigenvalues and m runs over mb_states.

    Parameters
    ----------
    omega : float or array
        Frequency or frequency grid.
    lanczos_eigenvalues : array, shape (M,)
        Eigenvalues of the tridiagonal matrix T from Lanczos.
    lanczos_vectors : array, shape (N_full, M)
        Lanczos basis vectors in the original Hilbert space.
    lanczos_eigenvector : array, shape (M, M)
        Matrix that diagonalizes T.
    eta : float
        Broadening parameter.
    mb_states : int or array of int, optional
        Indices m labeling reference eigenstates |psi_m> in the truncated basis.
        If None, uses [0] (ground state).
    hamiltonian : array, optional
        H in original basis (not used in this implementation but kept for API).
    operator_a : array
        Operator A in the original basis (matrix elements <i|A|j>).
    operator_b : array or None
        Operator B in the original basis.
        If None, B is taken as A^\dagger (conjugate transpose).
    backend : str
        Backend selector for get_backend.
    kind : {"retarded", "advanced"}
        Type of Green function. "retarded" by default.

    Returns
    -------
    G : array
        G(m, omega) in general. Reduced to 1D or scalar if inputs are scalar.
    """
    be = get_backend(backend)
    
    # scalar omega -> array
    if isinstance(omega, (int, float, complex)):
        omega_arr = be.asarray([omega], dtype=be.complex128)
    else:
        omega_arr = be.asarray(omega, dtype=be.complex128)

    # if operator_b is None => fast Hermitian Lanczos path
    if operator_b is None:
        return _greens_lanczos_single_chain(
            omega_arr,
            lanczos_eigenvalues,
            lanczos_eigenvector,
            eta=eta,
            mb_states=mb_states,
            backend=backend,
            kind=kind,
        )

    # else build bi-Lanczos
    if hamiltonian is None or operator_a is None:
        raise ValueError("For general A,B Green function provide hamiltonian, operator_a, operator_b.")
    
    return _greens_lanczos_bilanczos(
        omega_arr,
        hamiltonian,
        operator_a,
        operator_b,
        eta=eta,
        mb_states=mb_states,
        backend=backend,
        kind=kind,
    )

def greens_function_lanczos_finite_T(
        omega               : Union[float, Array],
        hamiltonian         : Array,
        operator_a          : Array,
        eta                 : float             = 0.01,
        *,
        beta                : float             = 1.0,
        operator_b          : Optional[Array]   = None,
        n_random            : int               = 50,
        max_krylov          : int               = 100,
        backend             : str               = "default",
        kind                : str               = "retarded",
        lehmann_full        : bool              = False,
        seed                : Optional[int]     = None) -> Array:
    r"""
    Finite-Temperature Lanczos Method (FTLM) for many-body Green's functions.

    Method of Jaklic and Prelovsek, Phys. Rev. B 49, 5065 (1994).

    Instead of exact diagonalization, this method approximates thermal traces
    using stochastic sampling with random states:

        Tr[e^{-\beta H} O] ~ (N_Hilbert / R) \sum_{r=1}^R <r|e^{-\beta H} O|r>

    For each random state |r>, a Lanczos run gives approximate eigenpairs
    {\epsilon_i^{(r)}, |\phi_i^{(r)}>} within a small Krylov subspace.

    The Green's function is then:

        G_{AB}(\omega) ~ (N_Hilbert / ZR) \sum_{r,i,j} e^{-\beta \epsilon_i^{(r)}}
                         <r|\phi_i^{(r)}><\phi_i^{(r)}|A^\dagger|\phi_j^{(r)}>
                         <\phi_j^{(r)}|B|r> / (\omega + i\eta - (\epsilon_j - \epsilon_i))

    This captures finite-T behavior with:
    - R ~ 50 random states
    - M ~ 100 Krylov dimension
    Much cheaper than full ED for large Hilbert spaces.

    Parameters
    ----------
    omega : float or array
        Frequency grid for Green's function.
    hamiltonian : array, shape (N_Hilbert, N_Hilbert)
        Full Hamiltonian matrix in the many-body basis.
    operator_a : array, shape (N_Hilbert, N_Hilbert)
        Operator A in the many-body basis.
    eta : float, optional
        Broadening parameter (default: 0.01).
    beta : float, optional
        Inverse temperature \beta = 1/T (default: 1.0).
    operator_b : array, optional
        Operator B. If None, uses A^\dagger (default: None).
    n_random : int, optional
        Number of random states R for stochastic trace (default: 50).
    max_krylov : int, optional
        Maximum Krylov dimension M for each Lanczos run (default: 100).
    backend : str, optional
        Numerical backend (default: "default").
    kind : str, optional
        Type of Green's function: "retarded" or "advanced" (default: "retarded").
    lehmann_full : bool, optional
        If True (default), include both +/- frequency terms (full Lehmann).
        If False, only positive frequency (single-pole, for spectral functions).
    seed : int, optional
        Random seed for reproducibility (default: None).

    Returns
    -------
    Array
        Finite-temperature Green's function G(\omega).
        Shape: (len(omega),) if omega is array, scalar if omega is scalar.

    Notes
    -----
    FTLM advantages:
    - Scales to much larger systems than full ED
    - Statistical error decreases as 1/sqrt(R)
    - Each Lanczos run is independent (parallelizable)

    **Note**: Current implementation is functional but may need tuning for optimal
    accuracy. For production use, test convergence with your specific system.
    Recommended: n_random >= 50, max_krylov >= 100.

    For very large systems (>10^6 states), consider TPQ (thermal pure quantum)
    method which uses a single |psi_T> = e^{-\beta H/2}|r> instead.

    References
    ----------
    Jaklic & Prelovsek, Phys. Rev. B 49, 5065 (1994)
    Prelovsek & Bonca, "Ground State and Finite Temperature Lanczos Methods"

    Examples
    --------
    >>> # FTLM at finite temperature
    >>> G_ftlm = greens_function_lanczos_finite_T(
    ...     omega=omega_grid, hamiltonian=H, operator_a=A_op,
    ...     beta=10.0, n_random=50, max_krylov=100, eta=0.05
    ... )
    """
    be = get_backend(backend)
    
    # Setup
    H = be.asarray(hamiltonian, dtype=be.complex128)
    A = be.asarray(operator_a, dtype=be.complex128)
    
    if operator_b is None:
        B = A.conj().T
    else:
        B = be.asarray(operator_b, dtype=be.complex128)
    
    N_Hilbert = H.shape[0]
    
    # Omega array
    if isinstance(omega, (int, float, complex)):
        w = be.asarray([omega], dtype=be.complex128)
        scalar_omega = True
    else:
        w = be.asarray(omega, dtype=be.complex128)
        scalar_omega = False
    
    # Initialize result
    G_total = be.zeros((len(w),), dtype=be.complex128)
    Z_total = 0.0  # Partition function accumulator
    
    # Random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Loop over random states
    for r_idx in range(n_random):
        # Generate random state |r>
        if be.__name__ == 'numpy':
            r_vec = np.random.randn(N_Hilbert) + 1j * np.random.randn(N_Hilbert)
            r_vec = r_vec / np.linalg.norm(r_vec)
        else:
            # For JAX, use numpy then convert
            r_vec = np.random.randn(N_Hilbert) + 1j * np.random.randn(N_Hilbert)
            r_vec = r_vec / np.linalg.norm(r_vec)
            r_vec = be.asarray(r_vec, dtype=be.complex128)
        
        # Lanczos tridiagonalization starting from |r>
        alpha_list = []
        beta_list = []
        V_lanczos = []  # Lanczos vectors
        
        v_prev = be.zeros(N_Hilbert, dtype=be.complex128)
        v = r_vec
        V_lanczos.append(v)
        
        for j in range(max_krylov):
            Hv = H @ v
            a = be.vdot(v, Hv).real  # Should be real for Hermitian H
            alpha_list.append(a)
            
            # Residual
            if j == 0:
                r_res = Hv - a * v
            else:
                r_res = Hv - a * v - beta_list[-1] * v_prev
            
            b = be.sqrt(be.vdot(r_res, r_res).real)
            
            if b < 1e-14:
                break
            
            beta_list.append(b)
            v_prev = v
            v = r_res / b
            V_lanczos.append(v)
        
        M = len(alpha_list)
        
        # Build tridiagonal matrix
        T = be.zeros((M, M), dtype=be.complex128)
        for i in range(M):
            T[i, i] = alpha_list[i]
            if i < M - 1:
                T[i, i+1] = beta_list[i]
                T[i+1, i] = beta_list[i]
        
        # Diagonalize T
        if be.__name__ == 'numpy':
            eps, U_T = np.linalg.eigh(np.asarray(T))
        else:
            # For JAX
            import jax.numpy as jnp
            eps, U_T = jnp.linalg.eigh(T)
        
        eps = be.asarray(eps, dtype=be.float64)
        U_T = be.asarray(U_T, dtype=be.complex128)
        
        # Compute Boltzmann weights
        # Shift for numerical stability
        eps_min = be.min(eps)
        exp_factors = be.exp(-beta * (eps - eps_min))
        Z_r = be.sum(exp_factors)
        Z_total += Z_r
        
        # Matrix elements in Lanczos basis
        # <r|\phi_i> = U_T[0, i] (first component, since |v_0> = |r>)
        overlap_r = U_T[0, :]  # <r|\phi_i>
        
        # Transform operators to Lanczos basis
        V_matrix = be.stack(V_lanczos[:M], axis=1)  # (N_Hilbert, M)
        
        A_lanczos = V_matrix.conj().T @ A @ V_matrix  # (M, M)
        B_lanczos = V_matrix.conj().T @ B @ V_matrix  # (M, M)
        
        # Transform to eigenbasis of T
        A_eig = U_T.conj().T @ A_lanczos @ U_T
        B_eig = U_T.conj().T @ B_lanczos @ U_T
        
        # Compute contribution to Green's function
        for i in range(M):
            weight_i = exp_factors[i] * be.abs(overlap_r[i])**2
            
            if weight_i < 1e-15:
                continue
            
            for j in range(M):
                # Overlap factors: <r|\phi_i> and <\phi_j|r>
                overlap_factor = overlap_r[i].conj() * overlap_r[j]
                
                # Matrix element <\phi_i|A|\phi_j>
                A_ij = A_eig[i, j]
                B_ji = B_eig[j, i]
                
                deltaE = eps[j] - eps[i]
                denom_pos = w + 1j * eta - deltaE
                
                # Positive frequency contribution
                G_total += weight_i * overlap_factor * A_ij * B_ji / denom_pos
                
                if lehmann_full:
                    # Negative frequency contribution (for full Lehmann)
                    denom_neg = w + 1j * eta + deltaE
                    G_total -= weight_i * overlap_factor * B_ji * A_ij / denom_neg
    
    # Normalize by partition function and number of random states
    G = (N_Hilbert / (n_random * Z_total)) * G_total
    
    # Handle kind
    if kind == "advanced":
        G = G.conj()
    elif kind != "retarded":
        raise ValueError(f"Unsupported kind '{kind}'. Use 'retarded' or 'advanced'.")
    
    # Return
    if scalar_omega:
        return G[0]
    return G


# =============================================================================
# Spectral Function Calculations
# =============================================================================

def spectral_function(  greens_function     : Optional[Array]   = None,
                        backend             : str               = "default",
                        eta                 : float             = 0.01,
                        *, 
                        kind                : Literal["retarded", "advanced", "greater", "lesser", "timeordered"] = "retarded",
                        omega               : Optional[Union[float, Array]] = None,
                        # for quadratic, if greens_function is None
                        q_eigenvalues       : Optional[Array]   = None,
                        q_eigenvectors      : Optional[Array]   = None,
                        q_operator_a        : Optional[Array]   = None,
                        q_operator_b        : Optional[Array]   = None,
                        q_occupations       : Optional[Array]   = None,
                        q_basis_transform   : bool              = True,
                        # for many-body, if greens_function is None
                        mb_eigenvalues      : Optional[Array]   = None,
                        mb_operator_a       : Optional[Array]   = None,
                        mb_operator_b       : Optional[Array]   = None,
                        mb_states           : Optional[Union[Array, int]] = None
                       ) -> Array:
    r"""
    Compute spectral function.
    
    If greens_function is provided:
        A(\omega) = -(1/\pi) Im[G(\omega)]
    
    Otherwise, compute Green's function first from eigenvalues/eigenvectors,
    then extract spectral function.
    
    Parameters
    ----------
    greens_function : array-like, dtype complex, optional
        Pre-computed Green's function G(\omega). If provided, spectral function
        is computed directly.
    backend : str, optional
        Numerical backend (default: 'default').
    eta : float, optional
        Broadening parameter (default: 0.01).
    kind : str, optional
        Type of Green's function (default: 'retarded').
    omega : float or array-like, optional
        Frequency or frequencies (required if greens_function is None).
    q_eigenvalues : array-like, optional
        Eigenvalues for quadratic system.
    q_eigenvectors : array-like, optional
        Eigenvectors for quadratic system.
    q_operator_a : array-like, optional
        Operator A for quadratic Green's function.
    q_operator_b : array-like, optional
        Operator B for quadratic Green's function.
    q_occupations : array-like, optional
        Occupation numbers for quadratic system.
    q_basis_transform : bool, optional
        Whether to transform to eigenbasis (default: True).
    mb_eigenvalues : array-like, optional
        Eigenvalues for many-body system.
    mb_operator_a : array-like, optional
        Operator A for many-body Green's function.
    mb_operator_b : array-like, optional
        Operator B for many-body Green's function.
    mb_states : array-like or int, optional
        Many-body states for Lehmann representation.
        
    Returns
    -------
    Array
        Spectral function A(\omega), same shape as input, dtype real.
        Always non-negative for physical systems.
        
    Notes
    -----
    The spectral function is computed as A(\omega) = -(1/\pi) Im[G(\omega)].
    
    Examples
    --------
    >>> # From pre-computed Green's function
    >>> A = spectral_function(greens_function=G)
    >>> 
    >>> # From quadratic system eigenvalues
    >>> A = spectral_function(omega=0.5, q_eigenvalues=E, q_eigenvectors=U, eta=0.01)
    """
    be  = get_backend(backend)
    
    # Case 1: Green's function provided directly
    if greens_function is not None:
        G = be.asarray(greens_function, dtype=be.complex128)
    
    # Case 2: Compute Green's function from eigenvalues
    else:
        if omega is None:
            raise ValueError("omega must be provided when greens_function is None.")
        
        # Determine which system type based on provided parameters
        # Priority: many-body if mb_states provided, else quadratic
        if mb_eigenvalues is not None and mb_states is not None:
            # Many-body system
            G = greens_function_manybody(
                omega               = omega,
                eigenvalues         = mb_eigenvalues,
                operator_a          = mb_operator_a,
                eta                 = eta,
                mb_states           = mb_states,
                operator_b          = mb_operator_b,
                backend             = backend,
                kind                = kind
            )
        elif q_eigenvalues is not None:
            # Quadratic system
            G = greens_function_quadratic(
                omega               = omega,
                eigenvalues         = q_eigenvalues,
                eigenvectors        = q_eigenvectors,
                eta                 = eta,
                operator_a          = q_operator_a,
                operator_b          = q_operator_b,
                occupations         = q_occupations,
                basis_transform     = q_basis_transform,
                backend             = backend
            )
        else:
            raise ValueError("Must provide either greens_function or eigenvalues (q_eigenvalues or mb_eigenvalues).")
    
    # Compute spectral function: A(omega) = -(1/pi) Im[G(omega)]
    A = -be.imag(G) / np.pi
    
    return A


# =============================================================================
# Momentum-Resolved Spectral Function
# =============================================================================

def spectral_function_k_resolved(
        omegas          : Array,
        k_points        : Array,
        eigenvalues_k   : Array,
        eta             : float = 0.01,
        backend         : str = "default") -> Array:
    r"""
    Compute momentum-resolved spectral function A(k, omega).
    
    For systems with momentum as a good quantum number.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Frequencies.
    k_points : array-like, shape (n_k, d) or (n_k,)
        k-points in Brillouin zone.
    eigenvalues_k : array-like, shape (n_k,) or (n_k, n_bands)
        Energy dispersion E(k). If 2D, includes multiple bands.
    eta : float, optional
        Broadening parameter (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Shape (n_k, n_omega) if eigenvalues_k is 1D,
        Shape (n_k, n_bands, n_omega) if eigenvalues_k is 2D.
        Dtype: real
        
    Examples
    --------
    >>> # 1D tight-binding dispersion
    >>> k_points = np.linspace(-np.pi, np.pi, 100)
    >>> E_k = -2 * np.cos(k_points)  # single band
    >>> omegas = np.linspace(-3, 3, 200)
    >>> A_k_omega = spectral_function_k_resolved(omegas, k_points, E_k)
    >>> # Shape: (100, 200) - band structure mapped to spectral function
    >>> plt.imshow(A_k_omega, aspect='auto')
    """
    be              = get_backend(backend)
    omegas          = be.asarray(omegas)
    eigenvalues_k   = be.asarray(eigenvalues_k)
    
    n_omega         = len(omegas)
    
    if eigenvalues_k.ndim == 1:
        # Single band: shape (n_k,)
        n_k     = len(eigenvalues_k)
        A       = be.zeros((n_k, n_omega), dtype=be.float64)
        
        for k_idx in range(n_k):
            E_k     = eigenvalues_k[k_idx]
            A_k     = (eta / np.pi) / ((omegas - E_k)**2 + eta**2)
            if JAX_AVAILABLE and backend == "jax":
                A   = A.at[k_idx, :].set(A_k)
            else:
                A[k_idx, :] = A_k
    
    elif eigenvalues_k.ndim == 2:
        # Multiple bands: shape (n_k, n_bands)
        n_k, n_bands    = eigenvalues_k.shape
        A               = be.zeros((n_k, n_bands, n_omega), dtype=be.float64)
        
        for k_idx in range(n_k):
            for band in range(n_bands):
                E_k = eigenvalues_k[k_idx, band]
                A_k = (eta / np.pi) / ((omegas - E_k)**2 + eta**2)
                if JAX_AVAILABLE and backend == "jax":
                    A = A.at[k_idx, band, :].set(A_k)
                else:
                    A[k_idx, band, :] = A_k
    else:
        raise ValueError("eigenvalues_k must be 1D or 2D")
    
    return A

# =============================================================================
# Integrated Spectral Weight
# =============================================================================

def integrated_spectral_weight(
        spectral_function   : Array,
        omega_grid          : Array,
        omega_min           : Optional[float] = None,
        omega_max           : Optional[float] = None,
        backend             : str = "default") -> Union[float, Array]:
    r"""
    Compute integrated spectral weight over an energy window.
    
    W = int_{omega_min}^{omega_max} domega A(omega)
    
    Parameters
    ----------
    spectral_function : array-like
        Spectral function A(omega), shape (..., n_omega).
    omega_grid : array-like, shape (n_omega,)
        Frequency grid (should be uniformly or quasi-uniformly spaced).
    omega_min, omega_max : float, optional
        Integration limits. If None, integrates over full range.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    float or Array
        Integrated spectral weight. Returns scalar if input is 1D,
        otherwise returns array integrated along last axis.
        
    Notes
    -----
    Uses trapezoidal rule for integration.
    For a complete spectrum, integrated weight should be close to N
    (number of states).
    """
    be = get_backend(backend)
    spectral_function = be.asarray(spectral_function, dtype=be.float64)
    omega_grid = be.asarray(omega_grid, dtype=be.float64)
    
    # Determine integration window
    if omega_min is None:
        omega_min = be.asarray(omega_grid[0])
    else:
        omega_min = be.asarray(omega_min, dtype=be.float64)
    
    if omega_max is None:
        omega_max = be.asarray(omega_grid[-1])
    else:
        omega_max = be.asarray(omega_max, dtype=be.float64)
    
    # Find indices for integration window
    mask = (omega_grid >= omega_min) & (omega_grid <= omega_max)
    omega_window = omega_grid[mask]
    
    # Extract spectral function in window
    if spectral_function.ndim == 1:
        A_window = spectral_function[mask]
    elif spectral_function.ndim == 2:
        A_window = spectral_function[:, mask]
    else:
        A_window = spectral_function[..., mask]
    
    # Integrate using trapezoidal rule
    # Note: np.trapz works with JAX arrays too
    return np.trapz(A_window, omega_window, axis=-1)

# =============================================================================
# Peak Detection and Analysis
# =============================================================================

def find_spectral_peaks(
        spectral_function: Array,
        omega_grid: Array,
        threshold: float = 0.1,
        min_distance: int = 5,
        backend: str = "default"
) -> Array:
    r"""
    Find peaks in spectral function.
    
    Parameters
    ----------
    spectral_function : array-like, shape (n_omega,)
        1D spectral function.
    omega_grid : array-like, shape (n_omega,)
        Frequency grid corresponding to spectral_function.
    threshold : float, optional
        Minimum relative height for peak detection (default: 0.1).
        Peaks below threshold * max(A) are ignored.
    min_distance : int, optional
        Minimum distance between peaks in grid points (default: 5).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array
        Frequencies omega where peaks are detected.
        
    Notes
    -----
    Uses simple local maximum detection. For more sophisticated peak finding,
    consider scipy.signal.find_peaks.
    
    Examples
    --------
    >>> omegas = np.linspace(-5, 5, 200)
    >>> evals = np.array([-2, -1, 1, 2])
    >>> A = spectral_function_multi_omega(omegas, evals, diagonal_only=True).sum(axis=1)
    >>> peaks = find_spectral_peaks(A, omegas, threshold=0.2)
    """
    be = get_backend(backend)
    spectral_function = be.asarray(spectral_function, dtype=be.float64)
    omega_grid = be.asarray(omega_grid, dtype=be.float64)
    
    # Find local maxima
    peaks = []
    A_max = be.max(spectral_function)
    threshold_val = threshold * A_max
    
    for i in range(min_distance, len(spectral_function) - min_distance):
        if (spectral_function[i] > threshold_val and
            be.all(spectral_function[i] >= spectral_function[i - min_distance:i]) and
            be.all(spectral_function[i] >= spectral_function[i + 1:i + min_distance + 1])):
            peaks.append(omega_grid[i])
    
    return be.asarray(peaks) if peaks else be.asarray([])

# =============================================================================
# Thermal Utilities (Shared with Susceptibility Module)
# =============================================================================

def thermal_weights(
        eigenvalues : Array,
        temperature : float = 0.0,
        backend     : str = "default"
) -> Array:
    r"""
    Compute thermal occupation weights (Boltzmann/Fermi/Bose factors).
    
    For temperature T > 0:
        ρ_n = exp(-β(E_n - E_0)) / Z, where β = 1/k_B T
    
    For temperature T = 0:
        ρ_0 = 1, all others = 0 (ground state only)
    
    Parameters
    ----------
    eigenvalues : array-like, shape (N,)
        Eigenvalues E_n. Must be real.
    temperature : float, optional
        Temperature in energy units (default: 0). 
        If 0, returns ground-state-only weights.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array, shape (N,), dtype float
        Normalized thermal weights (sum to 1).
        
    Examples
    --------
    >>> E = np.array([-2, -1, 0, 1, 2])
    >>> rho_T0 = thermal_weights(E, temperature=0.0)     # [1, 0, 0, 0, 0]
    >>> rho_T1 = thermal_weights(E, temperature=1.0)     # Boltzmann distribution
    """
    be = get_backend(backend)
    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    
    if temperature <= 0:
        # T=0: ground state only
        rho = be.zeros(len(eigenvalues), dtype=be.float64)
        rho_list = rho.tolist() if hasattr(rho, 'tolist') else list(rho)
        rho_list[0] = 1.0
        return be.asarray(rho_list, dtype=be.float64)
    else:
        # Finite temperature
        beta = 1.0 / temperature
        E_min = be.min(eigenvalues)
        rho = be.exp(-beta * (eigenvalues - E_min))
        Z = be.sum(rho)
        return rho / Z


# =============================================================================
# Operator-Projected Spectral Functions (Many-body)
# =============================================================================

def operator_spectral_function_lehmann(
        omega               : float,
        eigenvalues         : Array,
        eigenvectors        : Array,
        operator            : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0,
        backend             : str = "default"
) -> float:
    r"""
    Compute operator-projected spectral function using Lehmann representation.
    
    A_O(omega ) = Σ_{m,n} (ρ_m - ρ_n) |<m|O|n>|² delta (omega  - (E_n - E_m))
    
    This represents the contribution of operator O to the spectrum:
    - Used for spin/charge/current response in many-body systems
    - Naturally accounts for finite temperature via ρ_n
    - Gives matrix elements <m|O|n> between ALL many-body eigenstates
    
    Parameters
    ----------
    omega : float
        Frequency omega .
    eigenvalues : array-like, shape (N,)
        Hamiltonian eigenvalues E_n.
    eigenvectors : array-like, shape (N, N)
        Hamiltonian eigenvectors (columns = eigenstates).
    operator : array-like, shape (N, N)
        Many-body operator O (e.g., magnetization, density).
    eta : float, optional
        Broadening parameter (default: 0.01). Controls peak width.
    temperature : float, optional
        Temperature in energy units (default: 0).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    float
        Operator-projected spectral function A_O(omega ).
        
    Notes
    -----
    At T=0, only ground-state contributions (|<n|O|0>|²) appear.
    At T>0, thermal factors ρ_m ≠ ρ_n activate all transitions.
    
    This is the foundation for computing susceptibilities:
    χ_OO(omega ) can be reconstructed from this for all operators.
    
    See Also
    --------
    operator_spectral_function_multi_omega : Vectorized over frequencies
    thermal_weights : Thermal occupation factors
    
    Examples
    --------
    >>> E, V = np.linalg.eigh(H)
    >>> S_z = compute_spin_operator()  # Your magnetization
    >>> A_sz = operator_spectral_function_lehmann(omega=0.5, 
    ...                                             eigenvalues=E, 
    ...                                             eigenvectors=V,
    ...                                             operator=S_z,
    ...                                             temperature=1.0)
    """
    be = get_backend(backend)
    
    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    eigenvectors = be.asarray(eigenvectors, dtype=be.complex128)
    operator = be.asarray(operator, dtype=be.complex128)
    
    # Ensure operator is 2D and properly shaped
    if operator.ndim == 0:
        # Scalar
        operator = be.eye(len(eigenvalues), dtype=be.complex128) * operator
    elif operator.ndim == 1:
        # 1D array - make it diagonal
        operator = be.diag(operator)
    
    N = len(eigenvalues)
    
    # Transform operator to eigenbasis: O_nm = <n|O|m>
    O_eigen = eigenvectors.conj().T @ operator @ eigenvectors
    
    # Thermal weights
    rho = thermal_weights(eigenvalues, temperature, backend)
    
    # Lorentzian broadening kernel
    def lorentzian(delta_E):
        return (eta / np.pi) / (delta_E**2 + eta**2)
    
    # Lehmann sum over all transitions
    A = 0.0
    for m in range(N):
        for n in range(N):
            if be.abs(rho[m] - rho[n]) < 1e-14:
                continue
            delta_E = omega - (eigenvalues[n] - eigenvalues[m])
            matrix_element_sq = be.abs(O_eigen[m, n])**2
            A += (rho[m] - rho[n]) * matrix_element_sq * lorentzian(delta_E)
    
    return float(be.real(A))


def operator_spectral_function_multi_omega(
        omegas              : Array,
        eigenvalues         : Array,
        eigenvectors        : Array,
        operator            : Array,
        eta                 : float = 0.01,
        temperature         : float = 0.0,
        backend             : str = "default"
) -> Array:
    r"""
    Compute operator-projected spectral function for multiple frequencies.
    
    Vectorized version of operator_spectral_function_lehmann.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Frequency grid.
    eigenvalues : array-like, shape (N,)
        Eigenvalues.
    eigenvectors : array-like, shape (N, N)
        Eigenvectors.
    operator : array-like, shape (N, N)
        Operator O.
    eta : float, optional
        Broadening (default: 0.01).
    temperature : float, optional
        Temperature (default: 0).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array, shape (n_omega,), dtype float
        Spectral function A_O(omega ) for each omega .
        
    Examples
    --------
    >>> omegas = np.linspace(-5, 5, 200)
    >>> A_sz = operator_spectral_function_multi_omega(omegas, E, V, S_z, temperature=1.0)
    >>> plt.plot(omegas, A_sz)
    """
    be = get_backend(backend)
    omegas = be.asarray(omegas)
    
    n_omega = len(omegas)
    A = be.zeros(n_omega, dtype=be.float64)
    
    for i, omega in enumerate(omegas):
        # Convert omega to Python float (avoid numpy scalar issues)
        omega_val = float(omega) if hasattr(omega, '__float__') else omega
        A_i = operator_spectral_function_lehmann(
            omega_val, eigenvalues, eigenvectors, operator,
            eta=eta, temperature=temperature, backend=backend
        )
        if JAX_AVAILABLE and backend == "jax":
            A = A.at[i].set(A_i)
        else:
            A[i] = A_i
    
    return A


# =============================================================================
# Bubble Susceptibility (Quadratic / Mean-Field)
# =============================================================================

def susceptibility_bubble(
        omega               : float,
        eigenvalues         : Array,
        vertex              : Optional[Array] = None,
        occupation          : Optional[Array] = None,
        eta                 : float = 0.01,
        backend             : str = "default"
) -> complex:
    r"""
    Compute bare susceptibility (Lindhard function) from single-particle spectrum.
    
    χ⁰(omega ) = Σ_{m,n} (f_m - f_n) |V_{mn}|² / (omega  + i\eta - (E_n - E_m))
    
    This is the bubble diagram contribution, valid for quadratic or mean-field
    Hamiltonians where Wick's theorem holds.
    
    Parameters
    ----------
    omega : float or complex
        Frequency omega .
    eigenvalues : array-like, shape (N,)
        Single-particle energies E_n (not many-body eigenvalues!).
    vertex : array-like, shape (N, N), optional
        Vertex/coupling matrix V_{mn} (e.g., velocity for conductivity).
        If None, assumes identity (density-density response).
    occupation : array-like, shape (N,), optional
        Single-particle occupations f_n (Fermi/Bose factors).
        If None, assumes ground state (f_n = Θ(-E_n) for fermions at T=0).
    eta : float, optional
        Broadening (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    complex
        Bare susceptibility χ⁰(omega ) at this frequency.
        
    Notes
    -----
    The occupation f_n should be computed from the single-particle eigenenergies,
    NOT from the many-body Hamiltonian. Typical choices:
    
    - Fermions at T=0: f_n = Θ(-E_n) [filled orbitals below Fermi]
    - Fermions at T>0: f_n = 1/(1 + exp(β(E_n - μ)))
    - Bosons: f_n = 1/(exp(β E_n) - 1)
    
    For density-density response, vertex = I (identity).
    For charge/current response, vertex encodes the operator O_mn = <m|O|n>.
    
    See Also
    --------
    conductivity_kubo_bubble : Optical conductivity \sigma(omega ) from bubbles
    
    Examples
    --------
    >>> # Tight-binding chain: E_k = -2cos(k)
    >>> E_k = np.array([-2, -1.8, -1.5, 0, 1.5, 1.8, 2])
    >>> f = 1/(1 + np.exp(E_k/(0.1)))  # Fermi-Dirac at T=0.1
    >>> chi0 = susceptibility_bubble(omega=0.5, eigenvalues=E_k, occupation=f)
    """
    be = get_backend(backend)
    
    eigenvalues = be.asarray(eigenvalues, dtype=be.float64)
    omega_complex = be.asarray(omega, dtype=be.complex128)
    eta_complex = be.asarray(eta, dtype=be.complex128)
    
    N = len(eigenvalues)
    
    if vertex is None:
        vertex = be.eye(N, dtype=be.complex128)
    else:
        vertex = be.asarray(vertex, dtype=be.complex128)
    
    if occupation is None:
        # T=0: all states below Fermi filled (filled Fermi sea)
        occupation = be.where(eigenvalues < 0, 1.0, 0.0)
    else:
        occupation = be.asarray(occupation, dtype=be.float64)
    
    # Bubble: χ⁰ = Σ_{mn} (f_m - f_n) |V_{mn}|² / (omega  + i\eta - (E_n - E_m))
    chi = 0.0 + 0.0j
    for m in range(N):
        for n in range(N):
            if be.abs(occupation[m] - occupation[n]) < 1e-14:
                continue
            denom = omega_complex + 1j * eta_complex - (eigenvalues[n] - eigenvalues[m])
            V_mn_sq = be.abs(vertex[m, n])**2
            chi += (occupation[m] - occupation[n]) * V_mn_sq / denom
    
    return complex(chi)


def susceptibility_bubble_multi_omega(
        omegas              : Array,
        eigenvalues         : Array,
        vertex              : Optional[Array] = None,
        occupation          : Optional[Array] = None,
        eta                 : float = 0.01,
        backend             : str = "default"
) -> Array:
    r"""
    Compute bare susceptibility for multiple frequencies.
    
    Vectorized version of susceptibility_bubble.
    
    Parameters
    ----------
    omegas : array-like, shape (n_omega,)
        Frequency grid.
    eigenvalues : array-like, shape (N,)
        Single-particle energies.
    vertex : array-like, shape (N, N), optional
        Vertex matrix. If None, uses identity.
    occupation : array-like, shape (N,), optional
        Occupations. If None, assumes T=0 filled Fermi sea.
    eta : float, optional
        Broadening (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array, shape (n_omega,), dtype complex
        χ⁰(omega ) for each omega .
    """
    be = get_backend(backend)
    omegas = be.asarray(omegas)
    
    n_omega = len(omegas)
    chi = be.zeros(n_omega, dtype=be.complex128)
    
    for i, omega in enumerate(omegas):
        chi_i = susceptibility_bubble(
            float(omega), eigenvalues, vertex, occupation,
            eta=eta, backend=backend
        )
        if JAX_AVAILABLE and backend == "jax":
            chi = chi.at[i].set(chi_i)
        else:
            chi[i] = chi_i
    
    return chi


# =============================================================================
# Kubo Conductivity (Quadratic Systems)
# =============================================================================

def conductivity_kubo_bubble(
        omega               : float,
        eigenvalues         : Array,
        velocity_matrix     : Array,
        occupation          : Optional[Array] = None,
        eta                 : float = 0.01,
        backend             : str = "default"
) -> complex:
    r"""
    Compute optical conductivity from Kubo-Greenwood formula (bubble diagram).
    
    \sigma(omega ) = (1/(2omega )) Σ_{mn} (f_m - f_n) |<m|v|n>|² / (omega  + i\eta - (E_n - E_m))
    
    This is the single-particle bubble contribution, valid for non-interacting
    or mean-field Hamiltonians.
    
    Parameters
    ----------
    omega : float or complex
        Frequency omega  (should be real for physical conductivity).
    eigenvalues : array-like, shape (N,)
        Single-particle eigenenergies.
    velocity_matrix : array-like, shape (N, N)
        Velocity/momentum matrix elements v_mn = <m|v|n>.
        Typically: v_mn = ∂H/∂k in k-space, or related to hopping in real space.
    occupation : array-like, shape (N,), optional
        Single-particle occupations f_n.
        If None, assumes ground state (T=0).
    eta : float, optional
        Broadening/scattering rate (default: 0.01).
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    complex
        Conductivity \sigma(omega ).
        
    Notes
    -----
    The formula is:
        \sigma(omega ) = (1/(2omega )) χ⁰_{vv}(omega )
    
    Real part Re[\sigma(omega )] gives absorptive conductivity.
    Imaginary part Im[\sigma(omega )] gives reactive effects.
    
    For optical conductivity (intraband vs interband):
    - Intraband: f_m ≠ f_n within partially filled band
    - Interband: f_m ≠ f_n across band gap
    
    See Also
    --------
    susceptibility_bubble : General bubble susceptibility
    
    Examples
    --------
    >>> # Graphene-like: linear dispersion
    >>> E = np.array([-1, -0.5, 0, 0.5, 1])
    >>> v = np.array([[0, 1, 0, 0, 0],
    ...               [1, 0, 1, 0, 0],
    ...               [0, 1, 0, 1, 0],
    ...               [0, 0, 1, 0, 1],
    ...               [0, 0, 0, 1, 0]])
    >>> f = (E < 0).astype(float)  # filled below E=0
    >>> sigma = conductivity_kubo_bubble(omega=0.5, eigenvalues=E, 
    ...                                   velocity_matrix=v, occupation=f)
    """
    be = get_backend(backend)
    
    omega_complex = be.asarray(omega, dtype=be.complex128)
    
    # Compute χ⁰ from velocity vertex
    chi_vv = susceptibility_bubble(
        omega, eigenvalues, vertex=velocity_matrix,
        occupation=occupation, eta=eta, backend=backend
    )
    
    # \sigma(omega ) = (1/(2omega )) χ⁰_{vv}(omega )
    # Handle omega =0 carefully
    if be.abs(omega_complex) < 1e-14:
        return 0.0 + 0.0j
    
    sigma = chi_vv / (2.0 * omega_complex)
    return complex(sigma)


# =============================================================================
# Kramers-Kronig Relations
# =============================================================================

def kramers_kronig_transform(
        Im_chi           : Array,
        omega_grid       : Array,
        backend          : str = "default"
) -> Array:
    r"""
    Reconstruct Re[χ(omega )] from Im[χ(omega )] using Kramers-Kronig relations.
    
    Re[χ(omega )] = (2/π) P int_0^∞ domega ' (omega ' Im[χ(omega ')] / (omega '² - omega ²))
    
    where P denotes principal value.
    
    Parameters
    ----------
    Im_chi : array-like, shape (n_omega,)
        Imaginary part of susceptibility (must be real-valued).
    omega_grid : array-like, shape (n_omega,)
        Frequency grid. Should be dense and include positive/negative omega  symmetrically
        for best accuracy.
    backend : str, optional
        Numerical backend (default: 'default').
        
    Returns
    -------
    Array, shape (n_omega,)
        Real part Re[χ(omega )].
        
    Notes
    -----
    This uses a simple numerical Hilbert transform via integration.
    For high accuracy, use scipy.integrate.quad with analytic kernels.
    
    Assumes Im[χ(omega )] decays at large |omega |.
    
    See Also
    --------
    scipy.integrate.hilbert : More sophisticated Hilbert transform
    
    Examples
    --------
    >>> omegas = np.linspace(-10, 10, 500)
    >>> Im_chi = -0.1 / ((omegas - 1)**2 + 0.1**2)  # Lorentzian
    >>> Re_chi = kramers_kronig_transform(Im_chi, omegas)
    """
    from scipy import integrate
    
    be = get_backend(backend)
    Im_chi = np.asarray(Im_chi, dtype=np.float64)
    omega_grid = np.asarray(omega_grid, dtype=np.float64)
    
    n_omega = len(omega_grid)
    Re_chi = np.zeros(n_omega, dtype=np.float64)
    
    for i, omega in enumerate(omega_grid):
        # Avoid division by zero
        mask = np.abs(omega_grid - omega) > 1e-10
        omega_prime = omega_grid[mask]
        Im_chi_prime = Im_chi[mask]
        
        # Integrand: omega ' Im[χ(omega ')] / (omega '² - omega ²)
        integrand = omega_prime * Im_chi_prime / (omega_prime**2 - omega**2)
        
        # Integrate (could use more sophisticated method)
        integral = np.trapz(integrand, omega_prime)
        Re_chi[i] = (2.0 / np.pi) * integral
    
    return be.asarray(Re_chi, dtype=be.float64)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Green's functions
    'greens_function_diagonal',
    'greens_function_quadratic',
    'greens_function_quadratic_finite_T',
    'greens_function_manybody',
    'greens_function_lanczos',
    # Spectral functions
    'spectral_function',
    'spectral_function_k_resolved',
    # Integrated quantities
    'integrated_spectral_weight',
    # Analysis
    'find_spectral_peaks',
    # Thermal utilities
    'thermal_weights',
    # Operator-projected spectral (many-body)
    'operator_spectral_function_lehmann',
    'operator_spectral_function_multi_omega',
    # Bubble susceptibilities (quadratic)
    'susceptibility_bubble',
    'susceptibility_bubble_multi_omega',
    # Kubo conductivity
    'conductivity_kubo_bubble',
    # Kramers-Kronig
    'kramers_kronig_transform',
]

# =============================================================================
#! EOF
# =============================================================================