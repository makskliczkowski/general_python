r"""
Random matrix generators (GOE, GUE, COE/CRE, CUE) with robust fallbacks.

If TenPy is available, we use its implementations; otherwise we fall back to
well-known constructions:
- GOE: symmetric real Gaussian A = (X + X^T)/sqrt(2)
- GUE: Hermitian complex Gaussian H = (X + X^\dag)/2 with X Ginibre
- COE/CRE: Haar orthogonal via QR of real Ginibre with sign-fix on R
- CUE: Haar unitary via QR of complex Ginibre with phase-fix on R

Also exposes CUE_QR explicitly.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, Union
import numpy as np
import numpy.random as npr

# Optional JAX support
try:
    import jax
    import jax.numpy as jnp
    from jax import random as random_jp
    _JAX_AVAILABLE = True
    _JAX_DEFAULT_KEY = random_jp.PRNGKey(42)
except Exception:
    jnp = None  # type: ignore
    random_jp = None  # type: ignore
    _JAX_AVAILABLE = False
    _JAX_DEFAULT_KEY = None

# Optional TenPy support
try:
    from tenpy.linalg.random_matrix import COE as _T_COE, GUE as _T_GUE, GOE as _T_GOE, CRE as _T_CRE, CUE as _T_CUE
    _TENPY_AVAILABLE = True
except Exception:
    _TENPY_AVAILABLE = False


def _haar_unitary_qr_np(n: int, rng: Optional[npr.Generator] = None, simple: bool = True) -> np.ndarray:
    if rng is None:
        rng = npr.default_rng()
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    x /= np.sqrt(2)
    Q, R = np.linalg.qr(x)
    if not simple:
        d = np.diagonal(R)
        ph = d / np.abs(d)
        Q = Q @ np.diag(ph)
    return Q


def _haar_orthogonal_qr_np(n: int, rng: Optional[npr.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = npr.default_rng()
    x = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(x)
    # Fix signs to ensure uniform (Haar) distribution on O(n)
    d = np.sign(np.diag(R))
    d[d == 0.0] = 1.0
    Q = Q @ np.diag(d)
    return Q


def CUE_QR(n: int, simple: bool = True, rng: Optional[object] = None, backend: str = "numpy"):
    """Haar unitary via QR of complex Ginibre (NumPy or JAX)."""
    if backend in ("numpy", "np") or backend is np:
        return _haar_unitary_qr_np(n, rng=rng, simple=simple)
    if not _JAX_AVAILABLE:
        raise ImportError("JAX backend not available for CUE_QR")
    key = _JAX_DEFAULT_KEY if rng is None else rng
    key_re, key_im = random_jp.split(key)
    xr = random_jp.normal(key_re, shape=(n, n))
    xi = random_jp.normal(key_im, shape=(n, n))
    x = (xr + 1j * xi) / np.sqrt(2)
    Q, R = jnp.linalg.qr(x)
    if not simple:
        d = jnp.diagonal(R)
        ph = d / jnp.abs(d)
        Q = Q @ jnp.diag(ph)
    return Q


def goe(n: int, use_tenpy: bool = True) -> np.ndarray:
    """
    Generate a random matrix from the Gaussian Orthogonal Ensemble (GOE).

    If TenPy is available and `use_tenpy` is True, it uses `tenpy.linalg.random_matrix.GOE`.
    Otherwise, it constructs a symmetric real matrix: A = (X + X^T) / sqrt(2),
    where X has i.i.d. normal entries.

    Parameters
    ----------
    n : int
        Dimension of the matrix (n x n).
    use_tenpy : bool
        Whether to use TenPy implementation if available.

    Returns
    -------
    np.ndarray
        A sample from the GOE.
    """
    if use_tenpy and _TENPY_AVAILABLE:
        return np.asarray(_T_GOE((n, n)))
    rng = npr.default_rng()
    a = rng.normal(size=(n, n))
    return (a + a.T) / np.sqrt(2)


def gue(n: int, use_tenpy: bool = True) -> np.ndarray:
    """
    Generate a random matrix from the Gaussian Unitary Ensemble (GUE).

    If TenPy is available and `use_tenpy` is True, it uses `tenpy.linalg.random_matrix.GUE`.
    Otherwise, it constructs a Hermitian complex matrix: H = (X + X^dag) / 2,
    where X has i.i.d. complex normal entries.

    Parameters
    ----------
    n : int
        Dimension of the matrix (n x n).
    use_tenpy : bool
        Whether to use TenPy implementation if available.

    Returns
    -------
    np.ndarray
        A sample from the GUE.
    """
    if use_tenpy and _TENPY_AVAILABLE:
        return np.asarray(_T_GUE((n, n)))
    rng = npr.default_rng()
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = (x + x.conj().T) / 2.0
    return h


def coe(n: int, use_tenpy: bool = True) -> np.ndarray:
    """
    Generate a random matrix from the Circular Orthogonal Ensemble (COE).

    If TenPy is available and `use_tenpy` is True, it uses `tenpy.linalg.random_matrix.COE`.
    Otherwise, it generates a Haar-distributed orthogonal matrix via QR decomposition.

    Parameters
    ----------
    n : int
        Dimension of the matrix (n x n).
    use_tenpy : bool
        Whether to use TenPy implementation if available.

    Returns
    -------
    np.ndarray
        A sample from the COE (unitary symmetric or orthogonal, context dependent).
    """
    if use_tenpy and _TENPY_AVAILABLE:
        return np.asarray(_T_COE((n, n)))
    return _haar_orthogonal_qr_np(n)


def cre(n: int, use_tenpy: bool = True) -> np.ndarray:
    """
    Generate a random matrix from the Circular Real Ensemble (CRE).

    If TenPy is available and `use_tenpy` is True, it uses `tenpy.linalg.random_matrix.CRE`.
    Otherwise, falls back to COE (Haar orthogonal).

    Parameters
    ----------
    n : int
        Dimension of the matrix (n x n).
    use_tenpy : bool
        Whether to use TenPy implementation if available.

    Returns
    -------
    np.ndarray
        A sample from the CRE.
    """
    if use_tenpy and _TENPY_AVAILABLE:
        return np.asarray(_T_CRE((n, n)))
    # Fallback: treat CRE same as COE (orthogonal) in absence of TenPy
    return _haar_orthogonal_qr_np(n)


def cue(n: int, use_tenpy: bool = True, simple: bool = True) -> np.ndarray:
    """
    Generate a random matrix from the Circular Unitary Ensemble (CUE).

    If TenPy is available and `use_tenpy` is True, it uses `tenpy.linalg.random_matrix.CUE`.
    Otherwise, it generates a Haar-distributed unitary matrix via QR decomposition.

    Parameters
    ----------
    n : int
        Dimension of the matrix (n x n).
    use_tenpy : bool
        Whether to use TenPy implementation if available.
    simple : bool
        If True, uses a simpler QR method (may have phase bias). If False, corrects phases
        to ensure true Haar measure (passed to CUE_QR).

    Returns
    -------
    np.ndarray
        A sample from the CUE.
    """
    if use_tenpy and _TENPY_AVAILABLE:
        return np.asarray(_T_CUE((n, n)))
    return CUE_QR(n, simple=simple)


class RMT:
    GOE = "GOE"
    GUE = "GUE"
    COE = "COE"
    CRE = "CRE"
    CUE = "CUE"


def random_matrix(n: Union[int, Tuple[int, int]], kind: Union[str, RMT] = RMT.GOE, **kwargs) -> np.ndarray:
    if isinstance(n, tuple):
        n = n[0]
    kind = str(kind).upper()
    if kind == RMT.GOE:
        return goe(n, **kwargs)
    if kind == RMT.GUE:
        return gue(n, **kwargs)
    if kind == RMT.COE:
        return coe(n, **kwargs)
    if kind == RMT.CRE:
        return cre(n, **kwargs)
    if kind == RMT.CUE:
        return cue(n, **kwargs)
    raise ValueError(f"Unknown random matrix type: {kind}")


def list_capabilities() -> Dict[str, Tuple[str, ...]]:
    return {
        "ensembles": (RMT.GOE, RMT.GUE, RMT.COE, RMT.CRE, RMT.CUE),
        "backends": ("numpy",) + (("jax",) if _JAX_AVAILABLE else tuple()),
        "providers": ("tenpy" if _TENPY_AVAILABLE else "builtin",),
    }


__all__ = [
    "CUE_QR",
    "goe", "gue", "coe", "cre", "cue",
    "RMT", "random_matrix",
    "list_capabilities",
]