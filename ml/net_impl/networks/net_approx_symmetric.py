r"""
Approximately Symmetric Ansatz implementation.

Combo architecture (paper-aligned):
    chi (non-invariant local block) -> sigma (plaquette Wilson map) -> omega (invariant block)

The combo variant uses lattice geometry (neighbors + plaquettes) and is the default.

This architecture allows learning "unfattening" maps in the nonsymmetric block while
guaranteeing symmetry in the final wavefunction, as described in Kufel et al. (2025).

----------------------------------------------------------------
file        : general_python/ml/net_impl/networks/net_approx_symmetric.py
author      : Maksymilian Kliczkowski
date        : 2026-01-21
----------------------------------------------------------------
"""

from    __future__ import annotations

from    typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import  numpy as np

import  jax
import  jax.numpy as jnp
from    flax import linen as nn

try:
    from ...net_impl.activation_functions   import get_activation
    from ...net_impl.interface_net_flax     import FlaxInterface
except ImportError as exc:
    raise ImportError("Required modules from general_python package are missing.") from exc

try:
    from ....lattices import choose_lattice
except ImportError:
    choose_lattice = None

# -----------------------------------------------------------------------------
# Complex-safe activations used in the paper implementation
# -----------------------------------------------------------------------------

def _complex_map(x: jnp.ndarray, fun: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    if jnp.issubdtype(jnp.asarray(x).dtype, jnp.complexfloating):
        return fun(jnp.real(x)) + 1j * fun(jnp.imag(x))
    return fun(x)

def c_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """
    Complex-aware scaled sigmoid from the paper's supplementary text.

    phi(x) = (sigmoid(x) - 1/2) * (2 + 2e)/(e - 1)
    """

    scale = (2.0 + 2.0 * jnp.e) / (jnp.e - 1.0)
    return _complex_map(x, lambda z: (jax.nn.sigmoid(z) - 0.5) * scale)

def c_elu(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    """Complex ELU: ELU applied separately to real and imaginary parts."""

    return _complex_map(x, lambda z: nn.elu(z, alpha=alpha))

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _resolve_activation(spec: Any, *, default: Any) -> Callable:
    """Resolve string/callable activation specs with custom complex aliases."""

    use = default if spec is None else spec
    if callable(use):
        return use

    if isinstance(use, str):
        key = use.lower()
        if key in ("c_sigmoid", "complex_sigmoid"):
            return c_sigmoid
        if key in ("c_elu", "complex_elu"):
            return c_elu
        act, _ = get_activation(key)
        return act

    raise TypeError(f"Unsupported activation specification: {type(use)!r}")

def _real_dtype(dtype: Any) -> Any:
    dt = jnp.dtype(dtype)
    if dt == jnp.complex64:
        return jnp.float32
    if dt == jnp.complex128:
        return jnp.float64
    return dt

def _complex_normal_init(stddev: float) -> Callable:
    """Gaussian initializer that correctly handles complex parameter dtypes."""

    normal = nn.initializers.normal(stddev=stddev)

    def init(key, shape, dtype=jnp.float32):
        dt = jnp.dtype(dtype)
        if jnp.issubdtype(dt, jnp.complexfloating):
            kr, ki  = jax.random.split(key)
            rd      = _real_dtype(dt)
            r       = normal(kr, shape, rd)
            i       = normal(ki, shape, rd)
            return (r + 1j * i).astype(dt)
        return normal(key, shape, dt)

    return init

def _identity_center_kernel_init() -> Callable:
    """Initializer with center coefficient = 1 and all other coefficients = 0."""

    def init(key, shape, dtype=jnp.float32):
        del key
        out_ch, in_ch, ksz  = shape
        w                   = jnp.zeros((out_ch, in_ch, ksz), dtype=dtype)
        one                 = jnp.asarray(1.0, dtype=dtype)
        return w.at[:, :, 0].set(one)

    return init

def _valid_site(site: Any, num_sites: int) -> bool:
    try:
        s = int(site)
    except Exception:
        return False
    return 0 <= s < num_sites

def _safe_neighbors_from_lattice(lattice: Any, num_sites: int) -> List[List[int]]:
    """Build site adjacency from lattice nearest-neighbor lists."""

    adjacency: List[List[int]] = [[] for _ in range(num_sites)]
    if lattice is None:
        return adjacency

    for i in range(num_sites):
        neigh_i = []
        try:
            raw = lattice.get_nn(i)
        except Exception:
            raw = []

        if raw is None:
            raw = []
        if isinstance(raw, (int, np.integer)):
            raw = [int(raw)]

        for n in raw:
            if _valid_site(n, num_sites):
                neigh_i.append(int(n))

        adjacency[i] = sorted(set(neigh_i))

    return adjacency

def _adjacency_from_edges(lattice: Any, num_sites: int) -> Optional[List[List[int]]]:
    """
    Build adjacency from generic lattice edge APIs when available.

    Supported APIs:
    - ``lattice.edges()``
    - ``lattice.bonds`` iterable with pairs
    - ``lattice.adjacency_matrix(...)``
    """
    if lattice is None:
        return None

    adjacency: List[set] = [set() for _ in range(num_sites)]
    used = False

    # 1) NetKet-like edges() helper
    if hasattr(lattice, "edges") and callable(getattr(lattice, "edges")):
        try:
            for edge in lattice.edges():
                if edge is None or len(edge) < 2:
                    continue
                i, j = int(edge[0]), int(edge[1])
                if _valid_site(i, num_sites) and _valid_site(j, num_sites) and i != j:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    used = True
        except Exception:
            pass

    # 2) bonds property fallback
    if not used and hasattr(lattice, "bonds"):
        try:
            bonds = lattice.bonds
            for edge in bonds:
                if edge is None or len(edge) < 2:
                    continue
                i, j = int(edge[0]), int(edge[1])
                if _valid_site(i, num_sites) and _valid_site(j, num_sites) and i != j:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    used = True
        except Exception:
            pass

    # 3) binary adjacency matrix fallback
    if not used and hasattr(lattice, "adjacency_matrix") and callable(getattr(lattice, "adjacency_matrix")):
        try:
            amat = lattice.adjacency_matrix(mode="binary", include_self=False, sparse=False, save=False)
            amat = np.asarray(amat)
            if amat.ndim == 2 and amat.shape[0] == num_sites:
                rows, cols = np.nonzero(amat)
                for i, j in zip(rows.tolist(), cols.tolist()):
                    if i != j and _valid_site(i, num_sites) and _valid_site(j, num_sites):
                        adjacency[i].add(j)
                        adjacency[j].add(i)
                        used = True
        except Exception:
            pass

    if not used:
        return None

    return [sorted(x) for x in adjacency]

def _canonical_cycle(cycle: Sequence[int]) -> Tuple[int, ...]:
    """
    Canonical representation of a cycle under rotation and reversal.
    """
    cyc = [int(x) for x in cycle]
    n = len(cyc)
    if n == 0:
        return tuple()

    variants = []
    for seq in (cyc, list(reversed(cyc))):
        for shift in range(n):
            variants.append(tuple(seq[shift:] + seq[:shift]))
    return min(variants)

def _shortest_path_excluding_edge(
    adjacency: Sequence[Sequence[int]],
    src: int,
    dst: int,
    forbidden_edge: Tuple[int, int],
) -> Optional[List[int]]:
    """
    BFS shortest path from src to dst while excluding one undirected edge.
    """
    u0, v0 = forbidden_edge
    queue = [src]
    prev = {src: -1}

    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        if u == dst:
            break
        for v in adjacency[u]:
            if (u == u0 and v == v0) or (u == v0 and v == u0):
                continue
            if v in prev:
                continue
            prev[v] = u
            queue.append(v)

    if dst not in prev:
        return None

    # Reconstruct path src -> dst
    path = [dst]
    cur = dst
    while prev[cur] != -1:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path

def _infer_plaquettes_from_cycles(
    adjacency: Sequence[Sequence[int]],
    *,
    max_cycle_len: int = 12,
    max_cycles: Optional[int] = None,
) -> List[List[int]]:
    """
    Infer plaquette-like loops from graph cycles.

    This is a generic fallback for lattices that do not implement
    ``calculate_plaquettes`` explicitly.
    """
    n_nodes = len(adjacency)
    if max_cycles is None:
        max_cycles = max(1, 4 * n_nodes)

    edges = [(u, v) for u in range(n_nodes) for v in adjacency[u] if u < v]
    seen = set()
    out: List[List[int]] = []

    for u, v in edges:
        path = _shortest_path_excluding_edge(adjacency, u, v, (u, v))
        if path is None:
            continue
        if len(path) < 3 or len(path) > max_cycle_len:
            continue
        if len(set(path)) != len(path):
            continue
        key = _canonical_cycle(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(path))
        if len(out) >= max_cycles:
            break

    return out

def _bfs_kernels(adjacency: Sequence[Sequence[int]], radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build padded neighborhood tables for each node with BFS radius.

    Returns
    -------
    kernel_idx : (N, K) int
        Padded neighborhood indices. First entry is always the node itself.
    kernel_mask : (N, K) bool
        True where the index is valid (not padding).
    """

    n_nodes = len(adjacency)
    neighborhoods: List[List[int]] = []

    for start in range(n_nodes):
        visited     = {start}
        order       = [start]
        frontier    = [start]

        for _ in range(max(0, radius)):
            nxt = set()
            for u in frontier:
                for v in adjacency[u]:
                    if v not in visited:
                        nxt.add(v)
            if not nxt:
                break
            nxt_sorted = sorted(nxt)
            visited.update(nxt_sorted)
            order.extend(nxt_sorted)
            frontier = nxt_sorted

        neighborhoods.append(order)

    max_k           = max(len(n) for n in neighborhoods)
    kernel_idx      = np.zeros((n_nodes, max_k), dtype=np.int32)
    kernel_mask     = np.zeros((n_nodes, max_k), dtype=bool)

    for i, neigh in enumerate(neighborhoods):
        k = len(neigh)
        kernel_idx[i, :k] = np.asarray(neigh, dtype=np.int32)
        if k < max_k:
            kernel_idx[i, k:] = i
        kernel_mask[i, :k] = True

    return kernel_idx, kernel_mask

def _normalize_plaquettes(plaquettes: Optional[Sequence[Sequence[int]]], num_sites: int) -> List[List[int]]:
    """Sanitize and deduplicate plaquette index lists."""

    cleaned: List[List[int]] = []
    seen = set()

    if plaquettes is None:
        plaquettes = []

    for p in plaquettes:
        vals = []
        for s in p:
            if _valid_site(s, num_sites):
                vals.append(int(s))
        if not vals:
            continue

        key = tuple(sorted(vals))
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(vals)

    # Fallback: no plaquettes available -> identity map over sites
    if not cleaned:
        cleaned = [[i] for i in range(num_sites)]

    return cleaned

def _pack_plaquettes(plaquettes: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Pack variable-length plaquette lists into padded arrays."""

    n_plaq = len(plaquettes)
    max_len = max(len(p) for p in plaquettes)

    p_idx = np.full((n_plaq, max_len), -1, dtype=np.int32)
    p_mask = np.zeros((n_plaq, max_len), dtype=bool)

    for i, p in enumerate(plaquettes):
        k = len(p)
        p_idx[i, :k] = np.asarray(p, dtype=np.int32)
        p_mask[i, :k] = True

    return p_idx, p_mask

def _plaquette_adjacency(plaquettes: Sequence[Sequence[int]]) -> List[List[int]]:
    """Dual-graph adjacency: plaquettes sharing at least two spins are neighbors."""

    n_plaq  = len(plaquettes)
    sets    = [set(p) for p in plaquettes]
    adj     = [set() for _ in range(n_plaq)]

    for i in range(n_plaq):
        for j in range(i + 1, n_plaq):
            if len(sets[i].intersection(sets[j])) >= 2:
                adj[i].add(j)
                adj[j].add(i)

    return [sorted(a) for a in adj]

def _infer_honeycomb_shape(num_sites: int) -> Tuple[int, int]:
    """Infer (Lx, Ly) for honeycomb where Ns = 2 * Lx * Ly."""

    n_cells = max(1, int(num_sites // 2))
    lx      = int(np.floor(np.sqrt(n_cells)))
    while lx > 1 and (n_cells % lx) != 0:
        lx -= 1
    ly = max(1, n_cells // lx)
    return int(lx), int(ly)


def _resolve_lattice(
    lattice: Any,
    *,
    lattice_type: Optional[str],
    lattice_shape: Optional[Tuple[int, int]],
    bc: str,
    num_sites: int,
):
    if lattice is not None:
        return lattice

    if choose_lattice is None or lattice_type is None:
        return None

    try:
        if lattice_shape is None:
            if str(lattice_type).lower() == "honeycomb":
                lattice_shape = _infer_honeycomb_shape(num_sites)
            else:
                side = int(np.sqrt(num_sites))
                lattice_shape = (max(1, side), max(1, num_sites // max(1, side)))

        lx, ly = int(lattice_shape[0]), int(lattice_shape[1])
        return choose_lattice(typek=lattice_type, lx=lx, ly=ly, bc=bc)
    except Exception:
        return None


def _extract_plaquettes(lattice: Any, num_sites: int) -> List[List[int]]:
    """Get plaquettes from lattice with BC-aware open/PBC switch when available."""
    if lattice is None:
        return _normalize_plaquettes([], num_sites)

    open_bc = True
    if hasattr(lattice, "periodic_flags"):
        try:
            pbc_flags = lattice.periodic_flags()
            open_bc = not (bool(pbc_flags[0]) and bool(pbc_flags[1]))
        except Exception:
            open_bc = True

    try:
        plaquettes = lattice.calculate_plaquettes(open_bc=open_bc)
    except TypeError:
        try:
            plaquettes = lattice.calculate_plaquettes(use_obc=open_bc)
        except Exception:
            try:
                plaquettes = lattice.calculate_plaquettes()
            except Exception:
                plaquettes = []
    except Exception:
        plaquettes = []

    # Generic fallback: infer loops from graph connectivity if lattice has no plaquette API
    if not plaquettes:
        adjacency = _adjacency_from_edges(lattice, num_sites)
        if adjacency is None:
            adjacency = _safe_neighbors_from_lattice(lattice, num_sites)
        plaquettes = _infer_plaquettes_from_cycles(adjacency)

    return _normalize_plaquettes(plaquettes, num_sites)

# -----------------------------------------------------------------------------

def build_combo_geometry(
    *,
    num_sites: int,
    chi_kernel_size     : int,
    omega_kernel_size   : int,
    lattice             : Any = None,
    plaquettes          : Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, np.ndarray]:
    """Prepare all static index/mask tables required by the combo architecture."""

    if plaquettes is None:
        plaq_list = _extract_plaquettes(lattice, num_sites)
    else:
        plaq_list = _normalize_plaquettes(plaquettes, num_sites)

    p_idx, p_mask = _pack_plaquettes(plaq_list)

    # Site (qubit) kernels for non-invariant block
    site_adj = _adjacency_from_edges(lattice, num_sites)
    if site_adj is None:
        site_adj = _safe_neighbors_from_lattice(lattice, num_sites)
    chi_radius = max(0, (max(1, int(chi_kernel_size)) - 1) // 2)
    chi_idx, chi_mask = _bfs_kernels(site_adj, chi_radius)

    # Plaquette kernels for invariant block
    dual_adj = _plaquette_adjacency(plaq_list)
    omega_radius = max(0, (max(1, int(omega_kernel_size)) - 1) // 2)
    omega_idx, omega_mask = _bfs_kernels(dual_adj, omega_radius)

    return {
        "chi_kernel_idx": chi_idx,
        "chi_kernel_mask": chi_mask,
        "plaquette_idx": p_idx,
        "plaquette_mask": p_mask,
        "omega_kernel_idx": omega_idx,
        "omega_kernel_mask": omega_mask,
        "n_plaquettes": np.asarray([len(plaq_list)], dtype=np.int32),
    }


# -----------------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------------


class _MaskedConv(nn.Module):
    """Shared masked convolution over graph neighborhoods represented by index tables."""

    kernel_idx: jnp.ndarray
    kernel_mask: jnp.ndarray
    out_channels: int
    dtype: Any
    kernel_init: Callable

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, nodes, in_channels)
        in_channels = int(x.shape[-1])
        kernel_size = int(self.kernel_idx.shape[-1])

        w = self.param(
            "kernel",
            self.kernel_init,
            (self.out_channels, in_channels, kernel_size),
            self.dtype,
        )
        b = self.param("bias", nn.initializers.zeros_init(), (self.out_channels,), self.dtype)

        # Gather local neighborhoods for each node
        gathered = jnp.take(x, self.kernel_idx, axis=1)  # (B, N, K, C_in)
        gathered = gathered * self.kernel_mask[None, :, :, None].astype(gathered.dtype)
        gathered = jnp.swapaxes(gathered, 2, 3)  # (B, N, C_in, K)

        y = jnp.einsum("bnik,oik->bno", gathered, w)
        return y + b


# -----------------------------------------------------------------------------
# Combo architecture (paper-aligned)
# -----------------------------------------------------------------------------


class ApproxSymmetricNet(nn.Module):
    """
    Geometry-aware approximately symmetric ansatz.

    psi(s) = Omega( Sigma( Chi(s) ) )

    - Chi: non-invariant local block on sites
    - Sigma: plaquette Wilson-product map (invariant by construction)
    - Omega: local block on plaquette dual graph
    """

    chi_channels: Sequence[int]
    omega_channels: Sequence[int]

    chi_kernel_idx: jnp.ndarray
    chi_kernel_mask: jnp.ndarray
    plaquette_idx: jnp.ndarray
    plaquette_mask: jnp.ndarray
    omega_kernel_idx: jnp.ndarray
    omega_kernel_mask: jnp.ndarray

    nib_act: Callable[[jnp.ndarray], jnp.ndarray]
    ib_act: Callable[[jnp.ndarray], jnp.ndarray]
    readout_act: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    nib_identity_init: bool = True
    ib_init_std: float = 0.02
    wilson_rescale: float = 10 ** 1.5
    wilson_separate_complex: bool = True

    pool_mode: str = "sum"
    dtype: Any = jnp.complex64
    islog: bool = True

    def _wilson_map(self, x: jnp.ndarray) -> jnp.ndarray:
        """Plaquette map Sigma: product over spins belonging to each plaquette."""

        mask = self.plaquette_mask
        safe_idx = jnp.where(mask, self.plaquette_idx, 0)

        gathered = jnp.take(x, safe_idx, axis=1)  # (B, Np, Psz, C)

        if self.wilson_separate_complex and jnp.issubdtype(jnp.asarray(x).dtype, jnp.complexfloating):
            mr = jnp.where(mask[None, :, :, None], jnp.real(gathered), 1.0)
            mi = jnp.where(mask[None, :, :, None], jnp.imag(gathered), 1.0)
            return self.wilson_rescale * jnp.prod(mr, axis=2) + 1j * self.wilson_rescale * jnp.prod(
                mi, axis=2
            )

        m = jnp.where(mask[None, :, :, None], gathered, jnp.ones_like(gathered))
        return self.wilson_rescale * jnp.prod(m, axis=2)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
        x = x.reshape((x.shape[0], -1))

        h = x.astype(self.dtype)
        h = h[:, :, None]  # (B, Ns, 1)

        # Non-invariant block Chi
        chi = [int(v) for v in self.chi_channels]
        if not chi:
            chi = [1]
        if chi[0] != h.shape[-1]:
            chi = [h.shape[-1]] + chi

        for li, out_ch in enumerate(chi[1:]):
            kernel_init = _identity_center_kernel_init() if self.nib_identity_init else _complex_normal_init(0.05)
            h = _MaskedConv(
                kernel_idx=self.chi_kernel_idx,
                kernel_mask=self.chi_kernel_mask,
                out_channels=int(out_ch),
                dtype=self.dtype,
                kernel_init=kernel_init,
                name=f"chi_conv_{li}",
            )(h)
            h = self.nib_act(h)

        # Plaquette-invariant nonlinearity Sigma
        h = self._wilson_map(h)

        # Invariant block Omega
        omg = [int(v) for v in self.omega_channels]
        if not omg:
            omg = [int(h.shape[-1])]
        if omg[0] != h.shape[-1]:
            omg = [int(h.shape[-1])] + omg

        for li, out_ch in enumerate(omg[1:]):
            h = _MaskedConv(
                kernel_idx=self.omega_kernel_idx,
                kernel_mask=self.omega_kernel_mask,
                out_channels=int(out_ch),
                dtype=self.dtype,
                kernel_init=_complex_normal_init(self.ib_init_std),
                name=f"omega_conv_{li}",
            )(h)
            h = self.ib_act(h)

        # Final reduction to log-amplitude scalar
        if str(self.pool_mode).lower() == "mean":
            out = jnp.mean(h, axis=(1, 2))
        else:
            out = jnp.sum(h, axis=(1, 2))

        if self.readout_act is not None:
            out = self.readout_act(out)

        if not self.islog:
            out = jnp.exp(out)

        return out


class AnsatzApproxSymmetric(FlaxInterface):
    """
    Flax interface for approximately symmetric ansatz.

    This class only exposes the geometry-aware combo architecture described in the paper.
    """

    def __init__(
        self,
        # Optional aliases
        chi_features: Optional[Sequence[int]] = None,
        readout_act: Any = None,
        chi_act: Any = None,
        # Combo architecture arguments
        architecture: str = "combo",
        chi_channels: Optional[Sequence[int]] = None,
        omega_channels: Optional[Sequence[int]] = None,
        chi_kernel_size: int = 3,
        omega_kernel_size: int = 15,
        nib_act: Any = "c_sigmoid",
        ib_act: Any = "c_elu",
        ib_init_std: float = 0.02,
        nib_identity_init: bool = True,
        wilson_rescale: float = 10 ** 1.5,
        pool_mode: str = "sum",
        wilson_separate_complex: bool = True,
        # Lattice / geometry
        lattice: Any = None,
        plaquettes: Optional[Sequence[Sequence[int]]] = None,
        lattice_type: Optional[str] = None,
        lattice_shape: Optional[Tuple[int, int]] = None,
        bc: str = "pbc",
        # Runtime
        input_shape: tuple = (10,),
        backend: str = "jax",
        dtype: Any = jnp.complex128,
        seed: int = 42,
        islog: bool = True,
        **kwargs,
    ):
        dt = jnp.dtype(dtype)
        arch = str(architecture).lower()
        if arch not in {"combo", "paper", "paper_combo"}:
            raise ValueError(
                f"Unsupported approx-symmetric architecture '{architecture}'. "
                "Only the geometry-aware combo architecture is available."
            )

        n_sites = int(np.prod(input_shape))
        resolved_lattice = _resolve_lattice(
            lattice,
            lattice_type=lattice_type,
            lattice_shape=lattice_shape,
            bc=bc,
            num_sites=n_sites,
        )

        geometry = build_combo_geometry(
            num_sites=n_sites,
            chi_kernel_size=chi_kernel_size,
            omega_kernel_size=omega_kernel_size,
            lattice=resolved_lattice,
            plaquettes=plaquettes,
        )

        if chi_channels is None:
            if chi_features is not None:
                chi_channels = (1,) + tuple(int(v) for v in chi_features)
            else:
                chi_channels = (1, 2, 4)
        if omega_channels is None:
            omega_channels = (4, 4, 4)

        # Preserve older kwarg aliases while using combo internals.
        nib_fun = _resolve_activation(nib_act if chi_act is None else chi_act, default="c_sigmoid")
        ib_fun = _resolve_activation(ib_act, default="c_elu")
        readout_fun = None if readout_act is None else _resolve_activation(readout_act, default="identity")

        net_module = ApproxSymmetricNet
        net_kwargs = {
            "chi_channels": tuple(int(v) for v in chi_channels),
            "omega_channels": tuple(int(v) for v in omega_channels),
            "chi_kernel_idx": jnp.asarray(geometry["chi_kernel_idx"], dtype=jnp.int32),
            "chi_kernel_mask": jnp.asarray(geometry["chi_kernel_mask"], dtype=bool),
            "plaquette_idx": jnp.asarray(geometry["plaquette_idx"], dtype=jnp.int32),
            "plaquette_mask": jnp.asarray(geometry["plaquette_mask"], dtype=bool),
            "omega_kernel_idx": jnp.asarray(geometry["omega_kernel_idx"], dtype=jnp.int32),
            "omega_kernel_mask": jnp.asarray(geometry["omega_kernel_mask"], dtype=bool),
            "nib_act": nib_fun,
            "ib_act": ib_fun,
            "readout_act": readout_fun,
            "nib_identity_init": bool(nib_identity_init),
            "ib_init_std": float(ib_init_std),
            "wilson_rescale": float(wilson_rescale),
            "wilson_separate_complex": bool(wilson_separate_complex),
            "pool_mode": str(pool_mode).lower(),
            "dtype": dt,
            "islog": bool(islog),
        }

        self._combo_meta = {
            "n_sites": n_sites,
            "n_plaquettes": int(geometry["n_plaquettes"][0]),
            "chi_kernel": int(geometry["chi_kernel_idx"].shape[1]),
            "omega_kernel": int(geometry["omega_kernel_idx"].shape[1]),
        }

        super().__init__(
            net_module=net_module,
            net_kwargs=net_kwargs,
            input_shape=input_shape,
            backend=backend,
            dtype=dt,
            seed=seed,
            **kwargs,
        )

    def __repr__(self) -> str:
        meta = self._combo_meta
        return (
            "AnsatzApproxSymmetric(architecture='combo', "
            f"n_sites={meta.get('n_sites')}, n_plaquettes={meta.get('n_plaquettes')}, "
            f"chi_kernel={meta.get('chi_kernel')}, omega_kernel={meta.get('omega_kernel')})"
        )


# ----------------------------------------------------------------------
#! END
# ----------------------------------------------------------------------
