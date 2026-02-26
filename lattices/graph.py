"""
Generic graph lattice implementation.

Provides a lightweight :class:`GraphLattice` that builds on the base
:class:`~general_python.lattices.lattice.Lattice` using a user-specified
adjacency matrix and optional geometric embedding.  This allows leveraging the
Hilbert-space symmetry machinery on irregular graphs while keeping the public
API consistent with regular lattices.
"""

from    __future__ import annotations

from    dataclasses import dataclass
from    typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import  numpy as np

try:
    from .lattice import Lattice, LatticeBC, LatticeDirection, LatticeType, BoundaryFlux
except ImportError as e:
    raise RuntimeError("Bad import graph.py") from e

def _normalize_adjacency(adj: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
    """Return a dense square adjacency matrix."""
    arr = np.asarray(adj, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Adjacency matrix must be square; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Adjacency matrix must contain only finite values.")
    return arr

def _normalize_coordinates(coords: Optional[Union[np.ndarray, Sequence[Sequence[float]]]], ns: int) -> np.ndarray:
    """
    Normalize coordinate input.  If ``coords`` is None, embed vertices on a
    one-dimensional chain.  Otherwise ensure shape ``(ns, dim)``.
    """
    if coords is None:
        return np.arange(ns, dtype=float).reshape(ns, 1)
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != ns:
        raise ValueError(f"Coordinates must have shape (Ns, dim); got {arr.shape}, expected ({ns}, dim).")
    return arr

# -------------------------------------------------------------------

@dataclass
class GraphMetadata:
    """
    Optional metadata associated with a :class:`GraphLattice`.
    """
    name: str = "graph"
    tags: Tuple[str, ...] = ()
    info: Optional[str] = None

# -------------------------------------------------------------------

class GraphLattice(Lattice):
    """
    Lattice backed by an adjacency matrix instead of structural formulas.

    Parameters
    ----------
    adjacency:
        Square (``Ns \times Ns``) adjacency matrix.  Non-zero entries denote
        connected pairs.  The absolute value of the weight is used to rank
        neighbor order (highest -> nearest).
    coords:
        Optional vertex embedding of shape ``(Ns, dim)``.  Defaults to a 1D
        chain ordering.  Used for plotting and distance heuristics.
    bc:
        Boundary condition descriptor.  Defaults to open boundaries.
    flux:
        Optional boundary flux phases - forwarded to :class:`Lattice`.
    metadata:
        Auxiliary metadata (name, tags, free-form info).
    """

    def __init__(
        self,
        adjacency   : Union[np.ndarray, Sequence[Sequence[float]]],
        coords      : Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        bc          : Optional[Union[str, LatticeBC]] = LatticeBC.OBC,
        flux        : Optional[Union[float, BoundaryFlux, Mapping[Union[str, LatticeDirection], float]]] = None,
        metadata    : Optional[GraphMetadata] = None,
        **kwargs,
    ):
        self._adjacency     = _normalize_adjacency(adjacency)
        ns                  = int(self._adjacency.shape[0])
        self._coord_array   = _normalize_coordinates(coords, ns)
        dim                 = int(self._coord_array.shape[1])
        self.metadata       = metadata or GraphMetadata()

        # Set up trivial lattice extents (lx=Ns ensures consistent indexing)
        super().__init__(
            dim=dim,
            lx=ns,
            ly=1,
            lz=1,
            bc=bc,
            adj_mat=self._adjacency,
            flux=flux,
            **kwargs,
        )

        self._type  = LatticeType.GRAPH
        self._ns    = ns
        self._dim   = dim
        self._lx    = ns
        self._ly    = 1
        self._lz    = 1
        self.init()

    # ------------------------------------------------------------------
    # Abstract implementations

    def site_index(self, x: int, y: int = 0, z: int = 0) -> int:
        """
        Return linear index for coordinates.  For graph lattices we interpret
        the first argument as the explicit vertex index.
        """
        if y != 0 or z != 0:
            raise ValueError("GraphLattice only supports 1D indexing: provide vertex id as `x`.")
        if not (0 <= x < self._ns):
            raise IndexError(f"Vertex index {x} out of bounds for Ns={self._ns}.")
        return int(x)

    def get_real_vec(self, x: int, y: int = 0, z: int = 0):
        idx = self.site_index(int(x), int(y), int(z))
        return np.asarray(self._coord_array[idx], dtype=float)

    def get_norm(self, x: int, y: int = 0, z: int = 0):
        return float(np.linalg.norm(self.get_real_vec(x, y, z)))

    def get_nn_direction(self, site, direction):
        neigh = self.get_nn(int(site))
        if neigh is None or len(neigh) == 0:
            return self.bad_lattice_site
        if isinstance(direction, LatticeDirection):
            idx = {LatticeDirection.X: 0, LatticeDirection.Y: 1, LatticeDirection.Z: 2}.get(direction, 0)
        else:
            idx = int(direction)
        if idx < 0 or idx >= len(neigh):
            return self.bad_lattice_site
        return int(neigh[idx])

    def calculate_coordinates(self):
        self.coordinates = [tuple(row) for row in self._coord_array]

    def calculate_reciprocal_vectors(self):
        # Reciprocal vectors are not well-defined on generic graphs.
        self._k1 = np.zeros(3, dtype=float)
        self._k2 = np.zeros(3, dtype=float)
        self._k3 = np.zeros(3, dtype=float)

    def calculate_r_vectors(self):
        self.rvectors = np.asarray(self._coord_array, dtype=float)

    def calculate_k_vectors(self):
        # For irregular graphs reciprocal vectors are not well-defined;
        # provide a zero placeholder to satisfy downstream usage.
        self.kvectors = np.zeros_like(self._coord_array, dtype=float)

    def calculate_dft_matrix(self):
        # Generic graphs do not have translationally-defined Fourier modes.
        # Keep a valid placeholder so lattice initialization can proceed.
        self._dft = np.eye(int(self._ns), dtype=complex)

    def calculate_norm_sym(self):
        self.sym_norm = {}
        self.sym_map = {}

    def calculate_nn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        del pbcx, pbcy, pbcz
        ns = int(self._adjacency.shape[0])
        self._nn = []
        self._nn_forward = []
        for i in range(ns):
            neigh = [int(j) for j in np.nonzero(self._adjacency[i])[0].tolist() if int(j) != i]
            neigh = sorted(set(neigh))
            self._nn.append(neigh)
            self._nn_forward.append([j for j in neigh if j > i])
        return self._nn

    def calculate_nnn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        del pbcx, pbcy, pbcz
        ns = int(self._adjacency.shape[0])
        if not hasattr(self, "_nn") or len(self._nn) != ns:
            self.calculate_nn_in(False, False, False)

        self._nnn = []
        self._nnn_forward = []
        for i in range(ns):
            first = set(self._nn[i])
            second = set()
            for j in first:
                second.update(self._nn[j])
            second.discard(i)
            second -= first
            nnn = sorted(second)
            self._nnn.append(nnn)
            self._nnn_forward.append([j for j in nnn if j > i])
        return self._nnn

    # ------------------------------------------------------------------

    @property
    def adjacency(self) -> np.ndarray:
        return self._adjacency

    def embedding(self) -> np.ndarray:
        """
        Return coordinate embedding with shape ``(Ns, dim)``.
        """
        return self._coord_array.copy()

    # ------------------------------------------------------------------
    # Convenience graph coupling API

    def _refresh_topology(self, *, reinitialize: bool = True) -> None:
        """
        Synchronize internal adjacency references and optionally rebuild neighbours.
        """
        self._adj_mat = self._adjacency
        if hasattr(self, "_adj_cache"):
            self._adj_cache = {}
        if reinitialize:
            self.init()

    def edge_list(self, *, include_weights: bool = False, threshold: float = 0.0) -> List[Tuple]:
        """
        Return unique undirected edges from adjacency.
        """
        out = []
        ns  = int(self._adjacency.shape[0])
        thr = float(threshold)
        for i in range(ns):
            for j in range(i + 1, ns):
                w = float(self._adjacency[i, j])
                if abs(w) <= thr:
                    continue
                if include_weights:
                    out.append((i, j, w))
                else:
                    out.append((i, j))
        return out

    def set_edge(
        self,
        i               : int,
        j               : int,
        weight          : float = 1.0,
        *,
        symmetric       : bool = True,
        add             : bool = False,
        reinitialize    : bool = True,
    ) -> None:
        """
        Set or add an edge coupling weight.
        """
        i, j = int(i), int(j)
        if i == j:
            raise ValueError("Self-edges are not supported in GraphLattice.")
        if not (0 <= i < self._ns and 0 <= j < self._ns):
            raise IndexError(f"Edge ({i}, {j}) out of range for Ns={self._ns}.")

        w = float(weight)
        if add:
            self._adjacency[i, j] += w
            if symmetric:
                self._adjacency[j, i] += w
        else:
            self._adjacency[i, j] = w
            if symmetric:
                self._adjacency[j, i] = w
        self._refresh_topology(reinitialize=reinitialize)

    def add_edge(self, i: int, j: int, weight: float = 1.0, *, reinitialize: bool = True) -> None:
        """Set an undirected edge weight."""
        self.set_edge(i, j, weight=weight, symmetric=True, add=False, reinitialize=reinitialize)

    def add_to_edge(self, i: int, j: int, weight: float, *, reinitialize: bool = True) -> None:
        """Increment an undirected edge weight by ``weight``."""
        self.set_edge(i, j, weight=weight, symmetric=True, add=True, reinitialize=reinitialize)

    def remove_edge(self, i: int, j: int, *, reinitialize: bool = True) -> None:
        """Remove edge by setting weight to zero."""
        self.set_edge(i, j, weight=0.0, symmetric=True, add=False, reinitialize=reinitialize)

    def clear_edges(self, *, reinitialize: bool = True) -> None:
        """Remove all couplings."""
        self._adjacency[:, :] = 0.0
        self._refresh_topology(reinitialize=reinitialize)

    def add_couplings(
        self,
        couplings       : Iterable[Tuple[int, int, float]],
        *,
        mode            : str = "set",
        reinitialize    : bool = True,
    ) -> None:
        """
        Bulk add/update couplings from ``(i, j, weight)`` tuples.

        ``mode``:
        - ``'set'``: overwrite values
        - ``'add'``: increment values
        """
        mode_l = str(mode).lower()
        if mode_l not in {"set", "add"}:
            raise ValueError("mode must be one of {'set', 'add'}.")

        for i, j, w in couplings:
            self.set_edge(
                int(i),
                int(j),
                weight=float(w),
                symmetric=True,
                add=(mode_l == "add"),
                reinitialize=False,
            )
        self._refresh_topology(reinitialize=reinitialize)
