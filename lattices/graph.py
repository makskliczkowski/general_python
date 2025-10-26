"""
Generic graph lattice implementation.

Provides a lightweight :class:`GraphLattice` that builds on the base
:class:`~QES.general_python.lattices.lattice.Lattice` using a user-specified
adjacency matrix and optional geometric embedding.  This allows leveraging the
Hilbert-space symmetry machinery on irregular graphs while keeping the public
API consistent with regular lattices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .lattice import Lattice, LatticeBC, LatticeDirection, LatticeType, BoundaryFlux


def _normalize_adjacency(adj: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
    """Return a dense square adjacency matrix."""
    arr = np.asarray(adj, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Adjacency matrix must be square; got shape {arr.shape}.")
    if np.any(arr < 0):
        raise ValueError("Adjacency matrix weights must be non-negative.")
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


@dataclass
class GraphMetadata:
    """
    Optional metadata associated with a :class:`GraphLattice`.
    """
    name: str = "graph"
    tags: Tuple[str, ...] = ()
    info: Optional[str] = None


class GraphLattice(Lattice):
    """
    Lattice backed by an adjacency matrix instead of structural formulas.

    Parameters
    ----------
    adjacency:
        Square (``Ns \times Ns``) adjacency matrix.  Non-zero entries denote
        connected pairs.  The absolute value of the weight is used to rank
        neighbor order (highest ⇒ nearest).
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
        adjacency: Union[np.ndarray, Sequence[Sequence[float]]],
        coords: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        bc: Optional[Union[str, LatticeBC]] = LatticeBC.OBC,
        flux: Optional[Union[float, BoundaryFlux, Mapping[Union[str, LatticeDirection], float]]] = None,
        metadata: Optional[GraphMetadata] = None,
        **kwargs,
    ):
        self._adjacency = _normalize_adjacency(adjacency)
        ns = int(self._adjacency.shape[0])
        self._coord_array = _normalize_coordinates(coords, ns)
        dim = int(self._coord_array.shape[1])
        self.metadata = metadata or GraphMetadata()

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

        self._type = LatticeType.GRAPH
        self._ns = ns
        self._dim = dim
        self._lx = ns
        self._ly = 1
        self._lz = 1
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

    def calculate_coordinates(self):
        self.coordinates = [tuple(row) for row in self._coord_array]

    def calculate_r_vectors(self):
        self.rvectors = np.asarray(self._coord_array, dtype=float)

    def calculate_k_vectors(self):
        # For irregular graphs reciprocal vectors are not well-defined;
        # provide a zero placeholder to satisfy downstream usage.
        self.kvectors = np.zeros_like(self._coord_array, dtype=float)

    def calculate_norm_sym(self):
        self.sym_norm = {}
        self.sym_map = {}

    def calculate_nnn_in(self, pbcx: bool, pbcy: bool, pbcz: bool):
        # Neighbor lists are derived from the adjacency matrix in `Lattice.init`.
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
