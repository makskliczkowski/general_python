r"""
Lattice Point-Group and Space-Group Symmetry Utilities
======================================================

Generate permutation tables for translation, point-group, and full space-group
symmetries of common lattice geometries.  The permutation tables are used by
the group-equivariant CNN (EquivariantGCNN) and symmetry-projected ansätze.

Supported lattice symmetries
-----------------------------
*   **Translations**:  $\mathbb{Z}_{L_x} \times \mathbb{Z}_{L_y}$
*   **Point group** (square lattice):  $D_4 = \{e, C_4, C_2, C_4^3, m_x, m_y, m_{d_1}, m_{d_2}\}$
*   **Space group**: semi-direct product  translations $\rtimes$ point group

References
----------
*   Sharma, Manna, Rao, Sreejith, arXiv:2505.23728v1 (2025)
*   Roth & MacDonald, arXiv:2104.05085 (2021)
*   Cohen & Welling, arXiv:1602.07576 (2016)

-------------------------------------------------------------------------------
File        : general_python/lattices/tools/lattice_symmetry.py
Author      : Maksymilian Kliczkowski
Date        : 2025-02-12
-------------------------------------------------------------------------------
"""

import numpy as np
from typing import Optional

__all__ = [
    "generate_translation_perms",
    "generate_point_group_perms_square",
    "generate_space_group_perms",
    "compute_cayley_table",
]

def generate_translation_perms(
    Lx: int,
    Ly: int,
    sites_per_cell: int = 1,
) -> np.ndarray:
    r"""
    Generate permutation table for the translation group
    $\mathbb{Z}_{L_x} \times \mathbb{Z}_{L_y}$.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.
    sites_per_cell : int
        Sites per unit cell (1 for square, 2 for honeycomb).

    Returns
    -------
    perm_table : ndarray, shape (Lx*Ly, Ns)
        Each row is a permutation of site indices corresponding to a
        lattice translation.
    """
    Ns = Lx * Ly * sites_per_cell
    perms = []
    for tx in range(Lx):
        for ty in range(Ly):
            perm = np.zeros(Ns, dtype=np.int32)
            for site in range(Ns):
                cell    = site // sites_per_cell
                sub     = site %  sites_per_cell
                x, y    = cell % Lx, cell // Lx
                nx      = (x + tx) % Lx
                ny      = (y + ty) % Ly
                perm[site] = (ny * Lx + nx) * sites_per_cell + sub
            perms.append(perm)
    return np.array(perms, dtype=np.int32)

def generate_point_group_perms_square(Lx: int, Ly: int) -> np.ndarray:
    r"""
    Generate point group $D_4$ permutations for a square lattice.

    $D_4 = \{e, C_4, C_2, C_4^3, m_x, m_y, m_{d_1}, m_{d_2}\}$

    Only generates non-trivial operations when ``Lx == Ly``.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.

    Returns
    -------
    perm_table : ndarray, shape (n_ops, Ns)
        Permutations for each point-group element.
    """
    Ns  = Lx * Ly
    ops = [np.arange(Ns, dtype=np.int32)]    # identity

    if Lx != Ly:
        return np.array(ops, dtype=np.int32)

    L = Lx

    def _s(x, y):
        return (y % L) * L + (x % L)

    # C4 rotations
    for rot_fn in [
        lambda x, y: (L - 1 - y, x),            # 90°
        lambda x, y: (L - 1 - x, L - 1 - y),    # 180°
        lambda x, y: (y, L - 1 - x),            # 270°
    ]:
        p = np.zeros(Ns, dtype=np.int32)
        for s in range(Ns):
            p[s] = _s(*rot_fn(s % L, s // L))
        ops.append(p)

    # Mirrors
    for mir_fn in [
        lambda x, y: (L - 1 - x, y),            # m_x
        lambda x, y: (x, L - 1 - y),            # m_y
        lambda x, y: (y, x),                    # m_{d1}
        lambda x, y: (L - 1 - y, L - 1 - x),    # m_{d2}
    ]:
        p = np.zeros(Ns, dtype=np.int32)
        for s in range(Ns):
            p[s] = _s(*mir_fn(s % L, s // L))
        ops.append(p)

    return np.array(ops, dtype=np.int32)

def generate_space_group_perms(
    Lx: int,
    Ly: int,
    sites_per_cell: int = 1,
    point_group: str = "full",
) -> np.ndarray:
    r"""
    Generate the full space group (translations $\rtimes$ point group).

    For square lattice with ``point_group='full'``, returns
    $p4m \cong \mathbb{Z}^2 \rtimes D_4$ with $|G| = 8 L^2$ (when $L_x = L_y$).

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions.
    sites_per_cell : int
        Sites per unit cell (1 for square, 2 for honeycomb).
    point_group : str
        ``'full'`` for maximal point group, ``'translations'`` for translations only.

    Returns
    -------
    perm_table : ndarray, shape (|G|, Ns)
        Permutation table for each group element.
    """
    trans = generate_translation_perms(Lx, Ly, sites_per_cell)

    if point_group == "translations" or sites_per_cell > 1:
        return trans

    point    = generate_point_group_perms_square(Lx, Ly)
    combined = []
    seen     = set()
    for t_perm in trans:
        for p_perm in point:
            c   = t_perm[p_perm]
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                combined.append(c)
    return np.array(combined, dtype=np.int32)

# -----------------------------------------------------------------------------

def compute_cayley_table(perm_table: np.ndarray) -> np.ndarray:
    r"""
    Compute the Cayley (multiplication) table of a finite group given as
    permutations:  ``cayley[i, j] = index of g_i^{-1} \circ g_j``.

    Parameters
    ----------
    perm_table : ndarray, shape (|G|, Ns)
        Each row is a permutation of ``Ns`` site indices.

    Returns
    -------
    cayley : ndarray, shape (|G|, |G|)
        ``cayley[i, j]`` is the group-element index of the composition
        $g_i^{-1} \circ g_j$.
    """
    n_group, Ns     = perm_table.shape

    # Inverse permutations
    inv_perms       = np.zeros_like(perm_table)
    for i in range(n_group):
        inv_perms[i, perm_table[i]] = np.arange(Ns, dtype=np.int32)

    # Permutation -> index lookup
    perm_to_idx     = {tuple(perm_table[i]): i for i in range(n_group)}
    cayley          = np.zeros((n_group, n_group), dtype=np.int32)
    
    for i in range(n_group):
        for j in range(n_group):
            composed      = inv_perms[i][perm_table[j]]
            cayley[i, j]  = perm_to_idx[tuple(composed)]

    return cayley

# -----------------------------------------------------------------------------
#! EOF
# -----------------------------------------------------------------------------