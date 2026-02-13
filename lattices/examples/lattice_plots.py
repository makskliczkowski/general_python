#!/usr/bin/env python3
"""
Comprehensive lattice region & k-space demo.

Generates publication-quality multi-panel figures for four lattice types
(square, honeycomb, hexagonal / armchair, triangular):

  1. Honeycomb (zig-zag) vs Hexagonal (armchair) structure comparison
  2. Bond-length validation — confirms every NN bond = 1.0
  3. Kitaev–Preskill tripartite sectors A, B, C
  4. Levin–Wen annular regions A (core), B (annulus), C (exterior)
  5. Disk regions centered at site 0
  6. Half-system bipartitions in x and y
  7. Full lattice structures under PBC
  8. First Brillouin zones with high-symmetry paths

Usage:
    python demo_lattice_plots.py           # saves PNGs to demo_plots/
    python demo_lattice_plots.py --show    # also opens interactive windows
"""

import sys, os
import numpy as np
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project import ──────────────────────────────────────────────────────
_QES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, _QES_ROOT)
from general_python.lattices import choose_lattice

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_plots")
os.makedirs(SAVE_DIR, exist_ok=True)
SHOW = "--show" in sys.argv

LATTICE_TYPES = ["square", "honeycomb", "hexagonal", "triangular"]


def savefig(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"  saved → {path}")
    if SHOW:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Honeycomb (zig-zag) vs Hexagonal (armchair) structure comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n═══ 1. Honeycomb vs Hexagonal structure ═══")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for idx, (name, title_lbl) in enumerate([
    ("honeycomb", "Honeycomb (zig-zag)"),
    ("hexagonal", "Hexagonal (armchair)"),
]):
    lat = choose_lattice(name, dim=2, lx=4, ly=4, bc="obc")
    lat.plot.structure(
        ax=axes[idx],
        show_indices=True,
        show_primitive_cell=True,
        partition_colors=("tab:blue", "tab:orange"),
        title=f"{title_lbl}  ({lat.Ns} sites, {lat.n_edges} bonds)",
    )
fig.suptitle("Honeycomb (zig-zag edge) vs Hexagonal (armchair edge)", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "01_honeycomb_vs_hexagonal_structure.png")


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Bond-length validation across all lattices
# ═══════════════════════════════════════════════════════════════════════════
print("\n═══ 2. Bond-length check ═══")

for ltype in LATTICE_TYPES:
    lat = choose_lattice(ltype, dim=2, lx=6, ly=6, bc="obc")
    bonds = lat.edges()
    if not bonds:
        print(f"  {ltype:12s}  ** no NN-forward bonds (known issue) **")
        continue
    dists = [lat.distance(i, j) for (i, j) in bonds]
    dmin, dmax, dmean = min(dists), max(dists), np.mean(dists)
    status = "OK" if abs(dmax - dmin) < 1e-6 else f"SPREAD {dmax-dmin:.4f}"
    print(f"  {ltype:12s}  bonds={len(bonds):4d}  "
          f"dist=[{dmin:.4f}, {dmax:.4f}]  mean={dmean:.4f}  {status}")


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Kitaev-Preskill regions — all lattice types
# ═══════════════════════════════════════════════════════════════════════════
print("\n═══ 3. Kitaev-Preskill regions (all lattices) ═══")

KP_DESCS = {"A": "sector −π..−π/3", "B": "sector −π/3..+π/3", "C": "sector +π/3..+π"}

fig_kp, axes_kp = plt.subplots(2, 2, figsize=(14, 13))
axes_kp = axes_kp.ravel()

for idx, ltype in enumerate(LATTICE_TYPES):
    lat = choose_lattice(ltype, dim=2, lx=8, ly=8, bc="pbc")
    kp = lat.regions.get_region("kitaev_preskill", radius=4.0)
    kp_abc = {k: v for k, v in kp.items() if len(k) == 1}

    n_str = ", ".join(f"{k}:{len(v)}" for k, v in kp_abc.items())
    lat.plot.regions(
        kp_abc,
        ax=axes_kp[idx],
        fill=True, fill_alpha=0.10,
        blob_radius=0.22, blob_alpha=0.10,
        show_bonds=True, show_complement=False,
        region_descriptions=KP_DESCS,
        title=f"{ltype.capitalize()} KP  ({n_str})",
        legend_loc="upper right", legend_fontsize=7,
    )

fig_kp.suptitle("Kitaev–Preskill Sectors (A, B, C)  —  r = 4.0, no PBC wrap",
                 fontsize=14, y=1.01)
fig_kp.tight_layout()
savefig(fig_kp, "03_kitaev_preskill_all.png")


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Levin-Wen annular regions — all lattice types
# ═══════════════════════════════════════════════════════════════════════════
print("\n═══ 4. Levin-Wen regions (all lattices) ═══")

LW_DESCS = {"A": "inner disk r<2", "B": "annulus 2≤r≤4.5", "C": "outer r>4.5"}

fig_lw, axes_lw = plt.subplots(2, 2, figsize=(14, 13))
axes_lw = axes_lw.ravel()

for idx, ltype in enumerate(LATTICE_TYPES):
    lat = choose_lattice(ltype, dim=2, lx=8, ly=8, bc="pbc")
    lw = lat.regions.get_region("levin_wen", inner_radius=2.0, outer_radius=4.5)
    lw_abc = {k: v for k, v in lw.items() if len(k) == 1}

    n_str = ", ".join(f"{k}:{len(v)}" for k, v in lw_abc.items())
    lat.plot.regions(
        lw_abc,
        ax=axes_lw[idx],
        fill=True, fill_alpha=0.12,
        blob_radius=0.22, blob_alpha=0.10,
        show_bonds=True, show_complement=False,
        region_descriptions=LW_DESCS,
        title=f"{ltype.capitalize()} LW  ({n_str})",
        legend_loc="upper right", legend_fontsize=7,
    )

fig_lw.suptitle("Levin–Wen Annular Regions (A, B, C)  —  r_in = 2.0, r_out = 4.5",
                 fontsize=14, y=1.01)
fig_lw.tight_layout()
savefig(fig_lw, "04_levin_wen_all.png")


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Disk regions — all lattice types
# ═══════════════════════════════════════════════════════════════════════════
print("\n═══ 5. Disk regions (all lattices) ═══")

fig_dk, axes_dk = plt.subplots(2, 2, figsize=(14, 13))
axes_dk = axes_dk.ravel()

for idx, ltype in enumerate(LATTICE_TYPES):
    lat = choose_lattice(ltype, dim=2, lx=8, ly=8, bc="pbc")
    disk = lat.regions.get_region("disk", origin=0, radius=3.5)

    lat.plot.regions(
        {"Disk": disk},
        ax=axes_dk[idx],
        blob_radius=0.22, blob_alpha=0.14,
        show_bonds=True, show_complement=False,
        region_descriptions={"Disk": f"r=3.5, {len(disk)} sites"},
        title=f"{ltype.capitalize()} Disk  ({len(disk)} sites)",
        legend_loc="upper right", legend_fontsize=7,
    )

fig_dk.suptitle("Disk Regions  —  origin = site 0,  r = 3.5 (PBC-wrapped distances)",
                 fontsize=14, y=1.01)
fig_dk.tight_layout()
savefig(fig_dk, "05_disk_all.png")


# ═══════════════════════════════════════════════════════════════════════════
#  6.  Half-system cuts — all lattice types
# ═══════════════════════════════════════════════════════════════════════════
print("\n═══ 6. Half-system cuts (all lattices) ═══")

fig_hf, axes_hf = plt.subplots(2, 2, figsize=(14, 13))
axes_hf = axes_hf.ravel()

for idx, ltype in enumerate(LATTICE_TYPES):
    lat = choose_lattice(ltype, dim=2, lx=8, ly=8, bc="pbc")
    half_x = lat.regions.get_region("half_x")
    half_y = lat.regions.get_region("half_y")

    lat.plot.regions(
        {"Half-x": half_x, "Half-y": half_y},
        ax=axes_hf[idx],
        fill=True, fill_alpha=0.08,
        show_bonds=False, show_complement=False,
        region_descriptions={"Half-x": f"x<med ({len(half_x)} sites)",
                             "Half-y": f"y<med ({len(half_y)} sites)"},
        title=f"{ltype.capitalize()} Half-cuts",
        legend_loc="upper right", legend_fontsize=7,
        marker_size=40,
    )

fig_hf.suptitle("Half-System Cuts (x and y directions)", fontsize=14, y=1.01)
fig_hf.tight_layout()
savefig(fig_hf, "06_half_cuts_all.png")


# ═══════════════════════════════════════════════════════════════════════════
#  7.  All lattice structures side-by-side (PBC)
# ═══════════════════════════════════════════════════════════════════════════
print("\n═══ 7. All structures (PBC) ═══")

fig_all, axes_all = plt.subplots(2, 2, figsize=(13, 12))
axes_all = axes_all.ravel()

for idx, ltype in enumerate(LATTICE_TYPES):
    L = 5 if ltype != "square" else 6
    lat = choose_lattice(ltype, dim=2, lx=L, ly=L, bc="pbc")
    lat.plot.structure(
        ax=axes_all[idx],
        show_indices=False,
        show_primitive_cell=True,
        partition_colors=("tab:blue", "tab:orange"),
        title=f"{ltype.capitalize()}  Ns={lat.Ns}  (PBC)",
    )

fig_all.suptitle("Lattice Structures (PBC)", fontsize=14, y=1.01)
fig_all.tight_layout()
savefig(fig_all, "07_all_lattice_structures.png")


# ═══════════════════════════════════════════════════════════════════════════
#  8.  Brillouin zone with high-symmetry points
# ═══════════════════════════════════════════════════════════════════════════
print("\n═══ 8. BZ with high-symmetry points ═══")

fig_bz, axes_bz = plt.subplots(2, 2, figsize=(13, 12))
axes_bz = axes_bz.ravel()

for idx, ltype in enumerate(LATTICE_TYPES):
    L = 10 if ltype in ("square", "triangular") else 8
    lat = choose_lattice(ltype, dim=2, lx=L, ly=L, bc="pbc")
    lat.plot.bz_high_symmetry(
        ax=axes_bz[idx],
        title=f"{ltype.capitalize()} BZ",
    )

fig_bz.suptitle("First Brillouin Zones with High-Symmetry Points", fontsize=14, y=1.01)
fig_bz.tight_layout()
savefig(fig_bz, "08_brillouin_zones.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n✓ All plots saved to {SAVE_DIR}/")
print("  Run with --show to display interactively.")
