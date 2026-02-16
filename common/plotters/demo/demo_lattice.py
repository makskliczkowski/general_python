#!/usr/bin/env python3
"""
Comprehensive lattice region & k-space demo.

Generates publication-quality multi-panel figures for four lattice types
(square, honeycomb, hexagonal / armchair, triangular):

  1. Honeycomb (zig-zag) vs Hexagonal (armchair) structure comparison
  2. Bond-length validation — confirms every NN bond = 1.0
  3. Kitaev-Preskill tripartite sectors A, B, C
  4. Levin-Wen annular regions A (core), B (annulus), C (exterior)
  5. Disk regions centered at site 0
  6. Half-system bipartitions in x and y
  7. Full lattice structures under PBC
  8. First Brillouin zones with high-symmetry paths

Usage:
    python demo_lattice_plots.py           # saves PNGs to demo_plots/
    python demo_lattice_plots.py --show    # also opens interactive windows
"""

import  sys, os
import  numpy as np
import  matplotlib as mpl
import  matplotlib.pyplot as plt

# Set the plotting style to a clean, publication-ready style using the 'science'
try:
    plt.style.use(['science', 'no-latex', 'colors5-light'])
except Exception:
    try:
        plt.style.use(['science', 'no-latex'])
    except Exception:
        # Fallback to default if science styles are missing
        pass
# PRL-style plotting defaults (serif fonts, inward ticks, clean lines)
mpl.rcParams.update({
    # "savefig.dpi"                   : 300,
    "font.family"                   : "serif",
    "font.serif"                    : ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset"              : "stix",
    "axes.linewidth"                : 0.8,
    "axes.labelsize"                : 10,
    "axes.titlesize"                : 10,
    'font.size'                     : 14,
    "xtick.direction"               : "in",
    "ytick.direction"               : "in",
    "xtick.major.size"              : 4,
    "ytick.major.size"              : 4,
    "xtick.minor.size"              : 2,
    "ytick.minor.size"              : 2,
    "xtick.major.width"             : 0.8,
    "ytick.major.width"             : 0.8,
    "xtick.minor.width"             : 0.6,
    "ytick.minor.width"             : 0.6,
    "xtick.top"                     : True,
    "ytick.right"                   : True,
    "legend.frameon"                : False,
    "legend.fontsize"               : 10,
    "lines.linewidth"               : 1.4,
    "lines.markersize"              : 4,
    "axes.grid"                     : False,
    "figure.constrained_layout.use" : True,
} )

if "--show" not in sys.argv:
    mpl.use("Agg")
    
import  matplotlib.pyplot as plt
from    pathlib import Path

#! project import
# Add project root to sys.path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))

# ----------------------------------------------------------------------------

try:
    from general_python.lattices    import choose_lattice
    from general_python.common.flog import get_global_logger
    SAVE_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "lattice_demo_plots")
    SAVE_NAME       = lambda lbl: f"demo_lattice_plots_{lbl}.png"
    SHOW            = "--show" in sys.argv
    ITER            = 0
    _LATTICE_TYPES  = ["square", "honeycomb", "hexagonal", "triangular"]
    logger          = get_global_logger()
    
    sizes           = {
        'square'    : (8, 8),
        'honeycomb' : (6, 6),
        'hexagonal' : (6, 6),
        'triangular': (8, 8),
    }
    bcs             = {
        'square'    : "pbc",
        'honeycomb' : "pbc",
        'hexagonal' : "pbc",
        'triangular': "pbc",
    }    
    
except ImportError as e:
    raise ImportError(f"Failed to import choose_lattice from general_python.lattices. "
                      f"Please ensure QES is installed and the QES_PYPATH_GEN_PYTHON environment variable is set correctly.") from e
    
# ----------------------------------------------------------------------------    

def savefig(fig, name):
    global ITER
    
    name = name.replace("demo_", f"demo_{ITER:02d}_")
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=180)
    logger.info(f"Saved figure: {path}", color="green", lvl=2)
    
    if SHOW:
        plt.show()
    plt.close(fig)
    
    # increment global iteration counter for unique filenames
    ITER += 1

# ═══════════════════════════════════════════════════════════════════════════
# 1. Honeycomb (zig-zag) vs Hexagonal (armchair) structure comparison
# ═══════════════════════════════════════════════════════════════════════════

def _hexagonal_vs_honeycomb_demo():
    logger.title("1. Honeycomb (zig-zag) vs Hexagonal (armchair) structure comparison")
    
    lattice_types = [
        ("honeycomb", "Honeycomb (zig-zag)", "obc"),
        ("hexagonal", "Hexagonal (armchair)", "obc"),
        ("honeycomb", "Honeycomb (zig-zag)", "pbc"),
        ("hexagonal", "Hexagonal (armchair)", "pbc"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), dpi=180)

    for idx, (name, title_lbl, bc) in enumerate([
        ("honeycomb", "Honeycomb (zig-zag)", "obc"),
        ("hexagonal", "Hexagonal (armchair)", "obc"),
        ("honeycomb", "Honeycomb (zig-zag)", "pbc"),
        ("hexagonal", "Hexagonal (armchair)", "pbc"),
    ]):
        lat = choose_lattice(name, dim=2, lx=3, ly=4, bc=bc)
        lat.plot.structure(
            ax                  =   axes[idx//2, idx%2],
            show_indices        =   True,
            show_primitive_cell =   False,
            partition_colors    =   ("tab:blue", "tab:orange"),
            bond_colors         =   {0: "tab:gray", 1: "tab:red", 2: "tab:green"},
            title               =   f"{title_lbl}  ({lat.Ns} sites, {lat.n_edges} bonds)",
        )
    savefig(fig, "demo_lattice_structure_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
#  2. Bond-length validation across all lattices
# ═══════════════════════════════════════════════════════════════════════════

def _bond_length_check():
    logger.title("2. Bond-length validation across all lattice types")

    for ltype in _LATTICE_TYPES:
        lat     = choose_lattice(ltype, dim=2, lx=6, ly=6, bc="obc")
        bonds   = lat.edges()
        if not bonds:
            logger.info(f"{ltype:12s}  ** no bonds found **", color="red", lvl=1)
            continue

        dists               = [lat.distance(i, j) for (i, j) in bonds]
        dmin, dmax, dmean   = min(dists), max(dists), np.mean(dists)
        status              = "OK" if abs(dmax - dmin) < 1e-6 else f"SPREAD {dmax-dmin:.4f}"
        logger.info(f"{ltype:12s}  bonds={len(bonds):4d} dist=[{dmin:.4f}, {dmax:.4f}]  mean={dmean:.4f}  {status}", color="green" if status == "OK" else "red", lvl=1)

# ═══════════════════════════════════════════════════════════════════════════
#  3. Kitaev-Preskill regions — all lattice types
# ═══════════════════════════════════════════════════════════════════════════

def _kitaev_preskill_sectors(radius=3.0):
    logger.title("3. Kitaev-Preskill sectors (A, B, C) across all lattice types")

    KP_DESCS        = {
        "A": r"sector $-\Pi..-\Pi/3$", 
        "B": r"sector $-\Pi/3..+\Pi/3$", 
        "C": r"sector $+\Pi/3..+\Pi$"
        }

    fig_kp, axes_kp = plt.subplots(2, 2, figsize=(14, 13))
    axes_kp         = axes_kp.ravel()
    
    for idx, ltype in enumerate(_LATTICE_TYPES):
        lat         = choose_lattice(ltype, dim=2, lx=sizes[ltype][0], ly=sizes[ltype][1], bc=bcs[ltype])
        kp          = lat.regions.get_region("kitaev_preskill", radius=radius)
        kp_abc      = {k: v for k, v in kp.items() if len(k) == 1}
        n_str       = ", ".join(f"{k}:{len(v)}" for k, v in kp_abc.items())
        
        lat.plot.regions(
            kp_abc,
            ax                      =   axes_kp[idx],
            fill                    =   True, 
            fill_alpha              =   0.05,
            blob_radius             =   0.22, blob_alpha=0.10,
            show_bonds              =   False, 
            show_complement         =   False,
            region_descriptions     =   KP_DESCS,
            title                   =   f"{ltype.capitalize()} KP ({n_str})",
            legend_loc              =   "lower right", 
            legend_fontsize         =   7,
            legend_bbox             =   (0.95, 0.05),
        )

    savefig(fig_kp, "demo_kitaev_preskill_all.png")

# ═══════════════════════════════════════════════════════════════════════════
#  4.  Levin-Wen annular regions — all lattice types
# ═══════════════════════════════════════════════════════════════════════════

def _levin_wen_regions(inner_radius=2.0, outer_radius=3.5):
    logger.title("4. Levin-Wen annular regions (A: core, B: annulus, C: exterior) across all lattice types")
    LW_DESCS        = {
        "A": fr"inner disk $r<{inner_radius}$", 
        "B": fr"annulus ${inner_radius}\leq r\leq {outer_radius}$", 
        "C": fr"outer $r>{outer_radius}$"
        }

    fig_lw, axes_lw = plt.subplots(2, 2, figsize=(14, 13))
    axes_lw         = axes_lw.ravel()

    for idx, ltype in enumerate(_LATTICE_TYPES):
        lat         = choose_lattice(ltype, dim=2, lx=sizes[ltype][0], ly=sizes[ltype][1], bc=bcs[ltype])
        lw          = lat.regions.get_region("levin_wen", inner_radius=inner_radius, outer_radius=outer_radius)
        lw_abc      = {k: v for k, v in lw.items() if len(k) == 1}

        n_str   = ", ".join(f"{k}:{len(v)}" for k, v in lw_abc.items())
        lat.plot.regions(
            lw_abc,
            ax                  =axes_lw[idx],
            fill                =True, 
            fill_alpha          =0.12,
            blob_radius         =0.22,
            blob_alpha          =0.10,
            show_bonds          =False,
            show_complement     =False,
            label_offset        =1.5,
            region_descriptions =LW_DESCS,
            title               =f"{ltype.capitalize()} LW  ({n_str})",
            legend_loc="upper right", legend_fontsize=7,
        )

    savefig(fig_lw, "demo_levin_wen_all.png")


# ═══════════════════════════════════════════════════════════════════════════
#  5. Disk regions — all lattice types
# ═══════════════════════════════════════════════════════════════════════════

def _disk_regions():
    logger.title("5. Disk regions (r=3.5 from site 0) across all lattice types (PBC-wrapped distances)")

    fig_dk, axes_dk = plt.subplots(2, 2, figsize=(14, 13))
    axes_dk         = axes_dk.ravel()

    for idx, ltype in enumerate(_LATTICE_TYPES):
        lat         = choose_lattice(ltype, dim=2, lx=sizes[ltype][0], ly=sizes[ltype][1], bc=bcs[ltype])
        origin      = lat.get_coordinates(lat.ns//4)  # choose a site near the center for better visualization
        disk        = lat.regions.get_region("disk", origin=origin, radius=1.5)

        lat.plot.regions(
            {"Disk": disk},
            ax=axes_dk[idx],
            blob_radius=0.22, blob_alpha=0.14,
            origin=origin,
            show_bonds=False, show_complement=False,
            region_descriptions={"Disk": f"r=3.5, {len(disk)} sites"},
            title=f"{ltype.capitalize()} Disk  ({len(disk)} sites)",
            legend_loc="upper right", legend_fontsize=7,
        )

    savefig(fig_dk, "demo_disk_regions_all.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Half-system cuts — all lattice types
# ═══════════════════════════════════════════════════════════════════════════

def _half_system_cuts():
    logger.title("6. Half-system cuts (x, y, xy, yx) across all lattice types")

    fig_hf, axes_hf = plt.subplots(4, 4, figsize=(20, 18))
    cuts            = ["half_x", "half_y", "half_xy", "half_yx"]
    cut_labels      = ["half-x", "half-y", "half-xy", "half-yx"]

    for row, ltype in enumerate(_LATTICE_TYPES):
        lat = choose_lattice(ltype, dim=2, lx=8, ly=8, bc="pbc")
        
        for col, (cut_key, cut_lbl) in enumerate(zip(cuts, cut_labels)):
            ax      = axes_hf[row, col]
            indices = lat.regions.get_region(cut_key)
            
            lat.plot.regions(
                {cut_lbl: indices},
                ax              =ax,
                fill            =True, 
                fill_alpha      =0.1,
                show_bonds      =False, 
                show_complement =True,
                title           =f"{ltype.capitalize()} {cut_lbl}" if row == 0 else cut_lbl,
                legend_loc      ="lower center",
                legend_fontsize =8,
                legend_bbox     =(0.5, -0.15),
                marker_size     =30,
            )

    savefig(fig_hf, "demo_half_cuts_all.png")

# ═══════════════════════════════════════════════════════════════════════════
#  7. All lattice structures side-by-side (PBC)
# ═══════════════════════════════════════════════════════════════════════════

def _all_structures():
    logger.title("7. All lattice structures side-by-side (PBC) across all lattice types")

    fig_all, axes_all   = plt.subplots(2, 2, figsize=(13, 12))
    axes_all            = axes_all.ravel()

    for idx, ltype in enumerate(_LATTICE_TYPES):
        L       = 5 if ltype != "square" else 6
        lat     = choose_lattice(ltype, dim=2, lx=L, ly=L, bc="pbc")
        lat.plot.structure(
            ax=axes_all[idx],
            show_indices=False,
            show_primitive_cell=True,
            partition_colors=("tab:blue", "tab:orange"),
            title=f"{ltype.capitalize()}  Ns={lat.Ns}  (PBC)",
        )

    savefig(fig_all, "demo_all_structures.png")

# ═══════════════════════════════════════════════════════════════════════════

#  8.  Brillouin zone and High-symmetry Path Analysis

# ═══════════════════════════════════════════════════════════════════════════



def _bz_path_comprehensive_demo():

    logger.title("8. Brillouin Zones and Path Coordinates Analysis")

    

    from general_python.lattices.tools.lattice_kspace import brillouin_zone_path

    

    # 4 rows (lattices) x 2 columns (BZ and Coordinates)

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))

    

    for idx, ltype in enumerate(_LATTICE_TYPES):

        lat = choose_lattice(ltype, dim=2, lx=8, ly=8, bc="pbc")

        

        # Panel A: 2D BZ plot

        ax_bz = axes[idx, 0]

        lat.plot.bz_high_symmetry(ax=ax_bz, title=f"{ltype.capitalize()} BZ & Path", show_kpoints=True)

        

        # Panel B: Path coordinates components

        ax_path = axes[idx, 1]

        hs = lat.high_symmetry_points()

        if hs is None: continue

        

        k_path, k_dist, labels, k_frac = brillouin_zone_path(lat, hs.get_default_path_points(), points_per_seg=100)

        

        # Plot kx and ky components

        ax_path.plot(k_dist, k_path[:, 0], label=r'$k_x$ (Cartesian)', lw=2.5, color='C0')

        ax_path.plot(k_dist, k_path[:, 1], label=r'$k_y$ (Cartesian)', lw=2.5, color='C2')

        

        # Highlight and annotate high-symmetry points

                for i_pt, lbl in labels:

                    idx_p = min(i_pt, len(k_path)-1)

                    dist = k_dist[idx_p]

                    kx, ky = k_path[idx_p][:2]

                    

                    # Vertical reference line

            ax_path.axvline(dist, color='gray', linestyle='--', alpha=0.3)

            # Dots at the actual component values

            ax_path.scatter([dist, dist], [kx, ky], color=['C0', 'C2'], s=50, zorder=5, edgecolors='white')

            

            # Coordinate text annotation (kx, ky)

            coord_str = f"({kx:.2f}, {ky:.2f})"

            y_max = max(kx, ky)

            ax_path.annotate(coord_str, xy=(dist, y_max), xytext=(3, 10),

                           textcoords='offset points', rotation=45, fontsize=9,

                           fontweight='bold', ha='left', va='bottom')



        ax_path.set_xticks([k_dist[min(i, len(k_dist)-1)] for i, _ in labels])

        ax_path.set_xticklabels([lbl for _, lbl in labels], fontsize=13, fontweight='bold')

        ax_path.set_title(f"{ltype.capitalize()} Momentum Components vs. Path", fontsize=15)

        ax_path.set_ylabel("Momentum (1/A)", fontsize=12)

        ax_path.legend(loc='lower left', frameon=True, framealpha=0.9)

        ax_path.grid(True, alpha=0.15)

        

    fig.suptitle("Lattice K-Space Analysis: Brillouin Zones and High-Symmetry Path Components", 

                 fontsize=22, fontweight='bold', y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    savefig(fig, "demo_bz_path_analysis_comprehensive.png")



# ═══════════════════════════════════════════════════════════════════════════
#  9. Extended Brillouin zones
# ═══════════════════════════════════════════════════════════════════════════

def _extended_brillouin_zones():
    logger.title("9. Extended Brillouin zones with background copies across all lattice types")
    fig_bz, axes_bz = plt.subplots(2, 2, figsize=(13, 12))
    axes_bz         = axes_bz.ravel()
    
    for idx, ltype in enumerate(_LATTICE_TYPES):
        L   = 6 if ltype in ("square", "triangular") else 4
        lat = choose_lattice(ltype, dim=2, lx=L, ly=L, bc="pbc")
        lat.plot.bz_high_symmetry(
            ax=axes_bz[idx],
            extend=True,
            nx=1, ny=1,
            show_background_bz=True,
            title=f"{ltype.capitalize()} Extended BZ",
        )
    savefig(fig_bz, "demo_extended_brillouin_zones.png")

# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ══==═══════════════════════════════════════════════════════════════════════



if __name__ == "__main__":
    logger.title("Lattice Visualization Demo - Summary of Plots")
    logger.info("This demo generates a series of publication-quality plots showcasing various lattice structures, regions, and Brillouin zones across multiple lattice types (square, honeycomb, hexagonal, triangular). Each plot is saved to the 'tmp/lattice_demo_plots/' directory with descriptive filenames. Use the '--show' flag to also display the plots interactively.", color="cyan", lvl=0)
    os.makedirs(SAVE_DIR, exist_ok=True)
    _bond_length_check()
    _hexagonal_vs_honeycomb_demo()
    _kitaev_preskill_sectors()
    _levin_wen_regions()
    _disk_regions()
    _half_system_cuts()
    _all_structures()
    _bz_path_comprehensive_demo()
    _extended_brillouin_zones()
    
# --------------------------------------------------------------------------
#! EOF
# --------------------------------------------------------------------------