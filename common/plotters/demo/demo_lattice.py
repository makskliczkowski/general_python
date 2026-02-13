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

# get the QES path from the environment variable if set
gen_python_path = Path(os.environ.get("QES_PYPATH_GEN_PYTHON", "/usr/local/QES/Python/general_python")).resolve()
if not gen_python_path.exists() or not gen_python_path.is_dir():
    raise FileNotFoundError(f"QES QES_PYPATH_GEN_PYTHON '{gen_python_path}' does not exist or is not a directory. "
                            f"If QES is installed, please set the QES_PYPATH_GEN_PYTHON environment variable.")

print(f"Using QES path: {gen_python_path}")
cwd         = Path.cwd()
file_path   = file_path = cwd
mod_path    = file_path.parent.resolve()
lib_path    = gen_python_path.parent
gen_python  = gen_python_path
extra_paths = [file_path, mod_path, lib_path, gen_python]
for p, label in zip(extra_paths, ["file_path", "mod_path", "lib_path", "gen_python"]):
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Required path '{p}' does not exist or is not a directory. "
                                f"Please ensure QES is installed correctly.")
    print(f"-> Adding to sys.path - {label}: {p}")
    sys.path.insert(0, str(p))

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
#  2.  Bond-length validation across all lattices
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
#  3.  Kitaev-Preskill regions — all lattice types
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
            show_bonds              =   True, 
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
            show_bonds          =True,
            show_complement     =False,
            label_offset        =10,
            region_descriptions =LW_DESCS,
            title               =f"{ltype.capitalize()} LW  ({n_str})",
            legend_loc="upper right", legend_fontsize=7,
        )

    savefig(fig_lw, "demo_levin_wen_all.png")


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Disk regions — all lattice types
# ═══════════════════════════════════════════════════════════════════════════

def _disk_regions():
    logger.title("5. Disk regions (r=3.5 from site 0) across all lattice types (PBC-wrapped distances)")

    fig_dk, axes_dk = plt.subplots(2, 2, figsize=(14, 13))
    axes_dk         = axes_dk.ravel()

    for idx, ltype in enumerate(_LATTICE_TYPES):
        lat     = choose_lattice(ltype, dim=2, lx=sizes[ltype][0], ly=sizes[ltype][1], bc=bcs[ltype])
        disk    = lat.regions.get_region("disk", origin=0, radius=3.5)

        lat.plot.regions(
            {"Disk": disk},
            ax=axes_dk[idx],
            blob_radius=0.22, blob_alpha=0.14,
            show_bonds=True, show_complement=False,
            region_descriptions={"Disk": f"r=3.5, {len(disk)} sites"},
            title=f"{ltype.capitalize()} Disk  ({len(disk)} sites)",
            legend_loc="upper right", legend_fontsize=7,
        )

    savefig(fig_dk, "demo_disk_regions_all.png")


# ═══════════════════════════════════════════════════════════════════════════
#  6.  Half-system cuts — all lattice types
# ═══════════════════════════════════════════════════════════════════════════

def _half_system_cuts():
    logger.title("6. Half-system cuts (x and y) across all lattice types (PBC-wrapped distances)")

    fig_hf, axes_hf = plt.subplots(2, 2, figsize=(14, 13))
    axes_hf         = axes_hf.ravel()

    for idx, ltype in enumerate(_LATTICE_TYPES):
        lat         = choose_lattice(ltype, dim=2, lx=8, ly=8, bc="pbc")
        half_x      = lat.regions.get_region("half_x")
        half_y      = lat.regions.get_region("half_y")

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

    savefig(fig_hf, "demo_half_cuts_all.png")

# ═══════════════════════════════════════════════════════════════════════════
#  7.  All lattice structures side-by-side (PBC)
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
#  8.  Brillouin zone with high-symmetry points
# ═══════════════════════════════════════════════════════════════════════════

def _brillouin_zones():
    logger.title("8. First Brillouin zones with high-symmetry points across all lattice types")

    fig_bz, axes_bz = plt.subplots(2, 2, figsize=(13, 12))
    axes_bz         = axes_bz.ravel()

    for idx, ltype in enumerate(_LATTICE_TYPES):
        L = 10 if ltype in ("square", "triangular") else 8
        lat = choose_lattice(ltype, dim=2, lx=L, ly=L, bc="pbc")
        lat.plot.bz_high_symmetry(
            ax=axes_bz[idx],
            title=f"{ltype.capitalize()} BZ",
        )

    savefig(fig_bz, "demo_brillouin_zones.png")

# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════

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
    _brillouin_zones()