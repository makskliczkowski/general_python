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
    "axes.labelsize"                : 12,
    "axes.titlesize"                : 14,
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

SHOW = False
if "--show" in sys.argv:
    SHOW = True

if not SHOW:
    mpl.use("Agg")
    
from    pathlib import Path
import  argparse

#! project import
# Add project root to sys.path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))

try:
    from general_python.lattices import choose_lattice
except ImportError:
    raise ImportError("Could not import choose_lattice from general_python.lattices. Please ensure the module is available.")

ITER        = 0
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "lattice_demo_plots", '2d')


def savefig(fig, name):
    global ITER
    
    name = f"{ITER:03d}_demo_{name}"
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=180)
    
    if SHOW:
        plt.show()
    plt.close(fig)
    
    # increment global iteration counter for unique filenames
    ITER += 1

# --------------------------------------------------------------------------------
#! Main script
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script for lattice sweeps.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    parser.add_argument("--type", type=str, default="square", help="Type of lattice to generate (e.g., square, honeycomb, triangular).")
    parser.add_argument("--lx_min", type=int, default=2, help="Minimum lattice size in x direction.")
    parser.add_argument("--lx_max", type=int, default=4, help="Maximum lattice size in x direction.")
    parser.add_argument("--ly_min", type=int, default=2, help="Minimum lattice size in y direction.")
    parser.add_argument("--ly_max", type=int, default=4, help="Maximum lattice size in y direction.")
    args = parser.parse_args()
    
    # Create directory for saving plots if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Choose the lattice type based on the argument
    for Lx in range(args.lx_min, args.lx_max + 1):
        for Ly in range(args.ly_min, args.ly_max + 1):
            
            # Create the lattice and plot its structure
            lat         = choose_lattice(args.type, lx=Lx, ly=Ly, dim=2)
            fig, ax     = plt.subplots(2, 3, figsize=(24, 12.0))
            ax          = ax.flatten()
            lat.plot.structure(
                ax                  =   ax[0],
                show_indices        =   True,
                show_primitive_cell =   False,
                partition_colors    =   ("tab:blue", "tab:orange"),
                bond_colors         =   {0: "tab:gray", 1: "tab:red", 2: "tab:green"},
                title               =   "Full Structure",
                node_size           =   60,
            )
            
            common_kw = dict(
                fill            = True, 
                fill_alpha      = 0.25,
                blob_radius     = 0.35, 
                blob_alpha      = 0.25,
                marker_size     = 60,
                edge_width      = 2.5,
                show_bonds      = True, 
                show_complement = True,
                show_indices    = True,
                legend_fontsize = 10,
                label_fontsize  = 14,
                legend_loc      = "upper center",
                legend_bbox     = (0.5, -0.15),
            )

            # Plot the half-system size regions
            lat.plot.regions('half-x',  ax=ax[1], **common_kw, title="Half-x")
            lat.plot.regions('half-y',  ax=ax[2], **common_kw, title="Half-y")
            lat.plot.regions('half-xy', ax=ax[3], **common_kw, title="Half-xy")
            lat.plot.regions('half-yx', ax=ax[4], **common_kw, title="Half-yx")
            
            # last will be kitaev-preskill
            regions             = lat.get_region('kitaev', radius=2.0)
            kp_abc              = {k: v for k, v in regions.items() if len(k) == 1}
            lat.plot.regions(kp_abc, ax=ax[5], **common_kw, title="Kitaev-Preskill")
            
            fig.suptitle(f"{args.type.capitalize()} Lattice: Lx={Lx}, Ly={Ly}", fontsize=22, fontweight='bold')
            print(f"Generated {args.type} lattice with Lx={Lx}, Ly={Ly}, Ns={lat.Ns}, n_edges={lat.n_edges}")

            if not args.show:
                savefig(fig, f"lattice_{args.type}_Lx{Lx}_Ly{Ly}.png")
                
# -----------------------------------------------------------------------------
#!End of script
# -----------------------------------------------------------------------------