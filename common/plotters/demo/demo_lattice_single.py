"""
Plot single lattice demonstration.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add project root to sys.path
from pathlib import Path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
print(f"Adding project root to sys.path: {_QES_ROOT}")
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))
    
SHOW        = "--show" in sys.argv or os.environ.get("DISPLAY")
if not SHOW:
    plt.switch_backend("Agg")  # Use non-interactive backend for headless environments

from general_python.lattices import choose_lattice, LatticeType

def demo_lattice_single(lx, ly, lz, ltype=LatticeType.SQUARE, bc="obc"):
    """
    Demo for plotting a single lattice configuration.
    """
    
    # Example: Square lattice, 3x3, OBC
    lat             = choose_lattice(ltype, lx=lx, ly=ly, lz=lz, bc=bc)
    
    # Plot the lattice with sites colored by region (if any)
    fig, ax         = plt.subplots(figsize=(5, 5))
    lat.plot.structure(
        ax                  =   ax,
        show_indices        =   True,
        show_primitive_cell =   False,
        partition_colors    =   ("tab:blue", "tab:orange"),
        bond_colors         =   {0: "tab:gray", 1: "tab:red", 2: "tab:green"},
        title               =   f"{str(lat)}",
        node_size           =   60,
    )
    
    # Save or show the plot
    if SHOW:
        plt.show()
    else:
        # Ensure the directory exists
        save_path = _CWD / "tmp" / "lattices" / f"{str(lat)}_demo.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path)
        print(f"Saved demo plot to {save_path}")
        
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo script for plotting a single lattice configuration.")
    parser.add_argument("--show", action="store_true",          help="Show plot interactively.")
    parser.add_argument("--type",   type=str, default="square", help="Type of lattice to generate (e.g., square, honeycomb, triangular).")
    parser.add_argument("--lx",     type=int, default=3,        help="Lattice size in x direction.")
    parser.add_argument("--ly",     type=int, default=3,        help="Lattice size in y direction.")
    parser.add_argument("--lz",     type=int, default=1,        help="Lattice size in z direction (for 3D lattices).")
    parser.add_argument("--bc",     type=str, default="obc",    help="Boundary conditions (e.g., obc, pbc).")
    args = parser.parse_args()
    
    demo_lattice_single(args.lx, args.ly, args.lz, ltype=args.type, bc=args.bc)
    
# ----------------------------------------------------------------------------
#! EOF 
# ----------------------------------------------------------------------------