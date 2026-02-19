"""
Physics-relevant demo: static and dynamic structure factors on lattices
- Builds a polynomial-decay correlator in real space and computes S(k) on a k-mesh
- Builds a simple dispersion and Lorentzian-lineshape dynamic S(k,omega) along HS path
- Uses lattice reciprocal vectors so k-mesh is physically correct for each lattice

Run as module from project root:
    python -m general_python.common.plotters.demo_physics_demo

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
from    typing import Optional, Tuple, List, Callable
from    pathlib import Path

#! project import
# Add project root to sys.path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))
    
try:
    from    general_python.lattices                         import choose_lattice, Lattice
    from    general_python.common.plotters.plot_helpers     import plot_kspace_intensity, plot_static_structure_factor
    from    general_python.lattices.tools.lattice_kspace    import extend_kspace_data
except ImportError:
    import  traceback
    print("Error importing lattice modules. Make sure to run this from the project root and that the general_python package is available.")
    traceback.print_exc()
    raise

# ------------------------------------------
# small helper functions
# ------------------------------------------

def build_real_positions(lat, dim: int = 2):
    ''' Build real-space positions for a 1D/2D lattice. Tries multiple approaches for robustness. '''
    
    if hasattr(lat, 'positions'):
        pos = np.asarray(lat.positions)
        if pos.shape[1] >= dim:
            return pos[:,:dim]

    # try method for 2D lattices with get_position(i)
    coords = []
    for i in range(lat.Ns):
        try:
            # try attribute mapping
            coords.append(lat.get_position(i)[:dim])
        except Exception:
            break
        
    # if we got valid coords, return them
    if coords:
        return np.asarray(coords)
    
    # last resort: approximate grid from Lx,Ly
    pts = []
    for y in range(lat.Ly):
        for x in range(lat.Lx):
            pts.append(x * lat.a * np.array([1.0, 0.0]) + y * (lat.a * np.array([0.5, np.sqrt(3)/2])))
    return np.asarray(pts)[:,:dim]

# ------------------------------------------

def make_kmesh_from_reciprocal(lattice: Lattice, n_grid: int = 16):
    ''' 
    Build a k-mesh in the 2D Brillouin zone defined by reciprocal vectors b1,b2.
    The mesh is a regular grid of points in the unit cell defined by b1,b2, with nx points along b1 and ny points along b2. The points are generated as linear
    combinations of b1 and b2.
    '''
    k_cart  = np.asarray(lattice.kvectors, float)
    r_cart  = np.asarray(lattice.rvectors, float)
    
    k2      = k_cart[:, 2]
    r2      = r_cart[:, 2]
    
    phase   = np.exp(-1j * (k2 @ r2.T))
    phase_c = np.conj(phase)

    KX, KY  = np.meshgrid(np.linspace(-np.pi * 2, np.pi * 2, n_grid), np.linspace(-np.pi * 2, np.pi * 2, n_grid))
    
    return (KX, KY), (phase, phase_c), (k_cart, r_cart)
    
# ------------------------------------------
# Computation    
# ------------------------------------------
    
def compute_static_structure_factor(positions, correlator_func: Callable[[np.ndarray], np.ndarray], kpoints: np.ndarray, extend: bool = True, k1=None, k2=None) -> np.ndarray:
    ''' 
    Compute static structure factor S(k) for a set of k-points given real-space positions and a correlator function. 
    This is a simple implementation that 
    
    Parameters:
    -----------
    positions: 
        (Ns,2) array of real-space positions of the lattice sites
    correlator_func: 
        function that takes an array of distances R and returns the corresponding correlation values
    kpoints: 
        (Nk,2) array of k-points where S(k) should be evaluated
    '''
    # positions: (Ns,2), correlator_func(R) -> scalar
    Ns      = positions.shape[0]
    
    # precompute R_ij and correlationsaa
    Rij     = positions[:, None, :] - positions[None, :, :]  # (Ns,Ns,2)a
    Rnorm   = np.linalg.norm(Rij, axis=-1)
    Cij     = correlator_func(Rnorm)
    Sk      = np.zeros(kpoints.shape[0], dtype=float)
        
    for ik, k in enumerate(kpoints):
        phase   = np.exp(1j * (Rij @ k))  # (Ns,Ns)
        val     = np.sum(Cij * phase)
        Sk[ik]  = val.real
        
    return Sk

# ---------------------------------------------

class Correlators:

    @staticmethod
    def polynomial_correlator(alpha=1.5, oscillate=False):
        ''' Generate a correlator function that decays as 1/(1+R^alpha) with optional oscillations. The function returned takes an array of distances R and returns the corresponding correlation values.'''
        
        def f(R):
            # R is array
            r   = np.asarray(R)
            out = 1.0 / (1.0 + r**alpha)
            if oscillate:
                # staggered sign for AF-like correlations
                out = out * np.cos(np.pi * r)
            return out
        return f

    @staticmethod
    def exponential_correlator(xi=1.0):
        ''' Generate a correlator function that decays exponentially as exp(-R/xi). The function returned takes an array of distances R and returns the corresponding correlation values.'''
        
        def f(R):
            r = np.asarray(R)
            return np.exp(-r/xi)
        return f
    
    @staticmethod
    def gaussian_correlator(xi=1.0):
        ''' Generate a correlator function that decays as a Gaussian exp(-R^2/(2*xi^2)). The function returned takes an array of distances R and returns the corresponding correlation values.'''
        
        def f(R):
            r = np.asarray(R)
            return np.exp(-r**2/(2*xi**2))
        return f
    
    @staticmethod
    def oscillatory_correlator(k0=1.0, xi=1.0):
        ''' Generate a correlator function that oscillates with wavevector k0 and decays exponentially with correlation length xi. The function returned takes an array of distances R and returns the corresponding correlation values.'''
        
        def f(R):
            r = np.asarray(R)
            return np.cos(k0 * r) * np.exp(-r/xi)
        return f
    
    @staticmethod
    def correlator(R, type='polynomial', **params):
        if type == 'polynomial':
            return Correlators.polynomial_correlator(**params)(R)
        elif type == 'exponential':
            return Correlators.exponential_correlator(**params)(R)
        elif type == 'gaussian':
            return Correlators.gaussian_correlator(**params)(R)
        elif type == 'oscillatory':
            return Correlators.oscillatory_correlator(**params)(R)
        else:
            raise ValueError(f"Unknown correlator type: {type}")
    
class Dispersions:
    
    @staticmethod
    def lorentzian(omega, omega0, gamma=0.2):
        return (1.0/np.pi) * (0.5*gamma) / ((omega - omega0)**2 + (0.5*gamma)**2)
    
    @staticmethod
    def gaussian(omega, omega0, sigma=0.3):
        return (1.0/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((omega - omega0)/sigma)**2)
    
    @staticmethod
    def delta(omega, omega0):
        # approximate delta function with a narrow Gaussian
        sigma = 0.1
        return (1.0/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((omega - omega0)/sigma)**2)

    @staticmethod
    def broadening(omega, omega0, gamma=0.2, type='lorentzian'):
        if type == 'lorentzian':
            return Dispersions.lorentzian(omega, omega0, gamma)
        elif type == 'gaussian':
            return Dispersions.gaussian(omega, omega0, sigma=gamma)
        elif type == 'delta':
            return Dispersions.delta(omega, omega0)
        else:
            raise ValueError(f"Unknown broadening type: {type}")

# ---------------------------------------------
#! TEST FUNCTIONS
# ---------------------------------------------

def lattice_loop(lattice_types: List[str] = ['square', 'honeycomb', 'hexagonal', 'triangular'], 
                # lattice settings
                lx: int = 3, ly: int = 3, bc: str = 'pbc', 
                # correlator settings
                correlator_type: str = 'polynomial', correlator_params: dict = {'alpha': 1.5, 'oscillate': True},
                # dispersion settings
                broadening: float = 0.2, broadening_type: str = 'lorentzian',
                # k-mesh settings
                extend: bool = True):

    OUT_DIR = Path(file_path) / 'tmp'
    OUT_DIR.mkdir(exist_ok=True)
    OUT_NAME = lambda ltype: OUT_DIR / f'demo_kspace_{ltype}.png'

    # use slightly smaller systems to keep compute and visual clarity
    for ltype in lattice_types:
        print(f"Processing {ltype}...")
        
        # use smaller systems for quick demos
        lat         = choose_lattice(ltype, dim=2, lx=lx, ly=ly, bc=bc) # build lattice
        positions   = build_real_positions(lat)                         # (Ns,2)           
        Ns          = lat.ns
        
        # reciprocal vectors
        try:
            b1      = np.asarray(lat.k1, float).ravel()[:2]
            b2      = np.asarray(lat.k2, float).ravel()[:2]
        except Exception:
            raise Exception(f"Could not extract reciprocal vectors for lattice type {ltype}. Make sure the lattice class has k1 and k2 attributes that can be converted to 2D vectors.")

        # correlator
        corr        = Correlators.correlator(0.0, type=correlator_type, **correlator_params)  # get correlator function with specified parameters
        
        #! 1. Static structure factor S(k)
        Sk          = compute_static_structure_factor(positions, corr, lat.kvectors[:, :2], k1=b1, k2=b2, extend=False)
        fig, ax     = plt.subplots(1,1, figsize=(6,6))
        
        #? a) plot S(k) as intensity in k-space
        plot_kspace_intensity(ax, lat.kvectors[:, :2], Sk, lattice=lat, show_extended_bz=extend, bz_copies=1)        
        plt.savefig(OUT_NAME(ltype).with_name(OUT_NAME(ltype).stem + '_static.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # build high-symmetry path
        hs      = lat.high_symmetry_points()
        # choose canonical path depending on lattice
        if ltype == 'square':
            seq = ['G', 'M', 'X', 'G']
        else:
            seq = ['G', 'M', 'K', 'G']

        # map names: try common alternatives
        hs_map      = {k.upper(): np.asarray(v)[:2] for k,v in (hs.items() if isinstance(hs, dict) else [])}
        path_nodes  = [hs_map.get(n, None) for n in seq]
        path_nodes  = [p for p in path_nodes if p is not None]
        
        # make linear path between nodes - if we have at least 2 valid nodes, otherwise fallback to a simple line in kx
        kpath = []
        if len(path_nodes) >= 2:
            for a,b in zip(path_nodes[:-1], path_nodes[1:]):
                for t in np.linspace(0,1,120,endpoint=False):
                    kpath.append((1-t)*a + t*b)
            kpath   = np.vstack(kpath)
        else:
            kpath   = np.linspace(-np.pi, np.pi, 240).reshape(-1,1) * np.array([1.0,0.0])

        # compute S(k) along path
        Sk_path     = compute_static_structure_factor(positions, corr, kpath, k1=b1, k2=b2, extend=False)
        
        #? b) plot S(k) along the path as a 1D intensity plot
        # plot S(k) along path (as 1D intensity)
        fig, ax     = plt.subplots(1,1, figsize=(8,3))
        dist_along  = np.arange(len(Sk_path))

        ax.plot(dist_along, Sk_path, '-', color='navy')
        ax.set_xlabel('k-path index')
        ax.set_ylabel('S(k)')
        ax.set_title(f"{ltype.capitalize()} S(k) along HS path")
        plt.savefig(OUT_NAME(ltype).with_name(OUT_NAME(ltype).stem + '_path.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

        #! Dynamic structure factor along the path
        # define omega grid
        omega       = np.linspace(0.0, 8.0, 200)

        def omega0_of_k(k):
            w = lat.dispersion(k)
            w = np.asarray(w)
            if w.shape == ():
                return float(w)
            return float(w[0])
        
        #? c) build S(k,omega) along path using a simple Lorentzian lineshape centered at the dispersion omega0(k) with amplitude proportional to S(k)
                
        Skw     = np.zeros((kpath.shape[0], omega.size), dtype=float)
        for ik, k in enumerate(kpath):
            A_k         = max(1e-6, Sk_path[ik])
            w0          = omega0_of_k(k)
            Skw[ik,:]   = A_k * Dispersions.broadening(omega, w0, gamma=broadening, type=broadening_type)

        # Plot dynamic S(k,omega) as a color map with k on x-axis and omega on y-axis
        fig, ax     = plt.subplots(1,1, figsize=(10,4))
        k_vals      = np.linspace(0, len(Sk_path)-1, len(Sk_path))
        plot_static_structure_factor(ax, k_vals, omega, Skw, mode='kpath', path_info={'label_positions':[0, len(k_vals)//3, 2*len(k_vals)//3, len(k_vals)-1], 'label_texts':seq, 'description':'Dynamic S(k,ω) along HS path'})
        plt.savefig(OUT_NAME(ltype).with_name(OUT_NAME(ltype).stem + '_dynamic.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        

# ---------------------------------------------

if __name__ == "__main__":
    lattice_loop()
    