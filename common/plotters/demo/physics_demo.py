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
import  matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
    
import  matplotlib.pyplot as plt

from    general_python.lattices import choose_lattice
from    general_python.common.plotters.plot_helpers import plot_kspace_intensity, plot_static_structure_factor

OUTDIR = os.path.join(os.path.dirname(__file__), 'demo_plots_physics')
os.makedirs(OUTDIR, exist_ok=True)

# small helper functions

def build_real_positions(lat):
    ''' Build real-space positions for a 2D lattice. Tries multiple approaches for robustness. '''
    
    if hasattr(lat, 'positions'):
        pos = np.asarray(lat.positions)
        if pos.ndim == 2:
            return pos[:, :2]
    # fallback: reconstruct from lattice vectors for simple 2D cell
    coords = []
    for i in range(lat.Ns):
        try:
            # try attribute mapping
            coords.append(lat.get_position(i)[:2])
        except Exception:
            break
    if coords:
        return np.asarray(coords)
    # last resort: approximate grid from Lx,Ly
    pts = []
    for y in range(lat.Ly):
        for x in range(lat.Lx):
            pts.append(x * lat.a * np.array([1.0, 0.0]) + y * (lat.a * np.array([0.5, np.sqrt(3)/2])))
    return np.asarray(pts)


def make_kmesh_from_reciprocal(b1, b2, nx=80, ny=80):
    # create coefficients alpha,beta in [-0.5,0.5)
    a = np.linspace(-0.5, 0.5, nx, endpoint=False)
    b = np.linspace(-0.5, 0.5, ny, endpoint=False)
    A, B = np.meshgrid(a, b, indexing='ij')
    ks = (A.reshape(-1,1) * b1.reshape(1,2) + B.reshape(-1,1) * b2.reshape(1,2)).reshape(-1,2)
    return ks


def compute_static_S_k(positions, correlator_func, kpoints):
    # positions: (Ns,2), correlator_func(R) -> scalar
    Ns = positions.shape[0]
    # precompute R_ij and correlations
    Rij = positions[:, None, :] - positions[None, :, :]  # (Ns,Ns,2)
    Rnorm = np.linalg.norm(Rij, axis=-1)
    Cij = correlator_func(Rnorm)
    Sk = np.zeros(kpoints.shape[0], dtype=float)
    for ik,k in enumerate(kpoints):
        phase = np.exp(1j * (Rij @ k))  # (Ns,Ns)
        val = np.sum(Cij * phase)
        Sk[ik] = val.real
    return Sk


def polynomial_correlator(alpha=1.5, oscillate=False):
    def f(R):
        # R is array
        r = np.asarray(R)
        out = 1.0 / (1.0 + r**alpha)
        if oscillate:
            # staggered sign for AF-like correlations
            out = out * np.cos(np.pi * r)
        return out
    return f


def dispersion_simple(k):
    # simple nearest-neighbor like dispersion on square lattice
    kx, ky = k[...,0], k[...,1]
    return 2.0 * (2.0 - np.cos(kx) - np.cos(ky))


def lorentzian(omega, omega0, gamma=0.2):
    return (1.0/np.pi) * (0.5*gamma) / ((omega - omega0)**2 + (0.5*gamma)**2)


# Demo loop
LATTICE_TYPES = ['square', 'honeycomb', 'hexagonal', 'triangular']
# use slightly smaller systems to keep compute and visual clarity
for ltype in LATTICE_TYPES:
    print(f"Processing {ltype}...")
    # use smaller systems for quick demos
    lat = choose_lattice(ltype, dim=2, lx=3, ly=3, bc='pbc')
    positions = build_real_positions(lat)  # (Ns,2)
    Ns = positions.shape[0]
    # reciprocal vectors
    try:
        b1 = np.asarray(lat.k1, float).ravel()[:2]
        b2 = np.asarray(lat.k2, float).ravel()[:2]
    except Exception:
        # fallback to simple square reciprocal vectors
        b1 = np.array([2*np.pi/lat.a, 0.0])
        b2 = np.array([0.0, 2*np.pi/lat.a])

    # build k-mesh (moderate resolution for demo)
    kmesh = make_kmesh_from_reciprocal(b1, b2, nx=16, ny=16)

    # correlator: polynomial decay with exponent 1.5
    corr = polynomial_correlator(alpha=1.5, oscillate=True)
    Sk = compute_static_S_k(positions, corr, kmesh)

    # plot static S(k) intensity
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    plot_kspace_intensity(ax, kmesh, Sk, lattice=lat, show_extended_bz=True, bz_copies=1)
    ax.set_title(f"{ltype.capitalize()} Static S(k)")
    out = os.path.join(OUTDIR, f"static_Sk_{ltype}.png")
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # build high-symmetry path
    hs = lat.high_symmetry_points()
    # choose canonical path depending on lattice
    if ltype == 'square':
        seq = ['G', 'M', 'X', 'G']
    else:
        seq = ['G', 'M', 'K', 'G']
    # map names: try common alternatives
    hs_map = {k.upper(): np.asarray(v)[:2] for k,v in (hs.items() if isinstance(hs, dict) else [])}
    path_nodes = [hs_map.get(n, None) for n in seq]
    path_nodes = [p for p in path_nodes if p is not None]
    # make linear path
    kpath = []
    if len(path_nodes) >= 2:
        for a,b in zip(path_nodes[:-1], path_nodes[1:]):
            for t in np.linspace(0,1,120,endpoint=False):
                kpath.append((1-t)*a + t*b)
        kpath = np.vstack(kpath)
    else:
        # fallback: sample a line in kx
        kpath = np.linspace(-np.pi, np.pi, 240).reshape(-1,1) * np.array([1.0,0.0])

    # compute S(k) along path
    Sk_path = compute_static_S_k(positions, corr, kpath)

    # plot S(k) along path (as 1D intensity)
    fig, ax = plt.subplots(1,1, figsize=(8,3))
    # use simple line plot for clarity
    dist_along = np.arange(len(Sk_path))
    ax.plot(dist_along, Sk_path, '-', color='navy')
    ax.set_xlabel('k-path index')
    ax.set_ylabel('S(k)')
    ax.set_title(f"{ltype.capitalize()} S(k) along HS path")
    outp = os.path.join(OUTDIR, f"static_Sk_path_{ltype}.png")
    fig.savefig(outp, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Dynamic structure factor along the path
    # define omega grid
    omega = np.linspace(0.0, 8.0, 200)
    # dispersion: use simple lattice-dependent function
    def omega0_of_k(k):
        # Prefer lattice-provided dispersion when available
        try:
            w = lat.dispersion(k)
            # ensure scalar
            w = np.asarray(w)
            if w.shape == ():
                return float(w)
            # if vector returned, take first element
            return float(w)
        except Exception:
            kx, ky = k[0], k[1]
            if ltype == 'honeycomb' or ltype == 'hexagonal':
                return 1.5 * (2 - np.cos(kx) - np.cos(ky))
            elif ltype == 'triangular':
                return 1.8 * (1 - 0.5*np.cos(kx) - 0.5*np.cos(ky) - 0.5*np.cos(kx-ky))
            else:
                return 2.0 * (2 - np.cos(kx) - np.cos(ky))

    Skw = np.zeros((kpath.shape[0], omega.size), dtype=float)
    for ik, k in enumerate(kpath):
        A_k = max(1e-6, Sk_path[ik])
        w0 = omega0_of_k(k)
        Skw[ik,:] = A_k * lorentzian(omega, w0, gamma=0.18)

    # Plot dynamic S(k,omega) as a color map with k on x-axis and omega on y-axis
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    k_vals = np.linspace(0, len(Sk_path)-1, len(Sk_path))
    plot_static_structure_factor(ax, k_vals, omega, Skw, mode='kpath', path_info={'label_positions':[0, len(k_vals)//3, 2*len(k_vals)//3, len(k_vals)-1], 'label_texts':seq, 'description':'Dynamic S(k,ω) along HS path'})
    ax.set_title(f"{ltype.capitalize()} Dynamic S(k, ω)")
    outd = os.path.join(OUTDIR, f"dynamic_Skomega_{ltype}.png")
    fig.savefig(outd, dpi=180, bbox_inches='tight')
    plt.close(fig)

print(f"Demo outputs written to {OUTDIR}")
