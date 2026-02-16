#!/usr/bin/env python3
"""
Demo: Reduced Density Matrix (RDM) and Schmidt Decomposition.

This demo showcases:
1.  Computing RDM for contiguous and masked subsystems.
2.  Comparing NumPy and Numba backends.
3.  Verifying results for GHZ and Random states.
4.  Schmidt decomposition and entanglement spectrum.

Usage:
    python demo_density_matrix.py
"""

import  sys, os
import  numpy as np
import  matplotlib.pyplot as plt
from    pathlib import Path

# Add project root to sys.path
_CWD        = Path(__file__).resolve().parent
_QES_ROOT   = _CWD.parents[3]
if str(_QES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QES_ROOT))
    
try:
    from QES.general_python.physics.density_matrix  import rho, rho_spectrum
    from QES.general_python.common.flog             import get_global_logger
except ImportError:
    print("Error: Could not import QES modules.")
    sys.exit(1)

logger      = get_global_logger()
SAVE_DIR    = _CWD / "tmp"

# --------------------------------------------------------------------------------
#! Density matrix and Schmidt decomposition functions
# --------------------------------------------------------------------------------

def test_ghz():
    ''' 
    Test RDM and Schmidt decomposition on the 3-qubit GHZ state.
    '''
    logger.title("Testing RDM and Schmidt decomposition on GHZ state")
    ns      = 3
    # |GHZ> = 1/sqrt(2) (|000> + |111>)
    psi     = np.zeros(2**ns, dtype=complex)
    psi[0]  = 1.0/np.sqrt(2)
    psi[7]  = 1.0/np.sqrt(2)
    
    # RDM of first qubit (site 0)
    # contiguous=True, va=1 means sites [0]
    rho_0   = rho(psi, va=1, ns=ns, contiguous=True)
    logger.info(f"RDM of site 0:\n{rho_0.real}")
    
    # RDM of sites 0 and 2
    rho_02  = rho(psi, va=[0, 2], ns=ns)
    logger.info(f"RDM of sites 0 and 2:\n{rho_02.real}")
        
    # Verifying purity
    purity  = np.trace(rho_0 @ rho_0).real
    logger.info(f"Purity of RDM for site 0: {purity:.4f} (should be 0.5)", color="green", lvl=1)
    
    assert np.isclose(purity, 0.5), "Purity should be 0.5 for a GHZ state."
    
    # Entropy verification
    eigvals = rho_spectrum(rho_0)
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-15)).real
    logger.info(f"Von Neumann entropy of RDM for site 0: {entropy:.4f} (should be ln(2) ~ 0.6931)", color="green", lvl=1)
    assert np.isclose(entropy, np.log(2)), "Entropy should be ln(2) for a GHZ state."
    
def test_bell_state():
    ''' 
    Test RDM and Schmidt decomposition on the 2-qubit Bell state.
    '''
    logger.title("Testing RDM and Schmidt decomposition on Bell state")
    ns      = 2
    # |Bell> = 1/sqrt(2) (|00> + |11>)
    psi     = np.zeros(2**ns, dtype=complex)
    psi[0]  = 1.0/np.sqrt(2)
    psi[3]  = 1.0/np.sqrt(2)
    
    # RDM of first qubit (site 0)
    rho_0   = rho(psi, va=1, ns=ns, contiguous=True)
    logger.info(f"RDM of site 0:\n{rho_0.real}")
    
    # Verifying purity
    purity  = np.trace(rho_0 @ rho_0).real
    logger.info(f"Purity of RDM for site 0: {purity:.4f} (should be 0.5)", color="green", lvl=1)
    assert np.isclose(purity, 0.5), "Purity should be 0.5 for a Bell state."
    
    # Entropy verification
    eigvals = rho_spectrum(rho_0)
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-15)).real
    logger.info(f"Von Neumann entropy of RDM for site 0: {entropy:.4f} (should be ln(2) ~ 0.6931)", color="green", lvl=1)
    assert np.isclose(entropy, np.log(2)), "Entropy should be ln(2) for a Bell state."

def test_single_state():
    ''' 
    Test RDM and Schmidt decomposition on a single-site state.
    '''
    logger.title("Testing RDM and Schmidt decomposition on single-site state")
    ns      = 1
    # |psi> = |0>
    psi     = np.zeros(2**ns, dtype=complex)
    psi[0]  = 1.0
    
    # RDM of the single site
    rho_0   = rho(psi, va=1, ns=ns, contiguous=True)
    logger.info(f"RDM of the single site:\n{rho_0.real}")
    
    # Verifying purity
    purity  = np.trace(rho_0 @ rho_0).real
    logger.info(f"Purity of RDM for the single site: {purity:.4f} (should be 1.0)", color="green", lvl=1)
    
    assert np.isclose(purity, 1.0), "Purity should be 1 for a pure state."
    
    # Entropy verification
    eigvals = rho_spectrum(rho_0)
    entropy = -np.sum(eigvals * np.log(eigvals + 1e-15)).real
    logger.info(f"Von Neumann entropy of RDM for the single site: {entropy:.4f} (should be 0.0)", color="green", lvl=1)
    assert np.isclose(entropy, 0.0), "Entropy should be 0 for a pure state."
    
def test_performance(how_many: int = 5):
    logger.title("Performance comparison for RDM computation")
    import time
    ns      = 14
    psi     = np.random.randn(2**ns) + 1j*np.random.randn(2**ns)
    psi    /= np.linalg.norm(psi)
    
    va      = list(range(ns // 2))
    
    # NumPy
    times   = []
    for _ in range(how_many):
        t0      = time.time()
        rho_np  = rho(psi, va=va, ns=ns)
        t1      = time.time()
        times.append(t1-t0)

    av_time = sum(times) / how_many
    logger.info(f"Average time for RDM computation (NumPy): {av_time*1000:.4f} miliseconds")
    return times

def plot_random_spectrum():
    ''' 
    Plot the entanglement spectrum of a random state for different subsystem sizes.
    
    The entanglement spectrum is defined as xi_i = -ln(lambda_i), where lambda_i are
    the eigenvalues of the reduced density matrix rho_A.
    '''
    logger.title("Plotting entanglement spectrum of a random state")
    ns      = 12
    psi     = np.random.randn(2**ns) + 1j*np.random.randn(2**ns)
    psi    /= np.linalg.norm(psi)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for size_a in [2, 4, 6]:
        rho_A   = rho(psi, va=size_a, ns=ns, contiguous=True)
        spec    = rho_spectrum(rho_A)
        ax.plot(range(len(spec)), -np.log(spec), 'o-', label=f"|A|={size_a}")
        
    ax.set_xlabel("Index")
    ax.set_ylabel(r"Entanglement level $\xi = -\ln \lambda$")
    ax.set_title(f"Random State Entanglement Spectrum (N={ns})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = SAVE_DIR / "entanglement" / "demo_random_spectrum.png"
    fig.savefig(path, dpi=150)

# -------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # make sure save directory exists
    (SAVE_DIR / "entanglement").mkdir(parents=True, exist_ok=True)
    
    try:
        test_ghz()
        test_bell_state()
        test_single_state()
        # performance test...
        test_performance()
        plot_random_spectrum()
        logger.info("Demo completed successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error running demo: {e}")
        raise
    
# -------------------------------------------------------------------------------
#! EOF
# -------------------------------------------------------------------------------