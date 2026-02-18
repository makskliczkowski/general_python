"""
Random Matrix Theory and specialized sampling utilities.

This module complements `algebra.ran_wrapper` by providing specific ensembles
like Circular Unitary Ensemble (CUE) matrices via QR decomposition.

Input/Output Contracts
----------------------
- `CUE_QR`: Returns a unitary complex matrix of shape (n, n).

Numerical Stability
-------------------
The QR method for CUE is generally stable but phase adjustment is needed
for true Haar measure compliance (controlled by `simple=False`).
"""

import numpy as np

##################################### RANDOM #####################################

def CUE_QR( n       :   int,
            simple  =   True,
            rng     =   None):
    """
    Create the CUE matrix using QR decomposition.
    - n     : size of the matrix (n X n)
    - simple: use the straightforward method
    """
    if rng is None:
        rng = np.random.default_rng()
    # Complex Ginibre matrix with i.i.d. N(0, 1/2) entries for real and imaginary parts
    x       =   rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    x       /=  np.sqrt(2)
    Q, R    =   np.linalg.qr(x)
    if not simple:
        # Adjust phases to ensure Haar measure on U(n)
        d       =   np.diagonal(R)
        ph      =   d / np.abs(d)
        Q       =   Q @ np.diag(ph)
    return Q

# -------------------------------------------------------------------------------
#! End of file
# -------------------------------------------------------------------------------