'''

'''

import numpy as np

##################################### RANDOM #####################################

def CUE_QR( n       :   int,
            simple  =   True,
            rng     =   None):
    '''
    Create the CUE matrix using QR decomposition
    - n     : size of the matrix (n X n)
    - simple: use the straightforward method
    '''
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