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
    x       =   rng.random.Generator.normal(size = (n, n)) + 1j * rng.random.Generator.normal(size = (n, n))
    x       /=  np.sqrt(2)
    Q, R    =   np.linalg.qr(x)
    if not simple:
        d       =   np.diagonal(R)
        ph      =   d / np.abs(d)
        Q       =   np.matmul(Q, ph) * Q
    return Q
