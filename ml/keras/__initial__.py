# include general tensorflow imports
from .__incl__ import *
import numpy as np

'''
https://arxiv.org/pdf/1903.06733.pdf
sig = 0.6
P = beta(2, 1)
'''
def RAI(shape, dtype = None):
    sig             = 0.6007
    V                       = None
    if len(shape) == 1:
        V                   = np.random.normal(0.0, sig**2, size = shape) / shape[0] ** 0.5
        k                   = np.random.randint(0, shape[0])
        V[k]                = np.random.beta(2, 1)
    elif len(shape) == 2:
        # weights
        V                   = np.random.randn(shape[0], shape[1]) * sig / shape[0] ** 0.5  
        for i in range(shape[1]):
            k               = np.random.randint(0, shape[0])
            V[k, i]         = np.random.beta(2, 1) 
    elif len(shape) == 3:
        V                   = np.random.randn(shape[0], shape[1], shape[2]) * sig / shape[0] ** 0.5
        for i in range(shape[0]):
            for j in range(shape[1]):
                k           = np.random.randint(0, shape[0])
                V[k, j, i]  = np.random.beta(2, 1) 
    return tf.convert_to_tensor(V, dtype = dtype)
                

