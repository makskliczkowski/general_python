"""
Eigenlevel statistics and entropy calculators for quantum systems.

This module contains tools for:
- Reduced density matrix calculation (direct or Schmidt decomposition).
- Entanglement entropy (von Neumann).
- Level statistics (gap ratios).
- Statistical measures of eigenstates (participation ratio, moments).

Input/Output Contracts
----------------------
- States are typically 1D or 2D NumPy arrays (basis size, number of states).
- Reduced density matrices return shape (dimA, dimA).
- Entropies return scalar floats.
- Gap ratios return a dictionary with mean, std, and raw values.

Numerical Stability
-------------------
Entropy calculations handle small eigenvalues by clipping or conditional checks to avoid log(0).
Schmidt decomposition is preferred for reduced density matrices when possible for stability and efficiency.
"""

import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.special import polygamma
from scipy.special import erf, erfinv
from scipy.optimize import curve_fit
from scipy.linalg import svd
from ..maths.statistics import Fraction

# gap ratio average values
rgoe        = 0.5307
pois        = 0.386294

####################################################### REDUCED DENSITY MATRIX #######################################################

'''
Calculate the reduced density matrix out of a state
'''
def reduced_density_matrix(state    : np.ndarray, 
                           A_size   : int,
                           L        : int):
        dimA    = int(( (2 **      A_size ) ))
        dimB    = int(( (2 ** (L - A_size)) ))
        N       = dimA * dimB
        rho     = np.zeros((dimA, dimA), dtype = complex)
        for n in range(0, N, 1):					
            counter                     = 0
            for m in range(n % dimB, N, dimB):
                idx                     =   n // dimB
                rho[idx, counter]       +=  np.conj(state[n]) * state[m]
                counter                 +=  1
        return rho

'''
Calculates the reduced density matrix via the Schmidt decomposition
'''
def reduced_density_matrix_schmidt(state     :   np.ndarray, 
                                    L        :   int, 
                                    La       :   int):    
    dimA        =   2 ** La
    dimB        =   2 ** (L-La)
    N           =   dimA * dimB

    # reshape array to matrix
    rho         =   state.reshape(dimA, dimB)

    # get schmidt coefficients from singular-value-decomposition
    U, schmidt_coeff, _ = svd(rho)

    # return square
    return np.square(schmidt_coeff)

######################################################### ENTANGLEMENT ENTROPY #########################################################

'''
Calculate the bipartite entanglement entropy
'''
def entropy_vonNeuman(  state       :   np.ndarray, 
                        L           :   int, 
                        La          :   int,
                        TYP         =   "SCHMIDT"):    
    entropy = 0
    eV      = None
    if TYP == "SCHMIDT":
        eV  = reduced_density_matrix_schmidt(state, L, La)
    else:
        rho = reduced_density_matrix(state, La, L)
        eV  = np.linalg.eigvals(rho)
        
    for i in range(len(eV)):
        entropy += ((-eV[i] * np.log(eV[i])) if (abs(eV[i]) > 0) else 0)
    return entropy
    # return -np.multiply(eV, np.log(eV)).sum()


# def entro_old_rho(rho : np.ndarray):
#     eig_sym = np.linalg.eigvals(rho)
#     ent = -np.multiply(eig_sym, np.log(eig_sym)).sum()
#     return ent

# def entro_old(state : np.ndarray, L, La):

#     ent = 
#     return ent
####################################################### CALCULATORS #######################################################

def gap_ratio(en: np.ndarray,
            fraction                = 0.3,
            use_mean_lvl_spacing    = True
        ):
    r'''
    Calculate the gap ratio of the eigenvalues as:
        $\gamma = \frac{min(\Delta_n, \Delta_{n+1})}{max(\Delta_n, \Delta_{n+1})}$
    - en                    : eigenvalues
    - fraction              : fraction of the eigenvalues to use
    - use_mean_lvl_spacing  : divide by mean level spacing
    '''
    
    mean            = np.mean(en)
    energies        = Fraction.take_fraction(frac=fraction, data=en, around=mean)
    d_en            = energies[1:]-energies[:-1]
    if use_mean_lvl_spacing:
        d_en /= np.mean(d_en)
    
    # calculate the gapratio
    gap_ratios = np.minimum(d_en[:-1], d_en[1:]) / np.maximum(d_en[:-1], d_en[1:])
    return {
        'mean': np.mean(gap_ratios),
        'std' : np.std(gap_ratios),
        'vals': gap_ratios
    }

'''
Calculate the average entropy in a given DataFrame
- df : DataFrame with entropies
- row : row number (-1 for half division of a system)
'''
def mean_entropy(df : pd.DataFrame, row : int):
    # return np.mean(df.loc[row] if row != -1 else df.iloc[row])
    ent_np = df.to_numpy()
    return np.mean(ent_np[row])

class HamiltonianProperties:
    
    @staticmethod
    def hilbert_schmidt_norm(mat : np.ndarray):
        """ Creates the Hilbert-Schmidt norm of the matrix.
        Args:
            mat (np.ndarray): matrix to calculate the norm of

        Returns:
            _type_: The Hilbert-Schmidt norm of the matrix.
        """
        # return np.trace(mat * mat) / mat.shape[0]
        return np.trace(np.matmul(mat, np.conj(mat).T)) / mat.shape[0]

class StatMeasures:

    @staticmethod
    def moments(arr : np.ndarray, axis = None):
        '''
        Calculate the moments of the array
        - arr : array to calculate the moments
        - axis : axis to calculate the moments
        '''
        if axis is not None:
            S   = np.mean(arr, axis = axis)
            S2  = np.mean(arr**2, axis = axis)
            V   = S2 - S**2
            S4  = np.mean(arr**4, axis = axis)
            B   = 1.0 - (S4 / (3.0 * S2**2))
            return S, S2, V, S4, B
        S   = np.mean(arr)
        S2  = np.mean(arr**2)
        V   = S2 - S**2
        S4  = np.mean(arr**4)
        B   = 1.0 - (S4 / (3.0 * S2**2))
        return S, S2, V, S4, B

    @staticmethod
    def gaussianity(arr : np.ndarray, axis = None):
        '''
        Calculate the gaussianity <|Oab|^2>/<|Oab|>^2 -> for normal == pi/2
        - arr : array to calculate the gaussianity
        - axis : axis to calculate the gaussianity
        '''
        if axis is not None:
            return np.mean(np.square(arr), axis = axis) / np.square(np.mean(arr, axis = axis))
        return np.mean(np.square(arr))/np.square(np.mean(arr))
    
    ##############################################################
    
    @staticmethod
    def binder_cumulant(arr : np.ndarray, axis = None):
        '''
        Calculate the binder cumulant <|Oab|^4>/(3 * <|Oab|^2>^2) -> for normal == 2/3
        - arr : array to calculate the binder cumulant
        '''
        if axis is not None:
            return np.mean(np.power(arr, 4), axis = axis) / (3 * np.square(np.mean(np.square(arr), axis = axis)))
        return np.mean(np.power(arr, 4)) / (3 * np.square(np.mean(np.square(arr))))
    
    ##############################################################
    
    '''
    Calculate the modulus fidelity - should be 2/pi for gauss
    - states : np.array of eigenstates
    '''
    @staticmethod
    def modulus_fidelity(states : np.ndarray):
        Ms = []
        for i in range(0, states.shape[-1] - 1):
            Ms.append(np.dot(states[:, i], states[:, i+1]))
        return np.mean(Ms)

'''
Calculate the information entropy for given states
'''
@staticmethod
def info_entropy(states : np.ndarray, model_info : str):
    try:
        entropies = []
        for state in states.T:
            square = np.square(np.abs(state))
            entropies.append(-np.sum(square * np.log(square)))
        return np.mean(entropies)
    except:
        print(f'\nHave some problem in {model_info}\n')
        return -1.0