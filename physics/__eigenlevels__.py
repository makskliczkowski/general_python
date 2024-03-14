import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.special import polygamma
from scipy.special import erf, erfinv
from scipy.optimize import curve_fit
from scipy.linalg import svd
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

'''
Calculate the gap ratio around the mean energy in a sector
'''
def gap_ratio(en, fraction = 0.3, use_mean_lvl_spacing = True, return_mean = True):
    mean = np.mean(en)
    mean_idx = find_nearest_idx_np(en, mean)
    #print(mean, en[mean_idx])
    
    bad, _, lower, upper = get_values_num(fraction, np.arange(1, len(en)), 14, mean_idx)
    if bad: 
        return -1
    energies = en[lower:upper]
    #print(lower, upper, mean_idx, len(en), len(energies))
    # delta energies
    d_en = energies[1:]-energies[:-1]
    # if we use mean level spacing divide by it
    if use_mean_lvl_spacing:
        d_en /= np.mean(d_en)
    
    # calculate the gapratio
    gap_ratios = np.minimum(d_en[:-1], d_en[1:]) / np.maximum(d_en[:-1], d_en[1:])
            
    return np.mean(gap_ratios) if return_mean else gap_ratios.flatten()

'''
Calculate the average entropy in a given DataFrame
- df : DataFrame with entropies
- row : row number (-1 for half division of a system)
'''
def mean_entropy(df : pd.DataFrame, row : int):
    # return np.mean(df.loc[row] if row != -1 else df.iloc[row])
    ent_np = df.to_numpy()
    return np.mean(ent_np[row])

'''
Calculate the gaussianity <|Oab|^2>/<|Oab|>^2 -> for normal == pi/2
'''
def gaussianity(arr : np.ndarray, axis = None):
    if axis is not None:
        return np.mean(np.square(arr), axis = axis) / np.square(np.mean(arr, axis = axis))
    return np.mean(np.square(arr))/np.square(np.mean(arr))

'''
Calculate the modulus fidelity - should be 2/pi for gauss
- states : np.array of eigenstates
'''
def modulus_fidelity(states : np.ndarray):
    Ms = []
    for i in range(0, states.shape[-1] - 1):
        Ms.append(np.dot(states[:, i], states[:, i+1]))
    return np.mean(Ms)

'''
Calculate the information entropy for given states
'''
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