# Adds higher directory to python modules path.
from general_python.common.hdf5_lib import *
from scipy.special import digamma, polygamma, binom, psi
import numpy as np

def page_var(LA, LB):
    d_a = 2**LA
    d_b = 2**LB
    return (((d_a + d_b)/(d_a*d_b + 1.0)) * polygamma(1, d_b + 1)) - polygamma(1, d_a*d_b + 1) - ((d_a-1)*(d_a + 2.0 * d_b - 1.0))/(4.0 * d_b * d_b * (d_a * d_b  + 1.0))

######################################################################### AVERAGES OVER WHOLE SPECTRUM #####################

# def SranSz0(L, La, f):
#     return La * np.log(2.0) + (f + np.log(1.0 - f)) / 2.0 - 0.5 * binom(2.0 * La, La) / binom(L, L//2)

class EntropyPredictions:

    @staticmethod
    def entro_volume_law(L, f = 0.5):
        '''
        Volume law
        - L : system size
        - f : filling
        '''
        return (f * L) * np.log(2)

    @staticmethod
    def entro_quadratic_chaotic(L : int, f : float):
        '''
        PRL 125, 180604 (2020)
        - La: subsystem size
        - f : filling
        '''
        La = int(L * f)
        return (1. - (1. + (1. - f) * np.log(1. - f) / f) / np.log(2.)) * La * np.log(2.)

    '''
    PRL 119, 020601 (2017)
    '''
    entro_ff = 0.5378

    ################################# TYPICAL ##############################

    @staticmethod
    def entro_random_gaussian(L : int, f : float):
        '''
        Random Gaussian states in thermodynamic limit
        - L : system size
        - f : filling
        '''
        La = int(L * f)
        return  (L - 0.5) * psi(2 * L) + (0.5 + La - L) * psi(2 * L - 2 * La) + (0.25 - La) * psi(L) - 0.25 * psi(L - La) - La
        # return L * (f * (np.log(2.) - 1.) + (f - 1.) * np.log(1. - f)) + 0.5 * f + 0.25 * np.log(1. - f)

    @staticmethod
    def entro_random_gaussian_u1(L : int, f : float, n = 0.5):
        '''
        Random Gaussian states with U(1) conservation in thermodynamic limit
        - L : system size
        - f : filling
        '''
        return L * ((f - 1.0) * np.log(1.0 - f) + f * ((n - 1.0) * np.log(1.0 - n) - n * np.log(n) - 1)) + (f * (1.0 - f + n * (1.0 - n))) / (12 * (1.0 - f) * (1.0 - n) * n * V)

    ################################# MB CHAOS ##############################

    @staticmethod
    def entro_page_th(L : int, f : float):
        '''
        Page value in thermodynamic limit.
        - L : system size
        - f : filling
        '''
        return f * L * np.log(2) - binom(2 * f * L, f * L) / binom(L, L/2) / 2

    @staticmethod
    def entro_page(f    : float, 
                   L    : int):
        '''
        Page value for a given subsystem sizes.
        - La : subsystem size
        - Lb : subsystem size
        '''
        La = int(L * f)
        Lb = L - La
        da = 2**La
        db = 2**Lb
        return digamma(da * db + 1) - digamma(db + 1) - (da - 1) / (2*db)

    @staticmethod
    def entro_page_u1(f : float, 
                      L : int, 
                      n = 0.5):

        '''
        Page result with the correction for U1.
        - La : subsystem size
        - Lb : subsystem size
        - n  : fermionic filling
        '''
        pass
        # Sval = 0
        # L_tot = int(La + Lb)
        # N = int(L_tot * n)
        # for na in range(0, min(N, L_a) + 1):
        #     d_a = binom(La, na)
        #     d_b = binom(Lb, N - na)
        #     d_N = binom(L_tot, N)
        #     Sval += d_a * d_b / d_N * ( page_result2(d_a, d_b) + digamma(d_N + 1) - digamma(d_a * d_b + 1) )
        # return Sval

    @staticmethod
    def entro_page_u1_th(f  : float,
                         L  : int,
                         n  = 0.5):
        '''
        Page results with U1 correction in thermodynamic limit
        - f : filling
        - L : system size
        - n : fermionic filling
        '''
        return ((n-1.0) * np.log(1.0-n) - n*np.log(n))*f*L - np.sqrt(n*(1.0-n)/2.0/np.pi) * np.abs(np.log((1.0-n)/n)) * (1.0 if f == 0.5 else 0.) * np.sqrt(L) + (f+np.log(1-f))/2. - 0.5 * (1. if f == 0.5 else 0) * (1. if n == 0.5 else 0.0)

###############################

class Fractal:
    
    @staticmethod
    def fractal_dim_s_info(S_lp1, S_l, lp1, l):
        '''
        Calculate the fractal dimension out of the information entropy of the system.
        - S_lp1 : Entropy of the system with L+1 sites.
        - S_l   : Entropy of the system with L sites.
        '''
        return (S_lp1 - S_l) / (np.log(2**lp1) - np.log(2**l))
    
    @staticmethod
    def fractal_dim_s_info_mean(dq_lp1, dq_l):
        '''
        Calculate the fractal dimension out of the information entropy of the system.
        Average over system sizes to get the mean value.
        - dq_lp1    : fractal of the system with L+1 sites.
        - dq_l      : fractal of the system with L sites.
        '''
        return (dq_lp1 - dq_l) / 2.0

    ################################
    
    @staticmethod
    def fractal_dim_pr(pr_lp1, pr_l, q):
        '''
        Calculate the fractal dimension out of the information entropy of the system.
        - pr_lp1: Entropy of the system with L+1 sites.
        - pr_l: Entropy of the system with L sites.
        '''
        if q != 1.0:    
            return (np.log2(pr_lp1) - np.log2(pr_l)) / (1 - q)
        else:
            return (np.log2(pr_lp1) - np.log2(pr_l))