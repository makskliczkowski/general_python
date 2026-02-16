'''
Constants and semi-analytical values for entropy predictions.
'''

import  numpy as np
from    scipy.special   import digamma, polygamma, binom, psi

# Entropy value from PRL 119, 020601 (2017)
ENTRO_FF_HALF   = 0.5378

# Small numerical epsilon
EPS_DEFAULT     = 1e-15


class EntropyPredictions:
    r"""
    A collection of static methods for predicting entanglement entropy in quantum systems under various physical scenarios.

    This class provides analytical and semi-analytical formulas for entanglement entropy, including volume law, chaotic, random Gaussian, and Page values, with and without U(1) conservation. The methods are based on results from quantum statistical mechanics and random matrix theory.

    Attributes:
        entro_ff (float): Entropy value from PRL 119, 020601 (2017).

    Methods:
        volume_law(L, f=0.5):
            Computes the volume law entropy for a system of size L and filling f.

        quadratic_chaotic(L: int, f: float):
            Computes the entropy for quadratic chaotic systems as described in PRL 125, 180604 (2020).

        entro_random_gaussian(L: int, f: float):
            Computes the entropy for random Gaussian states in the thermodynamic limit.

        entro_random_gaussian_u1(L: int, f: float, n=0.5):
            Computes the entropy for random Gaussian states with U(1) conservation in the thermodynamic limit.

        entro_page_th(L: int, f: float):
            Computes the Page value (average entropy of a subsystem) in the thermodynamic limit.

        entro_page(f: float, L: int):
            Computes the Page value for a given subsystem size.

        entro_page_u1(f: float, L: int, n=0.5):
            Placeholder for Page value computation with U(1) correction.

        entro_page_u1_th(f: float, L: int, n=0.5):
            Computes the Page value with U(1) correction in the thermodynamic limit.
    """

    # ---------------------------------
    
    class Mean:
        
        @staticmethod
        def volume_law(L, f = 0.5):
            """
            Calculates the entanglement entropy according to the volume law for a system of size L.

            Parameters:
                L (int or float):
                    The size of the system (e.g., number of sites or particles).
                f (float, optional):
                    Fraction of the system considered. Default is 0.5.

            Returns:
                float: The entanglement entropy given by (f * L) * log(2).

            Notes:
                The volume law states that the entanglement entropy scales linearly with the size of the subsystem.
            """
            return (f * L) * np.log(2)

        @staticmethod
        def quadratic_chaotic(L : int, f : float):
            """
            Computes the entropy for quadratic chaotic systems.

            Based on PRL 125, 180604 (2020).

            Parameters
            ----------
            L : int
                System size.
            f : float
                Filling fraction.

            Returns
            -------
            float
                Entanglement entropy.
            """
            La = int(L * f)
            return (1. - (1. + (1. - f) * np.log(1. - f) / f) / np.log(2.)) * La * np.log(2.)

        @staticmethod
        def free_fermions_half(L : int, f : float):
            r"""
            Computes the entropy for free fermions at half filling.

            Based on PRL 119, 020601 (2017).

            Parameters
            ----------
            L : int
                System size.
            f : float
                Filling fraction.

            Returns
            -------
            float
                Entanglement entropy constant (0.5378).
            """
            return 0.5378
    
        @staticmethod
        def random_gaussian_th(L: int, f: float):
            r"""
            Random Gaussian states in the thermodynamic limit.

            Parameters
            ----------
            L : int
                System size.
            f : float
                Filling fraction.

            Returns
            -------
            float
                Entanglement entropy for random Gaussian states.
            """
            return L * (f * (np.log(2.) - 1.) + (f - 1.) * np.log(1. - f)) + 0.5 * f + 0.25 * np.log(1. - f)

        @staticmethod
        def random_gaussian_u1_th(L: int, f: float, n: float = 0.5):
            r"""
            Random Gaussian states with U(1) conservation in the thermodynamic limit.

            Parameters
            ----------
            L : int
                System size.
            f : float
                Filling fraction.
            n : float, optional
                Fermionic filling. Default is 0.5.

            Returns
            -------
            float
                Entanglement entropy for random Gaussian states with U(1) conservation.
            """
            # Note: The variable V is not defined in the original code. Please define V if needed.
            return L * ((f - 1.0) * np.log(1.0 - f) + f * ((n - 1.0) * np.log(1.0 - n) - n * np.log(n) - 1))  # + (f * (1.0 - f + n * (1.0 - n))) / (12 * (1.0 - f) * (1.0 - n) * n * V)

        @staticmethod
        def page_th(L: int, f: float):
            r"""
            Page value in the thermodynamic limit.

            Parameters
            ----------
            L : int
                System size.
            f : float
                Filling fraction.

            Returns
            -------
            float
                Page value.
            """
            return f * L * np.log(2) - binom(2 * f * L, f * L) / binom(L, L / 2) / 2

        @staticmethod
        def page(da: int, db: int):
            r"""
            Page value for given subsystem sizes.

            Parameters
            ----------
            da : int
                Size of subsystem A.
            db : int
                Size of subsystem B.

            Returns
            -------
            float
                Page value.
            """
            return digamma(da * db + 1) - digamma(db + 1) - (da - 1) / (2 * db)

        @staticmethod
        def page_u1(La: int, Lb: int, n: float = 0.5):
            r"""
            Page result with the correction for U(1) symmetry.

            Parameters
            ----------
            La : int
                Subsystem A size.
            Lb : int
                Subsystem B size.
            n : float, optional
                Fermionic filling fraction. Default is 0.5 - half filling.

            Returns
            -------
            float
                Page value with U(1) correction.

            Notes
            -----
            This implements the Page value calculation for systems with U(1) charge conservation,
            following the formula:

            .. math::
                \langle S_A \rangle_N = \sum \frac{d_A d_B}{d_N} \left[ \langle S_A \rangle + \psi(d_N + 1) - \psi(d_A d_B + 1) \right]
            """
            
            if La <= 0 or Lb <= 0:
                return 0.0
            
            Sval        = 0.0
            L_tot       = La + Lb
            N           = int(L_tot * n)
            
            # Ensure N is within valid bounds
            N           = max(0, min(N, L_tot))
            
            for na in range(max(0, N - Lb), min(N, La) + 1):
                nb              = N - na
                d_a, d_b, d_N   = binom(La, na), binom(Lb, nb), binom(L_tot, N)
                if d_N > 0:
                    weight      = d_a * d_b / d_N
                    sa_mean     = EntropyPredictions.Mean.page(d_a, d_b)
                    correction  = digamma(d_N + 1) - digamma(d_a * d_b + 1)
                    Sval       += weight * (sa_mean + correction)
            return Sval

        @staticmethod
        def page_u1_th(f: float, L: int, n: float = 0.5):
            """
            Page results with U(1) correction in the thermodynamic limit.

            Parameters
            ----------
            f : float
                Filling fraction.
            L : int
                System size.
            n : float, optional
                Fermionic filling. Default is 0.5.

            Returns
            -------
            float
                Page value with U(1) correction in the thermodynamic limit.
            """
            return ((n - 1.0) * np.log(1.0 - n) - n * np.log(n)) * f * L \
            - np.sqrt(n * (1.0 - n) / (2.0 * np.pi)) * np.abs(np.log((1.0 - n) / n)) * (1.0 if f == 0.5 else 0.0) * np.sqrt(L) \
            + (f + np.log(1 - f)) / 2.0 \
            - 0.5 * (1.0 if f == 0.5 else 0.0) * (1.0 if n == 0.5 else 0.0)

    # ---------------------------------
    
    class Var:
        """
        A collection of static methods for calculating the variance of entanglement entropy in quantum systems.

        This class provides methods to compute the variance of entanglement entropy based on the Page value and other parameters.

        Methods:
            page_var(LA, LB):
                Computes the variance of entanglement entropy based on subsystem sizes LA and LB.
        """

        # ---------------------------------

        @staticmethod
        def page_var(LA, LB):
            """
            Computes the variance of the entanglement entropy (Page variance) for subsystems of sizes LA and LB.

            Parameters
            ----------
            LA : int
                Size of subsystem A.
            LB : int
                Size of subsystem B.

            Returns
            -------
            float
                Variance of the entanglement entropy.
            """
            d_a     = 2 ** LA
            d_b     = 2 ** LB
            term1   = ((d_a + d_b) / (d_a * d_b + 1.0)) * polygamma(1, d_b + 1)
            term2   = polygamma(1, d_a * d_b + 1)
            term3   = ((d_a - 1) * (d_a + 2.0 * d_b - 1.0)) / (4.0 * d_b ** 2 * (d_a * d_b + 1.0))
            return term1 - term2 - term3
    
        # ---------------------------------
        
    mean = Mean()
    var  = Var()
    
    # ---------------------------------

    ################################# TYPICAL ##############################

    @staticmethod
    def entro_random_gaussian(L : int, f : float):
        """
        Random Gaussian states in thermodynamic limit.

        Parameters
        ----------
        L : int
            System size.
        f : float
            Filling fraction.

        Returns
        -------
        float
            Entropy.
        """
        La = int(L * f)
        return  (L - 0.5) * psi(2 * L) + (0.5 + La - L) * psi(2 * L - 2 * La) + (0.25 - La) * psi(L) - 0.25 * psi(L - La) - La
        # return L * (f * (np.log(2.) - 1.) + (f - 1.) * np.log(1. - f)) + 0.5 * f + 0.25 * np.log(1. - f)

    @staticmethod
    def entro_random_gaussian_u1(L : int, f : float, n = 0.5):
        """
        Random Gaussian states with U(1) conservation in thermodynamic limit.

        Parameters
        ----------
        L : int
            System size.
        f : float
            Filling fraction.

        Returns
        -------
        float
            Entropy.
        """
        return L * ((f - 1.0) * np.log(1.0 - f) + f * ((n - 1.0) * np.log(1.0 - n) - n * np.log(n) - 1)) + (f * (1.0 - f + n * (1.0 - n))) / (12 * (1.0 - f) * (1.0 - n) * n * L)

    ################################# MB CHAOS ##############################

    @staticmethod
    def entro_page_th(L : int, f : float):
        """
        Page value in thermodynamic limit.

        Parameters
        ----------
        L : int
            System size.
        f : float
            Filling fraction.

        Returns
        -------
        float
            Page entropy.
        """
        return f * L * np.log(2) - binom(2 * f * L, f * L) / binom(L, L/2) / 2

    @staticmethod
    def entro_page(f    : float, 
                   L    : int):
        """
        Page value for a given subsystem sizes.

        Parameters
        ----------
        f : float
            Filling fraction.
        L : int
            System size.

        Returns
        -------
        float
            Page entropy.
        """
        La = int(L * f)
        Lb = L - La
        da = 2**La
        db = 2**Lb
        return digamma(da * db + 1) - digamma(db + 1) - (da - 1) / (2*db)

    @staticmethod
    def entro_page_u1(f : float, 
                      L : int, 
                      n = 0.5):

        """
        Page result with the correction for U1.

        **Not Implemented.**

        Parameters
        ----------
        f : float
            Filling fraction.
        L : int
            System size.
        n : float, optional
            Fermionic filling.

        Returns
        -------
        None
        """
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
        """
        Page results with U1 correction in thermodynamic limit.

        Parameters
        ----------
        f : float
            Filling fraction.
        L : int
            System size.
        n : float, optional
            Fermionic filling.

        Returns
        -------
        float
            Page entropy with U1 correction.
        """
        return ((n - 1.0) * np.log(1.0 - n) - n * np.log(n)) * f * L \
            - np.sqrt(n * (1.0 - n) / (2.0 * np.pi)) * np.abs(np.log((1.0 - n) / n)) * (1.0 if f == 0.5 else 0.0) * np.sqrt(L) \
            + (f + np.log(1 - f)) / 2.0 \
            - 0.5 * (1.0 if f == 0.5 else 0.0) * (1.0 if n == 0.5 else 0.0)
