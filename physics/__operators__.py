'''
Allows one to parse physical operators created in the simulation. 

'''

import numpy as np

class Spectral:
    '''
    For the spectral statistics and its properties. Allows one to take a fraction of the data
    and calculate the mean of the data.
    
    '''
    
    @staticmethod
    def diagonal_cutoff(Nh, nu, minimal_frac = 0.1):
        '''
        The diagonal cutoff for the spectral statistics
        - Ns: system size
        - Nh: Hilbert space dimension
        - nu: number of eigenvalues - if it's less than 1, then it's a fraction of the Hilbert space dimension
        - minimal_frac: minimal fraction of the Hilbert space dimension
        Returns:
        - The diagonal cutoff for the spectral statistics
        '''
        return min(int(minimal_frac * Nh), nu if nu >= 1 else int(nu * Nh))
    
    @staticmethod
    def take_fraction(nu : float, data, middle = None):
        '''
        Take a fraction of the data for the spectral statistics. 
        - nu: number of eigenvalues - if it's less than 1, then it's a fraction of the Hilbert space dimension
        - data: data to take the fraction from
        - middle: middle of the data (if None, then it's the middle of the data)
        Returns:
        - The fraction of the data
        '''
        sizeData    = len(data)
        middle      = sizeData // 2 if middle is None else middle
        nu          = Spectral.diagonal_cutoff(sizeData, nu)
        return data[max(middle - nu // 2, 0) : min(middle + nu // 2, sizeData)]
    
    @staticmethod
    def take_fraction_arr(nu : float, data, middle = None):
        '''
        Take a fraction of the data for the spectral statistics.
        - nu: number of eigenvalues - if it's less than 1, then it's a fraction of the Hilbert space dimension
        - data: data to take the fraction from
        - middle: middle of the data (if None, then it's the middle of the data)
        Returns:
        - The fraction of the data
        '''
        sizeRep     = data.shape[0]
        sizeData    = data.shape[-1]
        middle      = np.ones(sizeData) * sizeData // 2 if middle is None else middle
        nu          = Spectral.diagonal_cutoff(sizeData, nu)
        out         = np.array([data[i][max(middle[i] - nu // 2, 0) : min(middle[i] + nu // 2, sizeData)] for i in range(sizeRep)])
        return out
    
    @staticmethod
    def mean_fraction(nu : float, data, middle = None, axis = 0):
        '''
        Take a fraction of the data for the spectral statistics. Calculates the mean of the data.
        - nu: number of eigenvalues - if it's less than 1, then it's a fraction of the Hilbert space dimension
        - data: data to take the fraction from
        - middle: middle of the data (if None, then it's the middle of the data)
        - axis: axis to calculate the mean
        Returns:
        - The mean of the fraction of the data
        '''
        return np.mean(Spectral.take_fraction(nu, data, middle), axis = axis)
        
    ##########################
    
class Operators:
    '''
    Operator class that contains the method to parse Operator names
    both many-body and in single particle sector.
    '''
    OPERATOR_SEP		= "/"
    OPERATOR_SEP_CORR	= "-"
    OPERATOR_SEP_MULT 	= ","
    OPERATOR_SEP_DIFF	= "m"
    OPERATOR_SEP_RANGE	= ":"
    OPERATOR_SEP_RANDOM	= "r"
    OPERATOR_SEP_DIV	= "_"
    OPERATOR_PI			= "pi"
    OPERATOR_SITE		= "l"
    OPERATOR_SITEU    	= "L"
    OPERATOR_SITE_M_1   = True


    # for Hilbert space dimension (if it's smaller, then for sure the value needs to be changed)
    # to Full Hilbert space dimension
    OPERATOR_ED_LIMIT   = 16
    
    ##########################
    
    @staticmethod
    def resolve_hilbert(Ns, local_hilbert = 2):
        '''
        Resolves the Hilbert space dimension based on the
        inner limit -- OPERATOR_ED_LIMIT
        '''
        return local_hilbert**Ns if Ns <= Operators.OPERATOR_ED_LIMIT else Ns
    
    ##########################
    
    @staticmethod
    def resolveSite(site : str, _dimension = 1):
        '''
        Resolves the site to the proper one
        '''
        if len(site) == 0:
            return site 

        # check if site is L or l already - then return the dimension (L-1)
        if site == Operators.OPERATOR_SITE or site == Operators.OPERATOR_SITEU:
            return _dimension - (1 if Operators.OPERATOR_SITE_M_1 else 0)
        
        # check if the site is PI
        elif site == Operators.OPERATOR_PI:
            return np.pi
        
        # check if the site can be divided - then divide it
        elif Operators.OPERATOR_SEP_DIV in site:
            # contains L or l
            _div = Operators.resolveSite(site.split(Operators.OPERATOR_SEP_DIV)[1], _dimension)
            if Operators.OPERATOR_SITEU in site or Operators.OPERATOR_SITE in site:
                return int(_dimension / _div)
            # contains PI
            elif Operators.OPERATOR_PI in site:
                return np.pi / _div
    
        # check if the site is a difference
        elif Operators.OPERATOR_SEP_DIFF in site:
            _diff = Operators.resolveSite(site.split(Operators.OPERATOR_SEP_DIFF)[1], _dimension)
            return int(max(0.0, _dimension - _diff - (1 if Operators.OPERATOR_SITE_M_1 else 0)))

        # simply return the site as a number
        _siteInt = int(site)
        if _siteInt < 0 or _siteInt >= _dimension:
            raise Exception("The site: " + site + " is out of range. The dimension is: " + str(_dimension))
        return int(site)
    
    ##########################

    @staticmethod
    def resolve_operator(f_name, dimension):
        if Operators.OPERATOR_SEP in f_name:
            f_name_split = f_name.split(Operators.OPERATOR_SEP)
            elem_part    = f_name_split[-1]
            return Operators.OPERATOR_SEP.join(f_name_split[:-1]) + Operators.OPERATOR_SEP + str(Operators.resolveSite(elem_part, dimension))
        return f_name
    
    ##########################
    
    @staticmethod
    def name2title(f_name):
        if Operators.OPERATOR_SEP in f_name:
            f_name_split = f_name.split(Operators.OPERATOR_SEP)
            elem_part    = f_name_split[-1]
            
            if Operators.OPERATOR_SEP_DIFF in elem_part:
                elem_part= elem_part.replace("m", '-')
                
        f_name_k     = "${}^{{{}}}_{{{}}}$".format(f_name_split[0], elem_part, f_name)
        return f_name_k
