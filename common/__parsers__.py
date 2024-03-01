import pandas as pd

####################################################### DATAFRAME PARSE #######################################################

def parse_dataframe(df      : pd.DataFrame, 
                    params  : dict,
                    copy    = False):
    '''
    Parses the dataframe according to some given dictionary of parameters
    The dictionary shall contain a given column name : [list of possible values]
    - df        : pandas.DataFrame
    - params    : dictionary of parameters
    - copy      : shall copy the dataframe
    '''
    tmp     = pd.DataFrame()
    if copy:
        tmp = df.copy()
    else:
        tmp = df
    # go through keys
    for key in params.keys():
        tmp = tmp[tmp[key].isin(params[key])]
    return tmp

######################################################## STRING PARSER ########################################################

class StringParser:   
    '''
    A class that handles parsing of a string to different types
    '''
    
    
    #######################
    @staticmethod
    def e(  s    : float,
            prec : int):
        '''
        Prints the value in the scientific notation
        - s     : float to be printed
        - prec  : number of decimal places
        '''
        return format(s, f'.{prec}E').replace("E-0", "E-").replace("E+0", "E+")
    
    #######################
    @staticmethod
    def ls( lst,
            elemParser  = lambda s: str(s),
            joinElem    = ',',
            withoutBrcts= False):
        '''
        Parses the lists and returns the joint version of it.
        - lst           :   list of strings to be joined
        - elemParser    :   should we do something with the elements? lambda function!
        - joinElem      :   how those should be joined together within a string
        '''
        if withoutBrcts:
            return joinElem.join([elemParser(s) for s in lst])
        return "[" + joinElem.join([elemParser(s) for s in lst]) + "]"
    
    ############## FILENAME CONVENTIONS ##############
    @staticmethod
    def parseFileNameFloor( f        :   str,
                            whichEq  :   int,
                            whichPar :   int):
        '''
        Parses the file using the convention that main elements are separated with "_"
        and we choose which split gives us the part containing "{param}={value}" separated by ","
        - f           :   string to parse
        - whichEq     :   which part in split("_") contains the "{param}={value}" elements
        - whichPar    :   which "{param}={value}" element to take
        '''
        tmp = f.split("_")[whichEq].split(",")[whichPar].split("=")
        return tmp[0], float(tmp[1])