import pandas as pd

####################################################### DATAFRAME PARSE #######################################################

'''
Parses the dataframe according to some given dictionary of parameters
'''
def parse_dataframe(df : pd.DataFrame, params : dict):
    tmp         =       df.copy()
    for key in params.keys():
        tmp     =       tmp[tmp[key].isin(params[key])]
    return tmp

######################################################## STRING PARSER ########################################################

class StringParser:
    
    @staticmethod
    def e(  s,
            prec : int):
        '''
        Prints the value in the scientific notation
        '''
        return format(s, f'.{prec}E').replace("E-0", "E-").replace("E+0", "E+")
    
    @staticmethod
    def ls(
            lst,
            elemParser  = lambda s: str(s),
            joinElem    = ','
        ):
        '''
        Parses the lists and returns the joint version of it.
        '''
        return "[" + joinElem.join([elemParser(s) for s in lst]) + "]"