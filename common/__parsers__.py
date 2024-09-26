import pandas as pd
# filter warnings
from warnings import simplefilter
simplefilter(action = "ignore", category= pd.errors.PerformanceWarning)

import traceback

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

######################################################## EXCEPTIONS! ########################################################

import sys
import os

class ExceptionHandler:
    
    @staticmethod
    def handle(e, msg : str, *args):
        '''
        Handles the exception. 
        - e     : exception
        - msg   : message to be printed
        - skip  : list of exceptions to be skipped
        '''
        show_exception = True
        
        for s in args:
            if isinstance(e, s):
                show_exception = False
                break
        
        if show_exception:
            exc_type, exc_obj, exc_tb   = sys.exc_info()
            print("----------------------------------------------------")
            
            # frame print
            if exc_tb.tb_frame is not None:
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                if exc_tb is not None:
                    print(f"Error in file {fname}, line {exc_tb.tb_lineno}")
                    
            print(f"Error: {e}")
            print(f"Msg: {msg}")
            print(traceback.format_exc())
            
            print("----------------------------------------------------")
            
         
######################################################## STRING PARSER ########################################################

PARSER_STRING_DIVIDER       = "_"
PARSER_STRING_CORRELATION   = "-"
PARSER_STRING_MULTIPLE      = ","
PARSER_STRING_RANGES        = "."
PARSER_STRING_SEPARATOR     = "/"
PARSER_STRING_SEPARATOR_ALT = "/"

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