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