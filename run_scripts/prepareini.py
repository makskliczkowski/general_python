def prepareIni(script : str, 
               first  : str,
               middle : str,
               time   : str,
               mem    : str,
               cpus   : int,
               fun    : int):
    '''
    Prepares the ini file for the run
    - script : the script to be executed
    - first  : the first argument
    - middle : the middle argument
    - time   : the time argument
    - mem    : the memory argument
    - cpus   : the cpus argument
    - fun    : the function to be executed
    '''
    return f'sh {script} {first} "{middle}" {time} {mem} {cpus} {fun}'