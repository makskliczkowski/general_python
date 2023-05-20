import inspect

############################################### PRINT THE OUTPUT BASED ON CONDITION ###############################################

'''
Prints silently
'''
def printV(what : str, v = True, tabulators = 0):
    if v == True:
        for i in range(tabulators):
            print("\t")
        print("->" + what)


######################################################## PRINT THE DICTIONARY ######################################################

'''
Prints dictionary with a key and value
'''
def printDictionary(dict):
    string = ""
    for key in dict:
        value = dict[key]
        string += f'{key},{value},'
    return string[:-1]

###################################################### PRINT ELEMENTS WITH ADJUST ##################################################

"""
[summary] 
Function that can print a list of elements creating indents
The separator also can be used to clean the indents.
- width is governing the width of each column. 
- endline, if true, puts an endline after last element of the list
- scientific allows for scientific plotting
"""
def justPrinter(file, sep="\t", elements=[], width=8, endline=True, scientific = False):
    for item in elements:
        if not scientific:  
            file.write((str(item) + sep).ljust(width))
        else:
            file.write(("{:e}".format(item) + sep).ljust(width))
    if endline:
        file.write("\n")