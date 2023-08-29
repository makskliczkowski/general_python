import inspect
import logging
import os
from datetime import datetime

############################################### PRINT THE OUTPUT WITH A GIVEN LEVEL ###############################################

class Logger:
    def __init__(self, logfile : str, lvl = logging.DEBUG):
        self.now         = datetime.now()
        self.nowStr      = str(self.now.strftime("%d_%m_%Y_%H-%M_%S"))
        self.lvl         = lvl
        # set logger
        self.logger      = logging.getLogger(__name__)  
        self.logfile     = logfile
        self.working     = False
    
    '''
    Creates the log file and handler.
    '''
    def configureLog(self, directory : str):
        self.logfile     = directory + os.sep + f'{self.nowStr if len(self.logfile) == 0 else self.logfile}.log'
        # with open(self.logfile, 'w+') as f:
        #     f.write('Log started!')
        self.fHandler    = logging.FileHandler(self.logfile)
        # set level
        self.fHandler.setLevel(self.lvl)
        # set formater
        self.fFormat     = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
        self.fHandler.setFormatter(self.fFormat)
        # set handler
        self.logger.addHandler(self.fHandler)
        self.working     = True
        
        logging.basicConfig(format      = '%(asctime)s-%(levelname)s-%(message)s',
                            datefmt     = "%d_%m_%Y_%H-%M_%S",
                            filename    = self.logfile,
                            level       = self.lvl)
    '''
    Print standard message with a tabulator
    '''
    @staticmethod
    def printTab(lvl = 0):
        ret = ""
        for _ in range(lvl):
            ret += '\t'
        if lvl > 0:
            ret += '->'
        return ret
    
    @staticmethod
    def print(msg, lvl = 0):
        now         = datetime.now()
        nowStr      = now.strftime("%d/%m/%Y %H:%M:%S")
        return "[" + nowStr + "]" + Logger.printTab(lvl) + msg
    
    def info(self, msg : str, lvl = 0):
        if logging.INFO >=self.lvl:
            print(Logger.print(msg, lvl))
        logging.info(Logger.print(msg, lvl))

    def debug(self, msg : str, lvl = 0):
        if logging.DEBUG >= self.lvl:
            print(Logger.print(msg, lvl))
        logging.debug(Logger.print(msg, lvl))
    
    def warning(self, msg : str, lvl = 0):
        if logging.WARNING >= self.lvl:
            print(Logger.print(msg, lvl))
        logging.warning(Logger.print(msg, lvl))
        
    def error(self, msg : str, lvl = 0):
        if logging.ERROR >= self.lvl:
            print(Logger.print(msg, lvl))
        logging.error(Logger.print(msg, lvl))

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