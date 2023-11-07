from datetime import datetime
import logging
import os

############################################### PRINT THE OUTPUT WITH A GIVEN LEVEL ###############################################

class Logger:
    
    '''
    Levels for printing
    '''
    LEVELS = {
        0 : 'info',
        1 : 'debug',
        2 : 'warning',
        3 : 'error'
    }
    
    '''
    Logger class for reporting
    '''
    def __init__(self, logfile : str, lvl = logging.DEBUG):
        self.now         = datetime.now()
        self.nowStr      = str(self.now.strftime("%d_%m_%Y_%H-%M_%S"))
        self.lvl         = lvl
        # set logger
        self.logger      = logging.getLogger(__name__)  
        self.logfile     = logfile
        self.working     = False
    
    def configureLog(self, directory : str):
        '''
        Creates the log file and handler.
        '''
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
        
    @staticmethod
    def printTab(lvl = 0):
        '''
        Print standard message with a tabulator
        '''
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
    
    def say(self,
            *args,
            end = True,
            log = 0,
            lvl = 0):
        '''
        Print multiple informations
        '''
        argss   =   [str(a) for a in args]
        out     =   ' '.join(argss)
        if not end:
            if log == 0:
                self.info(out, lvl)
            elif log == 1:
                self.debug(out, lvl)
            elif log == 2:
                self.warning(out, lvl)
            else:
                self.error(out, lvl)
        else:
            for out in argss:
                if log == 0:
                    self.info(out, lvl)
                elif log == 1:
                    self.debug(out, lvl)
                elif log == 2:
                    self.warning(out, lvl)
                else:
                    self.error(out, lvl)
            
    def info(self, msg : str, lvl = 0):
        if logging.INFO >=self.lvl:
            print(Logger.print(msg, lvl))
        logging.info(Logger.print(msg, lvl))

    def debug(self, msg : str, lvl = 0):
        if logging.DEBUG >= self.lvl:
            print(Logger.print(msg, lvl))
        logging.debug(Logger.print(msg, lvl))

    def warning(self, msg : str, lvl = 0):
        '''
        Create warning log
        '''   
        if logging.WARNING >= self.lvl:
            print(Logger.print(msg, lvl))
        logging.warning(Logger.print(msg, lvl))
    
    def error(self, msg : str, lvl = 0):
        '''
        Create error log
        '''   
        if logging.ERROR >= self.lvl:
            print(Logger.print(msg, lvl))
        logging.error(Logger.print(msg, lvl))
    
    @staticmethod
    def BREAK(n : int):
        '''
        Create n breaklines in the log
        '''
        for i in range(n):
            Logger.print("\n")
    
    def TITLE(self, tail :str, desiredSize : int, fill : str, lvl = 0):
        '''
        Create a title for - printing in the middle
        '''
        tailLength  = len(tail)
        lvlLen      = 2 + lvl * 3 * 2
        # check the length
        if tailLength + lvlLen >= desiredSize:
            self.info(tail, lvl)
            return
        
        # check the size of the fill
        fillSize    = desiredSize // len(fill) - tailLength // 2 - lvlLen 
        fillSize    = fillSize      + (0 if not tailLength == 0 else 2)
        fillSize    = fillSize      - (1 if not tailLength % 2 == 0 else 0)
        
        out         = ""
        for i in range(fillSize):
            out     = out + fill
        
        # append text
        if not tailLength == 0:
            out     = out + " " + tail + " "
            
        # append last
        for i in range(fillSize):
            out     = out + fill\
        
        self.info(out, lvl)
        
############################################### PRINT THE OUTPUT BASED ON CONDITION ###############################################

def printV(what : str, v = True, tabulators = 0):
    '''
    Prints silently
    '''
    if v == True:
        for i in range(tabulators):
            print("\t")
        print("->" + what)


######################################################## PRINT THE DICTIONARY ######################################################

def printDictionary(dict):
    '''
    Prints dictionary with a key and value
    dict - dictionary to be printed
    '''
    string = ""
    for key in dict:
        value = dict[key]
        string += f'{key},{value},'
    return string[:-1]

###################################################### PRINT ELEMENTS WITH ADJUST ##################################################

def justPrinter(file, sep="\t", elements=[], width=8, endline=True, scientific = False):
    """
    [summary] 
    Function that can print a list of elements creating indents
    The separator also can be used to clean the indents.
    - width is governing the width of each column. 
    - endline, if true, puts an endline after last element of the list
    - scientific allows for scientific plotting
    """
    for item in elements:
        if not scientific:  
            file.write((str(item) + sep).ljust(width))
        else:
            file.write(("{:e}".format(item) + sep).ljust(width))
    if endline:
        file.write("\n")