from datetime import datetime
import logging
import os

############################################### PRINT THE OUTPUT WITH A GIVEN COLOR ###############################################

class Colors:
    black   = r"\033[30m"
    red		= r"\033[31m"
    green	= r"\033[32m"
    yellow	= r"\033[33m"
    blue	= r"\033[34m"
    white   = r"\033[0m"
    
    def __init__(self, color : str):
        self.color = color

    def __str__(self) -> str:
        if self.color == "black":
            return Colors.black
        elif self.color == "red":
            return Colors.red
        elif self.color == "green":
            return Colors.green
        elif self.color == "yellow":
            return Colors.yellow
        elif self.color == "blue":
            return Colors.blue
        else:
            return Colors.white
        
    def __repr__(self) -> str:
        return str(self)

############################################### PRINT THE OUTPUT WITH A GIVEN LEVEL ###############################################

class Logger:
    """
    Logger class for handling console and file logging with verbosity control.
    """
    
    LEVELS = {
        0: 'info',
        1: 'debug',
        2: 'warning',
        3: 'error'
    }
    
    def __init__(self, logfile: str, lvl=logging.DEBUG):
        """
        Initialize the logger instance.

        Args:
            logfile (str): Name of the log file.
            lvl (int): Logging level (default: logging.DEBUG).
        """
        self.now = datetime.now()
        self.nowStr = self.now.strftime("%d_%m_%Y_%H-%M_%S")
        self.lvl = lvl
        self.logger = logging.getLogger(__name__)
        self.logfile = logfile
        self.working = False

    @staticmethod
    def colorize(txt: str, color: str):
        """
        Apply color to the given text (for console output).

        Args:
            txt (str): Text to colorize.
            color (str): Color name.

        Returns:
            str: Colorized text.
        """
        return str(Colors(color)) + str(txt) + Colors.white

    def configure(self, directory: str):
        """
        Configure the logger to use a specific directory for log files.

        Args:
            directory (str): Path to the directory where log files will be stored.
        """
        self.logfile = directory + os.sep + f'{self.nowStr if len(self.logfile) == 0 else self.logfile}.log'
        os.makedirs(directory, exist_ok=True)
        with open(self.logfile, 'w+') as f:
            f.write('Log started!')
            
        self.fHandler = logging.FileHandler(self.logfile)
        self.fHandler.setLevel(self.lvl)
        self.fFormat = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
        self.fHandler.setFormatter(self.fFormat)
        self.logger.addHandler(self.fHandler)
        self.working = True
        
        logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s',
                            datefmt="%d_%m_%Y_%H-%M_%S",
                            filename=self.logfile,
                            level=self.lvl)

    @staticmethod
    def printTab(lvl=0):
        """
        Generate indentation for message formatting.

        Args:
            lvl (int): Number of tabulators.

        Returns:
            str: Indented string.
        """
        return '\t' * lvl + ('->' if lvl > 0 else '')

    @staticmethod
    def print(msg: str, lvl=0):
        """
        Format a message with a timestamp.

        Args:
            msg (str): Message to format.
            lvl (int): Indentation level.

        Returns:
            str: Formatted message.
        """
        now = datetime.now()
        nowStr = now.strftime("%d/%m/%Y %H:%M:%S")
        return f"[{nowStr}]{Logger.printTab(lvl)}{msg}"

    def say(self, *args, end=True, log=0, lvl=0, verbose=True):
        """
        Print and log multiple messages if verbosity is enabled.

        Args:
            *args: Messages to log.
            end (bool): Append newline (default: True).
            log (int): Log level (0: info, 1: debug, 2: warning, 3: error).
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
        """
        if not verbose:
            return
        messages = [str(arg) for arg in args]
        # if end is True, append newline
        if end:
            for msg in messages:
                self._log_message(log, msg, lvl)
        # else join messages with a space
        else:
            self._log_message(log, ' '.join(messages), lvl)
        

    def _log_message(self, log_level, msg, lvl):
        """
        Internal helper to log messages based on log level.

        Args:
            log_level (int): Log level (0: info, 1: debug, etc.).
            msg (str): Message to log.
            lvl (int): Indentation level.
        """
        if log_level == 0:
            self.info(msg, lvl)
        elif log_level == 1:
            self.debug(msg, lvl)
        elif log_level == 2:
            self.warning(msg, lvl)
        elif log_level == 3:
            self.error(msg, lvl)

    def info(self, msg: str, lvl=0, verbose=True):
        """
        Log an informational message if verbosity is enabled.

        Args:
            msg (str): Message to log.
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
        """
        if not verbose:
            return
        print(Logger.print(msg, lvl))
        self.logger.info(Logger.print(msg, lvl))

    def debug(self, msg: str, lvl=0, verbose=True):
        """
        Log a debug message if verbosity is enabled.

        Args:
            msg (str): Message to log.
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
        """
        if not verbose:
            return
        print(Logger.print(msg, lvl))
        self.logger.debug(Logger.print(msg, lvl))

    def warning(self, msg: str, lvl=0, verbose=True):
        """
        Log a warning message if verbosity is enabled.

        Args:
            msg (str): Message to log.
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
        """
        if not verbose:
            return
        print(Logger.print(msg, lvl))
        self.logger.warning(Logger.print(msg, lvl))

    def error(self, msg: str, lvl=0, verbose=True):
        """
        Log an error message if verbosity is enabled.

        Args:
            msg (str): Message to log.
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
        """
        if not verbose:
            return
        print(Logger.print(msg, lvl))
        self.logger.error(Logger.print(msg, lvl))

    @staticmethod
    def breakline(n: int):
        """
        Print multiple break lines.

        Args:
            n (int): Number of break lines.
        """
        for _ in range(n):
            print()

    def title(self, tail: str, desiredSize: int, fill: str, lvl=0, verbose=True):
        """
        Create a formatted title with filler characters if verbosity is enabled.

        Args:
            tail (str): Text in the middle of the title.
            desiredSize (int): Total width of the title.
            fill (str): Character used for filling.
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
        """
        if not verbose:
            return
        tailLength = len(tail)
        lvlLen = 2 + lvl * 3 * 2
        if tailLength + lvlLen > desiredSize:
            self.info(tail, lvl, verbose)
            return
        
        fillSize = (desiredSize - tailLength - 2) // (2 * len(fill))
        out = (fill * fillSize) + f" {tail} " + (fill * fillSize)
        self.info(out, lvl, verbose)
        
################################################ PRINT THE OUTPUT BASED ON CONDITION ###############################################

def printV(what         : str, 
           v            = True, 
           tabulators   = 0):
    '''
    Prints silently if necessary
    - what         : message to be printed
    - v            : verbosity
    - tabulators   : number of tabulators
    '''
    if v:
        for _ in range(tabulators):
            print("\t")
        print("->" + what)

######################################################## PRINT THE DICTIONARY ######################################################

def printDictionary(dict):
    '''
    Prints dictionary with a key and value
    dict - dictionary to be printed
    '''
    string      = ""
    for key in dict:
        value   = dict[key]
        string  += f'{key},{value},'
    return string[:-1]

###################################################### PRINT ELEMENTS WITH ADJUST ##################################################

def printJust(file, 
              sep           =   "\t",
              elements      =   [],
              width         =   8, 
              endline       =   True, 
              scientific    =   False):
    """
    [summary] 
    Function that can print a list of elements creating indents
    The separator also can be used to clean the indents.
    - width     :   is governing the width of each column. 
    - endline   :   if true, puts an endline after last element of the list
    - scientific:   allows for scientific printing
    """
    for item in elements:
        if not scientific:  
            file.write((str(item) + sep).ljust(width))
        else:
            file.write(("{:e}".format(item) + sep).ljust(width))
    if endline:
        file.write("\n")