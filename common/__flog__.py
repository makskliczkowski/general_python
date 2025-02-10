from datetime import datetime

import logging
# set the logging level to WARNING
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("flax").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import os
import functools

try:
    from colorama import init
    init(autoreset=True)
    __HAS_COLORAMA = True
except ImportError:
    __HAS_COLORAMA = False
    pass

############################################### PRINT THE OUTPUT WITH A GIVEN COLOR ###############################################

class Colors:
    """
    Class for defining ANSI colors for console output.
    """
    # Use normal string literals so escape sequences are interpreted
    black   = "\033[30m"
    red     = "\033[31m"
    green   = "\033[32m"
    yellow  = "\033[33m"
    blue    = "\033[34m"
    white   = "\033[0m"  # Reset / default color
    
    def __init__(self, color : str):
        self.color = color


    def __str__(self) -> str:
        mapping = {
            "black" : Colors.black,
            "red"   : Colors.red,
            "green" : Colors.green,
            "yellow": Colors.yellow,
            "blue"  : Colors.blue,
            "white" : Colors.white
        }
        return mapping.get(self.color, Colors.white)
        
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
            logfile (str): Name of the log file (without extension if empty, a timestamp will be used).
            lvl (int): Logging level (default: logging.DEBUG).
            
        """
        self.now            = datetime.now()
        self.nowStr         = self.now.strftime("%d_%m_%Y_%H-%M_%S")
        self.lvl            = lvl
        self.logger         = logging.getLogger(__name__)
        self.logfile        = logfile
        self.working        = False
        self.handler_added  = False  # Prevent adding multiple handlers
        
    # --------------------------------------------------------------
    
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

    # --------------------------------------------------------------
    
    def configure(self, directory: str):
        """
        Configure the logger to use a specific directory for log files.

        Args:
            directory (str): Path to the directory where log files will be stored.
        """
        
        # If logfile name is empty, use the timestamp
        base_name       = self.nowStr if len(self.logfile) == 0 else self.logfile
        self.logfile    = os.path.join(directory, f'{base_name}.log')
        
        # Create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)
        
        # Write initial log file header
        with open(self.logfile, 'w+') as f:
            f.write('Log started!\n')
            
        # Create a file handler only once
        if not self.handler_added:
            self.fHandler   = logging.FileHandler(self.logfile)
            self.fHandler.setLevel(self.lvl)
            self.fFormat    = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                                datefmt="%d_%m_%Y_%H-%M_%S")
            self.fHandler.setFormatter(self.fFormat)
            self.logger.addHandler(self.fHandler)
            self.handler_added = True
        
        # Also set up basic configuration (this might be redundant if multiple loggers are used)
        logging.basicConfig(format  =   '%(asctime)s - %(levelname)s - %(message)s',
                            datefmt =   "%d_%m_%Y_%H-%M_%S",
                            filename=   self.logfile,
                            level   =   self.lvl)
        self.working = True

    # --------------------------------------------------------------
    
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

    # --------------------------------------------------------------
    
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

    # --------------------------------------------------------------
    
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
    
    # --------------------------------------------------------------
    
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

    # --------------------------------------------------------------
    
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

    # --------------------------------------------------------------
    
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

    # --------------------------------------------------------------

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

    # --------------------------------------------------------------

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
        
    # --------------------------------------------------------------
    
    @staticmethod
    def breakline(n: int):
        """
        Print multiple break lines.

        Args:
            n (int): Number of break lines.
        """
        for _ in range(n):
            print()

    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    
    def timing(self, func):
        """
        Decorator to measure and log the execution time of functions.
        Parameters:
            func : function to be timed
        """
        
        @functools.wraps(func)  # Decorator to preserve the original function's metadata
        
        def wrapper(*args, **kwargs):
            self.debug(f"Starting '{func.__name__}'...")
            
            # Measure the execution time
            start_time  = datetime.now()
            result      = func(*args, **kwargs)
            end_time    = datetime.now()
            # end of measuring
            
            # Calculate the duration in seconds
            duration    = (end_time - start_time).total_seconds()
            self.debug(f"Finished '{func.__name__}' in {duration:.4f} seconds.")
            return result
        return wrapper

    # --------------------------------------------------------------
    
################################################ PRINT THE OUTPUT BASED ON CONDITION ###############################################

def printV(what: str, v=True, tabulators=0):
    '''
    Prints the message only if verbosity is enabled.
    '''
    if v:
        print("\t" * tabulators + "->" + what)

######################################################## PRINT THE DICTIONARY ######################################################

def printDictionary(d: dict) -> str:
    '''
    Returns a formatted string representation of a dictionary.
    '''
    return ', '.join(f'{key}: {value}' for key, value in d.items())

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
    
    Arguments:
    - width     :   governing the width of each column. 
    - endline   :   if true, puts an endline after the last element of the list.
    - scientific:   allows for scientific printing.
    """
    if not elements:
        return
    
    for item in elements:
        if not scientific:  
            file.write((str(item) + sep).ljust(width))
        else:
            file.write(("{:e}".format(item) + sep).ljust(width))
    if endline:
        file.write("\n")
        
#####################################################################################################################################
