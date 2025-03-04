from datetime import datetime

import os
import functools
import logging

# set the logging level to WARNING
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("flax").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

try:
    from colorama import init
    init(autoreset=True)
    __HAS_COLORAMA = True
except ImportError:
    __HAS_COLORAMA = False

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
        logging.DEBUG   : 'debug',
        logging.INFO    : 'info',
        logging.WARNING : 'warning',
        logging.ERROR   : 'error'
    }
    LEVELS_R = {v: k for k, v in LEVELS.items()}
    
    def __init__(self, logfile: str, lvl=logging.INFO, append_ts = False):
        """
        Initialize the logger instance.

        Args:
            logfile (str): Name of the log file (without extension if empty, a timestamp will be used).
            lvl (int): Logging level (default: logging.INFO).
            
        """
        self.now            = datetime.now()
        self.now_str        = self.now.strftime("%d_%m_%Y_%H-%M_%S")
        self.lvl            = lvl
        self.logger         = logging.getLogger(__name__)
        logging.basicConfig(level=lvl, format='%(asctime)s [%(levelname)s] %(message)s', datefmt="%d_%m_%Y_%H-%M_%S")
        self.logfile        = (logfile.split('.log')[0] if logfile.endswith('.log') else f'{logfile}') if len(logfile) > 0 else self.now_str
        if append_ts:
            self.logfile    += self.now_str
        self.handler_added  = False
        self.configure("./log")
        
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
        base_name       = self.now_str if len(self.logfile) == 0 else self.logfile
        self.logfile    = os.path.join(directory, f'{base_name}.log')
        
        # Create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)
        # reset file handler
        self.working    = False
        
        # Write initial log file header
        with open(self.logfile, 'w+') as f:
            f.write('This is the log file for the current session.\n')
            f.write(f'Log file created on {self.now_str}.\n')
            f.write(f'Log level set to: {self.LEVELS[self.lvl]}.\n')
            f.write(f"Author: {os.getlogin()}\n")
            f.write(f"Machine: {os.uname().nodename}\n")
            f.write(f"OS: {os.uname().sysname} {os.uname().release} {os.uname().version}\n")
            f.write('--------------------------------------------------\n')
            
        # Create a file handler only once

        
        # Write initial log file header if the log file is created
        if not self.handler_added:
            self._f_handler = logging.FileHandler(self.logfile, encoding='utf-8')
            self._f_handler.setLevel(Logger.LEVELS_R.get(self.lvl, logging.INFO))
            self._f_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt="%d_%m_%Y_%H-%M_%S")
            self._f_handler.setFormatter(self._f_format)
            self.logger.addHandler(self._f_handler)

            # Add a stream handler for console output
            # self._s_handler = logging.StreamHandler()
            # self._s_handler.setLevel(Logger.LEVELS_R.get(self.lvl, logging.INFO))
            # self._s_handler.setFormatter(self._f_format)  # Use the same formatter
            # self.logger.addHandler(self._s_handler)

            self.handler_added = True
            self._log_message(logging.INFO, f"Log file created: {self.logfile}")
            self._log_message(logging.INFO, f"Log level set to: {self.LEVELS[self.lvl]}")
            
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
        now             = datetime.now()
        # nowStr          = now.strftime("%d/%m/%Y %H:%M:%S")
        formatted_msg   = f"{Logger.printTab(lvl)}{msg}"
        return formatted_msg

    # --------------------------------------------------------------
    
    def say(self, *args, end=True, log=logging.INFO, lvl=0, verbose=True):
        """
        Print and log multiple messages if verbosity is enabled.

        Args:
            *args: Messages to log.
            end (bool)      : Append newline (default: True).
            log (int)       : Log level (10 : info, 20 : debug, 30 : warning, 40 : error) (default: 10).
            lvl (int)       : Indentation level.
            verbose (bool)  : Log if True (default: True).
        """
        if not verbose or log < self.lvl:
            return
        
        messages = [str(arg) for arg in args]
        
        # if end is True, append newlin
        # Log the combined message only once.
        combined_message = ' '.join(messages) if not end else '\n'.join(messages)
        self._log_message(log, combined_message, lvl)
    
    # --------------------------------------------------------------
    
    def _log_message(self, log_level, msg, lvl = 0):
        """
        Internal helper to log messages based on log level.

        Args:
            log_level (int): Log level (0: info, 1: debug, etc.).
            msg (str): Message to log.
            lvl (int): Indentation level.
        """
        log_function = getattr(self.logger, self.LEVELS.get(log_level, 'info'))  # Get the appropriate log function
        log_function(Logger.print(msg, lvl))

    # --------------------------------------------------------------
    
    def info(self, msg: str, lvl=0, verbose=True):
        """
        Log an informational message if verbosity is enabled.

        Args:
            msg (str)       : Message to log.
            lvl (int)       : Indentation level.
            verbose (bool)  : Log if True (default: True).
        """
        if not verbose:
            return
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
            
    def _breakline(self, n: int):
        """
        Log multiple break lines.

        Args:
            n (int): Number of break lines.
        """
        for _ in range(n):
            self.logger.info('')
            
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
        tailLength  = len(tail)
        lvlLen      = 2 + lvl * 3 * 2
        if tailLength + lvlLen > desiredSize:
            self.info(tail, lvl, verbose)
            return
        
        fillSize    = (desiredSize - tailLength - 2) // (2 * len(fill))
        out         = (fill * fillSize) + f" {tail} " + (fill * fillSize)
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

######################################################

# Global logger instance (starts as None)
_G_LOGGER = None

def get_global_logger():
    """
    Lazily initializes and returns the global logger instance.
    
    Returns:
    - Logger instance.
    """
    global _G_LOGGER
    if _G_LOGGER is None:
        _G_LOGGER = Logger("global")
        _G_LOGGER.title("Global logger initialized.", 50, '#', 0)
    return _G_LOGGER

######################################################

# Example usage
# logger = get_global_logger()
# logger.info("This is an informational message.")
# logger.debug("This is a debug message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.say("This is a message.", log=0, lvl=0, verbose=True)
# logger.say("This is a message.", log=1, lvl=0, verbose=True)
# logger.say("This is a message.", log=2, lvl=0, verbose=True)
# logger.say("This is a message.", log=3, lvl=0, verbose=True)

def get_example_usage():
    """
    Returns example usage of the global logger.

    Returns:
    - str: A string containing example usage of the logger.
    """
    return 'logger = get_global_logger()' + \
    '\nlogger.info("This is an informational message.")' + \
    '\nlogger.debug("This is a debug message.")' + \
    '\nlogger.warning("This is a warning message.")' + \
    '\nlogger.error("This is an error message.")' + \
    '\nlogger.say("This is a message.", log=0, lvl=0, verbose=True)' + \
    '\nlogger.say("This is a message.", log=1, lvl=0, verbose=True)' + \
    '\nlogger.say("This is a message.", log=2, lvl=0, verbose=True)...'
    
######################################################