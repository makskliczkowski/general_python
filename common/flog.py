'''
This module provides a logger class for handling console and file logging with verbosity control.
It includes methods for printing messages with different log levels, formatting titles, and measuring execution time.
and it is designed to be used in a quantum computing context, specifically for the Quantum EigenSolver package.

@note If one wants to use file logging, the environment variable PYLOGFILE should be set to a non-zero value.
@note If one wants to disable colored output, the environment variable PYLOGCOLORS should be set to '0'.

-------------------------------------------------------
file        :   general_python/common/flog.py
author      :   Maksymilian Kliczkowski
email       :   maksymilian.kliczkowski@pwr.edu.pl
date        :   2025-05-01
description :   This module provides a logger class for handling console and file logging with verbosity control.
-------------------------------------------------------
'''

__name__        = "flog"
__version__     = "1.0.0"
__author__      = "Maksymilian Kliczkowski"
__email__       = "maksymilian.kliczkowski@pwr.edu.pl"
__date__        = "2025-05-01"
__description__ = "This module provides a logger class for handling console and file logging with verbosity control."
__license__     = "MIT"
__status__      = "Development"
__all__         = [
    "Logger",
    "Colors",
    "printV",
    "printJust",
    "printDictionary",
    "print_arguments",
    "log_timing_summary",
    "get_global_logger"
]

import os
import re
import sys
import functools
import logging
import getpass
import threading
from datetime import datetime
from typing import Optional, Dict, List

# set the logging level to WARNING
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("flax").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

try:
    # from colorama import init
    # init(autoreset=True)
    __HAS_COLORAMA = False
except ImportError:
    __HAS_COLORAMA = False

######################################################
#! NOTEBOOK / IPYTHON DETECTION
######################################################

def _is_interactive_notebook() -> bool:
    """
    Check if running inside an IPython/Jupyter notebook environment.
    This is used to prevent duplicate log handlers that notebooks often create.
    
    Returns:
        bool: True if running in a notebook or IPython shell, False otherwise.
    """
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is None:
            return False
        # Check for ZMQInteractiveShell (Jupyter) or TerminalInteractiveShell
        return ipy.__class__.__name__ in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except (ImportError, NameError, AttributeError):
        return False

# Track already configured logger names to prevent duplicate handlers
_CONFIGURED_LOGGERS = set()

# In notebook mode, we use a SINGLE shared console handler for ALL loggers
# This prevents duplicate output when multiple Logger instances are created
_SHARED_CONSOLE_HANDLER = None
_NOTEBOOK_MODE_CHECKED = False
_IS_NOTEBOOK = False

######################################################
#! PRINT THE OUTPUT WITH A GIVEN COLOR
######################################################

class Colors:
    """
    Class for defining ANSI colors for console output.
    This class provides a way to colorize text in the terminal using ANSI escape codes.
    
    Attributes:
        black (str):
            ANSI escape code for black text.
        red (str):
            ANSI escape code for red text.
        green (str):
            ANSI escape code for green text.
        yellow (str):
            ANSI escape code for yellow text.
        blue (str):
            ANSI escape code for blue text.
        white (str):
            ANSI escape code for resetting color to default.
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
    
    def __call__(self, text: str) -> str:
        """
        Apply the color to the given text.

        Args:
            text (str): Text to colorize.

        Returns:
            str: Colorized text.
        """
        return f"{self.color}{text}{Colors.white}"
    
    def __len__(self) -> int:
        """
        Get the length of the color string.

        Returns:
            int: Length of the color string.
        """
        return len(self.color)

# Regex for ANSI colour codes (CSI sequences: ESC [ … m)
_ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

class StripAnsiFormatter(logging.Formatter):
    def format(self, record):
        # Format the message using the parent class (inserts time, level, etc.)
        msg = super().format(record)
        # Strip the ANSI codes from the entire formatted string
        return _ansi_escape.sub('', msg)
    
######################################################
#! PRINT THE OUTPUT WITH A GIVEN LEVEL
######################################################

ENV_LOGGER_FILE     = 'PYLOGFILE'
ENV_LOGGER_COLORS   = 'PYLOGCOLORS'

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
    
    def __init__(self,
                name            : str           = "Global",
                logfile         : Optional[str] = None,
                lvl             : int           = logging.INFO,
                append_ts       : bool          = False,
                use_ts_in_cmd   : bool          = False):
        """
        Initialize the logger instance.

        Args:
            logfile (str):
                Name of the log file (without extension if empty, a timestamp will be used).
            lvl (int):
                Logging level (default: logging.INFO).
            append_ts (bool):
                Whether to append a timestamp to the log file name (default: False).
            use_ts_in_cmd (bool):
                Whether to use a timestamp in console output (default: False).
        """
        self.now                = datetime.now()
        self.now_str            = self.now.strftime("%d_%m_%Y_%H-%M_%S")
        self.lvl                = Logger.LEVELS_R.get(lvl, logging.INFO) if isinstance(lvl, str) else lvl
        self.handler_added      = False
        self.use_console_ts     = use_ts_in_cmd
        self.has_colors         = sys.stdout.isatty() and os.environ.get(ENV_LOGGER_COLORS, '1') != '0'
        
        # Set up logging: always show timestamp in console if use_ts_in_cmd is True
        self.logger             = logging.getLogger(name or __name__)
        self.logger.setLevel(self.lvl)
        self.logger.propagate   = False
        
        # Check if this logger was already configured (prevents duplicates in notebooks)
        global _SHARED_CONSOLE_HANDLER, _NOTEBOOK_MODE_CHECKED, _IS_NOTEBOOK
        
        logger_name = name or __name__
        console_fmt = '%(asctime)s [%(levelname)s] %(message)s' if use_ts_in_cmd else '[%(levelname)s] %(message)s'
        
        # Check notebook mode once
        if not _NOTEBOOK_MODE_CHECKED:
            _IS_NOTEBOOK = _is_interactive_notebook()
            _NOTEBOOK_MODE_CHECKED = True
            
            # In notebook environments, clear root logger StreamHandlers
            if _IS_NOTEBOOK:
                root_logger = logging.getLogger()
                for h in list(root_logger.handlers):
                    if isinstance(h, logging.StreamHandler):
                        root_logger.removeHandler(h)
        
        # Clear any existing handlers on this specific logger
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
            h.close()
        
        # In notebook mode, share ONE console handler across ALL loggers
        if _IS_NOTEBOOK:
            if _SHARED_CONSOLE_HANDLER is None:
                _SHARED_CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
                _SHARED_CONSOLE_HANDLER.setLevel(self.lvl)
                _SHARED_CONSOLE_HANDLER.setFormatter(logging.Formatter(console_fmt, datefmt="%d_%m_%Y_%H-%M_%S"))
            self.logger.addHandler(_SHARED_CONSOLE_HANDLER)
        else:
            # Normal mode: each logger gets its own handler (standard behavior)
            if logger_name not in _CONFIGURED_LOGGERS:
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(self.lvl)
                ch.setFormatter(logging.Formatter(console_fmt, datefmt="%d_%m_%Y_%H-%M_%S"))
                self.logger.addHandler(ch)
                _CONFIGURED_LOGGERS.add(logger_name)
            else:
                # Re-add handler if needed
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(self.lvl)
                ch.setFormatter(logging.Formatter(console_fmt, datefmt="%d_%m_%Y_%H-%M_%S"))
                self.logger.addHandler(ch)
        
        # Set the log file name
        if logfile is not None and os.environ.get(ENV_LOGGER_FILE, '0') != '0':
            self.logfile = (logfile.split('.log')[0] if logfile.endswith('.log') else f'{logfile}') if len(logfile) > 0 else self.now_str
            if append_ts:
                self.logfile += f'_{self.now_str}'
            self.configure("./log")
        else:
            self.logfile = self.now_str
        
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
        if not color or color.lower() == 'white' or len(color) == 0:
            return str(txt)
        
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
            author = getpass.getuser() or "unknown"
            f.write('--------------------------------------------------\n')
            f.write('This is the log file for the current session.\n')
            f.write(f'Log file created on {self.now_str}.\n')
            f.write(f'Log level set to: {self.LEVELS[self.lvl]}.\n')
            f.write(f"Author: {author}\n")
            f.write(f"Machine: {os.uname().nodename}\n")
            f.write(f"OS: {os.uname().sysname} {os.uname().release} {os.uname().version}\n")
            f.write(f"Python version: {os.sys.version}\n")
            f.write(f"Python executable: {os.sys.executable}\n")
            f.write(f"Current working directory: {os.getcwd()}\n")
            f.write(f"Log file: {self.logfile}\n")
            f.write(f"Log level: {self.LEVELS[self.lvl]}\n")
            f.write(f"Log file created on {self.now_str}\n")
            f.write(f"Log level set to: {self.LEVELS[self.lvl]}\n")
            f.write('--------------------------------------------------\n')
            
        # Create a file handler only once
        
        # Write initial log file header if the log file is created
        if not self.handler_added:
            self._f_handler = logging.FileHandler(self.logfile, encoding='utf-8')
            self._f_handler.setLevel(Logger.LEVELS_R.get(self.lvl, logging.INFO))
            self._f_handler.setFormatter(StripAnsiFormatter('%(asctime)s [%(levelname)s] %(message)s', datefmt="%d_%m_%Y_%H-%M-%S"))
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
    def print_tab(lvl=0):
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
        # now             = datetime.now()
        # nowStr          = now.strftime("%d/%m/%Y %H:%M:%S")
        formatted_msg   = f"{Logger.print_tab(lvl)}{msg}"
        return formatted_msg

    # --------------------------------------------------------------
    
    def say(self, *args, end=True, log=logging.INFO, lvl=0, verbose=True, color=None):
        """
        Print and log multiple messages if verbosity is enabled.

        Args:
            *args: Messages to log.
            end (bool)      : Append newline (default: True).
            log (int)       : Log level (10 : info, 20 : debug, 30 : warning, 40 : error) (default: 10).
            lvl (int)       : Indentation level.
            verbose (bool)  : Log if True (default: True).
        """
        if isinstance(log, str):
            log = log.lower()
            if log.startswith('i'):
                log = logging.INFO
            elif log.startswith('e'):
                log = logging.ERROR
            elif log.startswith('w'):
                log = logging.WARNING
            elif log.startswith('d'):
                log = logging.DEBUG
            else:
                log = logging.DEBUG
        
        if not verbose or log < self.lvl:
            return
        
        messages = [str(arg) for arg in args]
        
        # if end is True, append newlin
        # Log the combined message only once.
        combined_message = ' '.join(messages) if not end else '\n'.join(messages)
        if color is not None and self.has_colors:
            combined_message = self.colorize(combined_message, color)
        self._log_message(log, combined_message, lvl)
    
    # --------------------------------------------------------------
    
    def _log_message(self, log_level, msg, lvl = 0):
        """
        Internal helper to log messages based on log level.

        Args:
            log_level (int):
                Log level (0: info, 1: debug, etc.).
            msg (str):
                Message to log.
            lvl (int):
                Indentation level.
        """
        log_function = getattr(self.logger, self.LEVELS.get(log_level, 'info'))  # Get the appropriate log function
        log_function(Logger.print(msg, lvl))

    # --------------------------------------------------------------
    
    def info(self, msg: str, lvl=0, verbose=True, color=None):
        """
        Log an informational message if verbosity is enabled.

        Args:
            msg (str)       : Message to log.
            lvl (int)       : Indentation level.
            verbose (bool)  : Log if True (default: True).
        """
        if not verbose:
            return
        if color is not None and self.has_colors:
            msg = self.colorize(msg, color)
        self.logger.info(Logger.print(msg, lvl))
        
    def inf(self, msg: str, lvl=0, verbose=True, color=None):   return self.info(msg, lvl, verbose, color)

    # --------------------------------------------------------------
    
    def debug(self, msg: str, lvl=0, verbose=True, color=None):
        """
        Log a debug message if verbosity is enabled.

        Args:
            msg (str): Message to log.
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
            color (str): Optional color for the message.
        """
        if not verbose:
            return
        if color is not None and self.has_colors:
            msg = self.colorize(msg, color)
        self.logger.debug(Logger.print(msg, lvl))
        
    def dbg(self, msg: str, lvl=0, verbose=True, color=None):   return self.debug(msg, lvl, verbose, color)

    # --------------------------------------------------------------
    
    def warning(self, msg: str, lvl=0, verbose=True, color='yellow'):
        """
        Log a warning message if verbosity is enabled.

        Args:
            msg (str): Message to log.
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
        """
        if not verbose:
            return
        if self.has_colors:
            msg = self.colorize(msg, color)
        self.logger.warning(Logger.print(msg, lvl))
        
    def warn(self, msg: str, lvl=0, verbose=True, color='yellow'): return self.warning(msg, lvl, verbose, color)

    # --------------------------------------------------------------

    def error(self, msg: str, lvl=0, verbose=True, color='red'):
        """
        Log an error message if verbosity is enabled.

        Args:
            msg (str): Message to log.
            lvl (int): Indentation level.
            verbose (bool): Log if True (default: True).
            color (str): Optional color for the message.
        """
        if not verbose:
            return
        if self.has_colors:
            msg = self.colorize(msg, 'red')
        self.logger.error(Logger.print(msg, lvl))
        
    def err(self, msg: str, lvl=0, verbose=True, color='red'): return self.error(msg, lvl, verbose, color)    
        
    # --------------------------------------------------------------
    
    @classmethod
    def endl(cls, n: int):
        return cls.breakline(n)
    
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

    def title(self, tail: str, desired_size: int=50, fill: str = '=', lvl=0, verbose=True, color=None):
        """
        Create a formatted title with filler characters if verbosity is enabled.

        Args:
            tail (str):
                Text in the middle of the title.
            desired_size (int):
                Total width of the title.
            fill (str):
                Character used for filling.
            lvl (int):
                Indentation level.
            verbose (bool):
                Log if True (default: True).
            color (str):
                Optional color for the title.
        """
        if not verbose:
            return
        tailLength  = len(tail)
        lvlLen      = 2 + lvl * 3 * 2
        if tailLength + lvlLen > desired_size:
            self.info(tail, lvl, verbose)
            return
        
        fillSize    = (desired_size - tailLength) // (2 * len(fill))
        out         = (fill * fillSize) + f"{tail}" + (fill * fillSize)
        
        if len(out) < desired_size:
            out += fill[0] * (desired_size - len(out) - 1)
        elif len(out) > desired_size:
            out = out[:desired_size]
            
        self.info(out, lvl, verbose, color)

    # --------------------------------------------------------------
    
    def timing(self, func):
        """
        Decorator to measure and log the execution time of functions.
        Parameters:
            func : function to be timed
            
        Use as:
            @logger.timing
            def my_function(...):
                ...
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

_G_LOGGER     = None
_G_LOGGER_PID = None
_G_LOCK       = threading.Lock()

def get_global_logger(**kwargs) -> Logger:
    """
    One Logger wrapper per process (PID), safe across threads/forks.
    Prints the banner only once per entire program via env sentinel.
    
    Args:
        **kwargs: Arguments to pass to the Logger constructor.
        - name (str): Name of the logger (default: "Global").
        - lvl (int): Logging level (default: logging.INFO).
        - append_ts (bool): Whether to append timestamps (default: True).
        - use_ts_in_cmd (bool): Whether to use timestamps in commands (default: True).
        - logfile (str or None): Path to a logfile (default: None).
    
    Returns:
        Logger: The global logger instance.
    
    Example
    -------
        >>> logger = get_global_logger()
        >>> logger.info("This is an informational message.")
        >>> logger.debug("This is a debug message.", color='blue')
    """
    global  _G_LOGGER, _G_LOGGER_PID
    pid     = os.getpid()
    
    if _G_LOGGER is not None and _G_LOGGER_PID == pid:
        return _G_LOGGER

    with _G_LOCK:
        if _G_LOGGER is not None and _G_LOGGER_PID == pid:
            return _G_LOGGER

        logger = Logger(
            name            = kwargs.get("name", "Global"),
            lvl             = kwargs.get("lvl",             logging.INFO),
            append_ts       = kwargs.get("append_ts",       True),
            use_ts_in_cmd   = kwargs.get("use_ts_in_cmd",   True),
            logfile         = kwargs.get("logfile",         None),
        )

        # Print the banner only once per program (env is inherited by forked workers)
        if os.environ.get("GEN_PYTHON_LOGGER_INIT_DONE", "0") != "1":
            os.environ["GEN_PYTHON_LOGGER_INIT_DONE"] = "1"
            
            # Print the banner            
            if os.environ.get("PY_BACKEND_INFO", "0") != "0":
                logger.title("Global Logger initialized!", 50, '#', 0)

        _G_LOGGER       = logger
        _G_LOGGER_PID   = pid
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

def get_example_usage() -> str:
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
#! Parsing helper
######################################################

def print_arguments(parser,
                    logger      : Optional[Logger] = None,
                    title       : str = "Options for the script",
                    columnsize  : int = 30) -> None:
    """
    Prints the arguments of a parser in a formatted table using the provided logger.
    Args:
        parser (argparse.ArgumentParser):
            The argument parser containing the script's options.
        logger (Optional[Logger]):
            The logger instance used to print the formatted output.
        title (str, optional):
            The title to display above the options table. Defaults to "Options for the script".
        columnsize (int, optional):
            The width of the "Option" column in the table. Defaults to 30.
    Returns:
        None
    """
    _default_size       = 15
    _description_size   = 70
    if logger is None:
        # Use print instead of logger
        print("\n")
        print(f"Title: {title}")
        
        # table
        print(
            f"|{'-' * (columnsize + 2)}|{'-' * (_default_size + 2)}|{'-' * (_description_size + 2)}|"
        )
        print(
            f"| {'Option':<{columnsize}} | {'Default':<{_default_size}} | {'Description':<{_description_size}} |"
        )
        print(
            f"|{'-' * (columnsize + 2)}|{'-' * (_default_size + 2)}|{'-' * (_description_size + 2)}|"
        )
        # Print the options
        for action in parser._actions:
            option      = f"| {action.dest:<{columnsize}} | {str(action.default):<{_default_size}} |"
            description = f" {action.help:<{_description_size}} |"
            print(option + description)
        print(f"|{'-' * (columnsize + 2)}|{'-' * (_default_size + 2)}|{'-' * (_description_size + 2)}|")
    else:
        # Print the options at the beginning
        logger.breakline(1)
        logger.title(title, 50, '#', 0)
        
        # table
        logger.info(f"|{'-' * (columnsize + 2)}|{'-' * (_default_size + 2)}|{'-' * (_description_size + 2)}|", lvl=0)
        logger.info(
            f"| {'Option':<{columnsize}} | {'Default':<{_default_size}} | {'Description':<{_description_size}} |",
            lvl=0
        )
        logger.info(
            f"|{'-' * (columnsize + 2)}|{'-' * (_default_size + 2)}|{'-' * (_description_size + 2)}|",
            lvl=0
        )
        
        # Print the options
        for action in parser._actions:
            option      = f"| {action.dest:<{columnsize}} | {str(action.default):<{_default_size}} |"
            description = f" {action.help:<{_description_size}} |"
            logger.info(option + description, lvl=0)
        logger.info(f"|{'-' * (columnsize + 2)}|{'-' * (_default_size + 2)}|{'-' * (_description_size + 2)}|", lvl=0)
        logger.breakline(1)
        
######################################################

def log_timing_summary(
    logger              : Logger,
    phase_durations     : Dict[str, float],
    total_duration      : Optional[float] = None,
    title               : str = "Timing Summary",
    phase_col_width     : int = 18,      # Adjusted width for potentially longer names + "Total"
    duration_col_width  : int = 14,      # Width for "Duration (s)" column header and values
    duration_precision  : int = 4,       # Decimal places for duration
    lvl                 : int = 0,
    add_total_row       : bool = True,   # Flag to add a "Total" row if total_duration is provided
    extra_info          : Optional[List[str]] = None
):
    """
    Logs a timing summary in a tabular format using the provided logger.

    Parameters:
    logger:
        Logger instance to log the timing summary.
    phase_durations:
        Dictionary mapping phase names (str) to their durations (float).
    total_duration:
        Total duration of the process (optional). If provided and
        add_total_row is True, a "Total" row will be added.
    title:
        Title for the summary table.
    phase_col_width:
        Width for the 'Phase' column.
    duration_col_width:
        Width for the 'Duration (s)' column.
    duration_precision:
        Decimal precision for duration values.
    lvl:
        Base logging level for the summary.
    add_total_row:
        Whether to include a 'Total' row using total_duration.
    extra_info:
        Optional list of strings to log after the table (e.g., notes, performance).
    """
    if not logger:
        print("Error: Logger instance is required for log_timing_summary.")
        return

    # Table Structure
    phase_header        = "Phase"
    duration_header     = "Duration (s)"
    # Ensure columns are wide enough for headers
    phase_col_width     = max(phase_col_width, len(phase_header))
    duration_col_width  = max(duration_col_width, len(duration_header))

    # Horizontal separator line
    separator           = f"|{'-' * (phase_col_width + 2)}|{'-' * (duration_col_width + 2)}|"
    # Header format string (Phase left-aligned, Duration right-aligned)
    header_fmt          = f"| {phase_header:<{phase_col_width}} | {duration_header:>{duration_col_width}} |"
    # Data row format string (Phase left-aligned, Duration right-aligned with precision)
    row_fmt             = f"| {{phase_name:<{phase_col_width}}} | {{duration:>{duration_col_width}.{duration_precision}f}} |"

    # Logging
    logger.title(f"{title}", 50, '#', lvl)
    
    # Print extra info above the table if desired (e.g., JAX note)
    perf_string         = None
    if extra_info:
        # Filter out performance string if present, handle it later
        filtered_extra_info = []
        for info in extra_info:
            if "samples/sec" in info.lower() or "performance" in info.lower() :
                perf_string = info
            else:
                filtered_extra_info.append(info)

        for info in filtered_extra_info:
            logger.info(info, lvl=lvl + 1)

    # Print table header
    logger.info(separator, lvl=lvl + 1)
    logger.info(header_fmt, lvl=lvl + 1)
    logger.info(separator, lvl=lvl + 1)

    # Print phase durations
    calculated_sum = 0.0
    if phase_durations:
        for name, duration in phase_durations.items():
            logger.info(row_fmt.format(phase_name=name, duration=duration), lvl=lvl + 1)
            calculated_sum += duration
    else:
        logger.info(f"| {'No phases timed':<{phase_col_width + duration_col_width + 3}} |", lvl=lvl+1) # Indicate if dict is empty


    # Add Total row if requested and possible
    if add_total_row:
        logger.info(separator, lvl=lvl + 1)
        actual_total = total_duration if total_duration is not None else calculated_sum
        # Warn if provided total differs significantly from sum of phases
        if total_duration is not None and not np.isclose(total_duration, calculated_sum, rtol=1e-3, atol=1e-4):
            logger.warning(f"Provided total duration ({total_duration:.4f}s) differs from sum of phases ({calculated_sum:.4f}s). Using provided total.", lvl=lvl+2)

        if actual_total is not None:
            logger.info(row_fmt.format(phase_name="Total", duration=actual_total), lvl=lvl + 1)
        else:
            logger.info(f"| {'Total duration not available':<{phase_col_width + duration_col_width + 3}} |", lvl=lvl+1)


    # Print final table separator
    logger.info(separator, lvl=lvl + 1)

    # Print performance string (if captured earlier) below the table
    if perf_string:
        logger.info(perf_string, lvl=lvl+1)
        
########################################################
