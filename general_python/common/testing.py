'''
file: general_python/common/testing.py
Contains an abstract class for testing the common functions in the general_python/common module.
'''

import unittest
import time
from typing import Union, Callable
from abc import ABC, abstractmethod
from ..common.flog import get_global_logger

class GeneralTests:
    '''
    Abstract class for testing
    '''
    
    # ---------------------------------------------------------------------------------------------
    
    def __init__(self, backend = 'default'):
        self.backendstr = backend
        self.backend    = backend
        self.logger     = get_global_logger()
        self.tests      = []
        self.tests_dict = {}
        self.change_backend(backend)
        self.test_count = 0
    
    # ---------------------------------------------------------------------------------------------
        
    def _log(self, message, test_number, color="white", log=0, lvl=1):
        '''
        Logs the message with the test number and colorizes the message
        Parameters:
            message: str
                The message to log
            test_number: int
                The test number
            color: str
                The color to colorize the message
            log: int
                The log level
            lvl: int
                The verbosity level
        '''
        
        msg = self.logger.colorize(f"[TEST {test_number}] {message}", color)
        self.logger.say(msg, log=log, lvl=lvl)
        
    # ---------------------------------------------------------------------------------------------
    
    def run_test(self, number = Union[str, int]):
        '''
        Runs the test with the given number
        Parameters:
            number: int
                The test number
        '''
        self._log(f"Running test {number}", number, color="green", log=1)
        t0 = time.time()
        if isinstance(number, str):
            self.tests_dict[number]()
        else:
            self.tests[number]()
        t1 = time.time()
        self._log(f"Test {number} passed in {t1 - t0:.2f} seconds", self.test_count, color="green", log=0)
    
    # ---------------------------------------------------------------------------------------------
    
    def change_backend(self, backend : str):
        ''' Adds the backend to the tests 
        May be overridden by the child class...
        '''
        pass
    
    @abstractmethod
    def add_tests(self):
        '''
        Abstract method to add tests
        '''
        self.tests.append(self.dummy_test)
        self.tests_dict[self.dummy_test.__name__] = self.dummy_test
    
    # ---------------------------------------------------------------------------------------------
    
    def dummy_test(self):
        '''
        Dummy test
        '''
        self._log("Dummy test", 0, color="white", log=1)
    
    def run(self, backend = 'default'):
        '''
        Runs the tests
        '''
        self.change_backend(backend)
        for i, test in enumerate(self.tests):
            # go through the tests
            try:
                self.run_test(i)
            except Exception as e:
                self._log(f"Test {i + 1} failed: {e}", i + 1, color="red", log=1)
                continue
        self._log("All tests passed", self.current_t, color="green", log=0)
            
    # ---------------------------------------------------------------------------------------------
    
###################################################################################################
# Abstract class for testing the algebraic operations
###################################################################################################

class GeneralAlgebraicTest(GeneralTests):
    '''
    Abstract class for testing the algebraic operations
    '''
    
    # ---------------------------------------------------------------------------------------------
    
    def __init__(self, backend = 'default'):
        super().__init__(backend)
    
    # ---------------------------------------------------------------------------------------------
    
    @abstractmethod
    def add_tests(self):
        '''
        Abstract method to add tests
        '''
        pass
    
    # ---------------------------------------------------------------------------------------------
    
    def change_backend(self, backend : str):
        ''' 
        Adds the backend to the tests 
        May be overridden by the child class...
        '''
        from ..algebra.utils import get_backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        else:
            self.backend = backend

    # ---------------------------------------------------------------------------------------------
