import os
import random
from .__flog__ import *

kPS = os.sep

#######################################################################################################################

class Directories(str):
    """ 
    Class representing a directory handler
    - static methods are represented with camel case
    - class methods are represented with underscore
    """
    
    def __new__(cls, *args):
        return str.__new__(cls, cls.makeDir(*args))
    
    def __init__(self, *args) -> None:
        '''
        Initialize a directory handler.
        '''
        super(Directories, self).__init__()
    
    ############################################################################
    
    def format_str(self, *args, **kwargs) -> "Directories":
        if not args and not kwargs:
            return Directories("")
        return Directories(self.format(*args, **kwargs))
    
    ############################################################################
    
    @staticmethod
    def win(st : str) -> "Directories":
        return Directories(*st.split('\\'))
    
    ############################################################################
    
    @staticmethod
    def makeDir(*args, create = False):
        '''
        [summary] 
        Given a set of folder it creates a directory with correct path separator
        '''
        directory = ""
        for i, arg in enumerate(args):
            if i == 0 and arg.endswith(kPS):
                directory += arg
            else:
                directory += arg + kPS
        # creates if necessary
        if create:
            Directories.createFolder(directory)
        return directory
    
    def make_dir(self, *args, create = False):
        return Directories(Directories.makeDir(*args, create = create))

    ############################################################################
    
    @staticmethod
    def upDir(direct):
        '''
        [summary] 
        Reproduction of ../
        
        [parameters]
        - dir : directory
        '''
        tmp = direct
        
        # check if the directory has path separator at the end already
        if tmp[-1] == kPS:
            tmp = tmp[:-1]
        # remove while we don't get the path separator or to the end of the directory
        while tmp[-1] != kPS and tmp[-1] != '.' and len(tmp) > 0:
            tmp = tmp[:-1]
        # check if it's just a current directory 
        if tmp == '.':
            return ".." + kPS
        elif tmp[-2] + tmp[-1] == '..':
            return ".." + kPS + tmp + kPS
        
        if isinstance(direct, str):
            return Directories(tmp)
        elif isinstance(direct, Directories):
            return str(tmp)
    
    def up_dir(self):
        return Directories(Directories.upDir(self))
        
    ############################################################################
    
    @staticmethod
    def createFolder(folder : str, silent = False):
        '''
        Create single folder listed as a directory
        - folder : folder to be created
        - silent : talk?
        '''
        try:
            if not os.path.isdir(folder):
                os.makedirs(folder)
                if not silent:
                    printV(f"Created a directory : {folder}", silent)
        except OSError:
            printV("Creation of the directory %s failed" % folder, silent)      
    
    @staticmethod
    def createFolders(directories : list, silent = False):
        '''
        Create folders from the list of directories.
        '''
        for folder in directories:
            try:
                Directories.createFolder(folder, silent)
            except OSError:
                printV("Creation of the directory %s failed" % folder, silent)      
    
    def create_folder(self, silent = False):
        Directories.createFolder(self, silent)
        
    ############################################################################

    @staticmethod
    def getRandomFile(folder : str, cond, relative = False):
        '''
        [summary]
        Getting a random file from a folder
        - folder        : folder to read from
        - cond          : lambda function to be applied to the file (if we shall consider it or not)
        - relative      : give path without the folder
        '''
        choice  = random.choice(os.listdir(folder))
        maxlen  = len(os.listdir(folder))
        counter = 0
        while not cond(choice):
            choice  = random.choice(os.listdir(folder))
            if counter > maxlen:
                raise Exception("Outside file scope")
            counter += 1
        if relative:
            return choice
        else:
            return folder + choice
        
    def get_random_file(self, cond, relative = False):
        '''
        [summary]
        Getting a random file from a folder
        - cond          : lambda function to be applied to the file (if we shall consider it or not)
        - relative      : give path without the folder
        '''
        return Directories.getRandomFile(self, cond, relative)
        
    ############################################################################

    @staticmethod
    def clearFiles(directory : str, files = [], empty = True):
        '''
        Clears the empty files in a directory
        - directory     : shall clear this up!
        - files         : fileToGoThrough
        - empty         : removes empty files if set to true
        '''
        filelist = list(os.listdir(directory)) if len(files) == 0 else files
        fileLeft = []
        # go through list of files
        for filename in filelist:
            removing = not empty
            try:
                # try removing if empty file
                removing = (not removing) and (os.stat(directory + filename).st_size == 0)
                if removing:
                    os.remove(directory + filename)
                    printV(f"removed {directory + filename}", True, 2)
                else:
                    fileLeft.append(filename)
            except Exception as inst:
                printV(f"Problem with reading: {inst} - {directory}/{filename}", True, 1)
        return fileLeft

    def clear_files(self, files = [], empty = True):
        return Directories.clearFiles(self, files, empty)
    
    ############################################################################

    @staticmethod
    def listDir(directory      :   str, 
                clearEmpty     =   False, 
                conditions     =   [],
                sortCondition  =   None):
        '''
        Lists a specific directory and gives the files that match a condition.
        - directory     : directory to be listed
        - clearEmpty    : shall clear empty files?
        - conditions    : lambda functions to be applied to filenames or files
        '''
        files       =   list(os.listdir(directory))

        # go through conditions
        for condition in conditions:
            files   =   list(filter(condition, files))

        # check clear
        if clearEmpty:
            files   =   Directories.clearFiles(directory, files, clearEmpty)
            
        if sortCondition is not None:
            files   =   sorted(files, key = sortCondition)
        return list(files)
    
    def list_dir(self, clearEmpty = False, conditions = [], sortCondition = None):
        return Directories.listDir(self, clearEmpty, conditions, sortCondition)
    
    ############################################################################