import os
import random
from .__flog__ import *

####################################### OS SEPARATOR AND OS BASED DEFINITIONS #########################################
kPS = os.sep

################################################### MAKE DIRECTORY ####################################################

def makeDir(*args):
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
    return directory

################################################## GO UP DIRECTORY ####################################################

def upDir(dir : str):
    '''
    [summary] 
    Reproduction of ../
    
    [parameters]
    - dir : directory
    '''
    tmp = dir
      
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
    return tmp

################################################### CREATE DIRECTORY ##################################################

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
        
def createFolders(directories : list, silent = False):
    '''
    Create folders from the list of directories.
    '''
    for folder in directories:
        try:
            createFolder(folder, silent)
        except OSError:
            printV("Creation of the directory %s failed" % folder, silent)      
            
########################################### READ A RANDOM FILE FROM A DIRECTORY #######################################

def readRandomFile(folder : str, cond, withoutFolder = False):
    '''
    Reading a random file from a folder
    - folder        : folder to read from
    - withoutFolder : give path without the folder
    '''
    choice = random.choice(os.listdir(folder))
    maxlen = len(os.listdir(folder))
    counter = 0
    while not cond(choice):
        choice = random.choice(os.listdir(folder))
        if counter > maxlen:
            raise Exception("Outside file scope")
        counter += 1
    if withoutFolder:
        return choice
    else:
        return folder + choice
    
############################################## CLEAR THE FILES FROM A DIRECTORY ########################################

def clear_files(directory : str, filelist, empty = True):
    '''
    Clears the empty files in a directory
    - directory     : shall clear this up!
    - fileList      : fileToGoThrough
    - empty         : removes empty files if set to true
    '''
    filelist = list(os.listdir(directory)) if len(filelist) == 0 else filelist
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
            
################################################### LIST DIRECTORY ####################################################

def list_dir(directory      :   str, 
             clearEmpty     =   False, 
             conditions     =   [],
             sortCondition  =   None):
    '''
    Lists a specific directory and gives the files that match a condition.
    - directory     : directory to be listed
    - clearEmpty    : shall clear empty files?
    - conditions    : lambda functions to be applied to filenames or files
    '''
    files   =   list(os.listdir(directory))
    # check clear
    if clearEmpty:
        files   =   clear_files(directory, files)

    # go through conditions
    for condition in conditions:
        files   =   list(filter(condition, files))
        
    if sortCondition is not None:
        files   =   sorted(files, key = sortCondition)
    return list(files)
    
################################################### SAVE DIRECTORY ####################################################

def make_saving_dir(directory      :   str, 
                    verbose        =   False):
    '''
    Makes saving directory with its creation
    '''
    createFolder(directory, not verbose)
    return directory

################################################### SAVE DIRECTORY ####################################################
