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

'''
Create single folder listed as a directory
'''
def createFolder(folder : str, silent = False):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        if not silent:
           printV(f"Created a directory : {folder}", silent)

'''
Create folders from the list of directories.
'''
def createFolders(directories : list, silent = False):
    for folder in directories:
        try:
            createFolder(folder, silent)
        except OSError:
            printV("Creation of the directory %s failed" % folder, silent)      
            
########################################### READ A RANDOM FILE FROM A DIRECTORY #######################################

'''
Reading a random file
'''
def readRandomFile(folder : str, cond, withoutFolder = False):
    choice = random.choice(os.listdir(folder))
    maxlen = len(os.listdir(folder))
    counter = 0
    while not cond(choice):
        choice = random.choice(os.listdir(folder))
        if counter > maxlen:
            raise
        counter += 1
    if withoutFolder:
        return choice
    else:
        return folder + choice
    
############################################## CLEAR THE FILES FROM A DIRECTORY ########################################

'''
Clears the empty files in a directory
'''
def clear_files(directory : str, filelist = [], empty = True):
    filelist = list(os.listdir(directory)) if len(filelist) == 0 else filelist
    for filename in filelist:
        removing = not empty
        removing = (not removing) and (os.stat(directory + filename).st_size == 0)
        if removing:
            os.remove(directory + filename)
            printV(f"removed {directory + filename}", True, 2)