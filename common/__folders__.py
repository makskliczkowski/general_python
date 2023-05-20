import os

'''
Create folders from the list of directories.
'''
def createFolder(directories : list, silent = False):
    for folder in directories:
        try:
            if not os.path.isdir(folder):
                os.makedirs(folder)
                if not silent:
                    printV(f"Created a directory : {folder}", silent)
        except OSError:
            print("Creation of the directory %s failed" % folder, silent)      