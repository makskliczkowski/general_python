##################### G E N E R A L   I M P O R T S #####################
import sys
# Adds higher directory to python modules path.
sys.path.append("./common/")

""" Standard numerical """
import numpy as np
import itertools
import os

"""       Flogger      """
from common.__flog__ import *
loggingLvl  =   logging.INFO
logger      =   Logger("", loggingLvl)
logger.info("Including common files to the project.", 0)

############################################### CHECK ALL EQUAL ELEMENTS ###############################################
def allEqual(iterator):
    '''
    In iterable, check if all items are equal.
    - iterator : iterator to be checked
    '''
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

############################################ CHECK THE INDICES OF DUPLICATES ###########################################
def listDups(seq, item):
    '''
    In a given sequence find the list of duplicate indices
    - seq   : sequence to find in
    - item  : item that is repeated
    '''
    start_at    = -1
    locs        = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs