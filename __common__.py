""" Standard numerical """
import numpy as np
import itertools
import os

############################################### CHECK ALL EQUAL ELEMENTS ###############################################
'''
In iterable, check if all items are equal.
- iterator : iterator to be checked
'''
def allEqual(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

