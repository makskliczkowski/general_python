# Path: Python/QES/general_python/tests/__init__.py
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# append top level directory to path
sys.path.append("..")

# import all test modules
from .. import algebra
from .. import common
from .. import lattices