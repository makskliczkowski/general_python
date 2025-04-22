# Path: Python/QES/general_python/tests/__init__.py
import sys
import os
__current_dir     = os.path.dirname(__file__)
__parent_dir      = os.path.abspath(os.path.join(__current_dir, ".."))
sys.path.append(__parent_dir)

# import all test modules
import algebra
import common
import lattices