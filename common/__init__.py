# //general_python/common/__init__.py
# """
# This module contains the following submodules:
# """

# import modules
from . import __binary__        as BinaryModule
from . import directories   as DirectoriesModule
from . import plot          as PlotModule
from . import __datat__         as DataModule
from . import __hdf5_read__     as HDF5Module

def get_module_description(module_name):
    """
    Get the description of a specific module in the common package.
    
    Parameters:
    - module_name (str): The name of the module.
    
    Returns:
    - str: The description of the module.
    """
    descriptions = {
        "__directories__"   : "Provides the Directories class for handling directory operations.",
        "__plot__"          : "Provides the Plotter class for various plotting functions, PlotterSave for saving plot data, and MatrixPrinter for printing matrices and vectors.",
        "__datat__"         : "Provides the DataHandler class for handling data operations.",
        "__hdf5_read__"     : "Provides the HDF5Handler class for reading HDF5 files.",
        "__binary__"        : "Provides the Binary class for handling binary data operations."
    }
    return descriptions.get(module_name, "Module not found.")

def list_available_modules():
    """
    List all available modules in the common package.
    
    Returns:
    - list: A list of available module names.
    """
    return ["__directories__", "__plot__", "__datat__", "__hdf5_read__", "__binary__"]

# Example usage
# print(get_module_description("__directories__"))
# print(list_available_modules())
# print(Directories)
# - Directories class is not imported directly
# print(Plotter)
# print(PlotterSave)
# print(MatrixPrinter)
# - MatrixPrinter class is not imported directly