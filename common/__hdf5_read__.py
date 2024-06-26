import sys
# Adds higher directory to python modules path.

from .__directories__ import *
import numpy as np
import h5py
import os

####################################################### READ HDF5 FILE #######################################################

def allbottomkeys(obj):
    bottomkeys  =   []
    def allkeys(obj):
        keys = (obj.name,)
        if isinstance(obj, h5py.Group):
            for key, value in obj.items():
                if isinstance(value, h5py.Group):
                    keys = keys + allkeys(value)
                else:
                    keys = keys + (value.name,)
                    bottomkeys.append(value.name)
        return keys
    allkeys(obj)
    return bottomkeys

def read_hdf5(file, keys = [], verbose = False, removeBad = False):
    '''
    Read the hdf5 saved file
    - keys : if we input keys, they will be used for reading. Otherwise use the available ones.
    '''
    data = {}
    if not os.path.exists(file):
        logging.error(f"directory {file} does not exist")
        return data
    try:
        # check the file
        if file.endswith('.h5') or file.endswith('.hdf5') or file.endswith('.hdf'):
            with h5py.File(file, "r") as f:
                # all root level object names (aka keys) 
                # these can be group or dataset names 
                #keys = f.keys()
                # get object names/keys; may or may NOT be a group
                logging.info(f'keys:{list(f.keys())}', 1)
                a_group_keys = list(allbottomkeys(f)) if len(keys) == 0 else keys
                
                
                logging.debug(f'my_keys:{a_group_keys}', 1)
                # get the object type for a_group_key: usually group or dataset
                #print(type(f[a_group_key])) 

                # If a_group_key is a dataset name, 
                # this gets the dataset values and returns as a list
                #data = list(f[a_group_key])
                
                # preferred methods to get dataset values:
                #ds_obj = f[a_group_key]      # returns as a h5py dataset object
                #ds_arr = f[a_group_key][()]  # returns as a numpy array
                # iterate the keys
                # print(f.keys(), a_group_keys)
                for i in a_group_keys:
                    data[i] = np.array(f[i][()]) 
                
        else:
            logging.info(f"Can't open hdf5 file: {file}")
        return data
        
    except Exception as e:
        print(f"Can't open hdf5 file: {file}")
        print(str(e))
        if "truncated" in str(e) or "doesn't exist" in str(e) and removeBad:
            logging.info(f"Removing {file}")
            os.remove(file)
        elif "setting an array element with a sequence." in str(e):
            logging.info("WTFFF\n\n\nn\nn\n\"}")
            if removeBad:
                logging.info(f"Removing {file}")
                os.remove(file)
        else:
            logging.info(f"Can't open hdf5 file: {file}")
        return {}

####################################################### SAVE HDF5 FILE #######################################################
    
def save_hdf5(directory, filename, data : np.ndarray, shape : tuple, keys = []):
    '''
    Creates and saves an ndarray as hdf5 file.
    - filename : name of the file to be saved
    - data : data to be saved
    - shape : shape of which the data shall be
    - keys : if len(keys) == len(data) we sample that and save each iteration
    '''
    # create a file first
    hf = h5py.File(directory + filename + '.h5', 'w')
    
    # create the labels
    labels = keys if len(keys) == len(data) else ([keys[0]] if len(keys) != 0 else 'green')
    
    # save the file
    if len(labels) == 1:
        hf.create_dataset(labels[0], data=data)
    else:
        for i, lbl in enumerate(labels):
            hf.create_dataset(lbl, data=data[i].reshape(shape))
    # close
    hf.close()

def app_hdf5(directory, filename, data : np.ndarray, key : str):
    '''
    Appends hdf5 file
    - key : given key to append the data with
    '''
    # create a file first
    hf = h5py.File(directory + filename + '.h5', 'a')
    # save
    hf.create_dataset(key, data=data)
    # close
    hf.close()