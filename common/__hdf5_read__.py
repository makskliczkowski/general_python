import sys
# Adds higher directory to python modules path.

from .__directories__ import *
import numpy as np
import h5py

####################################################### READ HDF5 FILE #######################################################

'''
Read the hdf5 saved file
- keys : if we input keys, they will be used for reading. Otherwise use the available ones.
'''
def read_hdf5(file, keys = [], verbose = False):
    data = {}
    if not os.path.exists(file):
        print(f"directory {file} does not exist")
        return data
    try:
        # check the file
        if file.endswith('.h5'):
            with h5py.File(file, "r") as f:
                # all root level object names (aka keys) 
                # these can be group or dataset names 
                #keys = f.keys()
                # get object names/keys; may or may NOT be a group
                printV(f'keys:{list(f.keys())}', verbose)
                a_group_keys = list(f.keys()) if len(keys) == 0 else keys
                printV(f'my_keys:{a_group_keys}', verbose)
                # get the object type for a_group_key: usually group or dataset
                #print(type(f[a_group_key])) 

                # If a_group_key is a dataset name, 
                # this gets the dataset values and returns as a list
                #data = list(f[a_group_key])
                
                # preferred methods to get dataset values:
                #ds_obj = f[a_group_key]      # returns as a h5py dataset object
                #ds_arr = f[a_group_key][()]  # returns as a numpy array
                # iterate the keys
                for i in a_group_keys:
                    data[i] = np.array(f[i][()])     
        return data
    except:
        print("can't open")
        return {}

####################################################### SAVE HDF5 FILE #######################################################
    
'''
Creates and saves an ndarray as hdf5 file.
- filename : name of the file to be saved
- data : data to be saved
- shape : shape of which the data shall be
- keys : if len(keys) == len(data) we sample that and save each iteration
'''
def save_hdf5(directory, filename, data : np.ndarray, shape : tuple, keys = []):
    # create a file first
    hf = h5py.File(directory + filename + '.h5', 'w')
    
    # create the labels
    labels = keys if len(keys) == len(data) else ['green']
    
    # save the file
    if len(labels) == 1:
        hf.create_dataset(labels[0], data=data)
    else:
        for i, lbl in enumerate(labels):
            hf.create_dataset(lbl, data=data[i].reshape(shape))
    # close
    hf.close()

'''
Appends hdf5 file
- key : given key to append the data with
'''
def app_hdf5(directory, filename, data : np.ndarray, key : str):
    # create a file first
    hf = h5py.File(directory + filename + '.h5', 'a')
    # save
    hf.create_dataset(key, data=data)
    # close
    hf.close()