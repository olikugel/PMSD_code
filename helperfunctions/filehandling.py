import os 
#import cv2 
import numpy as np
#from libtiff import TIFFfile
#from libtiff import TIFF
import re
import pickle
#import nibabel as nib


#%%
def psave(path, variable):
    '''
    psave(path, variable)
    
    Takes a variable (given as string with its name) and saves it to a file as specified in the path.
    The path must at least contain the filename (no file ending needed), and can also include a 
    relative or an absolute folderpath, if the file is not to be saved to the current working directory.
    
    # ToDo: save several variables (e.g. take X args, store them to special DICT, and save to file)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    cwd = os.getcwd().replace('\\','/')
    if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
        path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    file = open(path, 'wb')
    pickle.dump(variable,file,protocol=4)


#%%
def pload(path):
    '''
    variable = pload(path)
    
    Loads a variable from a file that was specified in the path. The path must at least contain the 
    filename (no file ending needed), and can also include a relative or an absolute folderpath, if 
    the file is not to located in the current working directory.
    
    # ToDo: load several variables (e.g. load special DICT from file and return matching entries)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    cwd = os.getcwd().replace('\\','/')
    if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
        path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    file = open(path, 'rb')
    return pickle.load(file)


#%% little helper function for ID padding, returns ID as string
def pad_ID(ID):
    
    if ID >= 0 and ID < 10:
        ID_padded = '000' + str(ID)
    elif ID >= 10 and ID < 100:
        ID_padded = '00' + str(ID)
    elif ID >= 100 and ID < 1000:
        ID_padded = '0' + str(ID)
    elif ID >= 1000 and ID < 10000:
        ID_padded = str(ID)
    else:
        print('pad_ID helperfunction is not implemented for IDs >= 10000 or < 0')
        return None
    
    return ID_padded
        
