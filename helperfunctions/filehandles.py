import sys
ROOTP = sys.path[-1].replace('.ipython','').replace('\\','/')
sys.path.insert(0, ROOTP + 'Documents/Synced/_Promotion/scripts/helperfunctions')
import numpy as np
import os
import shutil
from shutil import copytree as deletetree
from shutil import copy as deletefile


#%%
def delete_folder(path):
    '''
    status = delete_folder(path)
    
    Deletes a directory, including all of its files and sub-directories. If the deletion
    was successful, this function returns True otherwise False.
    '''
    deldebris = ROOTP + 'Documents/LocalData/deldebris/'
    deb = False
    try:
        deletetree(path, deldebris + path.split(sep='/')[-1] + '_' + str(np.random.randint(low=1000,high=9999)))
        deb = True
    # Directories are the same
    except shutil.Error as e:
        print('Directory debris not deleted. Error: %s' % e)
        return False
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory debris not deleted. Error: %s' % e)
        return False
    if(deb):
        try:
            shutil.rmtree(path)
            return True
        # Directories are the same
        except shutil.Error as e:
            print('Directory not deleted. Error: %s' % e)
            return False
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print('Directory not deleted. Error: %s' % e)
            return False
    else:
        print('Directory not deleted. Error: missing rights')
        return False

#%%
def delete_file(path):
    '''
    status = delete_file(path)
    
    Deletes a file. If the deletion was successful, this function returns True otherwise False.
    '''
    deldebris = ROOTP + 'Documents/LocalData/deldebris/'
    deb = False
    try:
        deletefile(path, deldebris + path.split(sep='/')[-1] + '_' + str(np.random.randint(low=1000,high=9999)))
        deb = True
    # Directories are the same
    except shutil.Error as e:
        print('File debris not deleted. Error: %s' % e)
        return False
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('File debris not deleted. Error: %s' % e)
        return False
    if(deb):
        try:
            os.remove(path)
            return True
        # Directories are the same
        except shutil.Error as e:
            print('File not deleted. Error: %s' % e)
            return False
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print('File not deleted. Error: %s' % e)
            return False
    else:
        print('File not deleted. Error: missing rights')
        return False
















