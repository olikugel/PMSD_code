ROOTP = '/home/olikugel/' # root path
import sys
sys.path.insert(0, ROOTP + 'Leo/code/helperfunctions')
import filehandling
import blobanalysis
import dataconversions
import plotting
import p_leo
from plotting import print_dict as pd

# Standard python libraries
import numpy as np
import matplotlib.pyplot as plt


BASEP = ROOTP + 'Leo/' # base path for project folder

#%% Define parameters

region = filehandling.pload(BASEP + 'data/F15/region.pickledump')


print("The following files are the ones I want to keep:")
to_keep = 0
for patch in region['patches']:
    locationgroup = patch['locationgroup']
    patch_id = patch['id']
    if locationgroup != 'Outside': 
        to_keep += 1
        filename = "patchvolume_" + str(patch_id) + ".nii"
        print(filename + "\tlocation group: " + locationgroup)
        f = open("nifti_files_to_keep.txt", "a")
        f.write(filename + "\n")
        

print("The following files are the ones I want to delete:")
to_delete = 0
for patch in region['patches']:
    locationgroup = patch['locationgroup']
    patch_id = patch['id']
    if locationgroup == 'Outside': 
        to_delete += 1
        filename = "patchvolume_" + str(patch_id) + ".nii"
        print(filename + "\tlocation group: " + locationgroup)
        f = open("nifti_files_to_delete.txt", "a")
        f.write(filename + "\n")
        
total = to_keep + to_delete
print("\n")
print("In total there are " + str(total) + " Nifti files")
print("I want to keep " + str(to_keep))
print("I want to delete " + str(to_delete))
