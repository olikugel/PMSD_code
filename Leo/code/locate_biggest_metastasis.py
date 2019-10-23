ROOTP = '/home/olikugel/OlisIDP/' # root path
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
segmentation = filehandling.pload(BASEP + 'data/F15/segmentation.pickledump')
segmentation_patch_515 = filehandling.pload(BASEP + 'data/F15/segmentations/segmentation_patch_515.pickledump')

tumor_volume_515 = filehandling.readNifti(ROOTP + 'LocalData/F15/C02_tumorBoost/patchvolume_515.nii') # we're reading in a .nii file, so it's real image data, not just metadata

#%% Main code


# print dictionary in structured format to console
'''
print("--- Region (meta data for patch locations):")
plotting.print_dict(region)
print("\n\n")

print("--- Segmentation (aggregated list of metastases across all patches):")
plotting.print_dict(segmentation)
'''

print("\n\n")
print("--- Single segmentation (list of metastases for one patch):")
plotting.print_dict(segmentation_patch_515)

'''
# Print metastases of patch 515 (probably we don't need this)
print("\nHere are all the metastases of patch 515, described by ID and Volume")
metastases_in_patch515 = segmentation_patch_515['metastases']
for metastasis in metastases_in_patch515:
    print('ID:',str(metastasis['id']),'Volume:',str(metastasis['volume']))
print("There are " + str(len(metastases_in_patch515)) + " metastases in patch 515")
'''

# ALTERNATIVE:  Print metastases of patch 515
print("Here are all metastases of patch 515, described by their global ID and volume:")
all_metastases = segmentation['metastases']
metastases_in_p515 = {}
for metastasis in all_metastases:
    if metastasis['patch_id'] == 515: # given the metastasis is in patch 515
        metastases_in_p515[metastasis['global_id']] = metastasis['volume']
plotting.print_dict(metastases_in_p515)

ID_of_biggest_metastasis = max(metastases_in_p515, key=metastases_in_p515.get)
volume_of_biggest_metastasis = metastases_in_p515[ID_of_biggest_metastasis]

print("The biggest metastasis in this patch is metastasis " + str(ID_of_biggest_metastasis) \
+ " with a volume of " + str(volume_of_biggest_metastasis) + " pixels.")

# determine location of biggest mestastasis
for metastasis in all_metastases:
    if metastasis['global_id'] == ID_of_biggest_metastasis:
        location_of_biggest_metastasis = metastasis['offset'] + metastasis['CoM']
print("It is located at " + str(location_of_biggest_metastasis) + " within the patch.")
y_offset = location_of_biggest_metastasis[0]
x_offset = location_of_biggest_metastasis[1]
z_offset = location_of_biggest_metastasis[2]

# plot patch 515 as maximum intensity projections from three different axes
plotwidth = 14
plt.figure(num=0, figsize=(plotwidth,plotwidth)) # plt = plot
plt.clf() # clf = clear current figure

plt.subplot(2,2,1)
MIP_y = np.max(tumor_volume_515,0) # maximum intensity projection  (second argument is the axis)
MIP_y = np.clip(MIP_y,0,600) # adjust contrast
plt.annotate('Here',xy=(z_offset,x_offset), xycoords='data',
             xytext=(55,325), textcoords='data',
             arrowprops=dict(arrowstyle='->',color='white'), color='white')
plt.imshow(MIP_y, vmin=0,vmax=500)
plt.title('MIP along y-axis, adj. contrast, biggest metastasis')

plt.subplot(2,2,2)
MIP_x = np.max(tumor_volume_515,1) # maximum intensity projection
MIP_x = np.clip(MIP_x,0,600) # adjust contrast
plt.annotate('Here',xy=(z_offset,y_offset), xycoords='data',
             xytext=(55,325), textcoords='data',
             arrowprops=dict(arrowstyle='->',color='white'), color='white')
plt.imshow(MIP_x)
plt.title('MIP along x-axis, adj. contrast, biggest metastasis')

plt.subplot(2,2,3) 
MIP_z = np.max(tumor_volume_515,2) # maximum intensity projection
MIP_z = np.clip(MIP_z,0,600) # adjust contrast
plt.annotate('Here',xy=(x_offset,y_offset), xycoords='data',
             xytext=(55,325), textcoords='data',
             arrowprops=dict(arrowstyle='->',color='white'), color='white')
plt.imshow(MIP_z)
plt.title('MIP along z-axis, adj. contrast, biggest metastasis')

'''
plt.subplot(2,2,4)
CS_y = tumor_volume_515[175,:,:] # choose center slice of the patch (3D -> 2D)  [y=150,x=all,z=all]
plt.imshow(CS_y,vmin=0,vmax=500)
plt.title('Center slice (along y-axis)')
'''