import basepath
BASEPATH = basepath.get_basepath(printInfo=True)
import sys
sys.path.insert(0, BASEPATH + 'Leo/code/helperfunctions')
import filehandling
import dataconversions as dc
import cropping
import plotting
from plotting import print_dict
# Standard python libraries
import numpy as np
import matplotlib.pyplot as plt


#%% Define parameters, load metadata

whole_scan_metadata = filehandling.pload(BASEPATH + 'Leo/data/F15/region.pickledump')

segmentation_all_patches = filehandling.pload(BASEPATH + 'Leo/data/F15/segmentation.pickledump')

mets_of_all_patches = segmentation_all_patches['metastases']

ROI_width = 50

CURRENT_PATCH = 403 # good ones: 402, 403, 412, 501, 515, 452, 459, 522, 523, 361, 922

#%% Main code


'''
print('\n')
print("--- Whole scan meta data ---")
plotting.print_dict(whole_scan_metadata)
'''


# counters for number of metastases in patch
mets_count = 0
inner_mets_count = 0
edge_mets_count = 0

  
# print overview of all metastases of current patch
print('-------------------------------------------------------------------')
print('Metastases of patch ' + str(CURRENT_PATCH) + ':')
for met in mets_of_all_patches:
    patch_ID = met['patch_id']
    if patch_ID == CURRENT_PATCH:
        met_ID = met['id']
        met_location = met['offset'] + met['CoM']
        print('ID: ' + str(met_ID) + '\tLocation: ' + str(met_location))
        mets_count += 1   

if mets_count == 0:
    sys.exit("\nPatch " + str(CURRENT_PATCH) + " either has 0 metastases or its Nifti-file doesn't exist")

print('\nPatch ' + str(CURRENT_PATCH) + ' holds ' + str(mets_count) + ' metastases')

# determine neighbour patches
neighbours = cropping.get_neighbours(CURRENT_PATCH)
print('Patch ' + str(CURRENT_PATCH) + ' has ' + str(len(neighbours)) + ' neighbours')
#for neighbour in neighbours:
#    plotting.print_dict(neighbour)


# parameters for plotting
plot_width = 9
plot_rows = 40 # should roughly match the maximum mets_count of a patch
plot_cols = 3 # = number of axes 
plot_position = 1 # determines where on the plot-grid the subplot is placed
plt.figure(figsize=(plot_width,plot_width)) 
 
for met in mets_of_all_patches:
    patch_ID = met['patch_id']
    if patch_ID == CURRENT_PATCH:
        met_ID = met['id']
        met_location = met['offset'] + met['CoM']
        met_absolute_location = met['location']['center']
        print('---------------------------------------')
        # print('Metastasis ' + str(met_ID) + ', located at ' + str(met_location)) 
        print('Metastasis ' + str(met_ID) + ' of patch ' + str(patch_ID))
        print('Relative location: ' + str(met_location))
        print('Absolute location: ' + str(met_absolute_location))
        
        # >>>>>>>> call cropping function here <<<<<<<<
        ROI, zone = cropping.crop_ROI(patch_ID, met_location, met_absolute_location, ROI_width, 'tumorBoost') 
        
        if zone == 'inner':
            inner_mets_count += 1
        elif zone == 'edge' : 
            edge_mets_count += 1
        elif zone == 'no file':
            pass

        if ROI is None: continue
        
        try: 
            # plotting along y-axis
            plt.subplot(plot_rows,plot_cols,plot_position)
            plt.subplots_adjust(left=0.4, right=1.4, bottom=2, top=17)
            MIP_y = np.max(ROI,0) # maximum intensity projection along y
            plt.imshow(MIP_y)
            plt.title('Metastasis ' + str(met_ID) +', MIP along y-axis')
            
            # plotting along x-axis
            plt.subplot(plot_rows,plot_cols,plot_position+1)
            MIP_x = np.max(ROI,1) # maximum intensity projection along x
            plt.imshow(MIP_x)
            plt.title('Metastasis ' + str(met_ID) +', MIP along x-axis')
            
            # plotting along z-axis
            plt.subplot(plot_rows,plot_cols,plot_position+2) 
            MIP_z = np.max(ROI,2) # maximum intensity projection along z
            plt.imshow(MIP_z)
            plt.title('Metastasis ' + str(met_ID) +', MIP along z-axis')
            
            plot_position += 3
        except:
            print('Plotting error')
        
    
if not (inner_mets_count == 0 and edge_mets_count == 0):    
    print('_____________________________')
    print('Number of inner metastases: ' + str(inner_mets_count))
    print('Number of edge metastases: ' + str(edge_mets_count))





                
    