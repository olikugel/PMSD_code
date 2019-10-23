import basepath
BASEPATH = basepath.get_basepath()
DATAPATH = BASEPATH + 'Leo/data/F15/'
import sys
import os
sys.path.insert(0, BASEPATH + 'Leo/code/helperfunctions')
import filehandling
import dataconversions as dc
import cropping
from math import sqrt
from plotting import print_dict
import blobanalysis
# Standard python libraries
import numpy as np
import matplotlib.pyplot as plt

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
#%% Read in data

whole_scan_metadata = filehandling.pload(DATAPATH + 'region.pickledump')

segmentation_all_patches = filehandling.pload(DATAPATH + 'segmentation.pickledump')

mets_of_all_patches = segmentation_all_patches['metastases']

# manually add another metastasis to the dataset (one that I stumbled upon)
patch403_met16 = filehandling.pload(DATAPATH + 'm_403_16.pickledump')
patch403_met16['global_id'] = len(mets_of_all_patches)
patch403_met16['patch_id'] = 403
patch403_met16['location'] = {}
patch403_met16['location']['center'] = [4062, 1673, 2676]
mets_of_all_patches.append(patch403_met16)

segmentation_ambig_mets = filehandling.pload(DATAPATH + 'segmentation_ambig_mets.pickledump')

ambig_mets_of_all_patches = segmentation_ambig_mets['ambiguous_metastases']

#%% Parameters

ROI_width = 50
ROI_width_half = int(ROI_width / 2)
tolerance = int(ROI_width / 4)

#%%

def get_surrounding_patch(ROI_center):
    
    for patch in whole_scan_metadata['patches']:
        offset = patch['offset']
        is_within = [False, False, False]
        for coord in range(3): # 0 is y, 1 is x, 2 is z
            if ROI_center[coord] > (offset[coord] - 300) and ROI_center[coord] < (offset[coord] + 300):
                is_within[coord] = True 
        if is_within == [True, True, True]:
            # print('Found surrounding patch for location ' + str(location))
            return patch
        
    # print('No surrounding patch for location ' + str(location))
    return None

#%%

def get_patch_ID_given_offset(offset):
    
    offset = np.array(offset, dtype=np.int64)
    patches = dc.filter_dicts(whole_scan_metadata['patches'], 'offset', offset)
    patch = patches[0]
    patch_ID = patch['id']
    
    return patch_ID

#%%
    
def get_points_of_ROI(ROI, ROI_center):
    offset = np.subtract(ROI_center, [ROI_width_half, ROI_width_half, ROI_width_half])
    points_of_ROI = dc.volume2points(ROI, offset)
    # print('ROI center: ' + str(ROI_center))
    # print('Offset: ' + str(offset))
    return points_of_ROI
         
#%%   
         
def get_ROI_around_location(ROI_center):

    surr_patch = get_surrounding_patch(ROI_center)
    if surr_patch is None: return None
    surr_patch_ID = surr_patch['id']
    # print('PatchID of surrounding patch: ' + str(surr_patch_ID))
    surr_patch_offset = surr_patch['offset']
    
    abs_location = ROI_center # absolute location
    rel_location = np.subtract(abs_location, surr_patch_offset) # relative location within patch
    # print('Location of ROI_center within surrounding patch: ' + str(rel_location))
    
    with HiddenPrints():
        ROI_C02, _ = cropping.crop_ROI(surr_patch_ID, rel_location, abs_location, ROI_width, 'tumorBoost') 
        ROI_C00, _ = cropping.crop_ROI(surr_patch_ID, rel_location, abs_location, ROI_width, 'autofluo')

    return (ROI_C02, ROI_C00)
    # return (ROI_C02, None)
#%%

def is_point_partof_met(point):
   
    point_is_partof_met = False
    
    point_as_list = [point]
    
    for met in mets_of_all_patches:
        
        met_center = met['location']['center']
        distance = blobanalysis.point_dist(met_center, point)
        if distance > ROI_width: continue
        
        points_of_met_rel = met['points'] 
        patchID_of_patch_of_met = met['patch_id']
        patch_of_met = dc.filter_dicts(whole_scan_metadata['patches'], 'id', patchID_of_patch_of_met)[0]
        offset_of_patch_of_met = patch_of_met['offset']
        points_of_met = np.add(points_of_met_rel, offset_of_patch_of_met).tolist() 
        point_is_partof_met = blobanalysis.test_overlap(point_as_list, points_of_met)
        if point_is_partof_met: 
            print('Point ' + str(point) + ' is part of met ' + str(met['global_id']))
            break
    
    return point_is_partof_met
        
 #%%   
 
def is_met_in_ROI(met, ROI, ROI_center, is_ambig_met=False):
    
    met_is_in_ROI = False
    
    met_center = met['location']['center']
    distance = blobanalysis.point_dist(met_center, ROI_center)
    met_descr_string = 'Ambig-Met' if is_ambig_met else 'Met'
    met_descr_string += ' ' + str(met['id']) + ' of Patch ' + str(met['patch_id'])
    
    # is
    if distance < ROI_width_half:
        met_is_in_ROI = True
    
    # is or isn't
    elif distance < (sqrt(3) * ROI_width_half + tolerance):
        points_of_met_rel = met['points'] # pointlist of all points of met, relative to patch
        patchID_of_patch_of_met = met['patch_id']
        patch_of_met = dc.filter_dicts(whole_scan_metadata['patches'], 'id', patchID_of_patch_of_met)[0]
        offset_of_patch_of_met = patch_of_met['offset']
        points_of_met = np.add(points_of_met_rel, offset_of_patch_of_met).tolist() # pointlist of all points of met, abs. positions
        points_of_ROI = get_points_of_ROI(ROI, ROI_center) # pointlist of all points of ROI, abs. positions
        # overlap_points = blobanalysis.get_overlap(points_of_met, points_of_ROI)
        met_is_in_ROI = blobanalysis.test_overlap(points_of_met, points_of_ROI)
        
    # isn't
    else:
        met_is_in_ROI = False
         
    # if met_is_in_ROI: print(met_descr_string + ' is within ROI')
        
    return met_is_in_ROI

#%%  

def count_mets_in_ROI(ROI, ROI_center):

    number_of_mets = 0
    number_of_ambig_mets = 0
    
    for met in mets_of_all_patches:
        met_is_in_ROI = is_met_in_ROI(met, ROI, ROI_center)
        if met_is_in_ROI:
            number_of_mets += 1
      
    for ambig_met in ambig_mets_of_all_patches:
        ambig_met_is_in_ROI = is_met_in_ROI(ambig_met, ROI, ROI_center, is_ambig_met=True)
        if ambig_met_is_in_ROI:
            number_of_ambig_mets += 1
        
    return (number_of_mets, number_of_ambig_mets)

#%%  
    
# --- MAIN ---

'''
# provide desired number of mets, get suitable ROIs
desired_number_of_mets = 2   
for y in range(4100, 4500, 50):
    for x in range(1600, 1700, 50):
        for z in range(2600, 2800, 50):
    
            ROI_center = [y, x, z]
            ROI, _ = get_ROI_around_location(ROI_center)
            
            if ROI is not None:
                number_of_mets_in_ROI, number_of_ambig_mets_in_ROI = count_mets_in_ROI(ROI, ROI_center)
                print('ROI with ' + str(number_of_mets_in_ROI) +  ' metastases at ' + str(ROI_center), end='') 
                if number_of_mets_in_ROI == desired_number_of_mets:
                    print(' <--') 
                    cropping.plot_volume(ROI, title='ROI around ' + str(ROI_center), contrast_set=True, contrast=350)
                else: 
                    print()
                if number_of_ambig_mets_in_ROI != 0:
                    print('  (contains ' + str(number_of_ambig_mets_in_ROI) + ' ambiguous metastases)')       
       
'''

         
'''
# plot ROI around assumed metastasis
ROI_center = [4150, 1650, 2700] # [4062, 1673, 2676]
ROI, _ = get_ROI_around_location(ROI_center)    

for y in range(ROI.shape[0]):
    for x in range(ROI.shape[1]):
        for z in range(ROI.shape[2]):
            position = [y, x, z]
            intensity = ROI[y][x][z]
            if intensity >= 100 and intensity <= 200:
                print('Potentially part of dim-metastasis, at ' + str(position) + ', intensity ' + str(intensity))

number_of_mets_in_ROI = count_mets_in_ROI(ROI, ROI_center)
print('\nROI with ' + str(number_of_mets_in_ROI) +  ' metastases at ' + str(ROI_center)) 
cropping.plot_volume(ROI, title='ROI around ' + str(ROI_center), contrast_set=True, contrast=350) 
'''



'''
newROI_origin = np.subtract(ROI_center, [ROI_width_half, ROI_width_half, ROI_width_half])
newROI_center = np.add(newROI_origin, [23, 25, 16])
ROI_center = newROI_center
'''











