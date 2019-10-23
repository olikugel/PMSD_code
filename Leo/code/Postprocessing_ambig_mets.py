import basepath
BASEPATH = basepath.get_basepath()
DATAPATH = BASEPATH + 'Leo/data/F15/'
import sys
sys.path.insert(0, BASEPATH + 'Leo/code/helperfunctions')
import filehandling
import p_leo
import numpy as np

#%% Parameters

whole_scan_metadata = filehandling.pload(BASEPATH + 'Leo/data/F15/region.pickledump')

segmentation_all_patches = filehandling.pload(BASEPATH + 'Leo/data/F15/segmentation.pickledump')

mets_of_all_patches = segmentation_all_patches['metastases']

# manually add another metastasis to the dataset (one that I stumbled upon)
patch403_met16 = filehandling.pload(BASEPATH + 'Leo/data/F15/m_403_16.pickledump')
patch403_met16['global_id'] = len(mets_of_all_patches)
patch403_met16['patch_id'] = 403
patch403_met16['location'] = {}
patch403_met16['location']['center'] = [4062, 1673, 2676]
mets_of_all_patches.append(patch403_met16)

#%% Step 1) Merge all segmentations

segmentation_ambig_mets = {}
segmentation_ambig_mets['name'] = whole_scan_metadata['name']
segmentation_ambig_mets['dataset'] = whole_scan_metadata['dataset']
segmentation_ambig_mets['ambiguous_metastases'] = []
global_id = len(mets_of_all_patches) # = 488
for patch in whole_scan_metadata['patches']:
    patch_seg = p_leo.load_segmentation('F15', patch['id'], load_partial=False)
    if(len(patch_seg['ambiguous_metastases']) > 0):
        print("Adding ambig_mets of patch #" + str(patch['id']))
        localpointlist = []
        for ambig_metastasis in patch_seg['ambiguous_metastases']:
            ambig_metastasis['global_id'] = global_id
            ambig_metastasis['patch_id'] = patch['id']
            segmentation_ambig_mets['ambiguous_metastases'].append(ambig_metastasis)
            global_id += 1
       
filehandling.psave(DATAPATH + 'segmentation_ambig_mets', segmentation_ambig_mets)
print('Segmentations_ambig_mets merged & saved.')


#%% Step 2) For each metastasis, add all relevant meta information

for m, ambig_metastasis in enumerate(segmentation_ambig_mets['ambiguous_metastases']): # double loop as this will be needed for upcoming computations
    print('Adding information for ambig_metastasis #' + str(ambig_metastasis['global_id']))
    # add global location information to metastasis
    segmentation_ambig_mets['ambiguous_metastases'][m]['location'] = {}
    p_offset = np.asarray(whole_scan_metadata['patches'][ambig_metastasis['patch_id']]['offset'])
    segmentation_ambig_mets['ambiguous_metastases'][m]['location']['offset'] = (p_offset + ambig_metastasis['offset']).tolist()
    segmentation_ambig_mets['ambiguous_metastases'][m]['location']['center'] = (p_offset + ambig_metastasis['offset'] + ambig_metastasis['CoM']).tolist()
    
filehandling.psave(DATAPATH + 'segmentation_ambig_mets', segmentation_ambig_mets)
print('Meta information added for all ambiguous metastases and saved to file.')









