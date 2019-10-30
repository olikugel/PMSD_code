import sys
ROOTP = sys.path[-1].replace('.ipython','').replace('//','/')
sys.path.insert(0, ROOTP + 'Documents/Synced/_Promotion/scripts/helperfunctions')
import filehandling
import dataconversions
import blobanalysis
import p_leo
from plotting import print_dict as pd

import numpy as np
from shapely.geometry import Point, Polygon

BASEP = ROOTP + 'Documents/Synced/_Promotion/Projects/Leo/'

#%%
#
# This script adds any parameters/quantifications to the metastases that are needed for any analysis.
# Thus, this script should be run after finalizing the segmentations and before running any analysis.
# For each metastasis, we add
#     - a global ID as well as the patch ID
#     - information on Drug channel (only for F15)
#     - information on global location incl. organ registration (only for F15)
# Finally, all patch segmentations are merged into on global segmentation and saved to file. Also, 
# global projections of the segmentation file will be created. The individual patch segmentation 
# files will not be altered.
#
#%% Parameters
dataset = 'F15'
pathData     = BASEP + 'data/' + dataset + '/'
pathAutofluo = ROOTP + 'Documents/LocalData/' + dataset + '_C00_autofluo/'
pathDrug     = ROOTP + 'Documents/LocalData/' + dataset + '_C01_drug/' # only relevant for F15
pathOrgans   = BASEP + 'data/' + dataset + '/organrois/'
region    = filehandling.pload(BASEP + 'data/' + dataset + '/region')
if(dataset == 'F15'): OrganROIs = filehandling.pload(pathOrgans + '/OrganROIs')


#%% Step 1) Merge all segmentations

canvas = np.zeros(region['partitioning']['cropping_boundingbox'] + np.asarray(region['partitioning']['patch_overlap']),np.uint8)
segmentation = {}
segmentation['name'] = region['name']
segmentation['dataset'] = region['dataset']
segmentation['metastases'] = []
global_id = 0
for patch in region['patches']:
    patch_seg = p_leo.load_segmentation(dataset,patch['id'],load_partial=False)
    if(len(patch_seg['metastases']) > 0):
        print("Adding metastases of patch #" + str(patch['id']))
        localpointlist = []
        for metastasis in patch_seg['metastases']:
            metastasis['global_id'] = global_id
            metastasis['patch_id'] = patch['id']
            segmentation['metastases'].append(metastasis)
            localpointlist += metastasis['points']
            global_id += 1
        localvolume, offset, bb = dataconversions.points2volume(localpointlist, offset=[0,0,0], bb=region['partitioning']['patch_size'])
        offset = patch['offset'] - region['partitioning']['cropping_offset']
        canvas[offset[0]:offset[0]+patch['boundingbox'][0],
                   offset[1]:offset[1]+patch['boundingbox'][1],
                   offset[2]:offset[2]+patch['boundingbox'][2]] = localvolume.astype(np.uint8)

filehandling.psave(pathData + 'segmentation', segmentation)
print('Segmentations merged & saved.')

print('Creating global projections...')
canvas = canvas[0:region['partitioning']['cropping_boundingbox'][0],
                0:region['partitioning']['cropping_boundingbox'][1],
                0:region['partitioning']['cropping_boundingbox'][2]]
projections = {}
projections['y'] = np.max(canvas,0).astype(np.bool)
projections['x'] = np.max(canvas,1).astype(np.bool)
projections['z'] = np.max(canvas,2).astype(np.bool)
filehandling.psave(pathData + 'segmentationprojections', projections)
del canvas
print('Segmentation projections created & saved.')


#%% Step 2) For each metastasis, add all relevant meta information

for m, metastasis in enumerate(segmentation['metastases']): # double loop as this will be needed for upcoming computations
    print('Loop 1/2: Adding information for metastasis #' + str(metastasis['global_id']))
    # add global location information to metastasis
    segmentation['metastases'][m]['location'] = {}
    p_offset = np.asarray(region['patches'][metastasis['patch_id']]['offset'])
    segmentation['metastases'][m]['location']['offset'] = (p_offset + metastasis['offset']).tolist()
    segmentation['metastases'][m]['location']['center'] = (p_offset + metastasis['offset'] + metastasis['CoM']).tolist()
    
    # Add info from drug channel: are metastases significantly targeted?
    #  --> Yes, if mean(FG) significantly above 1.5*mean(BG)
    if(dataset == 'F15'):
        drugvol = filehandling.readNifti(pathDrug + 'patchvolume_' + str(metastasis['patch_id']) + '.nii')
        otherblobs = dataconversions.filter_dicts(segmentation['metastases'],'patch_id',metastasis['patch_id'])
        metastasis = p_leo.characterize_drugtargeting(metastasis,drugvol,min_padding=25,thr=1.5,otherblobs=otherblobs)

for m, metastasis in enumerate(segmentation['metastases']):
    print('Loop 2/2: Adding information for metastasis #' + str(metastasis['global_id']))
    
    # compute distance to nearest neighbor metastasis
    segmentation['metastases'][m]['location']['distNN'] = 99999
    segmentation['metastases'][m]['location']['distNNtargeted'] = 99999
    segmentation['metastases'][m]['location']['distNNuntargeted'] = 99999
    for neighbor in segmentation['metastases']:
        dist = blobanalysis.point_dist(metastasis['location']['center'],neighbor['location']['center'])
        if(metastasis['global_id'] is not neighbor['global_id']):
            segmentation['metastases'][m]['location']['distNN'] = np.min([segmentation['metastases'][m]['location']['distNN'],dist])
            if(dataset == 'F15'):
                if(neighbor['DrugCharacterization']['ttest_result']):
                    segmentation['metastases'][m]['location']['distNNtargeted'] = np.min([segmentation['metastases'][m]['location']['distNNtargeted'],dist])
                else:
                    segmentation['metastases'][m]['location']['distNNuntargeted'] = np.min([segmentation['metastases'][m]['location']['distNNuntargeted'],dist])
                
                
    # check whether metastasis is in any organ
    if(dataset == 'F15'):
        segmentation['metastases'][m]['location']['organ'] = None
        [y0,x0,z0] = segmentation['metastases'][m]['location']['offset']
        [y1,x1,z1] = segmentation['metastases'][m]['location']['offset'] + np.asarray(metastasis['boundingbox'])
        for organ in OrganROIs.keys():
            # Check whether metastasis is in BB of organ
            [oy0,ox0,oz0] = OrganROIs[organ]['offset']
            [oy1,ox1,oz1] = OrganROIs[organ]['offset'] + np.asarray(OrganROIs[organ]['boundingbox'])
            Yinrange = (oy0 <= y0 <= oy1) or (oy0 <= y1 <= oy1)
            Xinrange = (ox0 <= x0 <= ox1) or (ox0 <= x1 <= ox1)
            Zinrange = (oz0 <= z0 <= oz1) or (oz0 <= z1 <= oz1)
            if(Yinrange and Xinrange and Zinrange):
                # If so, check precisely whether it is within the polygon of the most outer Z-layers
                roi = OrganROIs[organ]['roi']
                point = Point(metastasis['location']['center'][0:2]) # YX coordinates of metastasis centerpoint
                pointlist_z0 = filehandling.getROIzslice(roi,z0) if(oz0 <= z0) else []
                polygon_z0 = Polygon(pointlist_z0)
                inz0 = polygon_z0.contains(point)
                pointlist_z1 = filehandling.getROIzslice(roi,z1) if(z1 <= oz1) else []
                polygon_z1 = Polygon(pointlist_z1)
                inz1 = polygon_z1.contains(point)
                if(inz0 or inz1):
                    segmentation['metastases'][m]['location']['organ'] = organ

filehandling.psave(pathData + 'segmentation', segmentation)
print('Meta information added for all metastases and saved to file.')









