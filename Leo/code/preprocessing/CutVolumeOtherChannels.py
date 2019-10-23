import sys
ROOTP = sys.path[-1].replace('.ipython','').replace('\\','/')
sys.path.insert(0, ROOTP + 'Documents/Synced/_Promotion/scripts/helperfunctions')
import filehandling

import os 
import numpy as np
from scipy.misc import imresize
import cv2
import multiprocessing

BASEP = ROOTP + 'Documents/Synced/_Promotion/Projects/Leo/'

#%% 
#
# Whereas CutVolume.py is designed for the tumor channel, this script has small
# modifications to make it work for other channels (i.e. where to save etc)
#
# Please note: adjust in lower part of code whether to cut ALL patches, or only CoreOfBody

#%% Define parameters

sourcefolder = '/media/oliverschoppe/HD No 1/Data Ertuerk Lab/Project_Leo - whole body cancer/F15/preprocessed channels/' # Folder with raw TIFF files --> TranslaTUM
#sourcefolder = '/mnt/C286054386053985/oschoppe/RawData/F15/preprocessed channels/' # Folder with raw TIFF files --> Grosshadern
localfolder  = ROOTP + 'Documents/LocalData/F15_OldMask/' # to save Niftis --> TranslaTUM
#localfolder  = '/mnt/C286054386053985/oschoppe/F15_OldMask/' # to save Niftis --> Grosshadern
syncfolder   = localfolder # to save region file --> for other channels, we save it locally only!
path_region  = BASEP + 'data/F15/' # region is needed to select CoreOfBody patches

region = {}
region['name']        = 'F15'
region['description'] = 'Whole-body scan of F15 mouse at original resolution'
thumbnail_ds          = 0.1 # Downsampling for thumbnails of patches
file_format           = 'Nifti' # choose from Nifti or TIFF
num_cores_to_use = multiprocessing.cpu_count()-2
# Parameters: data set
region['dataset'] = {}
region['dataset']['channelname']  = 'C03_tumor-masked'
region['dataset']['downsampling'] = 'None'
# Parameters: cropping & dicing
region['partitioning'] = {}
region['partitioning']['patch_size']               = [350, 350, 350]
region['partitioning']['patch_overlap']            = 50
if(region['name'] == 'F14'):
    region['partitioning']['cropping_offset']      = [2100,1350,3400]
    region['partitioning']['cropping_boundingbox'] = [5700,2100,2100]
elif(region['name'] == 'F15'):
    region['partitioning']['cropping_offset']      = [1550,1200,1350]
    region['partitioning']['cropping_boundingbox'] = [7200,2100,2100]



#%% Set up patches

# Compute further parameters
(fsize, dims, bitdepth) = filehandling.estimateVolSize(sourcefolder,[region['dataset']['channelname']])
region['dataset']['file_size'] = fsize
region['dataset']['bounding_box'] = dims
region['dataset']['bit_depth'] = bitdepth
patch_size = np.asarray(region['partitioning']['patch_size'],np.uint16)
patch_overlap = np.asarray(region['partitioning']['patch_overlap'],np.uint16)
n_patches = np.asarray(region['partitioning']['cropping_boundingbox'] / (patch_size - patch_overlap),np.uint8)
region['partitioning']['patches_per_dim'] = n_patches
# Initialize patches
region['patches'] = []
patch_id = 0
for ny in range(0,n_patches[0]):
    for nx in range(0,n_patches[1]):
        for nz in range(0,n_patches[2]):
            patch = {}
            patch['id'] = patch_id
            patch['patchstep'] = [ny, nx, nz]
            patch['offset'] = region['partitioning']['cropping_offset'] + (patch_size - patch_overlap) * patch['patchstep']
            patch['boundingbox'] = patch_size
            region['patches'].append(patch)
            patch_id += 1
# Find patches that still need to be cut 
##!!!!!!! Adjust whether you only want to cut patches from CoreOfBody <------
TumorRegion = filehandling.pload(path_region + 'region')
remaining_patches = []
for p, patch in enumerate(region['patches']):
    if(TumorRegion['patches'][p]['locationgroup'] == 'Core' or TumorRegion['patches'][p]['locationgroup']=='Boundary'): # <-- ADJUST AS NECESSARY!
        if(os.path.isfile(localfolder + 'patchvolume_' + str(patch['id']) + '.nii')==False and file_format == 'Nifti'):
            remaining_patches.append(patch)
        elif(os.path.isfile(localfolder + 'patchvolume_' + str(patch['id']) + '.tif')==False and file_format == 'TIFF'):
            remaining_patches.append(patch)

filehandling.psave(syncfolder + 'region', region) # saving initial region file (will be updated later on)
print('Patches initialized, now segmenting volume')

#%% Function definition

def save_patch(patch):
    bb = patch['boundingbox']
    patch['volume'] = np.zeros((bb[0],bb[1],bb[2]), 'uint'+str(bitdepth))
    flist = sorted(os.listdir(sourcefolder + region['dataset']['channelname']))
    print('Patch ' + str(patch['id']) + ' was taken by ' + multiprocessing.current_process().name)
    for z in range(0,bb[2]):
        file = flist[patch['offset'][2] + z]
        image = cv2.imread(sourcefolder + region['dataset']['channelname'] + '/' + file, 2) # '2' forces cv2 to keep original bitdepth
        y0 = patch['offset'][0]
        x0 = patch['offset'][1]
        patch['volume'][:,:,z] = image[y0:y0+bb[0],x0:x0+bb[1]]
        print('Patch ' + str(patch['id']) + ': ' + str(np.round(100*z/bb[2],1)) + '% - ' + file)
    
    filename = 'patchvolume_' + str(patch['id'])
    croppedfsize = bb[0].astype(np.float) * bb[1] * bb[2] * bitdepth /(8*1024**3)
    if(file_format == 'TIFF' and croppedfsize<4):
        filehandling.writeFijiTIFF(localfolder + filename + '.tif', patch['volume'])
        print('Patch volume saved as Fiji-readable TIFF.')
    else:
        filehandling.writeNifti(localfolder + filename + '.nii', patch['volume'])        
        print('Patch volume saved as Nifti file.')
        if(file_format == 'TIFF'):
            print('File had to be saved as Nifti instead of TIFF as its size exceeds 4GB, the maximum for TIFFs')


#%% Execution

# Cut volume into patches and save them to file
pool = multiprocessing.Pool(processes=num_cores_to_use)
pool.map(save_patch, remaining_patches)
pool.close()
print('Entire volume cut to patches and saved to file.')

# Create thumbnails for all patches
bby = int(region['partitioning']['cropping_boundingbox'][0]*thumbnail_ds)
bbx = int(region['partitioning']['cropping_boundingbox'][1]*thumbnail_ds)
bbz = int(region['partitioning']['cropping_boundingbox'][2]*thumbnail_ds)
mpy = np.zeros((bbx,bbz,region['partitioning']['patches_per_dim'][0]),np.float32)
mpx = np.zeros((bby,bbz,region['partitioning']['patches_per_dim'][1]),np.float32)
mpz = np.zeros((bby,bbx,region['partitioning']['patches_per_dim'][2]),np.float32)
dy = int((region['partitioning']['patch_size'][0] - region['partitioning']['patch_overlap'])*thumbnail_ds)
dx = int((region['partitioning']['patch_size'][1] - region['partitioning']['patch_overlap'])*thumbnail_ds)
dz = int((region['partitioning']['patch_size'][2] - region['partitioning']['patch_overlap'])*thumbnail_ds)
for patch in region['patches']:
    print('Creating thumbnail for patch #' + str(patch['id']))
    pvol = filehandling.readNifti(localfolder + 'patchvolume_' + str(patch['id']))
    pvol = pvol[0:int(dy/thumbnail_ds),0:int(dx/thumbnail_ds),0:int(dz/thumbnail_ds)]
    s = patch['patchstep']
    mpy[s[1]*dx:(s[1]+1)*dx, s[2]*dz:(s[2]+1)*dz, s[0]] = imresize(np.max(pvol,0),thumbnail_ds, mode='F')
    mpx[s[0]*dy:(s[0]+1)*dy, s[2]*dz:(s[2]+1)*dz, s[1]] = imresize(np.max(pvol,1),thumbnail_ds, mode='F')
    mpz[s[0]*dy:(s[0]+1)*dy, s[1]*dx:(s[1]+1)*dx, s[2]] = imresize(np.max(pvol,2),thumbnail_ds, mode='F')
    mpy[s[1]*dx,:,s[0]] = mpy[:,s[2]*dz,s[0]] = 0
    mpx[s[0]*dy,:,s[1]] = mpx[:,s[2]*dz,s[1]] = 0
    mpz[s[0]*dy,:,s[2]] = mpz[:,s[1]*dx,s[2]] = 0

# Update and save region overview to file
region['thumbnails'] = {}
region['thumbnails']['downsampling'] = thumbnail_ds
region['thumbnails']['MaxProjections_Y'] = mpy
region['thumbnails']['MaxProjections_X'] = mpx
region['thumbnails']['MaxProjections_Z'] = mpz
filehandling.psave(syncfolder + 'region', region)
print('Region file updated with thumbnails.')
print('Done.')







