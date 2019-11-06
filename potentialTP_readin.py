CODEPATH = '/home/oschoppe/Documents/OKugel/PMSD_code'
DATAPATH = '/home/oschoppe/Documents/OKugel/PMSD_data'
HEAVY_DATAPATH = '/mnt/12TB/oschoppe'

import sys
sys.path.insert(0, CODEPATH + '/helperfunctions')
sys.path.insert(0, CODEPATH + '/IDP_code')
import numpy as np
import filehandling
import dataconversions
import cropping
import matplotlib.pyplot as plt

#%% Parameters

ROI_width = 50
mice = ['H2030IC10dn573','IC2dn2','IC6dn1','IC6dn2', 'IC14dn1', 'MCF7IC21dn528']
channels = ['C00','C01','C02']

#%%

def crop_ROI(metastasis, mouse, channel):
    
    met = metastasis
    patch_ID = met['patch_id']
    met_ID = met['id']
    
    met_location = met['offset'] + met['CoM']
    met_absolute_location = met['location']['center']
    print('---------------------------------------')
    print('Metastasis ', met_ID, ', located at ', met_location, ' in patch ', patch_ID) 
        
    ROI, zone = cropping.crop_ROI(patch_ID, met_location, met_absolute_location, ROI_width, mouse, channel) 
    return ROI



def write_ROI_to_Nifti(ROI, metastasis, mouse, channel):
    
    met = metastasis
    patch_ID = met['patch_id']
    met_ID = met['id']
    
    filepath = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/' + channel + '/ROI_' + str(ROI_width) + '/NIFTI/' 
    patch_ID_padded = filehandling.pad_ID(patch_ID)
    met_ID_padded = filehandling.pad_ID(met_ID)
    filename_prefix = 'patch' + patch_ID_padded + '_met' + met_ID_padded
    filename = filename_prefix + '.nii'
    fileToWriteTo = filepath + filename
    filehandling.writeNifti(fileToWriteTo, ROI)
    return filename_prefix
   
    

def write_ROI_to_PNGs(ROI, filename_prefix):
    
    filepath = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/' + channel + '/ROI_' + str(ROI_width) + '/PNG/'

    MIP_y = np.max(ROI, 0) # maximum intensity projection along y
    axis = 'y'
    filename = filename_prefix + '_' + axis + '.png'
    fileToWriteTo = filepath + filename
    plt.imsave(fileToWriteTo, MIP_y, format='png')
    
    MIP_x = np.max(ROI, 1) # maximum intensity projection along x
    axis = 'x'
    filename = filename_prefix + '_' + axis + '.png'
    fileToWriteTo = filepath + filename
    plt.imsave(fileToWriteTo, MIP_x, format='png')
    
    MIP_z = np.max(ROI, 2) # maximum intensity projection along z
    axis = 'z'
    filename = filename_prefix + '_' + axis + '.png'
    fileToWriteTo = filepath + filename
    plt.imsave(fileToWriteTo, MIP_z, format='png')
   
    
 #%% Main   
 
 
for mouse in mice:
    prediction = filehandling.pload(DATAPATH + '/mice_metadata/' + mouse + '/reviewed_prediction.pickledump')
    candidates = prediction['metastases']
    potential_TPs = dataconversions.filter_dicts(candidates,'evaluation-manually_confirmed',True)
    confirmed_FPs = dataconversions.filter_dicts(candidates,'evaluation-manually_confirmed',False)

    for potential_TP_met in potential_TPs:
        for channel in channels:
            print('\n\n')
            print('### Mouse ', mouse, ' ###')
            print('### Channel ', channel, ' ###')
            ROI = crop_ROI(potential_TP_met, mouse, channel)
            if ROI is not None:
                filename_prefix = write_ROI_to_Nifti(ROI, potential_TP_met, mouse, channel)
                write_ROI_to_PNGs(ROI, filename_prefix)
            
