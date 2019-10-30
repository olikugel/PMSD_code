BASEPATH = '/home/olikugel/PMSD_code'
import sys
sys.path.insert(0, BASEPATH + '/Leo/code/helperfunctions')
sys.path.insert(0, BASEPATH + '/Leo/code/IDP_code')
import filehandling
import dataconversions
import cropping

#%% Parameters

MOUSEPATH = BASEPATH + '/mice_metadata' # path to 12TB HDD with all the scans
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
    # Write ROI to Nifti-file whose filename includes patch_ID and met_ID
    filepath = BASEPATH + '/Potential_TP_Metastases/' + mouse + '/' + channel + '/ROI_' + str(ROI_width) + '/' + 'NIFTI/' 
    patch_ID_padded = filehandling.pad_ID(patch_ID)
    met_ID_padded = filehandling.pad_ID(met_ID)
    filename = 'patch' + patch_ID_padded + '_met' + met_ID_padded + '.nii'
    fileToWriteTo = filepath + filename
    filehandling.writeNifti(fileToWriteTo, ROI)
  
'''    
def write_ROI_to_PNGs(ROI, title, classification):
    
    IMAGE_DIR = DATAPATH + 'ROI_images/' + classification + '/'
    
    MIP_y = np.max(ROI, 0) # maximum intensity projection along y
    axis = 'y'
    filepath = IMAGE_DIR + title + '_' + axis + '.png'
    plt.imsave(filepath, MIP_y, format='png')
    
    MIP_x = np.max(ROI, 1) # maximum intensity projection along x
    axis = 'x'
    filepath = IMAGE_DIR + title + '_' + axis + '.png'
    plt.imsave(filepath, MIP_x, format='png')
    
    MIP_z = np.max(ROI, 2) # maximum intensity projection along z
    axis = 'z'
    filepath = IMAGE_DIR + title + '_' + axis + '.png'
    plt.imsave(filepath, MIP_z, format='png')
'''    
    
 #%% Main   
 
 
for mouse in mice:
    prediction = filehandling.pload(MOUSEPATH + '/' + mouse + '/reviewed_prediction.pickledump')
    candidates = prediction['metastases']
    potential_TPs = dataconversions.filter_dicts(candidates,'evaluation-manually_confirmed',True)
    confirmed_FPs = dataconversions.filter_dicts(candidates,'evaluation-manually_confirmed',False)

    for potential_TP_met in potential_TPs:
        for channel in channels:
            ROI = crop_ROI(potential_TP_met, mouse, channel)
            write_ROI_to_Nifti(ROI, potential_TP_met, mouse, channel)
            
