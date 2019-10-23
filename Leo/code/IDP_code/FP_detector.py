import basepath
BASEPATH = basepath.get_basepath()
DATAPATH = BASEPATH + 'Leo/data/F15/'
import sys
sys.path.insert(0, BASEPATH + 'Leo/code/helperfunctions')
import filehandling
import met_counting
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%% Read in data

points_of_potential_mets = filehandling.pload(DATAPATH + 'points_of_interest_F15_merged.pickledump')

points_and_labels = {}

#%% 

def TP_FP_distinguisher(point, point_ID):

    ROI, ROI_af = met_counting.get_ROI_around_location(point) # af = autofluorescence channel
    
    if ROI is None: return ('ROI cannot be read', -1, -1)

    number_of_mets_in_ROI, number_of_ambig_mets_in_ROI = met_counting.count_mets_in_ROI(ROI, point)
    
    point_is_partof_met = met_counting.is_point_partof_met(point)
    
    if point_is_partof_met: 
        classification = 'TP'
    elif number_of_mets_in_ROI >= 1:
        classification = 'FPb'
    else:
        classification = 'FP'
      
    point_as_string = str(point[0]) + '-' + str(point[1]) + '-' + str(point[2])
    
    title = 'point' + filehandling.pad_ID(point_ID) + '_TB_' + point_as_string + '_' + classification 
    write_ROI_to_file(ROI, title, classification)
    
    title_af = 'point' + filehandling.pad_ID(point_ID) + '_AF_' + point_as_string + '_' + classification 
    write_ROI_to_file(ROI_af, title_af, classification)

    # create dictionary mapping point_ID to label
    if classification == 'FPb': 
        classif_for_file = 'FP'
    else: 
        classif_for_file = classification
    points_and_labels[point_ID] = (point_as_string, classif_for_file)

    return (classification, number_of_mets_in_ROI, number_of_ambig_mets_in_ROI)
        
#%% 

def write_ROI_to_file(ROI, title, classification):
    
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
   
#%%    
    
     
    
    
# --- MAIN ---
    
number_of_TPs  = 0
number_of_FPbs = 0
number_of_FPs  = 0
number_of_failed_ROI_reads = 0
point_ID_local = 0  
  
for point_metadata in points_of_potential_mets: 
    
    point = point_metadata['global_coordinates']
    point_ID = point_ID_local # point_metadata['ID']
    classification, number_of_mets, number_of_ambig_mets = TP_FP_distinguisher(point, point_ID)

    if classification == 'ROI cannot be read': 
        number_of_failed_ROI_reads += 1
        continue

    print(str(point_ID) + ': ' + str(point) + ' --> ', end='')
    
    if classification == 'TP':
        print('True Positive!')
        number_of_TPs += 1
    elif classification == 'FPb':
        print('False Positive! But with ' + str(number_of_mets) + ' metastases closeby.')
        number_of_FPbs += 1
    elif classification == 'FP':
        print('False Positive!')
        number_of_FPs += 1
    else:
        print()
    
    if number_of_ambig_mets > 0:
        print(str(number_of_ambig_mets) + ' ambiguous metastases closeby')
    print()

    point_ID_local += 1



f = open('points_and_labels.pkl','wb')
pickle.dump(points_and_labels, f)
f.close()

print('\n')
print('Number of true positives: ' + str(number_of_TPs))
print('Number of false positives with closeby metastases: ' + str(number_of_FPbs))
print('Number of false positives: ' + str(number_of_FPs))
print('Number of failed ROI-reads: ' + str(number_of_failed_ROI_reads))


    
    