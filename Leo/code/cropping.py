import basepath
BASEPATH = basepath.get_basepath()
import sys
sys.path.insert(0, BASEPATH + 'Leo/code/helperfunctions')
import filehandling
import numpy as np
from operator import add
import plotting
from plotting import print_dict
import naive_cropping
import matplotlib.pyplot as plt


#%% Define parameters

whole_scan_metadata = filehandling.pload(BASEPATH + 'Leo/data/F15/region.pickledump')

overlap = 50

#%% Main code


def crop_ROI(patch_ID, met_location, met_absolute_location, ROI_width, channel='tumorBoost'):

    patch = load_patch(patch_ID, channel)
    if patch is None: return (None, 'no file')
    
    print("Attempting cropping of ROI in patch " + str(patch_ID) + "...")
    
    ROI_width_half = int(ROI_width / 2)
    patch_width = patch.shape[0]
    
    y       = met_location[0]
    y_lower = met_location[0] - ROI_width_half
    y_upper = met_location[0] + ROI_width_half
    
    x       = met_location[1]
    x_lower = met_location[1] - ROI_width_half
    x_upper = met_location[1] + ROI_width_half
    
    z       = met_location[2]
    z_lower = met_location[2] - ROI_width_half
    z_upper = met_location[2] + ROI_width_half

    if   y_lower < 0 or y_upper >= patch_width \
      or x_lower < 0 or x_upper >= patch_width \
      or z_lower < 0 or z_upper >= patch_width:
        print('>> Metastasis lies at cube edge')
        zone = 'edge'
        print('y: ' + str(y_lower) + '------' + str(y_upper))
        print('x: ' + str(x_lower) + '------' + str(x_upper))
        print('z: ' + str(z_lower) + '------' + str(z_upper))
        
        yxz = {'y'       : y,
               'y_lower' : y_lower,
               'y_upper' : y_upper,
               'x'       : x,
               'x_lower' : x_lower,
               'x_upper' : x_upper,
               'z'       : z,
               'z_lower' : z_lower,
               'z_upper' : z_upper}
        ROI = glue_patches_for_ROI(patch, patch_ID, yxz, ROI_width, patch_width, met_absolute_location, channel)    
        
    else:
        print('>> Metastasis lies within inner zone')
        zone = 'inner'
        print('y: ' + str(y_lower) + '------' + str(y_upper))
        print('x: ' + str(x_lower) + '------' + str(x_upper))
        print('z: ' + str(z_lower) + '------' + str(z_upper))
    
        # ROI = cube that surrounds the metastasis 
        ROI = patch[y_lower:y_upper, x_lower:x_upper, z_lower:z_upper] 
    
    if ROI is None:
        print('\n' + 'ROI cropping failed')
        return (None, zone)
    elif ROI.shape != (ROI_width, ROI_width, ROI_width):
        print('\n' + 'ROI cropping failed, ROI has wrong shape')
        return (None, zone)
    else:
        print('\n' + 'ROI cropping succeeded')
        return (ROI, zone)
    
    
#%%  
    
def glue_patches_for_ROI(patch, patch_ID, yxz, ROI_width, patch_width, met_absolute_location, channel):

    patchstep_of_original_patch = get_patchstep(patch_ID)
    print('\nPatch ' + str(patch_ID) + ' has patchstep-location ' + str(patchstep_of_original_patch))

    #  neighbour_types is a list of all the neighbour_types present in yxz
    neighbour_types = determine_neighbour_types(yxz, patch_width) 
    
    all_typelabels = get_all_typelabels()
   
    # print typelabels of the neighbours that are necessary for glueing
    neighbour_typelabels = []
    for neighbour_type in neighbour_types:
        neighbour_typelabels.append(all_typelabels[neighbour_type])
    print('Neighbours necessary for glueing: ', end = '')
    print(str(neighbour_typelabels) + '\n')
        
    # -------------------------------------------------------------------------
    if len(neighbour_types) == 1:
        neighbour_type = neighbour_types[0]
        neighbour_typelabel = all_typelabels[neighbour_type]
        glue_neighbour = get_glue_neighbour(patch_ID, patchstep_of_original_patch, neighbour_type)
        # print_glue_neighbour(glue_neighbour)
        ROI = glue_two_patches(patch, patch_ID, glue_neighbour, neighbour_typelabel, yxz, ROI_width, patch_width, channel)

    else: # i.e. if len(neighbour_types) > 1:
        number_of_cubes_to_glue = 8
        eight_patch_cube, ulf_patchID = naive_cropping.build_8patch_cube(patch, patch_ID, patch_width, channel)
        ROI = naive_cropping.crop_multipatchcube(eight_patch_cube, number_of_cubes_to_glue, ulf_patchID, ROI_width, met_absolute_location)
    
    if ROI is None or ROI.shape != (ROI_width, ROI_width, ROI_width):    
        number_of_cubes_to_glue = 27
        twentyseven_cube, ulf_patchID = naive_cropping.build_27patch_cube(patch, patch_ID, patch_width, channel)
        ROI = naive_cropping.crop_multipatchcube(twentyseven_cube, number_of_cubes_to_glue, ulf_patchID, ROI_width, met_absolute_location)

    return ROI
    # -------------------------------------------------------------------------  


 #%%  
    
def glue_two_patches(patch, patch_ID, glue_neighbour, neighbour_typelabel, yxz, ROI_width, patch_width, channel):

    try: 
        partial_original_patch = get_partial_original_patch(patch, neighbour_typelabel, yxz, ROI_width, patch_width)
        partial_neighbour_patch = get_partial_neighbour_patch(glue_neighbour['patch_ID'], neighbour_typelabel, yxz, ROI_width, patch_width, channel)
    
        partial_original_shape = partial_original_patch.shape
        partial_neighbour_shape = partial_neighbour_patch.shape
    except: 
        return None

    print('3D-shape of partial-original-patch (' + str(patch_ID) + '): ' + str(partial_original_shape))
    print('3D-shape of partial-neighbour-patch (' + str(glue_neighbour['patch_ID']) + '): ' + str(partial_neighbour_shape))
    
    # determine glue-axis
    if neighbour_typelabel == 'up' or neighbour_typelabel == 'down':
        glue_axis = 0 # glue on y-axis
    elif neighbour_typelabel == 'left' or neighbour_typelabel == 'right':
        glue_axis = 1 # glue on x-axis
    elif neighbour_typelabel == 'front' or neighbour_typelabel == 'back':
        glue_axis = 2 # glue on z-axis
    else: print('Cannot glue in ' + neighbour_typelabel + '-direction')
        
    # glue the two partial patches together
    if neighbour_typelabel == 'up' or neighbour_typelabel == 'left' or neighbour_typelabel == 'front':
        ROI = np.concatenate([partial_neighbour_patch, partial_original_patch], glue_axis)
    else:
        ROI = np.concatenate([partial_original_patch, partial_neighbour_patch], glue_axis)
    
    print('Successfully glued the two together in ' + neighbour_typelabel + '-direction')
    print('3D-shape of ROI: ' + str(ROI.shape))
    return ROI


#%%

def plot_volume(volume, title, contrast_set=False, contrast=0):
    
    plt.figure(figsize=(9,9)) 
    # plotting along y-axis
    plt.subplot(1,3,1)
    plt.subplots_adjust(left=0.4, right=1.6, bottom=2, top=5)
    MIP_y = np.max(volume,0) # maximum intensity projection along y
    if contrast_set: plt.imshow(MIP_y, vmin=0, vmax=contrast)
    else: plt.imshow(MIP_y)
    plt.xlabel('z') 
    plt.ylabel('x')
    plt.title(title + ', along y')
    
    
    # plotting along x-axis
    plt.subplot(1,3,2)
    MIP_x = np.max(volume,1) # maximum intensity projection along x
    if contrast_set: plt.imshow(MIP_x, vmin=0, vmax=contrast)
    else: plt.imshow(MIP_x)
    plt.xlabel('z') 
    plt.ylabel('y')
    plt.title(title + ', along x')
    
    # plotting along z-axis
    plt.subplot(1,3,3) 
    MIP_z = np.max(volume,2) # maximum intensity projection along z
    if contrast_set: plt.imshow(MIP_z, vmin=0, vmax=contrast)
    else: plt.imshow(MIP_z)
    plt.xlabel('x') 
    plt.ylabel('y')
    plt.title(title + ', along z')


#%%

def determine_neighbour_types(yxz, patch_width):
    
    required_neighbour_types = [] # types of all the neighbours that yxz requires
    
    if yxz['y_lower'] < 0:
        required_neighbour_types.append(4) # up

    if yxz['y_upper'] >= patch_width:
        required_neighbour_types.append(21) # down

    if yxz['x_lower'] < 0:
        required_neighbour_types.append(10) # left
     
    if yxz['x_upper'] >= patch_width:
        required_neighbour_types.append(15) # right
 
    if yxz['z_lower'] < 0:
        required_neighbour_types.append(12) # front
        
    if yxz['z_upper'] >= patch_width:
        required_neighbour_types.append(13) # back
       
    return required_neighbour_types
    

#%%

def load_patch(patch_ID, channel):
    
    filename = 'patchvolume_' + str(patch_ID) + '.nii'
    
    if channel == 'tumorBoost':
        path_to_patchdata = BASEPATH + 'LocalData/F15/C02_' + channel + '/' + filename
    elif channel == 'autofluo':
        path_to_patchdata = BASEPATH + 'LocalData/F15/C00_' + channel + '/' + filename
    else:
        print("ERROR: unknown channel '" + channel + "', please select either 'tumorBoost' or 'autofluo'")
        return None
    
    try:
        patch = filehandling.readNifti(path_to_patchdata)
    except FileNotFoundError: 
        print("Nifti-file " + filename + " doesn't exist")
        return None
    
    return patch


#%%

def get_partial_original_patch(patch, neighbour_typelabel, yxz, ROI_width, patch_width):
    
    original_patch = patch
    if original_patch is None: return None
    
    if neighbour_typelabel == 'up':
        width_of_glue_piece = abs(yxz['y_lower'])
        partial_original_patch = original_patch[ 0 : ROI_width - width_of_glue_piece,
                                                 yxz['x_lower'] : yxz['x_upper'], 
                                                 yxz['z_lower'] : yxz['z_upper'] 
                                               ]  
    elif neighbour_typelabel == 'down':
        width_of_glue_piece = yxz['y_upper'] - patch_width
        partial_original_patch = original_patch[ patch_width - ROI_width + width_of_glue_piece : patch_width,
                                                 yxz['x_lower'] : yxz['x_upper'], 
                                                 yxz['z_lower'] : yxz['z_upper'] 
                                               ]
    elif neighbour_typelabel == 'left':
        width_of_glue_piece = abs(yxz['x_lower'])
        partial_original_patch = original_patch[ yxz['y_lower'] : yxz['y_upper'], 
                                                 0 : ROI_width - width_of_glue_piece,
                                                 yxz['z_lower'] : yxz['z_upper'] 
                                               ] 
    elif neighbour_typelabel == 'right':
        width_of_glue_piece = yxz['x_upper'] - patch_width
        partial_original_patch = original_patch[ yxz['y_lower'] : yxz['y_upper'], 
                                                 patch_width - ROI_width + width_of_glue_piece : patch_width,
                                                 yxz['z_lower'] : yxz['z_upper'] 
                                               ] 
    elif neighbour_typelabel == 'front':
        width_of_glue_piece = abs(yxz['z_lower'])
        partial_original_patch = original_patch[ yxz['y_lower'] : yxz['y_upper'], 
                                                 yxz['x_lower'] : yxz['x_upper'], 
                                                 0 : ROI_width - width_of_glue_piece
                                               ]  
    elif neighbour_typelabel == 'back':
        width_of_glue_piece = yxz['z_upper'] - patch_width
        partial_original_patch = original_patch[ yxz['y_lower'] : yxz['y_upper'], 
                                                 yxz['x_lower'] : yxz['x_upper'], 
                                                 patch_width - ROI_width + width_of_glue_piece : patch_width
                                               ]  
    else: return None
    
    return partial_original_patch


#%%

def get_partial_neighbour_patch(patch_ID, neighbour_typelabel, yxz, ROI_width, patch_width, channel):
    
    neighbour_patch = load_patch(patch_ID, channel)
    if neighbour_patch is None: return None
    
    ROI_width_half = int(ROI_width / 2)
    glue_param = patch_width - overlap - ROI_width_half 
    
    if neighbour_typelabel == 'up':
        width_of_glue_piece = abs(yxz['y_lower'])
        partial_neighbour_patch = neighbour_patch[ glue_param + yxz['y'] : glue_param + yxz['y'] + width_of_glue_piece,
                                                   yxz['x_lower'] : yxz['x_upper'],
                                                   yxz['z_lower'] : yxz['z_upper']
                                                 ]
    elif neighbour_typelabel == 'down':
        width_of_glue_piece = yxz['y_upper'] - patch_width 
        partial_neighbour_patch = neighbour_patch[ overlap : overlap + width_of_glue_piece,
                                                   yxz['x_lower'] : yxz['x_upper'], 
                                                   yxz['z_lower'] : yxz['z_upper']
                                                 ] 
    elif neighbour_typelabel == 'left':
        width_of_glue_piece = abs(yxz['x_lower'])
        partial_neighbour_patch = neighbour_patch[ yxz['y_lower'] : yxz['y_upper'], 
                                                   glue_param + yxz['x'] : glue_param + yxz['x'] + width_of_glue_piece,
                                                   yxz['z_lower'] : yxz['z_upper']
                                                 ]
    elif neighbour_typelabel == 'right':
        width_of_glue_piece = yxz['x_upper'] - patch_width 
        partial_neighbour_patch = neighbour_patch[ yxz['y_lower'] : yxz['y_upper'], 
                                                   overlap : overlap + width_of_glue_piece,
                                                   yxz['z_lower'] : yxz['z_upper']
                                                 ] 
    elif neighbour_typelabel == 'front':
        width_of_glue_piece = abs(yxz['z_lower'])
        partial_neighbour_patch = neighbour_patch[ yxz['y_lower'] : yxz['y_upper'],
                                                   yxz['x_lower'] : yxz['x_upper'],
                                                   glue_param + yxz['z'] : glue_param + yxz['z'] + width_of_glue_piece
                                                 ]
    elif neighbour_typelabel == 'back':
        width_of_glue_piece = yxz['z_upper'] - patch_width 
        partial_neighbour_patch = neighbour_patch[ yxz['y_lower'] : yxz['y_upper'], 
                                                   yxz['x_lower'] : yxz['x_upper'], 
                                                   overlap : overlap + width_of_glue_piece
                                                 ] 
    else: return None
    
    return partial_neighbour_patch


#%%
        
def get_glue_neighbour(patch_ID, patchstep_of_original_patch, neighbour_type):
       
    steps_to_neighbours = get_steps_to_neighbours()
    
    for steps in steps_to_neighbours.values():
        if steps == steps_to_neighbours[neighbour_type]:
            steps_to_neighbour_patch = steps
            
    patchstep_of_neighbour_patch = list(map(add, patchstep_of_original_patch, steps_to_neighbour_patch))

    neighbours = get_neighbours(patch_ID)
    
    glue_neighbour = None
    for neighbour in neighbours:
        if neighbour['patchstep'] == patchstep_of_neighbour_patch:
            glue_neighbour = neighbour
    
    if glue_neighbour is None: 
        print('Non-existing neighbour at patchstep-location ' + str(patchstep_of_neighbour_patch))
   
    return glue_neighbour
    

#%%

def print_glue_neighbour(glue_neighbour):
    
    for key in glue_neighbour.keys():
        print('> ' + str(key) + ': ' + str(glue_neighbour[key]))


#%%

def get_steps_to_neighbours():
    # create dictionary to identify the steps to the 26 neighbours of the current patch
    steps_to_neighbours = {}
    neighbour_type = 0
    one_steps = [-1, 0, 1]
    for y_step in one_steps:
        for x_step in one_steps:
            for z_step in one_steps:
                if [y_step, x_step, z_step] != [0,0,0]:
                    steps_to_neighbours[neighbour_type] = [y_step, x_step, z_step]
                    neighbour_type += 1
    # plotting.print_dict(steps_to_neighbours) 
    return steps_to_neighbours


#%%

def get_all_typelabels():
    # create labels for neighbour_types in order to give them meaning
    typelabels = {}
    for neighbour_type in range(26):
        if neighbour_type ==  0: typelabels[neighbour_type] = 'up-left-front'
        if neighbour_type ==  1: typelabels[neighbour_type] = 'up-left'
        if neighbour_type ==  2: typelabels[neighbour_type] = 'up-left-back'
        if neighbour_type ==  3: typelabels[neighbour_type] = 'up-front'
        if neighbour_type ==  4: typelabels[neighbour_type] = 'up'
        if neighbour_type ==  5: typelabels[neighbour_type] = 'up-back'
        if neighbour_type ==  6: typelabels[neighbour_type] = 'up-right-front'
        if neighbour_type ==  7: typelabels[neighbour_type] = 'up-right'
        if neighbour_type ==  8: typelabels[neighbour_type] = 'up-right-back'
        if neighbour_type ==  9: typelabels[neighbour_type] = 'left-front'
        if neighbour_type == 10: typelabels[neighbour_type] = 'left'
        if neighbour_type == 11: typelabels[neighbour_type] = 'left-back'
        if neighbour_type == 12: typelabels[neighbour_type] = 'front'
        if neighbour_type == 13: typelabels[neighbour_type] = 'back'
        if neighbour_type == 14: typelabels[neighbour_type] = 'right-front'
        if neighbour_type == 15: typelabels[neighbour_type] = 'right'
        if neighbour_type == 16: typelabels[neighbour_type] = 'right-back'
        if neighbour_type == 17: typelabels[neighbour_type] = 'down-left-front'
        if neighbour_type == 18: typelabels[neighbour_type] = 'down-left'
        if neighbour_type == 19: typelabels[neighbour_type] = 'down-left-back'
        if neighbour_type == 20: typelabels[neighbour_type] = 'down-front'
        if neighbour_type == 21: typelabels[neighbour_type] = 'down'
        if neighbour_type == 22: typelabels[neighbour_type] = 'down-back'
        if neighbour_type == 23: typelabels[neighbour_type] = 'down-right-front'
        if neighbour_type == 24: typelabels[neighbour_type] = 'down-right'
        if neighbour_type == 25: typelabels[neighbour_type] = 'down-right-back'
    # plotting.print_dict(typelabels)
    return typelabels


#%%

def get_patchstep(patch_ID):
     # create dictionary that maps patchID to patchstep
    patches_metadata = whole_scan_metadata['patches']
    
    patchsteps = {}
    for patch_metadata in patches_metadata:
        patchsteps[patch_metadata['id']] = patch_metadata['patchstep']
        
    patchstep = patchsteps[patch_ID]
    return patchstep
    

#%%

def get_patch_ID(patchstep):
    # create dictionary that maps patchstep-strings to patchID
    patches_metadata = whole_scan_metadata['patches']
    
    patch_IDs = {} 
    for patch_metadata in patches_metadata:
        patchstep_as_string = str(patch_metadata['patchstep'])
        patch_IDs[patchstep_as_string] = patch_metadata['id']  
        
    patch_ID = patch_IDs[str(patchstep)]
    return patch_ID


#%%

def get_patch_absolute_location(patch_ID):
    # create dictionary that maps patchID to offset
    patches_metadata = whole_scan_metadata['patches']
    
    patch_absolute_locs = {} 
    for patch_metadata in patches_metadata:
        patch_absolute_locs[patch_metadata['id']] = patch_metadata['offset']  
        
    patch_absolute_loc = patch_absolute_locs[patch_ID]
    return patch_absolute_loc


#%%

def get_neighbours(patch_ID):
    
    patchstep = get_patchstep(patch_ID) # patchstep-location of current patch
    
    # dictionary to identify the steps to the 26 neighbours of the current patch
    steps_to_neighbours = get_steps_to_neighbours()
    
    # add each step_triple to patchstep to determine all neighbour locations
    neighbours = {} # neighbours really should be called neighbours_patchstep_locations
    for neighbour_type, step_triple in enumerate(steps_to_neighbours.values()):
        neighbours[neighbour_type] = list(map(add, patchstep, step_triple))
    
    # check if neighbours are outside reference frame, i.e. non-existent
    for neighbour_type, neighbour in enumerate(neighbours.values()):
        if (neighbour[0] < 0) or (neighbour[1] < 0) or (neighbour[2] < 0):
            neighbours[neighbour_type] = None
    
    # create labels for neighbour_types in order to give them meaning, e.g. 4 means 'up'
    neighbour_typelabels = get_all_typelabels()
     
    # create neighbour-datastructure and make a list of them, this will be returned
    list_of_neighbours = []
    for neighbour_type, neighbour in enumerate(neighbours.values()): 
        if neighbour is not None:
            try:
                a_neighbour = {'patch_ID'  : get_patch_ID(neighbour),
                               'patchstep' : neighbour,
                               'type'      : neighbour_type,
                               'typelabel' : neighbour_typelabels[neighbour_type]}
                list_of_neighbours.append(a_neighbour)
            except KeyError as e:
                pass
                # print('Non-existing neighbour at patchstep-location ' + str(e).replace("'",""))
    return list_of_neighbours
        
                       
#%%
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










