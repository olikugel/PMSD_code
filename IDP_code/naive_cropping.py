import numpy as np
import cropping

#%% Define parameters

overlap = 50

#%% As a reference, these are the 26 neighbour-types
'''
    0: 'up-left-front'
    1: 'up-left'
    2: 'up-left-back'
    3: 'up-front'
    4: 'up'
    5: 'up-back'
    6: 'up-right-front'
    7: 'up-right'
    8: 'up-right-back'
    9: 'left-front'
   10: 'left'
   11: 'left-back'
   12: 'front'
   13: 'back'
   14: 'right-front'
   15: 'right'
   16: 'right-back'
   17: 'down-left-front'
   18: 'down-left'
   19: 'down-left-back'
   20: 'down-front'
   21: 'down'
   22: 'down-back'
   23: 'down-right-front'
   24: 'down-right'
   25: 'down-right-back'
'''

#%%

def build_27patch_cube(patch, patch_ID, patch_width, mouse, channel):

    print('Building 27-patch-cube')
    
    patchstep_original = cropping.get_patchstep(patch_ID)
    crop_width = patch_width - overlap
    original_patch = patch[0:crop_width, 0:crop_width, 0:crop_width]
    
    # load 26 neighbours
    neighbour_patches = []
    try:   
        for neighbour_type in range(26):
            neighbour = cropping.get_glue_neighbour(patch_ID, patchstep_original, neighbour_type)
            neighbour_patchID = neighbour['patch_ID']
            if neighbour_type == 0: up_left_front_patchID = neighbour_patchID
            neighbour_patch = cropping.load_patch(neighbour_patchID, mouse, channel)
            neighbour_patch = neighbour_patch[0:crop_width, 0:crop_width, 0:crop_width]
            neighbour_patches.append(neighbour_patch) 
    except:
        return (None, -1)
    
    # im1 = down-left-back ++ down-back ++ down-right-back # glue along x-axis
    im1 = np.concatenate([neighbour_patches[19], neighbour_patches[22], neighbour_patches[25]], 1)
    # im2 = down-left ++ down ++ down-right # glue along x-axis
    im2 = np.concatenate([neighbour_patches[18], neighbour_patches[21], neighbour_patches[24]], 1)
    # im3 = down-left-front ++ down-front ++ down-right-front # glue along x-axis
    im3 = np.concatenate([neighbour_patches[17], neighbour_patches[20], neighbour_patches[23]], 1)
    # down-layer = im3 ++ im2 ++ im1 # glue along z-axis
    down_layer = np.concatenate([im3, im2, im1], 2)
    
    # im4 = left-back + back ++ right-back # glue along x-axis
    im4 = np.concatenate([neighbour_patches[11], neighbour_patches[13], neighbour_patches[16]], 1)
    # im5 = left ++ original ++ right # glue along x-axis
    im5 = np.concatenate([neighbour_patches[10], original_patch, neighbour_patches[15]], 1)
    # im6 = left-front ++ front ++ right-front # glue along x-axis
    im6 = np.concatenate([neighbour_patches[9], neighbour_patches[12], neighbour_patches[14]], 1)
    # middle-layer = im6 ++ im5 ++ im4 # glue along z-axis
    middle_layer = np.concatenate([im6, im5, im4], 2)
    
    # im7 = up-left-back ++ up-back ++ up-right-back # glue along x-axis
    im7 = np.concatenate([neighbour_patches[2], neighbour_patches[5], neighbour_patches[8]], 1)
    # im8 = up-left ++ up ++ up-right # glue along x-axis
    im8 = np.concatenate([neighbour_patches[1], neighbour_patches[4], neighbour_patches[7]], 1)
    # im9 = up-left-front ++ up-front ++ up-right-front # glue along x-axis
    im9 = np.concatenate([neighbour_patches[0], neighbour_patches[3], neighbour_patches[6]], 1)
    # up-layer = im9 ++ im8 ++ im7 # glue along z-axis
    up_layer = np.concatenate([im9, im8, im7], 2)
    
    # 27patch_cube = up-layer ++ middle-layer ++ down-layer # glue along y-axis
    twentyseven_cube = np.concatenate([up_layer, middle_layer, down_layer], 0)

    return (twentyseven_cube, up_left_front_patchID)


#%%

def build_8patch_cube(patch, patch_ID, patch_width, mouse, channel):
    
    print('Building 8-patch-cube')
    '''
    We need to glue together the following 8 patches:
        - original
        - up
        - front
        - up-front
        - left
        - up-left
        - left-front
        - up-left-front
    '''
    try:
        patchstep_original = cropping.get_patchstep(patch_ID)
        original_patch = patch
        
        up_neighbour = cropping.get_glue_neighbour(patch_ID, patchstep_original, 4)
        up_patchID   = up_neighbour['patch_ID']
        up_patch     = cropping.load_patch(up_patchID, mouse, channel)
        
        front_neighbour = cropping.get_glue_neighbour(patch_ID, patchstep_original, 12)
        front_patchID   = front_neighbour['patch_ID']
        front_patch     = cropping.load_patch(front_patchID, mouse, channel)
        
        up_front_neighbour = cropping.get_glue_neighbour(patch_ID, patchstep_original, 3)
        up_front_patchID   = up_front_neighbour['patch_ID']
        up_front_patch     = cropping.load_patch(up_front_patchID, mouse, channel)
        
        left_neighbour = cropping.get_glue_neighbour(patch_ID, patchstep_original, 10)
        left_patchID   = left_neighbour['patch_ID']
        left_patch     = cropping.load_patch(left_patchID, mouse, channel)
        
        up_left_neighbour = cropping.get_glue_neighbour(patch_ID, patchstep_original, 1)
        up_left_patchID   = up_left_neighbour['patch_ID']
        up_left_patch     = cropping.load_patch(up_left_patchID, mouse, channel)
        
        left_front_neighbour = cropping.get_glue_neighbour(patch_ID, patchstep_original, 9)
        left_front_patchID   = left_front_neighbour['patch_ID']
        left_front_patch     = cropping.load_patch(left_front_patchID, mouse, channel)
        
        up_left_front_neighbour = cropping.get_glue_neighbour(patch_ID, patchstep_original, 0)
        up_left_front_patchID   = up_left_front_neighbour['patch_ID']
        up_left_front_patch     = cropping.load_patch(up_left_front_patchID, mouse, channel)
    
        crop_width = patch_width - overlap
            
        original_patch      = original_patch[0:crop_width, 0:crop_width, 0:crop_width]
        up_patch            = up_patch[0:crop_width, 0:crop_width, 0:crop_width]
        front_patch         = front_patch[0:crop_width, 0:crop_width, 0:crop_width]
        up_front_patch      = up_front_patch[0:crop_width, 0:crop_width, 0:crop_width]
        left_patch          = left_patch[0:crop_width, 0:crop_width, 0:crop_width]
        up_left_patch       = up_left_patch[0:crop_width, 0:crop_width, 0:crop_width]
        left_front_patch    = left_front_patch[0:crop_width, 0:crop_width, 0:crop_width]
        up_left_front_patch = up_left_front_patch[0:crop_width, 0:crop_width, 0:crop_width]   
    except:
        return (None, -1)
    '''
    Glueing order:
        intermed_1 = original ++ up
        intermed_2 = front ++ up-front
        intermed_3 = intermed_1 ++ intermed_2
        
        intermed_4 = left ++ up-left
        intermed_5 = left-front ++ up-left-front
        intermed_6 = intermed_4 ++ intermed_5
        
        eight_patch_cube = intermed_3 ++ intermed_6
    '''
    intermed_1 = np.concatenate([up_patch, original_patch], 0)    # 0 --> glue along y-axis  
    intermed_2 = np.concatenate([up_front_patch, front_patch], 0) # 0 --> glue along y-axis 
    intermed_3 = np.concatenate([intermed_2, intermed_1], 2)      # 2 --> glue along z-axis 
    
    intermed_4 = np.concatenate([up_left_patch, left_patch], 0)             # 0 --> glue along y-axis 
    intermed_5 = np.concatenate([up_left_front_patch, left_front_patch], 0) # 0 --> glue along y-axis 
    intermed_6 = np.concatenate([intermed_5, intermed_4], 2)                # 2 --> glue along z-axis 
    
    eight_patch_cube = np.concatenate([intermed_6, intermed_3], 1) # 1 --> glue along x-axis
    #plot_volume(eight_patch_cube, 'eight_patch_cube')
    
    return (eight_patch_cube, up_left_front_patchID)


#%%

def crop_multipatchcube(multipatchcube, number_of_cubes_to_glue, ulf_patchID, ROI_width, met_absolute_location):

    if multipatchcube is None: return None

    print('Shape of ' + str(number_of_cubes_to_glue) + '-patch-cube: ' + str(multipatchcube.shape))
    
    ROI_width_half = int(ROI_width / 2)
    
    # ulf = up-left-front
    ulf_absolute_location = cropping.get_patch_absolute_location(ulf_patchID)

    met_location = np.subtract(met_absolute_location, ulf_absolute_location)
    print('Met location within ' + str(number_of_cubes_to_glue) + '-patch-cube: ' + str(met_location))
    
    y_lower = met_location[0] - ROI_width_half
    y_upper = met_location[0] + ROI_width_half
    x_lower = met_location[1] - ROI_width_half
    x_upper = met_location[1] + ROI_width_half
    z_lower = met_location[2] - ROI_width_half
    z_upper = met_location[2] + ROI_width_half
    
    ROI = multipatchcube[y_lower:y_upper, x_lower:x_upper, z_lower:z_upper] 
    
    if ROI is None or ROI.shape != (ROI_width, ROI_width, ROI_width):  
        print('Cropping of ' + str(number_of_cubes_to_glue) + '-patch-cube failed')
        return None
    else:
        print('Successfully cropped ROI out of ' + str(number_of_cubes_to_glue) + '-patch-cube: ')
        print('3D-shape of ROI: ' + str(ROI.shape))
        return ROI


#%%







