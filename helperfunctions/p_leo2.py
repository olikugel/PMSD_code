import sys
sys.path.append('/home/oschoppe/.ipython')
ROOTP = sys.path[-1].replace('.ipython','').replace('\\','/')
sys.path.insert(0, ROOTP + 'Documents/Synced/_Promotion/scripts/helperfunctions')
import dataconversions
import blobanalysis
import filehandling

import numpy as np
import scipy.stats
import copy
import os.path

BASEP = ROOTP + 'Documents/Synced/_Promotion/Projects/Leo/'

#%%
# Library of customized functions specific for Project Leo that are used in several parts
# of the code and are thus, centrally organized in this file.


#%%
def load_segmentation(dataset,pid,load_partial,THR_dim=200):
    '''
    segmentation = load_segmentation(dataset,pid,load_partial,THR_dim=200)
    
    Not only loads segmentation file for desired patch, but also includes all (dim) metastases
    that belong to adjacent patches if they partially overlap with this patch. 
    
    For those cases, only the overlapping part of the metastasis is included, i.e. this function 
    may NOT BE USED to derive statistical information on the metastases as this will be false 
    for exactly those partial metastases.
    
    Also, if the part within the patch is small (less than 20% of total volume) or dim, it will
    only be included as a "dim" metastasis, i.e. detection will not be rewarded and missing it
    will not be punished; these metastases will have to be detected from the patch in which the
    majority of its volume is in
    
    Inputs:
     - dataset: string specifying dataset ("F15")
     - pid: integer for patch ID to be loaded
     - load_partial: boolean whether or not to load partially overlapping blobs from surrounding patches
     - THR_dim: threshold to check whether partial blobs are now considered dim because brighter parts are in other patch
    '''
    seg = filehandling.pload(BASEP + 'data/' + dataset + '/segmentations/segmentation_patch_' + str(pid))
    if(os.path.isfile('/mnt/C286054386053985/oschoppe/F15/patchvolume_' + str(pid) + '.nii') == False):
        print('WARNING: cannot load local patch files and thus, cannot consider partial blobs properly.')
        print('--> function load_segmentation() will be called with load_partial=False instead')
        print('ALTERED PATHS IN p_leo2 AND ALL HELPER FUNCTIONS!')
        load_partial = False
    if(load_partial):
        region = filehandling.pload(BASEP + 'data/' + dataset + '/region')
        [maxy,maxx,maxz] = np.asarray(region['partitioning']['patch_size']) - region['partitioning']['patch_overlap']
        [y0,x0,z0] = region['patches'][pid]['patchstep']
        cancervol = filehandling.readNifti('/mnt/C286054386053985/oschoppe/F15/patchvolume_' + str(pid))
        add_part_metastases = []
        add_part_dim_metastases = []
        # Go through neighboring patches whose buffer zones overlap with patch
        for dy in [-1,0]:
            for dx in [-1,0]:
                for dz in [-1,0]:
                    [y,x,z] = [y0+dy,x0+dx,z0+dz]
                    if((y>=0 and x>=0 and z>=0) and ((y==y0 and x==x0 and z==z0) is False)):
                        npid = dataconversions.filter_dicts(region['patches'],'patchstep',[y,x,z])[0]['id']
                        nseg = filehandling.pload(BASEP + 'data/' + dataset + '/segmentations/segmentation_patch_' + str(npid))
                        # check whether any metastases overlap with patch
                        dim_metastases = [] if('dim_metastases' not in nseg.keys()) else nseg['dim_metastases']
                        for m in nseg['metastases'] + dim_metastases:
                            abs_bb = m['offset'] + m['boundingbox']
                            if((abs_bb[0] >= (-1)*dy*maxy) and (abs_bb[1] >= (-1)*dx*maxx) and (abs_bb[2] >= (-1)*dz*maxz)):
                                # if so, then include THOSE points WITHIN the patch & adjust coordinate system to new patch
                                patchstep_offset = np.multiply([maxy,maxx,maxz],[dy,dx,dz])
                                shifted_points = m['points'] + patchstep_offset
                                filtered_points = shifted_points[np.min(shifted_points,axis=1) >= 0] # only take points fully within patch
                                if(len(filtered_points) > 0):
                                    # characterize new partial blob
                                    m_partial = {}
                                    m_partial['id'] = 100*npid + m['id']
                                    m_partial['points'] = filtered_points.tolist()
                                    m_partial = blobanalysis.characterize_blob(m_partial) # ~ 0.1 s
                                    m_partial = characterize_metastasis(m_partial,cancervol,min_padding=25,otherblobs=seg['metastases'])  # ~ 0.01 s
                                    # add to main list if NOT dim and at least 20% within patch
                                    if(m_partial['characterization']['maxFG'] > THR_dim and m_partial['volume'] >= 0.2*m['volume']): 
                                        m_partial['INFO'] = 'Partial, but substantial metastases from adjacent patch #' + str(npid)
                                        add_part_metastases.append(m_partial)
                                    else:
                                        m_partial['evaluation']['flag_dim'] = True
                                        m_partial['INFO'] = 'Partial, but dim/very small part of metastases from adjacent patch #' + str(npid)
                                        add_part_dim_metastases.append(m_partial)
        seg['metastases'] += add_part_metastases
        if('dim_metastases' in seg.keys()): seg['dim_metastases'] += add_part_dim_metastases
    
    return seg


#%%
def dilute_segmentation(seedblob,cancervol,min_padding,NofSTDs):
    ''' 
    dilutedblob = dilute_segmentation(seedblob,cancervol,min_padding)
    
    Takes a given blob as a seed to selectively dilute the segmentation within a local RoI within the cancervol.
    The segmentation will be diluted so that all voxels within the segmentation are at least 'NofSTDs' standard 
    deviations above the mean signal within the local RoI.The local RoI is dynamically growing if the segmentation 
    hits the border of the RoI. 

    Inputs:
        * seedblob - dict produced by blobanalysis.characterize_blob(). seedblob['points'] are absolute coordinates 
        * cancervol - 3D Numpy volume spanning the global coordinate system
        * min_padding - the minimum padding around the center of the metastasis to define its local RoI
        * NofSTDs - Integer setting the threshold for dilution, in multiples of standard deviations.
    
    --> Please note:
    implementation with hullpoints is faster, but can lead to issues with region growing (maybe..). To revert, replace
    touchingcandidates = blobanalysis.get_neighbors_of_blob(hullpoints,candidates_abs)
    with
    touchingcandidates = blobanalysis.get_neighbors_of_blob(currentpoints_abs,candidates_abs)
    and delete those lines starting with 'hullpoints = ...'
    '''
    dilutedblob = copy.deepcopy(seedblob)
    ## Step 2.1: Define initial RoI around seedblob
    padding = np.max([np.round(1*seedblob['max_dist']),min_padding]) # diameter of seedblob, but at least 5 voxel
    seedCoM_abs = seedblob['offset'] + seedblob['CoM']
    miny = np.clip(int(seedCoM_abs[0] - padding), 0, cancervol.shape[0])
    maxy = np.clip(int(seedCoM_abs[0] + padding), 0, cancervol.shape[0])
    minx = np.clip(int(seedCoM_abs[1] - padding), 0, cancervol.shape[1])
    maxx = np.clip(int(seedCoM_abs[1] + padding), 0, cancervol.shape[1])
    minz = np.clip(int(seedCoM_abs[2] - padding), 0, cancervol.shape[2])
    maxz = np.clip(int(seedCoM_abs[2] + padding), 0, cancervol.shape[2])
    
    ## Step 2.2: Grow RoI if needed, and dilute blob
    currentpoints_abs = copy.deepcopy(seedblob['points']) # start with seedblob
    hullpoints = copy.deepcopy(seedblob['points']) # In the first iteration, touching points are identified with full blob
    stopgrowing = 0
    while(stopgrowing==0):
        # Get list of candidates in RoI
        currentRoI = cancervol[miny:maxy,minx:maxx,minz:maxz]
        threshold = np.mean(currentRoI)+NofSTDs*np.std(currentRoI) # only add voxels with a signal x STDs above mean signal
        candidates_rel = np.asarray(np.where(currentRoI > threshold)).T.tolist() # find candidates for dilution
        if(len(candidates_rel)>0):
            # IF we found candidates:
            candidates_abs = blobanalysis.point_offset(candidates_rel,[miny,minx,minz],'add')
            candidates_abs = blobanalysis.delete_points(candidates_abs,currentpoints_abs) # Only check additional points
            stopdiluting = 0
            # Incrementally add touching neighbors to blob
            while(len(candidates_abs)>0 and stopdiluting==0):
                touchingcandidates = blobanalysis.get_neighbors_of_blob(hullpoints,candidates_abs)
                if(len(touchingcandidates)>0):
                    # If any candidates touch blob, add them & then remove them from the list
                    hullpoints = copy.deepcopy(touchingcandidates) # As of second iteration, touching points are identified with hull of blob; this could potentially affect iterative region growing...
                    currentpoints_abs = blobanalysis.add_points(currentpoints_abs, touchingcandidates)
                    candidates_abs = blobanalysis.delete_points(candidates_abs,touchingcandidates)
                else:
                    stopdiluting = 1  # If none of the candidates touch the blob, stop diluting
            # If blob touches border of RoI, grow the RoI and repeat
            bminy, bminx, bminz = np.min(np.asarray(currentpoints_abs),0)
            bmaxy, bmaxx, bmaxz = np.min(np.asarray(currentpoints_abs),0)
            blobatborder = (miny==bminy) or (minx==bminx) or (minz==bminz) or (maxy==bmaxy) or (maxx==bmaxx) or (maxz==bmaxz)
            if(blobatborder):
                prevminy = miny
                prevmaxy = maxy
                prevminx = minx
                prevmaxx = maxx
                prevminz = minz
                prevmaxz = maxz
                if(miny==bminy): miny = np.clip(bminy-1, 0, cancervol.shape[0])
                if(maxy==bmaxy): maxy = np.clip(bmaxy+1, 0, cancervol.shape[0])
                if(minx==bminx): minx = np.clip(bminx-1, 0, cancervol.shape[1])
                if(maxx==bmaxx): maxx = np.clip(bmaxx+1, 0, cancervol.shape[1])
                if(minz==bminz): minz = np.clip(bminz-1, 0, cancervol.shape[2])
                if(maxz==bmaxz): maxz = np.clip(bmaxz+1, 0, cancervol.shape[2])
                unchangedRoI = (miny==prevminy) and (minx==prevminx) and (minz==prevminz) and (maxy==prevmaxy) and (maxx==prevmaxx) and (maxz==prevmaxz)
                if(unchangedRoI): 
                    stopgrowing = 1 # We would need to grow beyond the volume --> stop growing RoI
            else:
                stopgrowing = 1 # If blob did not touch border, do not further grow RoI
        else:
            candidates_abs = [] # no candidates pass threshold --> blob will not be further diluted in currentRoI
            stopgrowing = 1
            if(len(currentpoints_abs)==len(seedblob['points'])): 
                currentpoints_abs = [] # if blob is only seedblob & below threshold, remove it
    dilutedblob['points'] = currentpoints_abs
    return dilutedblob


#%% 
def get_local_FG_and_BG(blob,volume,min_padding,otherblobs=None):
    '''
    FG, BG = get_local_FG_and_BG(blob,volume,min_padding,otherblobs=None)
    
    Returns the volume voxel values as segmented in blob as "FG" as 1D-array
    The "BG" are the volume voxel values of the local surrounding, excluding
    the "FG values and the "FG" values that might belong to otherblobs
    '''
    # Step 1: Define local RoI
    padding = np.max([np.round(1*blob['max_dist']),min_padding]) # diameter of blob, but at least X voxels
    CoM_abs = blob['offset'] + blob['CoM']
    miny = np.clip(int(CoM_abs[0] - padding), 0, volume.shape[0])
    maxy = np.clip(int(CoM_abs[0] + padding), 0, volume.shape[0])
    minx = np.clip(int(CoM_abs[1] - padding), 0, volume.shape[1])
    maxx = np.clip(int(CoM_abs[1] + padding), 0, volume.shape[1])
    minz = np.clip(int(CoM_abs[2] - padding), 0, volume.shape[2])
    maxz = np.clip(int(CoM_abs[2] + padding), 0, volume.shape[2])
    RoI = volume[miny:maxy,minx:maxx,minz:maxz]
    # Step 2: Get signal of foreground
    foregroundmask, _, _ = dataconversions.points2volume(blob['points'], offset=[miny,minx,minz], bb=[maxy-miny,maxx-minx,maxz-minz])
    FG = RoI[foregroundmask.astype(np.bool)]
    # Step 3: Get signal of background
    backgroundmask = np.ones(RoI.shape)
    backgroundmask = backgroundmask - foregroundmask # subtract segmentation of blob
    if(otherblobs is not None):      # And, if more blobs were provided, subtract their segmentation as well
        pointsofcloseblobs = []
        for otherblob in otherblobs:
            CoM_distance = blobanalysis.point_dist(CoM_abs,otherblob['offset']+otherblob['CoM'])
            if(CoM_distance < 3*np.max([padding,otherblob['max_dist']])):
                pointsofcloseblobs = blobanalysis.add_points(pointsofcloseblobs,otherblob['points'])
        if(len(pointsofcloseblobs)>0):
            otherforeground, _, _ = dataconversions.points2volume(pointsofcloseblobs, offset=[0,0,0], bb=volume.shape)
            backgroundmask = backgroundmask - otherforeground[miny:maxy,minx:maxx,minz:maxz] # Only subtract for local RoI
            backgroundmask = np.clip(backgroundmask,0,1)
    BG = RoI[backgroundmask.astype(np.bool)]
    return FG, BG

#%%
def characterize_metastasis(blob,cancervol,min_padding,otherblobs=None):
    ''' 
    metastasis = characterize_metastasis(blob,cancervol,min_padding)
    
    Adds additional fields to the dict 'blob', as returned by blobanalysis.get_blobs() and turns 
    this data structure into the data structure 'metastasis' by adding additional fields to the 
    dict and enhancing the 'characterization'-subdict with signal-specific measures.

    Inputs:
        * blob - dict produced by blobanalysis.characterize_blob(). blob['points'] are absolute coordinates 
        * cancervol - 3D Numpy volume spanning the global coordinate system
        * min_padding - the minimum padding around the center of the metastasis to define its local RoI
    
    Optional input:
        * otherblobs - list of dicts of all other segmentations in cancervol (either blobs or metastasis). This is only used to correct for any other segmentation that may overlap with the local RoI, i.e. when calculating statistics of foreground vs. background in the local RoI.

    These additional characterizations will be computed:
        * metastasis['characterization']['contrast']
        * metastasis['characterization']['avgSTDabovelocalmean']
        * metastasis['characterization']['minSTDabovelocalmean']
        * metastasis['characterization']['DR_FR']
        * metastasis['characterization']['minFR']
        * metastasis['characterization']['maxBG']
        * metastasis['characterization']['BGvol_above_minFR']
        
    These entries will be added for later use:
        * metastasis['characterization']['NofSTDs'] = None
        * metastasis['evaluation']['last_inspection'] = None
        * metastasis['evaluation']['manually_added'] = False
        * metastasis['evaluation']['manually_removed'] = False
        * metastasis['evaluation']['flag_interesting'] = False
        * metastasis['evaluation']['flag_problematic'] = False
    
    '''
    metastasis = blob.copy()
    # Step 1: Get foreground of metastasis and local background
    foreground, background = get_local_FG_and_BG(blob,cancervol,min_padding,otherblobs=otherblobs)
    # Step 2: Add further characterizations
    ROI = np.concatenate((foreground,background))
    if(np.mean(background)>0):
        metastasis['characterization']['contrast'] = np.mean(foreground) / np.mean(background)
    else:
        metastasis['characterization']['contrast'] = None
    metastasis['characterization']['avgSTDaboveBGmean'] = (np.mean(foreground) - np.mean(background)) / np.std(background)
    metastasis['characterization']['avgSTDabovelocalmean'] = (np.mean(foreground) - np.mean(ROI)) / np.std(ROI)
    metastasis['characterization']['minSTDabovelocalmean'] = (np.min(foreground)  - np.mean(ROI)) / np.std(ROI)
    if(np.min(foreground)>0):
        metastasis['characterization']['DR_FR'] = np.max(foreground) / np.min(foreground)
    else:
        metastasis['characterization']['DR_FR'] = None
    metastasis['characterization']['minFG'] = np.min(foreground)
    metastasis['characterization']['medianFG'] = np.median(foreground)
    metastasis['characterization']['maxFG'] = np.max(foreground)
    metastasis['characterization']['minBG'] = np.min(background)
    metastasis['characterization']['medianBG'] = np.median(background)
    metastasis['characterization']['maxBG'] = np.max(background)
    metastasis['characterization']['BGvol_above_minFR'] = np.sum(background>np.min(foreground))
    # Step 3: add further entries for later use
    if('NofSTDs' not in metastasis['characterization'].keys()):
        metastasis['characterization']['NofSTDs'] = None
    if('evaluation' not in metastasis.keys()):
        metastasis['evaluation'] = {}
        metastasis['evaluation']['last_inspection'] = None
        metastasis['evaluation']['manually_added'] = False
        metastasis['evaluation']['manually_removed'] = False
        metastasis['evaluation']['flag_interesting'] = False
        metastasis['evaluation']['flag_problematic'] = False
    return metastasis


#%%
def characterize_drugtargeting(metastasis,drugvol,min_padding,thr,otherblobs=None):
    ''' 
    metastasis = characterize_drugtargeting(metastasis,drugvol,thr,min_padding)
    
    Adds the subdict 'DrugCharacterization' to the dict 'metastasis', containing key parameters
    describing how the metastasis has been targeted by the drug molecules

    Inputs:
        * metastasis - dict produced by characterize_metastasis(). blob['points'] are absolute coordinates 
        * drugvol - 3D Numpy volume spanning the global coordinate system
        * min_padding - the minimum padding around the center of the metastasis to define its local RoI
        * thr - the "targeting threshold" to test, given as multiple of mean(BG) value
    
    Optional input:
        * otherblobs - list of dicts of all other metastases. This is only used to correct for any other segmentation that may overlap with the local RoI, i.e. when calculating statistics of foreground vs. background in the local RoI.
    '''
    # Step 1: Get foreground of metastasis and local background
    FG, BG = get_local_FG_and_BG(metastasis,drugvol,min_padding,otherblobs=otherblobs)
    # Step 2: Compute basic parameters
    metastasis['DrugCharacterization'] = {}
    metastasis['DrugCharacterization']['avgSTDaboveBGmean'] = (np.mean(FG) - np.mean(BG))/np.std(BG)
    if(np.mean(BG)>0):
        metastasis['DrugCharacterization']['contrast'] = np.mean(FG) / np.mean(BG)
    else:
        metastasis['DrugCharacterization']['contrast'] = None
    metastasis['DrugCharacterization']['minFG']    = np.min(FG)
    metastasis['DrugCharacterization']['medianFG'] = np.median(FG)
    metastasis['DrugCharacterization']['maxFG']    = np.max(FG)
    metastasis['DrugCharacterization']['minBG']    = np.min(BG)
    metastasis['DrugCharacterization']['medianBG'] = np.median(BG)
    metastasis['DrugCharacterization']['maxBG']    = np.max(BG)
    
    # Step 3: Compute Welch's t-test
    eps = 0.0001 # should only be needed if len(FG)=1 (which should not happen)
    rFG = np.std(FG)**2/len(FG)
    rBG = np.std(BG)**2/len(BG)
    df = (rFG+rBG)**2 / ( rFG**2/(len(FG)-1+eps) + rBG**2/(len(BG)-1+eps) )
    t = (np.mean(FG) - thr*np.mean(BG)) / np.sqrt(rFG + rBG)
    p = scipy.stats.t.sf(t,df)
    metastasis['DrugCharacterization']['ttest_t']  = t
    metastasis['DrugCharacterization']['ttest_p']  = p
    metastasis['DrugCharacterization']['ttest_DF'] = df
    if(metastasis['DrugCharacterization']['contrast'] >= thr and p <= 0.001):
        metastasis['DrugCharacterization']['ttest_result'] = True # significantly targeted
    else:
        metastasis['DrugCharacterization']['ttest_result'] = False # at least not significantly targeted
    return metastasis



