import sys
ROOTP = sys.path[-1].replace('.ipython','').replace('\\','/')
sys.path.insert(0, ROOTP + 'Documents/Synced/_Promotion/scripts/helperfunctions')

import numpy as np
import scipy.ndimage
import multiprocessing


#%%
def point_dist_corrected(p1,p2):
    ''' 
    dist = point_dist(p1,p2)
    
    Returns the distance (scalar) between two points, defined as their vector norm. 
    Points p1 and p2 need to be in p-dimensional vector format, i.e. a 1-D array 
    containing the coordinates of the corresponding pixel/voxel in the respective 
    p-dimensional space (e.g., an image or a volume)
    '''
    ndim = len(p1)
    distsum = 0
    for dim in range(0,ndim):
        distsum = distsum + abs(p1[dim]-p2[dim])**2
    dist = distsum**(1/2)
    return dist


#%%
def point_dist(p1,p2):
    ''' 
    dist = point_dist(p1,p2)
    
    Returns the distance (scalar) between two points, defined as their vector norm. 
    Points p1 and p2 need to be in p-dimensional vector format, i.e. a 1-D array 
    containing the coordinates of the corresponding pixel/voxel in the respective 
    p-dimensional space (e.g., an image or a volume)
    
    IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!######################################################################
    Use the corrected version above
    '''
    ndim = len(p1)
    distsum = 0
    for dim in range(0,ndim):
        distsum = distsum + abs(p1[dim]-p2[dim])**ndim
    dist = distsum**(1/ndim)
    return dist


#%%
def point_offset(old_pointlist,offsetpoint,mode):
    ''' 
    new_pointlist = point_offset(old_pointlist,offsetpoint,mode)
    
    Adds or subtracts offsetpoint from/to each point in old_pointlist. This can be
    used to switch between absolute and relative coordinates
    '''
    if(mode=='add'):
        new_pointlistarray = np.asarray(old_pointlist) + np.asarray(offsetpoint)
    elif(mode=='subtract'):
        new_pointlistarray = np.asarray(old_pointlist) - np.asarray(offsetpoint)
    return new_pointlistarray.tolist()

#%%
def delete_points(old_list,points):
    ''' 
    new_list = delete_points(old_list,points)
    
    Returns a list of all points in 'old_list' that are not part of the list 'points'.
    (The 'old_list' will not be modified by this function.)
    '''
    new_list = old_list.copy()
    for point in points:
        try: 
            new_list.remove(point)
        except:
            pass
    return new_list


#%%
def add_points(old_list,points):
    ''' 
    new_list = add_points(old_list,points)
    
    Returns a list of all points in 'old_list' plus those in 'points' that are not already in old_list
    (The 'old_list' will not be modified by this function.)
    '''
    new_list = old_list.copy()
    for point in points:
        if(point not in new_list): new_list.append(point)
    return new_list

#%%
def get_neighbors_of_point(startpoint,pointlist):
    ''' 
    neighborlist = get_neighbors_of_point(startpoint,pointlist)
    
    Returns a list of all points in 'pointlist' that are direct neighbors to 'startpoint'.
    (The 'pointlist' will not be modified by this function.)
    '''
    neighborlist = []
    n = len(pointlist)
    if(n>=26 and len(startpoint)==3):
        # any point in 3D has a maximum of 3**3-1=26 neighbors; if our pointlist is longer than that, it 
        # will be more efficient to check which of these 26 neighbors are in pointlist
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    point = [startpoint[0]+dy, startpoint[1]+dx, startpoint[2]+dz]
                    if(point in pointlist and point != startpoint): neighborlist.append(point)
        return neighborlist
    elif(n>=8 and len(startpoint)==2):
        # any point in 2D has a maximum of 3**2-1=8 neighbors; if our pointlist is longer than that, it 
        # will be more efficient to check which of these 8 neighbors are in pointlist
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                point = [startpoint[0]+dy, startpoint[1]+dx]
                if(point in pointlist and point != startpoint): neighborlist.append(point)
        return neighborlist
    else:
        # in all other cases, check distance from each point to startpoint
        for i in range(0,n):
            point = pointlist[i]
            dist = point_dist(startpoint,point)
            if(np.round(dist)==1): neighborlist.append(point)
        return neighborlist


#%%
def get_overlap(pointlist1,pointlist2):
    ''' 
    overlappingpoints = get_overlap(pointlist1,pointlist2)
    
    Returns a list of all points in 'pointlist2' that are overlapping with the list of
    points provided in 'pointlist1'.
    '''
    overlappingpoints = []
    for point in pointlist2:
        if(point in pointlist1): overlappingpoints.append(point)
    return overlappingpoints


#%%
def test_overlap(pointlist1,pointlist2):
    ''' 
    test_result = test_overlap(pointlist1,pointlist2)
    
    Checks whether any point in pointlist1 is also in pointlist2
    '''
    # First check trivial cases with fast methods
    pa1 = np.asarray(pointlist1)
    pa2 = np.asarray(pointlist2)
    ndims = pa1.shape[1]
    if(np.min(pa1[:,0]) > np.max(pa2[:,0])): return False
    if(np.min(pa2[:,0]) > np.max(pa1[:,0])): return False
    if(np.min(pa1[:,1]) > np.max(pa2[:,1]) and ndims >= 2): return False
    if(np.min(pa2[:,1]) > np.max(pa1[:,1]) and ndims >= 2): return False
    if(np.min(pa1[:,2]) > np.max(pa2[:,2]) and ndims >= 3): return False
    if(np.min(pa2[:,2]) > np.max(pa1[:,2]) and ndims >= 3): return False
    # Then check point by point overlap
    test_result = False
    maxind = len(pointlist1) - 1
    ind = 0
    while(test_result == False and ind <= maxind):
        if(pointlist1[ind] in pointlist2): test_result = True
        ind += 1
    return test_result


#%%
def get_neighbors_of_blob(pointlist1,pointlist2):
    ''' 
    touchingpoints = get_neighbors_of_blob(pointlist1,pointlist2)
    
    Returns a list of all points in 'pointlist2' that are direct neighbors to any of
    the points in the list 'pointlist1'. This includes points that are part of the blob
    
    ### Idea for speedup ###
    The current implementation calls get_overlap far more often than needed (at least for large 
    pointlists), as the 26 versions of currentpointlist are typically largely overlapping. However,
    the overhead involved in avoiding this may outweigh the benefits (1 implementation tested, but was not faster)
    '''
    if(len(pointlist2[0])!=3): raise ValueError('Please provide 3D inputs')
    touchingpoints = []
    pointlista = np.asarray(pointlist2)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                #shift points relative to pointlist1 & get overlapping points
                currentpointlist = (pointlista + [dy,dx,dz]).tolist()
                overlappingpoints = get_overlap(pointlist1,currentpointlist)
                # shift them back & add to list
                if(len(overlappingpoints)>0): overlappingpoints = (np.asarray(overlappingpoints) - [dy,dx,dz]).tolist()
                touchingpoints = add_points(touchingpoints,overlappingpoints)
    return touchingpoints


#%%
def add_all_neighbors(currentlist,startpoint,searchlist):
    ''' 
    currentlist = add_all_neighbors(currentlist,startpoint,searchlist)
    
    Adds all neighboring points of 'startpoint' and adds them to 'currenlist'.
    This is recursively repeated for all those added points until all points
    connected to the original 'startpoint' have been added to 'currenlist', 
    which then represents the entire blob.
    
    Thus, this function ultimatively returns a list of all points belonging
    to the same blob as the given 'startpoint'.
    
    Please note that 'currentlist' and 'searchlist' are modified by calling this function. Best practice to call it is thus:
        * mystartpoint = listofpoints[start_index]
        * mysearchlist = listofpoints.copy()
        * allneighbors = add_all_neighbors([],mystartpoint,mysearchlist)
    '''
    if(startpoint not in currentlist):
        currentlist.append(startpoint)
        searchlist.remove(startpoint) # only remove points from searchlist that have been actively added
    newneighbors = get_neighbors_of_point(startpoint,searchlist)
    if(len(newneighbors) > 0):
        for neighbor in newneighbors:
            add_all_neighbors(currentlist,neighbor,searchlist)
    return currentlist


#%%
def get_blobs(searchvolume):
    ''' 
    blobs = get_blobs(searchvolume)
    
    This function returns a list of dictionaries, in which each dictionary
    represents one blob in the given 'searchvolume'. A blob is defined as 
    a set of connected points. The 'searchvolume' is expected to be a 
    p-dimensional Numpy array of zero and non-zero values. All neighboring
    non-zero values will be treated as connected points, i.e. a blob.
    
    Each blob dictionary in the list 'blobs' has the following entries:
        * blob['id'] - Number of blob in searchvolume, starting with 0
        * blob['points'] - List of points in this blob. Each point is a 1D Numpy array with p coordinates (one per dimension)
        * blob['offset'] - Offset from bounding box to global coordinate system
        * blob['boundingbox'] - Size of 3D box enclosing the entire blob
        * blob['volume'] - Number of voxels in blob
        * blob['CoM'] - Center of Mass (within bounding box)
        * blob['max_dist'] - Largest distance between any two points of blob
        * blob['characterization'] - Dict of further characterizations
        
    NB: The runtime of this function is largely independent of size of the 
    searchvolume, but grows with the number as well as the size of blobs.
    For busy 3D volumes, get_blobs_fast() can >100 times faster (but might
    falsly merge two almost-overlapping points in rare cases)
    '''
    searchvolume = searchvolume.astype(np.bool) # makes values binary (zero / non-zero)
    totalsearchlist = np.asarray(np.where(searchvolume == True)).T.tolist() # returns list of pointers for all non-zero values
    blobs = []
    i = 0
    
    sys.setrecursionlimit(np.min([100000,np.sum(searchvolume)+3000]))
    try:
        while(len(totalsearchlist) > 0):
            startpoint = totalsearchlist[0].copy()
            searchlist = totalsearchlist.copy()
            allneighbors = add_all_neighbors([],startpoint,searchlist)
            blob = {}
            blob['id'] = i
            blob['points'] = allneighbors
            blob = characterize_blob(blob)
            blobs.append(blob)
            totalsearchlist = delete_points(totalsearchlist,allneighbors)
            i += 1
    except:
        print('Maximum recusion depth hit --> no blobs identified.')
    sys.setrecursionlimit(3000)

    return blobs


#%%
def get_single_blob(volume):
    ''' 
    blob = get_single_blob(volume)
    
    Returns blob-dict with all points in given volume. Only valid for
    volumes for which we already know to only contain a single blob.
    Result would be the same with get_blobs() but it's faster.
    '''
    allpoints = np.asarray(np.where(volume.astype(np.bool) == True)).T.tolist() # returns list of pointers for all non-zero values
    blob = {}
    blob['id'] = 0
    blob['points'] = allpoints
    blob = characterize_blob(blob)
    return blob


#%%
def get_blobs_2D(searcharea):
    '''
    TEST
    '''
    assert(len(searcharea.shape) == 2)
    connectedpixel = np.ones([3,3]) # Mask of pixels that count as connected to center pixel
    labeled,n_blobs = scipy.ndimage.label(searcharea,structure=connectedpixel) #very fast for large 2D, not for large 3D
    bloblist = []
    for b in range(0,n_blobs):
        blob = {}
        blob['points'] = (np.asarray(np.where(labeled==b+1)).T).tolist()
        blob = characterize_blob(blob,reduced=True) # only add minimal information
        bloblist.append(blob)
    return bloblist


#%%
def get_blobs_fast_new(volume):
    ''' 
    blobs = get_blobs_fast_new(volume)
    
    NEW VERSION - Yet undocumented, partially tested.
    '''
    blobs = []
    volume = volume.astype(np.float64) # makes values binary (zero / non-zero)
    # Subdivide volume into subvolumes that only contain a single blob
    Z_blobs = get_blobs_2D(np.max(volume,2)) # YX
    for Z_blob in Z_blobs:
        blobs = blobs + get_blobs_from_Z(volume)
    number_of_processes = 10#multiprocessing.cpu_count()-10
    results = multiprocessing.Pool(number_of_processes).map(createpdf, data)
    outputs = sum(results, [])
    return blobs

def get_blobs_from_Z(Z_blob,volume):
    blobs = []
    # crop along Y and X for given Z_blob
    dy       = Z_blob['offset'][0] # implicit offset-subtraction in y direction
    dx       = Z_blob['offset'][1] # implicit offset-subtraction in x direction
    Y_length = Z_blob['boundingbox'][0]
    X_length = Z_blob['boundingbox'][1]
    subvolume = np.zeros([Y_length,X_length,volume.shape[2]]) # reduce in y and x dimension
    pointlist_yx     = np.asarray(Z_blob['points'])
    pointlist_yx_rel = np.asarray(Z_blob['points']) - [dy,dx]
    subvolume[pointlist_yx_rel[:,0],pointlist_yx_rel[:,1],:]               = volume[pointlist_yx[:,0],pointlist_yx[:,1],:]
    Y_blobs = get_blobs_2D(np.max(subvolume,0))
    for Y_blob in Y_blobs:
        # from Y perspective, now also crop Z
        dz       = Y_blob['offset'][1] # implicit offset-subtraction in z direction
        Z_length = Y_blob['boundingbox'][1]
        subsubvolume = np.zeros([subvolume.shape[0],subvolume.shape[1],Z_length]) # only reduce in z dimension
        pointlist_xz     = np.asarray(Y_blob['points']) + [dx,0] # correct x offset from previous cut
        pointlist_xz_rel = np.asarray(Y_blob['points']) - [0,dz]
        subsubvolume[:,pointlist_xz_rel[:,0],pointlist_xz_rel[:,1]]        = volume[dy:dy+Y_length,pointlist_xz[:,0],pointlist_xz[:,1]]
        X_blobs = get_blobs_2D(np.max(subsubvolume,1))
        for X_blob in X_blobs:
            # from X perspective, further crop Y and Z
            dy2       = X_blob['offset'][0] # further implicit offset-subtraction in y direction
            dz2       = X_blob['offset'][1] # further implicit offset-subtraction in z direction
            Y_length2 = X_blob['boundingbox'][0]
            Z_length2 = X_blob['boundingbox'][1]
            subsubsubvolume = np.zeros([Y_length2,subsubvolume.shape[1],Z_length2]) # only keep x dimension
            pointlist_yz     = np.asarray(X_blob['points']) + [dy,dz]   # correct y and z offset from previous cuts
            pointlist_yz_rel = np.asarray(X_blob['points']) - [dy2,dz2]
            subsubsubvolume[pointlist_yz_rel[:,0],:,pointlist_yz_rel[:,1]] = volume[pointlist_yz[:,0],dx:dx+X_length,pointlist_yz[:,1]]
            
            blob = get_single_blob(subsubsubvolume)
            blob['id'] = len(blobs)
            blob['offset'] += [Z_blob['offset'][0],Z_blob['offset'][1],0]
            blob['points'] = (np.asarray(blob['points']) + [dy+dy2,dx,dz+dz2]).tolist()
            blobs.append(blob)
    return blobs

#%%
def get_blobs_fast(searchvolume):
    return get_blobs_fast_new(searchvolume) ###############################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def get_blobs_fast_OLD(searchvolume):
    ''' 
    blobs = get_blobs_fast(searchvolume)
    
    Much faster version of get_blobs() that solves 3D blob detection via 2D projections.
    Will be >100 times faster in busy 3D volumes but may potentially yield slightly wrong
    results. This will happen in cases where 2 neighboring blobs are so close and of such 
    a shape that their local 2D projections will always overlap (intertwined blobs)
    '''
    if(len(searchvolume.shape) == 2): return get_blobs(searchvolume) # no value add to use fast version (which only works for 3D)
    blobs = []
    searchvolume = searchvolume.astype(np.bool) # makes values binary (zero / non-zero)
    
    # Subdivide volume into subvolumes that only contain a single blob in Z projection
    Z_blobs = get_blobs(np.max(searchvolume,2)) # YX
    for Z_blob in Z_blobs:
        subvolume = np.zeros((Z_blob['boundingbox'][0],Z_blob['boundingbox'][1],searchvolume.shape[2]))
        pointlista = np.asarray(Z_blob['points'])
        pointlista_rel = np.asarray(Z_blob['points']) - Z_blob['offset']
        subvolume[pointlista_rel[:,0],pointlista_rel[:,1],:] = searchvolume[pointlista[:,0],pointlista[:,1],:]
        Y_blobs = get_blobs(np.max(subvolume,0))
        if(len(Y_blobs) == 1):
            # Extract single blob from subvolume
            blob = get_single_blob(subvolume)
            blob['id'] = len(blobs)
            blob['offset'] += [Z_blob['offset'][0],Z_blob['offset'][1],0]
            blob['points'] = (np.asarray(blob['points']) + [Z_blob['offset'][0],Z_blob['offset'][1],0]).tolist()
            blobs.append(blob)
        else:
            for Y_blob in Y_blobs:
                # Again split into subvolumes and then extract single blobs
                subsubvolume =  np.zeros(subvolume.shape)
                Z_start = Y_blob['offset'][1]
                Z_end = Z_start + Y_blob['boundingbox'][1]
                subsubvolume[:,:,Z_start:Z_end] = subvolume[:,:,Z_start:Z_end]
                blob = get_single_blob(subsubvolume)
                blob['id'] = len(blobs)
                blob['offset'] += [Z_blob['offset'][0],Z_blob['offset'][1],0]
                blob['points'] = (np.asarray(blob['points']) + [Z_blob['offset'][0],Z_blob['offset'][1],0]).tolist()
                blobs.append(blob)
    return blobs


#%%
def characterize_blob(blob,reduced=False):
    ''' 
    blob = characterize_blob(blob,reduced=False)
    
    This takes a dictonary 'blob' as an input, calculates various metrics
    to characterize the blob, and adds these metrics to the dictionary before
    returning it.
    
    For the input dictionary, only the field "points" must be given. It 
    should be a list of points in 3D space representing the blob. The points 
    must be given in absolute coordinates
    
    The returned dictionary will comprise the following metrics:
        * blob['offset'] - Offset from bounding box to global coordinate system
        * blob['boundingbox'] - Size of 3D box enclosing the entire blob
        * blob['volume'] - Number of voxels in blob
        * blob['CoM'] - Center of Mass (within bounding box)
        * blob['max_dist'] - Largest distance between any two points of blob
        * blob['characterization']['compactness'] - Volume of blob divided by volume of enclosing sphere
        * blob['characterization']['sphereness'] - Ratio of max_dist to diameter of a sphere with same volume as blob
        * blob['characterization']['stringness'] - Defined as "1-sphereness"; approaches 1 for string-like shapes
        * blob['characterization']['skewness'] - Approaches 1 if blob is thick/dense on one end and has large tail on other side
    '''
    # Crop to relevant region
    if(len(blob['points'])==0):
        print('WARNING: Blob is empty')
        blob['volume'] = 0
        blob['CoM'] = None
        blob['MOP'] = None
        blob['max_dist'] = 0
        blob['characterization'] = {}
        blob['characterization']['compactness'] = None
        blob['characterization']['sphereness'] = None
        blob['characterization']['stringness'] = None
        blob['characterization']['skewness'] = None
        return blob
    boundmin = np.min(blob['points'],axis=0,keepdims=True)
    boundmax = np.max(blob['points'],axis=0,keepdims=True)
    boundingbox = (boundmax-boundmin+1).flatten()
    relpointers = (blob['points'] - boundmin)
    blob['offset'] = boundmin.flatten()
    blob['boundingbox'] = boundingbox
    if(len(blob['points'][0])<3):
        #print('2D blobs are only partially characterized in current implementation.')
        return blob
    if(reduced):
        blob['volume'] = len(blob['points'][1])
        return blob 
    canvas = np.zeros(boundingbox,np.bool)
    canvas[relpointers[:,0],relpointers[:,1],relpointers[:,2]] = 1
    # Volume
    volume = np.sum(canvas)
    blob['volume'] = volume
    if(volume==1):
        blob['CoM'] = relpointers[0]
        blob['MOP'] = relpointers[0]
        blob['max_dist'] = 0
        blob['characterization'] = {}
        blob['characterization']['compactness'] = None
        blob['characterization']['sphereness'] = None
        blob['characterization']['stringness'] = None
        blob['characterization']['skewness'] = None
        return blob
    # Center of Mass
    CoM = np.uint32(np.round(np.mean(relpointers,axis=0,keepdims=True)).flatten())
    blob['CoM'] = CoM
    # Maximum distance between any two points of blob
    dist_to_MOP = 0
    for point in relpointers:
        dist = point_dist(CoM,point)
        if(dist>dist_to_MOP):
            dist_to_MOP = dist
            MOP = point
    max_dist = 0
    for point in relpointers:
        dist = point_dist(MOP,point)
        if(dist>max_dist):
            max_dist = dist
    blob['max_dist'] = max_dist
    # Create subdict, if not existent
    if('characterization' not in blob.keys()):
        blob['characterization'] = {}
    # Compactness
    compactness = volume/(4/3*np.pi*(max_dist/2)**3) # volume of blob divided by volume of enclosing sphere
    compactness = np.clip(compactness,0,1) # clip to 0 and 1 for rounding errors (discrete vs. continuous geometry)
    blob['characterization']['compactness'] = compactness
    # Sphereness
    d_min = 2*(volume/(4/3*np.pi))**(1/3) # diameter of a sphere with same volume
    sphereness = d_min/max_dist # will be 1 if blob is a sphere,
    sphereness = np.clip(sphereness,0,1) # clip to 0 and 1 for rounding errors (discrete vs. continuous geometry)
    blob['characterization']['sphereness'] = sphereness
    # Stringness/elongation
    stringness = 1-d_min/max_dist
    stringness = np.clip(stringness,0,1) # clip to 0 and 1 for rounding errors (discrete vs. continuous geometry)
    blob['characterization']['stringness'] = stringness
    # Skewness
    skewness = 2*(dist_to_MOP/max_dist - 0.5) # approaches 1 if blob is thick/dense on one end and has large tail
    blob['characterization']['skewness'] = skewness
    
    return blob
