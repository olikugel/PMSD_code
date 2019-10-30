import numpy as np

#%% Handy functions for list of dicts

def filter_dicts(listofdicts, fieldname, value):
    '''
    filtered_list = filter_dicts(listofdicts, fieldname, value)
    
    Returns those dicts where the fieldname matches the desired value
    
    If the desired field is embedded in another field, please provide fieldname
    in a form like 'outerfield-innerfield'
    '''
    
    if('-' in fieldname):
        outerfield = fieldname.split('-')[0]
        innerfield = fieldname.split('-')[1]
        if(type(value) is str and '>' in value):
            filtered_list = [d for d in listofdicts if d[outerfield][innerfield] > int(value.replace('>',''))]
        elif(type(value) is str and '<' in value):
            filtered_list = [d for d in listofdicts if d[outerfield][innerfield] < int(value.replace('<',''))]
        else:
            filtered_list = [d for d in listofdicts if d[outerfield][innerfield] == value]
    else:
        if(type(value) is str and '>' in value):
            filtered_list = [d for d in listofdicts if d[fieldname] > int(value.replace('>',''))]
        elif(type(value) is str and '<' in value):
            filtered_list = [d for d in listofdicts if d[fieldname] < int(value.replace('<',''))]
        elif(type(value) is np.ndarray):
            filtered_list = [d for d in listofdicts if (d[fieldname] == value).all()]
        else:
            filtered_list = [d for d in listofdicts if d[fieldname] == value]
    return filtered_list


def sort_dicts(listofdicts, fieldname):
    '''
    sorted_list = sort_dicts(listofdicts, fieldname)
    
    Sorts a list of dicts by the numeric values of fieldname
    '''
    return sorted(listofdicts, key=lambda k: k[fieldname]) 


def extract_from_dicts(listofdicts, fieldname):
    '''
    values = extract_from_dicts(listofdicts, fieldname)
    
    Iterates through a list of dicts and returns an array with all values
    of the specified dict-field. Values should be numbers
    
    If the desired field is embedded in another field, please provide fieldname
    in a form like 'outerfield-innerfield'
    '''
    values = []
    if('-' in fieldname):
        outerfield = fieldname.split('-')[0]
        innerfield = fieldname.split('-')[1]
        for index, mydict in enumerate(listofdicts):
            values.append(listofdicts[index][outerfield][innerfield])
    else:
        for index, mydict in enumerate(listofdicts):
            values.append(listofdicts[index][fieldname])
    return values

#%% Convert between lists of points <--> binary volumes
    
def points2volume(pointlist, offset='auto', bb='auto', dtype=np.float):
    ''' 
    volume, offset, bb = points2volume(pointlist, offset='auto', bb='auto')
    
    Takes a list of points and returns a volume with binary values, whereas voxels corresponding 
    to a point in the list will be 1, all others will be 0. 
    
    If no explicit offset was provided, the volume is as small as possible to include all points
    by offsetting the volume from the global origin. This offset is returned as a second output.
    
    If no explicit bounding box was provided, the volume is as small as possible to include all 
    points. This bounding box is returned as a third output.
    '''
    if(len(pointlist[0])!=3): raise ValueError('Please provide 3D inputs')
    pointlista = np.asarray(pointlist)
    if(str(offset)=='auto'):
        offset = np.min(pointlista,axis=0)
    else:
        if(offset[0] > np.min(pointlista[:,0])): raise ValueError('Provided offset cuts off points in y-direction')
        if(offset[1] > np.min(pointlista[:,1])): raise ValueError('Provided offset cuts off points in x-direction')
        if(offset[2] > np.min(pointlista[:,2])): raise ValueError('Provided offset cuts off points in z-direction')
    pointlista_rel = pointlista - offset
    if(str(bb)=='auto'):
        bb = np.max(pointlista,axis=0) - offset + [1,1,1]
    else:
        if(bb[0] < np.max(pointlista_rel[:,0])): raise ValueError('Provided bounding box cuts off points in y-direction')
        if(bb[1] < np.max(pointlista_rel[:,1])): raise ValueError('Provided bounding box cuts off points in x-direction')
        if(bb[2] < np.max(pointlista_rel[:,2])): raise ValueError('Provided bounding box cuts off points in z-direction')
    canvas = np.zeros([bb[0], bb[1], bb[2]],dtype=dtype)
    true_val = 1
    if('bool'   in str(dtype)): true_val = True
    if('uint8'  in str(dtype)): true_val = 2**8-1
    if('uint16' in str(dtype)): true_val = 2**16-1
    if('uint32' in str(dtype)): true_val = 2**32-1
    if('float'  in str(dtype)): true_val = 1
    canvas[pointlista_rel[:,0],pointlista_rel[:,1],pointlista_rel[:,2]] = true_val # we don't use bitscale for memory reasons
    return canvas, offset, bb


def volume2points(volume, offset=[0,0,0], threshold=None):
    ''' 
    pointlist = volume2points(volume, offset=[0,0,0], threshold=None)
    
    Takes a 3D volume and return coordinates of voxels with non-zero values. If an offset is 
    provided, it will be added to the coordinates within the volume. If a scalar threshold is 
    provided, the function returns coordinates of voxels equal to or above this threshold
    '''
    if(len(volume.shape)!=3): raise ValueError('Please provide 3D inputs')
    if(threshold is None):
        pointlist = (np.asarray(np.where(volume!=0)).T + offset).tolist()
    else:
        pointlist = (np.asarray(np.where(volume>=threshold)).T + offset).tolist()        
    return pointlist


#%%
def bitscale(x,desired_type):
    ''' 
    y = bitscaleto(x,desired_type)
    Converts and scales 'x' of any Numpy type to 'desired_type'. Desired type can be string or np.dtype
    
    E.g. 'x' is uint8 of value 1 (scale: 0 to 255)
    In this case, y = bitscaleto(x,'uint16') will return value of 257 (scale: 0 to 65,535)
    
    TODO: ################################################################################
        * --> float: scale to [0,1.0]
        float --> *: check whether input is in [0;1], then scale & convert
    ######################################################################################
    '''
    # convert
    # Determine dynamic range of input type
    if('numpy' not in str(type(x))): raise ValueError("Input must be Numpy dtype")
    if('float' not in x.dtype.name and 'int' not in x.dtype.name and 'bool' not in x.dtype.name): raise ValueError("Unknown Numpy dtype provided as input")
    if('bool'  in x.dtype.name):  x_max = 1;
    if('float'  in x.dtype.name): x_max = 1;
    if('int8'  in x.dtype.name):  x_max = 2**8  - 1;
    if('int16' in x.dtype.name):  x_max = 2**16 - 1;
    if('int32' in x.dtype.name):  x_max = 2**32 - 1;
    if('int64' in x.dtype.name):  x_max = 2**64 - 1;
    if('uint' in x.dtype.name or 'bool' in x.dtype.name or 'float' in x.dtype.name):  x_min = 0;
    if('uint'not in x.dtype.name and 'int' in x.dtype.name):  x_min = np.round(-x_max/2); x_max = np.round(x_max/2)-1;
    
    # Determine dynamic range of output type
    if('float' in str(desired_type)): y_max = 1;
    if('int8'  in str(desired_type)): y_max = 2**8  - 1;
    if('int16' in str(desired_type)): y_max = 2**16 - 1;
    if('int32' in str(desired_type)): y_max = 2**32 - 1;
    if('int64' in str(desired_type)): y_max = 2**64 - 1;
    if('uint' in str(desired_type) or 'float' in str(desired_type)):    y_min = 0;
    if('uint'not in str(desired_type) and 'int' in str(desired_type)):  y_min = np.round(-y_max/2); y_max = np.round(y_max/2)-1;  
    
    # Convert to float64 to enable scaling without information loss
    temp = x.astype(np.float64)
    
    # Sanity checks to avoid ill-defined behaviour
    if(x_min != y_min): raise ValueError("Conversion from signed to unsigned or vice versa is ill defined")
    if('float' in x.dtype.name):
        if(np.min(x) < 0): raise ValueError("Float values must be in [0,1], but input has values <0.");
        if(np.max(x) > 1): raise ValueError("Float values must be in [0,1], but input has values >1.");
    
    # Scale dynamic range, convert, and return
    if(x_min == 0):
        if('float' in str(desired_type)):
            temp = y_max/x_max * x
            if('int8'  in x.dtype.name):  temp = np.round(temp,3)  # Round only to avoid pseudo-accuracy
            if('int16'  in x.dtype.name): temp = np.round(temp,5)  # Round only to avoid pseudo-accuracy
            if('int32'  in x.dtype.name): temp = np.round(temp,10) # Round only to avoid pseudo-accuracy
        else:
            temp = np.round(y_max/x_max * x) # value must be rounded for any int type
    else:
        # negative values, by definition, only occur for int types --> will always be rounded
        mask = (x>=0)
        temp = np.round(y_min/x_min * x)
        temp[mask] = np.round(y_max/x_max * x[mask])
    if('class' in str(desired_type)):
        # desired type was provided as np.type
        y = temp.astype(desired_type)
    else:
        # desired type war provided as string
        y = eval('temp.astype(np.' + desired_type + ')')
    return y


#%%-----------------------------
def bitscale_OLD(x,desired_type):
    ''' 
    y = bitscaleto(x,desired_type)
    Converts and scales 'x' of any Numpy type to 'desired_type'.
    
    E.g. 'x' is uint8 of value 1 (scale: 0 to 255)
    In this case, y = bitscaleto(x,'uint16') will return value of 257 (scale: 0 to 65,535)
    
    TODO: ################################################################################
        * --> float: scale to [0,1.0]
        float --> *: check whether input is in [0;1], then scale & convert
    ######################################################################################
    '''
    
    # Determine dynamic range of input type
    if('numpy' not in str(type(x))): raise ValueError("Input must be Numpy dtype")
    if('float' not in x.dtype.name and 'int' not in x.dtype.name and 'bool' not in x.dtype.name): raise ValueError("Unknown Numpy dtype provided as input")
    if('bool'  in x.dtype.name):  x_max = 1;
    if('float'  in x.dtype.name): x_max = 1;
    if('int8'  in x.dtype.name):  x_max = 2**8  - 1;
    if('int16' in x.dtype.name):  x_max = 2**16 - 1;
    if('int32' in x.dtype.name):  x_max = 2**32 - 1;
    if('int64' in x.dtype.name):  x_max = 2**64 - 1;
    if('uint' in x.dtype.name or 'bool' in x.dtype.name or 'float' in x.dtype.name):  x_min = 0;
    if('uint'not in x.dtype.name and 'int' in x.dtype.name):  x_min = np.round(-x_max/2); x_max = np.round(x_max/2)-1;
    
    # Determine dynamic range of output type
    if('float'  in desired_type): y_max = 1;
    if('int8'  in desired_type):  y_max = 2**8  - 1;
    if('int16' in desired_type):  y_max = 2**16 - 1;
    if('int32' in desired_type):  y_max = 2**32 - 1;
    if('int64' in desired_type):  y_max = 2**64 - 1;
    if('uint' in desired_type or 'float' in desired_type):  y_min = 0;
    if('uint'not in desired_type and 'int' in desired_type):  y_min = np.round(-y_max/2); y_max = np.round(y_max/2)-1;  
    
    # Convert to float64 to enable scaling without information loss
    temp = x.astype(np.float64)
    
    # Case 1: * --> float
    # Scale dynamic range, convert, and return
    if('float' in desired_type):
        
        if(desired_type == 'float' or desired_type == 'float64'): y = temp; return y;
        if(desired_type == 'float32'): y = temp.astype(np.float32); return y;
        if(desired_type == 'float16'): y = temp.astype(np.float16); return y;
    
    # Case 2: float --> int
    # Check dynamic range and convert
    if('float' in x.dtype.name):
        if(np.min(x) < y_min):  
            raise ValueError("Smallest input value doesn't fit into desired output type; Scaling would be ill defined");
        if(np.max(x) > y_max):  
            raise ValueError("Largest input value doesn't fit into desired output type; Scaling would be ill defined");
        # Do not scale from float to int (ill defined)
        y = eval('temp.astype(np.' + desired_type + ')')
        return y
    
    # Case 3: int/bool --> int
    # Scale dynamic range, convert, and return
    if(x_min != y_min): raise ValueError("Conversion from signed to unsigned or vice versa is ill defined")
    if(x_min == 0):
        temp = np.round(y_max/x_max * x)
    else:
        mask = (x>=0)
        temp = np.round(y_min/x_min * x)
        temp[mask] = np.round(y_max/x_max * x[mask])
    y = eval('temp.astype(np.' + desired_type + ')')
    return y
