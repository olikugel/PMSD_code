import sys
ROOTP = sys.path[-1].replace('.ipython','').replace('//','/')
sys.path.insert(0, ROOTP + 'Documents/Synced/_Promotion/scripts/helperfunctions')
import filehandling
import plotting

import numpy as np
import matplotlib.pyplot as plt

BASEP = ROOTP + 'Documents/Synced/_Promotion/Projects/Leo/'

#%%
# Interactive tool to select all patches that should be ignored for any analysis because
# they are completely empty / only contain noise
# Opens thumbnails of given region, shows contents per patch (cancer channel only) and
# lets you set a flag in region-->patches-->'locationgroup' with a mouse click:
#  - Left: 'Core' (green)
#  - Middle: 'Boundary' (orange)
#  - Right: 'Outside' (red)
#
# You can switch from Z to X projection for a second run to refine the labels
#
# Important: excute last line after finishing to save results!
#

#%%
ShowXinsteadOfZ = True
dataset = 'F15'
path_data = BASEP + 'data/' + dataset + '/'

# Load data
region = filehandling.pload(path_data + 'region')
for p in region['patches']:
    if('locationgroup' not in p.keys()):
        p['locationgroup'] = 'Core' # create entry for any patches that don't have it yet.
TN_size = int((region['partitioning']['patch_size'][0] - region['partitioning']['patch_overlap']) * region['thumbnails']['downsampling'])
d1 = int(TN_size/2 - 1)
d2 = int(TN_size/2 + 2)
[Ny, Nx, Nz] = region['partitioning']['patches_per_dim']


# Define functions
def plotclick(clickdata):
    if(clickdata.inaxes is None):
        return
    else:
        z = axes.index(clickdata.inaxes)
        y = int(clickdata.ydata/TN_size)
        x = int(clickdata.xdata/TN_size)
        if(ShowXinsteadOfZ): x,z = z,x # swap
        #print(str([y,x,z]))
        patch = [p for p in region['patches'] if p['patchstep'] == [y,x,z]][0]
        plotting.print_dict(patch)
        if(clickdata.button is 1):
            set_to_core(patch['id'])
        elif(clickdata.button is 2):
            set_to_boundary(patch['id'])
        else:
            set_to_outside(patch['id'])

def set_to_outside(pid):
    global region, rgb, fig
    assert region['patches'][pid]['id'] == pid
    region['patches'][pid]['locationgroup'] = 'Outside'
    [y,x,z] = region['patches'][pid]['patchstep']
    if(ShowXinsteadOfZ): x,z = z,x # swap
    ax = axes[z]
    rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,:,z] = 0
    rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,0,z] = 1
    ax.imshow(rgb[:,:,:,z])
    plt.show()
    plt.pause(0.01)

def set_to_boundary(pid):
    global region, rgb, fig
    assert region['patches'][pid]['id'] == pid
    region['patches'][pid]['locationgroup'] = 'Boundary'
    [y,x,z] = region['patches'][pid]['patchstep']
    if(ShowXinsteadOfZ): x,z = z,x # swap
    ax = axes[z]
    rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,:,z] = 0
    rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,0,z] = 1
    rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,1,z] = 0.9
    ax.imshow(rgb[:,:,:,z])
    plt.show()
    plt.pause(0.01)

def set_to_core(pid):
    global region, rgb, fig
    assert region['patches'][pid]['id'] == pid
    region['patches'][pid]['locationgroup'] = 'Core'
    [y,x,z] = region['patches'][pid]['patchstep']
    if(ShowXinsteadOfZ): x,z = z,x # swap
    ax = axes[z]
    rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,:,z] = 0
    rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,1,z] = 1
    ax.imshow(rgb[:,:,:,z])
    plt.show()
    plt.pause(0.01)

# show plot
plt.close()
fig = plt.figure(num=101)
axes = []
rgb = np.zeros((Ny*TN_size,Nx*TN_size,3,Nz+1))
for z in range(0,Nz):
    ax = plt.subplot(1,Nz,z+1)
    axes.append(ax)
    projection = region['thumbnails']['MaxProjections_Z'][:,:,z].squeeze()
    if(ShowXinsteadOfZ): projection = region['thumbnails']['MaxProjections_X'][:,:,z].squeeze() # swap
    for j in range(0,3):
        rgb[:,:,j,z] = np.clip(projection,0,30)/90
    #rgb[:,:,1,z] += np.clip(projection,0,600)/901
    rgb[:,:,1,z] += np.clip(projection,0,3000)/4501
    plt.imshow(rgb[:,:,:,z])
    plt.title('Z = ' + str(z))
    if(ShowXinsteadOfZ): plt.title('X = ' + str(z)) # swap
    plt.yticks(np.arange(0*TN_size+d1,Ny*TN_size+d1,TN_size),[str(x) for x in range(0,Ny)])
    plt.xticks(np.arange(0*TN_size+d1,Nx*TN_size+d1,TN_size),[str(x) for x in range(0,Nx)])


# show status of all patches
try:
    segmentation = filehandling.pload(path_data + 'segmentation')
except:
    segmentation = None
N_outside = 0
N_boundary = 0
N_core = 0
N_m_outside = 0
N_m_boundary = 0
for p in region['patches']:
    [y,x,z] = p['patchstep']
    if(ShowXinsteadOfZ): x,z = z,x # swap
    if(p['locationgroup'] == 'Outside'):
        N_outside += 1
        rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,:,z] = 0
        rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,0,z] = 1
        try:
            seg = filehandling.pload(path_data + 'segmentations/segmentation_patch_' + str(p['id']))
            N_m_outside += len(seg['metastases'])
        except:
            pass
    elif(p['locationgroup'] == 'Boundary'):
        N_boundary += 1
        rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,:,z] = 0
        rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,0,z] = 1
        rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,1,z] = 0.9
        try:
            seg = filehandling.pload(path_data + 'segmentations/segmentation_patch_' + str(p['id']))
            N_m_boundary += len(seg['metastases'])
        except:
            pass
    else:
        N_core += 1
        rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,:,z] = 0
        rgb[y*TN_size+d1:y*TN_size+d2,x*TN_size+d1:x*TN_size+d2,1,z] = 1


for z in range(0,Nz):
    ax = plt.subplot(1,Nz,z+1)
    plt.imshow(rgb[:,:,:,z])
    
plt.tight_layout()
plt.suptitle(dataset + '\nRemeber to save results before refreshing/closing!')
fig.canvas.mpl_connect('button_press_event', plotclick)
plt.show()
plt.pause(0.01)


# filehandling.psave(path_data + 'region',region) # <-- RUN THIS LINE TO SAVE RESULTS



