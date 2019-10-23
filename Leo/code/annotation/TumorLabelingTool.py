############################ Tumor Labeling Tool ###################################15:15
#                                                                                  #
# This software is an interactive tool for lablling metastases in a 3D volume.     #
# To launch the tool, please select the desired dataset in the "Parameters"        #
# section and then press F5.                                                       #
#                                                                                  #
# In case of questions or errors, please just call Oliver (+49-175-318-8538)       #
#                                                                                  #
####################################################################################



#%% 





NameOfAnnotator = 'Oliver' # <-- Please enter your first name here before pressing F5 to launch the tool









#%% Import libraries
import sys
ROOTP = sys.path[-1].replace('.ipython','').replace('\\','/')
sys.path.insert(0, ROOTP + 'Documents/Synced/_Promotion/scripts/helperfunctions')
import filehandling
import plotting
import blobanalysis
import p_leo

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D # necessary to use ax = fig.gca(projection='3d')
import datetime 
import os.path

BASEP = ROOTP + 'Documents/Synced/_Promotion/Projects/Leo/'

#%% Parameters

dataset = 'F15'
path_segs     = BASEP + 'data/' + dataset + '/segmentations/'
path_patches  = ROOTP + 'Documents/LocalData/' + dataset + '/' if('oliver' in ROOTP) else '/mnt/C286054386053985/oschoppe/' + dataset + '/' # path switch Großhadern / TranslaTUM
path_autofluo = ROOTP + 'Documents/LocalData/' + dataset + '_C00_autofluo/' if('oliver' in ROOTP) else '/mnt/C286054386053985/oschoppe/' + dataset + '_C00_autofluo/' # path switch Großhadern / TranslaTUM
region = filehandling.pload(BASEP + 'data/' + dataset + '/region')
signalcap_default = 500 # cap for absolute intensity (idealy match to Fiji view of z-projection)
fs = [8, 10, 12, 14] if('oliver' in ROOTP) else [12, 14, 16, 18]
#sc = [124, 402, 419, 423, 451, 452, 459, 465, 466, 472, 500, 501, 502, 508, 509, 514, 515, 522, 552, 577, 593, 638, 650, 685, 690, 691, 838, 846, 863, 1126] # full list for F15
sc = [] #currently outstanding
plt.close('all')


#%% Define global variables with default values
print('Launching tool... please wait.')
# General
signalcap = signalcap_default
patch = None
next_patch = None
prev_patches = []
segmentation = None
volume = None
volume_vis = None
volume_autofluo = None
max_mID = None
metastases = None
DIMmetastases = None
removed_metastases = None
main_n_added = None
signalcap_FROZEN = False

# For overview window
overview_fig = plt.figure(num=100)
if('segs_completed' not in locals()):
    segs_completed = []
    segs_outstanding = []
    for p in region['patches']:
        if(os.path.isfile(path_patches + 'patchvolume_' + str(p['id']) + '.nii')):
            s = p_leo.load_segmentation(dataset,p['id'],load_partial=False)
            segs_completed.append(s['patch_id']) if(s['evaluation']['completed']) else segs_outstanding.append(s['patch_id'])


# For main window
main_fig = plt.figure(num=101)
main_ax_plt = main_fig.gca()
main_view = 'Z-projection'
main_mode = 'Inspect'
main_last_action = ''
main_w_flag_FROZEN = False

# For inspect window
inspect_fig = plt.figure(num=102)
inspect_ax_plt_list = []
inspect_view = 'Max. Projections'
inspect_maskvis = False
inspect_w_maskvis_FROZEN = False
inspect_w_masksize_FROZEN = False
inspect_w_flag_FROZEN = False
inspect_addremove_STATUS = None
OoI = None 
metastasis = None
CoM_abs = None
inspect_last_action = ''


#%% Define functions for overview window

def overview_next_patch(clickdata):
    global next_patch, prev_patches
    try:
        index = np.random.randint(0,len(segs_outstanding))
    except:
        print('\n\n\n\nThere are no more patches left to review.\n\n\n\n')
        next_patch = region['patches'][0]
        return
    next_patch_id = segs_outstanding[index]
    if(np.random.rand()>0.8 and NameOfAnnotator=="Madita" and False):
        priolist = blobanalysis.delete_points(sc,segs_completed)
        if(len(priolist)==0):
            file = open('Annotation variability/annstatus.txt', 'w+')
            file.write("all done ("+NameOfAnnotator+")")
            file.close()
        else:
            file = open('Annotation variability/annstatus.txt', 'w+')
            file.write("partially done: "+str(len(sc)-len(priolist))+" of "+str(len(sc)))
            file.close()
            next_patch_id = priolist[np.random.randint(0,len(priolist))]
    next_patch = next((p for p in region['patches'] if p["id"] == next_patch_id))
    prev_patches.append(next_patch['id'])
    overview_update_plot()


def overview_set_patch(desired_patch):
    global next_patch, prev_patches
    if(type(desired_patch)==int):
        next_patch = next((p for p in region['patches'] if p["id"] == desired_patch))
    elif(type(desired_patch)==list and len(desired_patch)==3):
        next_patch = next((p for p in region['patches'] if p["patchstep"] == desired_patch))
    prev_patches.append(next_patch['id'])
    overview_update_plot()


def overview_prev_patch(clickdata):
    global next_patch, prev_patches
    if(len(prev_patches)>0):
        next_patch = next((p for p in region['patches'] if p["id"] == prev_patches[-1]))
        del prev_patches[-1]
        overview_update_plot()


def overview_open_patch(clickdata):
    global signalcap, OoI, main_last_action, inspect_last_action
    update_sigcap(signalcap_default)
    load_patch(next_patch['id'])
    OoI = None
    main_last_action = ''
    inspect_last_action = ''
    main_update_title()
    main_update_plot()
    inspect_update_title()
    inspect_update_plot()


def overview_update_plot():
    global overview_ax1, overview_ax2, overview_ax3, overview_ax4, overview_ax5
    plt.figure(num=overview_fig.number)

    ds = region['thumbnails']['downsampling']
    [dy, dx, dz] = (np.asarray(region['partitioning']['patch_size']) - region['partitioning']['patch_overlap']) * ds
    
    overview_ax1.clear()
    overview_ax1.set_title('Top view full volume', fontsize=fs[2], fontweight='bold')
    plotting.intensity(np.max(region['thumbnails']['MaxProjections_Z'],2),ahandle=overview_ax1,cap=signalcap)
    overview_ax1.add_patch(patches.Rectangle((dx*next_patch['patchstep'][1],dy*next_patch['patchstep'][0]),dx,dy,linewidth=2,fill=False,edgecolor='red'))
    overview_ax1.set_xticks([])
    overview_ax1.set_yticks([])
    overview_ax1.set_ylabel('Y-Axis')
    overview_ax1.set_xlabel('X-Axis')
    
    overview_ax2.clear()
    overview_ax2.set_title('Side view of full volume', fontsize=fs[2], fontweight='bold')
    plotting.intensity(np.max(region['thumbnails']['MaxProjections_X'],2),ahandle=overview_ax2,cap=signalcap)
    overview_ax2.add_patch(patches.Rectangle((dz*next_patch['patchstep'][2],dy*next_patch['patchstep'][0]),dz,dy,linewidth=2,fill=False,edgecolor='red'))
    overview_ax2.set_xticks([])
    overview_ax2.set_yticks([])
    overview_ax2.set_xlabel('Z-Axis')
    
    overview_ax3.clear()
    overview_ax3.set_title('Top view of selected slice', fontsize=fs[2], fontweight='bold')
    plotting.intensity(region['thumbnails']['MaxProjections_Z'][:,:,next_patch['patchstep'][2]],ahandle=overview_ax3,cap=signalcap)
    overview_ax3.add_patch(patches.Rectangle((dx*next_patch['patchstep'][1],dy*next_patch['patchstep'][0]),dx,dy,linewidth=2,fill=False,edgecolor='red'))
    overview_ax3.set_xticks([])
    overview_ax3.set_yticks([])
    overview_ax3.set_xlabel('X-Axis')
    
    overview_ax4.clear()
    Z0 = next_patch['offset'][2]
    Z1 = next_patch['offset'][2] + (next_patch['boundingbox'][2]-region['partitioning']['patch_overlap'])
    X = int(next_patch['offset'][1] + 0.5* (next_patch['boundingbox'][1]-region['partitioning']['patch_overlap']))
    Y = int(next_patch['offset'][0] + 0.5* (next_patch['boundingbox'][0]-region['partitioning']['patch_overlap']))
    overview_text = 'The core body is partitioned into '+str(len(region['patches']))+' patches (x='+str(region['partitioning']['patches_per_dim'][1])+'; y='+str(region['partitioning']['patches_per_dim'][0])+'; z='+str(region['partitioning']['patches_per_dim'][2])+'). \n\nThe segmentations of '+str(len(segs_completed))+' patches ('+str(int(100*len(segs_completed)/len(region['patches'])))+'%) have been evaluated, \n'+str(len(segs_outstanding))+' are still outstanding. \n\nCurrent patch is #'+str(next_patch['id'])+' (x='+str(next_patch['patchstep'][1])+'; y='+str(next_patch['patchstep'][0])+'; z='+str(next_patch['patchstep'][2])+'). \n\nIn the full volume (i.e. if you open it in Fiji), this patch is \nbetween slices '+str(Z0)+' and '+str(Z1)+' and its centerpoint \nis at x='+str(X)+' and y='+str(Y)+'.'
    overview_ax4.text(0,0.2,overview_text, fontsize=fs[2])
    overview_ax4.axis('off')
    
    overview_ax5.clear()
    overview_ax5.set_title('Preview of current patch (#'+str(next_patch['id'])+') - open to see labels', fontsize=fs[2], style='italic')
    pvol = filehandling.readNifti(path_patches + 'patchvolume_' + str(next_patch['id']))
    plotting.intensity(np.max(pvol[0:int(dy/ds),0:int(dx/ds),0:int(dz/ds)],2),ahandle=overview_ax5,cap=signalcap)

    plt.pause(0.01)
    plt.draw()


#%% Define functions for main window

# Helper functions for main window

def main_update_title():
    global main_fig
    plt.figure(num=main_fig.number)
    if(segmentation is None):
        main_fig.suptitle('Please select a patch to proceed.')
    else:
        main_fig.suptitle(segmentation['dataset'] + ' Patch #' + str(segmentation['patch_id']) + ': #found=' + str(len(metastases)) + '; #added=' + str(main_n_added) + '; #removed=' + str(len(removed_metastases)) + ';\n Last action: ' + main_last_action)
    plt.pause(0.01)
    plt.draw()


def main_update_plot():
    global main_ax_plt
    plt.figure(num=main_fig.number)
    main_ax_plt.clear()
    if(volume_vis is None):
        return
    # Project background in green channel
    overlap = region['partitioning']['patch_overlap']
    if(main_view == 'Z-projection'):
        rgb = np.zeros([volume_vis.shape[0],volume_vis.shape[1],3])
        rgb[:,:,1] = np.max(volume_vis[:,:,:-overlap],2)
        vlabel = 'Y-axis'
        hlabel = 'X-axis'
    elif(main_view == 'Y-projection'):
        rgb = np.zeros([volume_vis.shape[1],volume_vis.shape[2],3])
        rgb[:,:,1] = np.max(volume_vis[:-overlap,:,:],0)
        vlabel = 'X-axis'
        hlabel = 'Z-axis'
    elif(main_view == 'X-projection'):
        rgb = np.zeros([volume_vis.shape[0],volume_vis.shape[2],3])
        rgb[:,:,1] = np.max(volume_vis[:,:-overlap,:],1)
        vlabel = 'Y-axis'
        hlabel = 'Z-axis'
    # Overlay segmentation as red channel
    for metastasis in metastases:
        # retrieve info from metastasis
        pointarray = np.asarray(metastasis['points'])
        offset = metastasis['offset']
        labelpadding = 1
        labely = np.clip(int(offset[0] - labelpadding), 0, volume_vis.shape[0])
        labelx = np.clip(int(offset[1] - labelpadding), 0, volume_vis.shape[1])
        labelz = np.clip(int(offset[2] - labelpadding), 0, volume_vis.shape[2])
        if(labely<2*labelpadding): labely = 2*labelpadding
        if(labelx<2*labelpadding): labelx = 2*labelpadding
        if(labelz<2*labelpadding): labelz = 2*labelpadding
        # for all segmented voxels, copy green channel value for red channel
        if(main_view == 'Z-projection'):
            rgb[pointarray[:,0],pointarray[:,1],0] = rgb[pointarray[:,0],pointarray[:,1],1] 
            main_ax_plt.text(labelx,labely,str(metastasis['id']),color=[1, 1, 1],fontsize=fs[0])
        elif(main_view == 'Y-projection'):
            rgb[pointarray[:,1],pointarray[:,2],0] = rgb[pointarray[:,1],pointarray[:,2],1]
            main_ax_plt.text(labelz,labelx,str(metastasis['id']),color=[1, 1, 1],fontsize=fs[0])
        elif(main_view == 'X-projection'):
            rgb[pointarray[:,0],pointarray[:,2],0] = rgb[pointarray[:,0],pointarray[:,2],1]
            main_ax_plt.text(labelz,labely,str(metastasis['id']),color=[1, 1, 1],fontsize=fs[0])
    
    # ALSO HIGHLIGHT DIM METASTASES AS RED
    for dim_met in DIMmetastases:
        # retrieve info from metastasis
        pointarray = np.asarray(dim_met['points'])
        offset = dim_met['offset']
        labelpadding = 1
        labely = np.clip(int(offset[0] - labelpadding), 0, volume_vis.shape[0])
        labelx = np.clip(int(offset[1] - labelpadding), 0, volume_vis.shape[1])
        labelz = np.clip(int(offset[2] - labelpadding), 0, volume_vis.shape[2])
        if(labely<2*labelpadding): labely = 2*labelpadding
        if(labelx<2*labelpadding): labelx = 2*labelpadding
        if(labelz<2*labelpadding): labelz = 2*labelpadding
        # for all segmented voxels, turn green pixels red
        if(main_view == 'Z-projection'):
            rgb[pointarray[:,0],pointarray[:,1],0] = rgb[pointarray[:,0],pointarray[:,1],1] 
            rgb[pointarray[:,0],pointarray[:,1],1] = 0.5*rgb[pointarray[:,0],pointarray[:,1],1]
            main_ax_plt.text(labelx,labely,str(dim_met['id']),color=[1, 0, 0],fontsize=fs[0])
        elif(main_view == 'Y-projection'):
            rgb[pointarray[:,1],pointarray[:,2],0] = rgb[pointarray[:,1],pointarray[:,2],1]
            rgb[pointarray[:,1],pointarray[:,2],1] = 0.5*rgb[pointarray[:,1],pointarray[:,2],1]
            main_ax_plt.text(labelz,labelx,str(dim_met['id']),color=[1, 0, 0],fontsize=fs[0])
        elif(main_view == 'X-projection'):
            rgb[pointarray[:,0],pointarray[:,2],0] = rgb[pointarray[:,0],pointarray[:,2],1]
            rgb[pointarray[:,0],pointarray[:,2],1] = 0.5*rgb[pointarray[:,0],pointarray[:,2],1]
            main_ax_plt.text(labelz,labely,str(dim_met['id']),color=[1, 0, 0],fontsize=fs[0])
    
    # Render updated plot
    rgb[-overlap:-overlap+1,:-overlap,0] = 1
    rgb[:-overlap,-overlap:-overlap+1,0] = 1
    main_ax_plt.imshow(rgb)
    main_ax_plt.set_ylabel(vlabel)
    main_ax_plt.set_xlabel(hlabel)
    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    main_flag_updaterender()
    main_update_title()


def main_getOoI(vpos,hpos):
    global main_last_action
    labelpadding = 1 # add tolarance so that click on label can be used to select
    vmult = 5 
    if(main_view == 'Z-projection'):
        ypos = vpos
        xpos = hpos
        mID = []
        for metastasis in metastases:
            miny = metastasis['offset'][0] - labelpadding
            maxy = metastasis['offset'][0] + metastasis['boundingbox'][0]
            minx = metastasis['offset'][1] - vmult * labelpadding
            maxx = metastasis['offset'][1] + metastasis['boundingbox'][1]
            if(miny<=ypos<=maxy and minx<=xpos<=maxx):
                mID.append(metastasis['id'])
    if(main_view == 'Y-projection'):
        xpos = vpos
        zpos = hpos
        mID = []
        for metastasis in metastases:
            minx = metastasis['offset'][1] - labelpadding
            maxx = metastasis['offset'][1] + metastasis['boundingbox'][1]
            minz = metastasis['offset'][2] - vmult * labelpadding
            maxz = metastasis['offset'][2] + metastasis['boundingbox'][2]
            if(minx<=xpos<=maxx and minz<=zpos<=maxz):
                mID.append(metastasis['id'])
    if(main_view == 'X-projection'):
        ypos = vpos
        zpos = hpos
        mID = []
        for metastasis in metastases:
            miny = metastasis['offset'][0] - labelpadding
            maxy = metastasis['offset'][0] + metastasis['boundingbox'][0]
            minz = metastasis['offset'][2] - vmult * labelpadding
            maxz = metastasis['offset'][2] + metastasis['boundingbox'][2]
            if(miny<=ypos<=maxy and minz<=zpos<=maxz):
                mID.append(metastasis['id'])
    if(len(mID)==0):
        if(main_view == 'Z-projection'):
            ypos = vpos
            xpos = hpos
            zpos = np.where(volume[ypos,xpos,:]==np.max(volume[ypos,xpos,:]))[0][0]
        elif(main_view == 'Y-projection'):
            xpos = vpos
            zpos = hpos
            ypos = np.where(volume[:,xpos,zpos]==np.max(volume[:,xpos,zpos]))[0][0]
        elif(main_view == 'X-projection'):
            ypos = vpos
            zpos = hpos
            xpos = np.where(volume[ypos,:,zpos]==np.max(volume[ypos,:,zpos]))[0][0]
        PoI = (ypos,xpos,zpos)
        return PoI
    if(len(mID)==1):
        return mID[0]
    if(len(mID)>1):
        main_last_action = 'There are several metastases here, a random one of them was selected.'
        main_update_title()
        print("Please switch view to select unambiguous metastasis.")
        return mID[np.random.randint(0,len(mID))]


def main_flag_updaterender():
    # Update the shown flags to match saved data
    global main_w_flag, main_w_flag_FROZEN
    main_w_flag_FROZEN = True # If the widget is updated for the next segmentation, don't let this be mistaken for a click
    status = main_w_flag.get_status()
    if(status[0] is not segmentation['evaluation']['completed']):
        main_w_flag.set_active(0)
    if(status[1] is not segmentation['evaluation']['flag_interesting']):
        main_w_flag.set_active(1)
    if(status[2] is not segmentation['evaluation']['flag_problematic']):
        main_w_flag.set_active(2)
    main_w_flag_FROZEN = False


# Interactive functions for main window
def main_view_change(label):
    global main_view
    main_view = label
    main_update_title()
    main_update_plot()


def main_mode_change(label):
    global main_mode
    main_mode = label


def main_flag_change(label):
    global main_last_action, segmentation
    if(main_w_flag_FROZEN == True):
        return # Don't do anything if widget is being updated
    if(label=='Completed'):
         segmentation['evaluation']['completed'] = not segmentation['evaluation']['completed']
         main_last_action = 'Updated completion status = ' +  str(segmentation['evaluation']['completed'])
    if(label=='Interesting'):
         segmentation['evaluation']['flag_interesting'] = not segmentation['evaluation']['flag_interesting']
         main_last_action = 'Updated flag interesting = ' +  str(segmentation['evaluation']['flag_interesting'])
    if(label=='Problematic'):
        segmentation['evaluation']['flag_problematic'] = not segmentation['evaluation']['flag_problematic']
        main_last_action = 'Updated flag problematic = ' +  str(segmentation['evaluation']['flag_problematic'])
    main_update_title()


def main_saveandclose(event):
    global segmentation, main_last_action, inspect_last_action
    global metastasis, OoI, patch, volume, volume_vis, max_mID, DIMmetastases, metastases, removed_metastases, main_n_added
    segmentation['evaluation']['last_inspection'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M") 
    segmentation['evaluation']['last_editor'] = NameOfAnnotator 
    segmentation['metastases'] = metastases
    segmentation['dim_metastases'] = DIMmetastases
    segmentation['removed_metastases'] = removed_metastases
    segmentation['max_mID'] = max_mID
    if(segmentation['evaluation']['completed']):
        try:
            segs_outstanding.pop(segs_outstanding.index(segmentation['patch_id']))
        except:
            pass # this will happen if we overwrite an previously completed patch
        segs_completed.append(segmentation['patch_id'])
    filehandling.psave(path_segs + 'segmentation_patch_' + str(patch['id']),segmentation)
    update_sigcap(signalcap_default)
    metastasis = None
    OoI = None
    patch = None
    segmentation = None
    volume = None
    volume_vis = None
    max_mID = None
    metastases = None
    DIMmetastases = None
    removed_metastases = None
    main_n_added = None
    main_last_action = ''
    inspect_last_action = ''
    inspect_update_plot()
    inspect_update_title()
    main_update_plot()
    main_update_title()
    print('Saved all changes to segmentation file.')


def main_plotclick(clickdata):z
    global OoI, metastasis, main_last_action, inspect_last_action
    metastasis = None
    if(clickdata.inaxes is not main_ax_plt or clickdata.button is not 3):
        return
    else:
        vpos = int(np.round(clickdata.ydata))
        hpos = int(np.round(clickdata.xdata))
        OoI = main_getOoI(vpos,hpos)
        if(type(OoI)==int):
            metastasis = next((m for m in metastases if m["id"] == OoI))
            if('flag_dim' not in metastasis['evaluation'].keys()): metastasis['evaluation']['flag_dim'] = False
    if(main_mode=='Inspect' and OoI is not None):
        if(type(OoI)==int):
            main_last_action = 'Inspected metastasis #' + str(OoI)
            inspect_last_action = ''
        else:
            main_last_action = 'Inspected region around ' + pointstring(OoI)
            inspect_last_action = ''
        main_update_title()
        inspect_update_plot()
    if(main_mode=='DirectAdd' and type(OoI)==tuple):
        add_metastasis()
    if(main_mode=='DirectRemoval' and type(OoI)==int):
        remove_metastasis()
    main_update_title()


#%% Define functions for Inspect window

# Interactive functions  for Inspect window
def inspect_update_title():
    global inspect_fig
    plt.figure(num=inspect_fig.number)
    if(OoI is None):
        inspect_fig.suptitle('Please click on an Object of Interest for further inspection')
    elif(type(OoI)==int):
        inspect_fig.suptitle('Metastasis # '+str(OoI)+': CoM at ' + pointstring(CoM_abs)+'\n Last action: ' + inspect_last_action)
    else:
        inspect_fig.suptitle('Region of Interest with center at ' + pointstring(OoI) +'\n Last action: ' + inspect_last_action)
    plt.pause(0.01)
    plt.draw()


def inspect_update_plot():
    global metastasis, CoM_abs, inspect_fig, inspect_ax_plt_list, inspect_addremove_STATUS, inspect_maskvis
    plt.figure(num=inspect_fig.number)
    try:
        for ax in inspect_ax_plt_list:
            inspect_fig.delaxes(ax)
    except:
        pass
    plt.draw()
    if(OoI is None):
        return
    min_padding = segmentation['autosegmentation']['min_padding']
    if(type(OoI)==int): 
        metastasis = next((m for m in metastases if m["id"] == OoI))
        if(metastasis['characterization']['NofSTDs'] is None): metastasis['characterization']['NofSTDs'] = 4
        padding = np.max([np.round(1*metastasis['max_dist']),min_padding]) # diameter of seedblob, but at least 5 voxel
        CoM_abs = metastasis['offset'] + metastasis['CoM']
        inspect_addremove_STATUS = 'remove'
    else:
        metastasis = None
        inspect_addremove_STATUS = 'add'
        inspect_maskvis = False
        padding = min_padding
        CoM_abs = OoI
    inspect_maskvis_updaterender()
    inspect_masksize_updaterender()
    inspect_flag_updaterender()
    inspect_addremove_updaterender()
    
    miny = np.clip(int(CoM_abs[0] - padding), 0, volume_vis.shape[0])
    maxy = np.clip(int(CoM_abs[0] + padding), 0, volume_vis.shape[0])
    minx = np.clip(int(CoM_abs[1] - padding), 0, volume_vis.shape[1])
    maxx = np.clip(int(CoM_abs[1] + padding), 0, volume_vis.shape[1])
    minz = np.clip(int(CoM_abs[2] - padding), 0, volume_vis.shape[2])
    maxz = np.clip(int(CoM_abs[2] + padding), 0, volume_vis.shape[2])
    mybox = volume_vis[miny:maxy,minx:maxx,minz:maxz]
    mask = np.zeros(mybox.shape).astype(np.bool)
    ax2 = None
    if(type(OoI)==int):
        abs_points = metastasis['points']
        pointarray = np.asarray(abs_points) - np.asarray([miny,minx,minz])
        mask[pointarray[:,0],pointarray[:,1],pointarray[:,2]] = 1
        plotting.print_dict(metastasis)
        #if(inspect_view is not '3D view'):
            #ax2 = inspect_fig.add_subplot(2,2,2)
            #ax2.text(0,0.3,plotting.print_dict(metastasis,filterlist='volume,max_dist,characterization,medianFG,maxFG,maxBG,avgSTDabovelocalmean,NofSTDs,evaluation,last_inspection,manually_added',printit=False),family='monospace',fontsize=fs[1])
            #ax2.axis('off')
    ############## Autofluo instead of metastasis info (which is the code above within the if statement)
    if(inspect_view is not '3D view'):
        ax2 = inspect_fig.add_subplot(2,2,2)
        autofluo_projection = np.max(volume_autofluo[miny:maxy,minx:maxx,minz:maxz],2)
        plotting.intensity(autofluo_projection,color='white',cap=np.max(autofluo_projection),ahandle=ax2)
        plt.title('Autofluorescence (z-Projection)')
    ##############
    
    if(inspect_view=='Max. Projections'):
        ax1 = inspect_fig.add_subplot(2,2,1)
        ax3 = inspect_fig.add_subplot(2,2,3)
        ax4 = inspect_fig.add_subplot(2,2,4)
        inspect_ax_plt_list = [ax1, ax3, ax4, ax2]
        plotting.projections(mybox, mask=mask, ahandles=inspect_ax_plt_list[0:3], outline=inspect_maskvis)
    if(inspect_view=='Cum. Projections'):
        ax1 = inspect_fig.add_subplot(2,2,1)
        ax3 = inspect_fig.add_subplot(2,2,3)
        ax4 = inspect_fig.add_subplot(2,2,4)
        inspect_ax_plt_list = [ax1, ax3, ax4, ax2]
        plotting.projections(mybox, mask=mask, ahandles=inspect_ax_plt_list[0:3], outline=inspect_maskvis, mode='cum')
    elif(inspect_view=='Slices through center'):
        ax1 = inspect_fig.add_subplot(2,2,1)
        ax3 = inspect_fig.add_subplot(2,2,3)
        ax4 = inspect_fig.add_subplot(2,2,4)
        inspect_ax_plt_list = [ax1, ax3, ax4, ax2]
        plotting.orthoslices(mybox, mask=mask, ahandles=inspect_ax_plt_list[0:3], outline=inspect_maskvis)
    elif(inspect_view=='3D view'):
        inspect_ax_plt_list = [inspect_fig.gca(projection='3d')]
        plotting.voxelintensity(mybox, mask=mask, ahandle=inspect_ax_plt_list[0],threshold=0.08)
        inspect_rotate_3dplot_to('YX')

    # Render updated plot
    plt.draw()
    plt.subplots_adjust(top=0.8)
    inspect_update_title()
    
    # Update last inspection
    if(type(OoI)==int):
        metastasis['evaluation']['last_inspection'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")  
        save_updated_metastasis()


def inspect_view_change(label):
    global inspect_view, inspect_last_action
    temp = inspect_last_action
    inspect_last_action = '......... PLEASE WAIT .........'
    inspect_update_title()
    inspect_last_action = temp
    inspect_view = label
    inspect_update_plot()


def inspect_maskvis_change(label):
    global inspect_maskvis
    if(inspect_w_maskvis_FROZEN):
        return
    if(label=='Highlight outline' and type(OoI) == int):
        inspect_maskvis = True
        inspect_update_plot()
    else:
        inspect_maskvis = False
        inspect_update_plot()


def inspect_masksize_change(label):
    global inspect_last_action, metastasis
    if(type(OoI) == tuple or inspect_w_masksize_FROZEN):
        return # Don't do anything if widget is being updated or OoI is not a metastasis
    NofSTDs = metastasis['characterization']['NofSTDs']
    inspect_last_action = '......... PLEASE WAIT .........'
    inspect_update_title()
    if(label=='Shrink outline'):
        NofSTDs = 1.5 * NofSTDs
        metastasis['points'] = [(metastasis['offset']+metastasis['CoM']).tolist()]
        inspect_last_action = 'Shrank outline to ' + str(np.round(NofSTDs,1)) + ' STDs above mean signal in local RoI'
    elif(label=='Expand outline'):
        NofSTDs = 1/1.5 * NofSTDs
        inspect_last_action = 'Expanded outline to ' + str(np.round(NofSTDs,1)) + ' STDs above mean signal in local RoI'
    else:
        if(NofSTDs == segmentation['autosegmentation']['NofSTDs']):
            return # We ARE already at the default value
        elif(NofSTDs < segmentation['autosegmentation']['NofSTDs']):
            metastasis['points'] = [(metastasis['offset']+metastasis['CoM']).tolist()] # to remove dispensable points
        NofSTDs = segmentation['autosegmentation']['NofSTDs']
        inspect_last_action = 'Outline was reset to ' + str(np.round(NofSTDs,1)) + ' STDs above mean signal in RoI surrounding'

    min_padding = segmentation['autosegmentation']['min_padding']
    dilutedblob = p_leo.dilute_segmentation(metastasis,volume,min_padding,NofSTDs)
    dilutedblob = blobanalysis.characterize_blob(dilutedblob)
    metastasis = p_leo.characterize_metastasis(dilutedblob,volume,min_padding,otherblobs=metastases)
    metastasis['characterization']['NofSTDs'] = NofSTDs
    save_updated_metastasis()
    inspect_update_plot()
    main_update_plot()
    

def inspect_flag_change(label):
    global inspect_last_action, metastasis
    if(type(OoI) == tuple or inspect_w_flag_FROZEN == True):
        return # Don't do anything if widget is being updated or OoI is not a metastasis
    if(label=='Dim (<200)'):
         metastasis['evaluation']['flag_dim'] = not  metastasis['evaluation']['flag_dim']
         inspect_last_action = 'Updated flag dim = ' +  str( metastasis['evaluation']['flag_dim'])
    if(label=='Interesting'):
         metastasis['evaluation']['flag_interesting'] = not  metastasis['evaluation']['flag_interesting']
         inspect_last_action = 'Updated flag interesting = ' +  str( metastasis['evaluation']['flag_interesting'])
    if(label=='Problematic'):
        metastasis['evaluation']['flag_problematic'] = not metastasis['evaluation']['flag_problematic']
        inspect_last_action = 'Updated flag problematic = ' +  str(metastasis['evaluation']['flag_problematic'])
    save_updated_metastasis()
    inspect_update_title()


def inspect_addremove(event):
    # add or remove current OoI, depending on status
    global inspect_addremove_STATUS
    if(inspect_addremove_STATUS=='add' and type(OoI)==tuple):
        add_metastasis()
        inspect_addremove_STATUS = 'remove'
        inspect_addremove_updaterender()
    elif(inspect_addremove_STATUS=='remove' and metastasis is not None):
        remove_metastasis()
        inspect_addremove_STATUS = 'add'
        inspect_addremove_updaterender()
    else:
        return
    

# Helper functions for Inspect window

def inspect_flag_updaterender():
    # Update the shown flags to match saved data
    global inspect_w_flag, inspect_w_flag_FROZEN
    inspect_w_flag_FROZEN = True # If the widget is updated for the next OoI, don't let this be mistaken for a click
    status = inspect_w_flag.get_status()
    if(metastasis is not None):
        if(status[0] is not metastasis['evaluation']['flag_dim']):
            inspect_w_flag.set_active(0)
        if(status[1] is not metastasis['evaluation']['flag_interesting']):
            inspect_w_flag.set_active(1)
        if(status[2] is not metastasis['evaluation']['flag_problematic']):
            inspect_w_flag.set_active(2)
    else:
        if(status[0] is True):
            inspect_w_flag.set_active(0)
        if(status[1] is True):
            inspect_w_flag.set_active(1)
        if(status[2] is True):
            inspect_w_flag.set_active(2)
    inspect_w_flag_FROZEN = False


def inspect_addremove_updaterender():
    # Update the shown button to match status
    global inspect_w_addremove
    if(inspect_addremove_STATUS is None):
        inspect_w_addremove.label.set_text('n/a' )
    elif(inspect_addremove_STATUS=='add'):
        inspect_w_addremove.label.set_text('Add it!')
    elif(inspect_addremove_STATUS=='remove'):
        inspect_w_addremove.label.set_text('Remove it!')


def inspect_masksize_updaterender():
    # Update the selected mask size to match metastasis
    global inspect_w_masksize, inspect_w_masksize_FROZEN
    inspect_w_masksize_FROZEN = True # If the widget is updated for the next OoI, don't let this be mistaken for a click
    if(type(OoI) == tuple):
        inspect_w_masksize.set_active(1)
        inspect_w_masksize_FROZEN = False
        return
    if(metastasis['characterization']['NofSTDs'] == segmentation['autosegmentation']['NofSTDs']):
        inspect_w_masksize.set_active(1)
    elif(metastasis['characterization']['NofSTDs'] > segmentation['autosegmentation']['NofSTDs']):
        inspect_w_masksize.set_active(0)
    elif(metastasis['characterization']['NofSTDs'] < segmentation['autosegmentation']['NofSTDs']):
        inspect_w_masksize.set_active(2)
    inspect_w_masksize_FROZEN = False


def inspect_maskvis_updaterender():
    # Update the selected mask visualisation to match status
    global inspect_w_maskvis, inspect_w_maskvis_FROZEN
    inspect_w_maskvis_FROZEN = True # If the widget is updated, don't let this be mistaken for a click
    if(inspect_maskvis==True):
        inspect_w_maskvis.set_active(1)
    else:
        inspect_w_maskvis.set_active(0)
    inspect_w_maskvis_FROZEN = False


def inspect_rotate_3dplot_to(view):
    inspect_fig = plt.figure(num=102)
    inspect_ax_plt_list = inspect_fig.gca(projection='3d')
    if(hasattr(inspect_ax_plt_list, 'get_zlim')):
        if(view=='perspective'):
            inspect_ax_plt_list.view_init(25.01, -55.01)
            plt.draw()
        if(view=='YX'):
            inspect_ax_plt_list.view_init(270.01, 90.01)
            plt.draw()
        if(view=='ZY'):
            inspect_ax_plt_list.view_init(0, 0)
            plt.draw()
        if(type(view)==list):
            inspect_ax_plt_list.view_init(view[0], view[1])
            plt.draw()
    plt.draw()


#%% General functions

def load_patch(patch_id):
    global region, patch, segmentation, volume, volume_vis, max_mID, metastases, DIMmetastases, removed_metastases, main_n_added, volume_autofluo
    patch = next((p for p in region['patches'] if p["id"] == patch_id))
    segmentation    = p_leo.load_segmentation(dataset,patch_id,load_partial=False)
    volume_autofluo = filehandling.readNifti(path_autofluo + 'patchvolume_' + str(patch_id))
    volume          = filehandling.readNifti(path_patches  + 'patchvolume_' + str(patch_id))
    volume_vis      = np.clip(volume, 0, signalcap) / signalcap # Cap & scaling for visualization
    
    if(patch_id == 9999):
        volume = volume[:,:,:,0]
        volume_vis = volume_vis[:,:,:,0]
        patch = {}
        patch['id'] = patch_id
        segmentation['patch_id'] = patch_id
        region = {}
        region['partitioning'] = {}
        region['partitioning']['patch_overlap'] = 50
    
    max_mID = segmentation['max_mID']
    metastases = segmentation['metastases']
    DIMmetastases = segmentation['dim_metastases'] if ('dim_metastases' in segmentation.keys()) else []
    removed_metastases = segmentation['removed_metastases']
    main_n_added = len(list((m for m in metastases if m['evaluation']['manually_added'] == True)))


def update_sigcap(sliderval):
    global volume_vis, signalcap, signalcap_FROZEN
    if(signalcap_FROZEN==False):
        signalcap_FROZEN = True
        signalcap = 100*np.round(sliderval/100,0)
        main_w_sigcap.poly.xy[2][0] = main_w_sigcap.poly.xy[3][0] = signalcap
        main_w_sigcap.valtext.set_text(str(int(signalcap)))
        try:
            volume_vis = np.clip(volume, 0, signalcap) / signalcap
        except:
            pass
        inspect_update_plot()
        main_update_plot()
        overview_update_plot()
        signalcap_FROZEN = False

def save_updated_metastasis():
    if(type(OoI) is None):
        return
    elif(type(OoI)==int):
        global metastasis
        indexinlist = next((index for (index, m) in enumerate(metastases) if m["id"] == OoI), None)
        metastases[indexinlist] = metastasis
        #print('Saved changes of metastasis #' + str(OoI))


def add_metastasis():
    global OoI, DIMmetastasis, metastasis, max_mID, metastases, DIMmetastases, inspect_last_action, main_last_action, main_n_added
    inspect_last_action = '......... PLEASE WAIT .........'
    inspect_update_title()
    main_last_action = '......... PLEASE WAIT .........'
    main_update_title()
    # Try to automatically segment metastasis
    seed = {}
    seed['points'] = [np.asarray(OoI).T.tolist()]
    seedblob = blobanalysis.characterize_blob(seed)
    min_padding = segmentation['autosegmentation']['min_padding']
    NofSTDs = segmentation['autosegmentation']['NofSTDs']
    dilutedblob = p_leo.dilute_segmentation(seedblob,volume,min_padding,NofSTDs)
    dilutedblob = blobanalysis.characterize_blob(dilutedblob)
    # Check for validity and add
    greedy_zone = patch['boundingbox'] - region['partitioning']['patch_overlap']
    m_upperleft = dilutedblob['offset']
    in_greedy_zone = (all(np.less(m_upperleft,greedy_zone)))
    cutoff = (any(np.equal(m_upperleft,0)))
    if(dilutedblob['volume'] < 1):
        print('No mestastasis was added, because no voxel in local RoI is above the threshold.')
        inspect_last_action = 'Failed to find metastasis around ' + pointstring(OoI)
        main_last_action = 'Failed to find metastasis around ' + pointstring(OoI)
        main_update_title()
        inspect_update_title()
    elif(not in_greedy_zone or cutoff):
        print('No mestastasis was added, because it belongs to an adjacent patch.')
        inspect_last_action = 'Metastasis belongs to an adjacent patch'
        main_last_action = 'Metastasis belongs to an adjacent patch'
        main_update_title()
        inspect_update_title()
    else:
        if(max_mID is None): max_mID = 0
        metastasis = p_leo.characterize_metastasis(dilutedblob,volume,min_padding,otherblobs=metastases)
        metastasis['characterization']['NofSTDs'] = NofSTDs
        metastasis['id'] = max_mID + 1
        metastasis['evaluation']['last_inspection'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        metastasis['evaluation']['manually_added'] = True
        if(metastasis['characterization']['maxFG'] < 200):
            metastasis['evaluation']['flag_dim'] = True
            DIMmetastases.append(metastasis)
            main_last_action = 'Metastasis was too dim (<200) to be added (saved, but will not be shown)'
            inspect_last_action = 'Metastasis was too dim (<200) to be added (saved, but will not be shown)'
        else:
            metastasis['evaluation']['flag_dim'] = False
            metastases.append(metastasis)
            main_last_action = 'Added new metastasis with ID ' + str(metastasis['id'])
            inspect_last_action = 'Added new metastasis with ID ' + str(metastasis['id'])
            OoI = metastasis['id']
        max_mID += 1
        main_n_added += 1
        main_update_plot()
        inspect_update_plot()


def remove_metastasis():
    global OoI, metastases, metastasis, max_mID, main_n_added, inspect_last_action, main_last_action
    metastases[:] = [m for m in metastases if m.get('id') != OoI] # remove from list of metastases
    if(metastasis['evaluation']['manually_added']==True):
        # We  added this metastasis, but we don't want to keep it --> revert adding
        main_n_added -= 1
        if(OoI==max_mID):
            max_mID -= 1 # only reduce max_mID if there is no metastasis with a higher ID
    else:
        # We are removing an automatically created segmentation
        metastasis['evaluation']['last_inspection'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        metastasis['evaluation']['manually_removed'] = True
        removed_metastases.append(metastasis) # this will be saved to segmentation upon save&close
    main_last_action = 'Removed metastasis #' + str(metastasis['id'])
    inspect_last_action = 'Removed metastasis #' + str(metastasis['id'])
    OoI = tuple((metastasis['offset'] + metastasis['CoM']))
    metastasis = None
    main_update_plot()
    inspect_update_plot()


def remove_all_metastases(event):
    global OoI, metastases, metastasis, inspect_last_action, main_last_action
    for m in metastases:
        m['evaluation']['last_inspection'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        m['evaluation']['manually_removed'] = False
        removed_metastases.append(metastasis) # this will be saved to segmentation upon save&close
    metastases = [] # remove from list of metastases
    main_last_action = 'Removed all metastases'
    inspect_last_action = ''
    OoI = None
    metastasis = None
    main_update_plot()
    inspect_update_plot()


def pointstring(point):
    return 'x='+str(point[1])+', y='+str(point[0])+', z='+str(point[2])


#%% Code for manual correction of DIM metastases
#
#def save_updated_DIMmetastasis(DIMmetastasis):
#    global DIMmetastases
#    indexinlist = next((index for (index, m) in enumerate(DIMmetastases) if m["id"] == DIMmetastasis['id']), None)
#    DIMmetastases[indexinlist] = copy.deepcopy(DIMmetastasis)
#    print('Saved changes of DIM metastasis #' + str(DIMmetastasis['id']))
#
#def delete_DIMmetastasis(ID):
#    global DIMmetastases
#    indexinlist = next((index for (index, m) in enumerate(DIMmetastases) if m["id"] == ID), None)
#    del DIMmetastases[indexinlist]
#    print('Deleted DIM metastasis #' + str(ID))
#
#
## code for changing size of DIM metastases -> execute part by part manually before running "save_updated_DIMmetastasis(myDim_d)"
#myDim = copy.deepcopy(dataconversions.filter_dicts(DIMmetastases,'id',34)[0])
#pd(myDim)
#
#NofSTDs = 1
#min_padding = segmentation['autosegmentation']['min_padding']
#myDim_d = p_leo.dilute_segmentation(myDim,volume,min_padding,NofSTDs)
#myDim_d = blobanalysis.characterize_blob(myDim_d)
#myDim_d = p_leo.characterize_metastasis(myDim_d,volume,min_padding,otherblobs=metastases)
#myDim_d['characterization']['NofSTDs'] = NofSTDs
#pd(myDim_d)
#
#save_updated_DIMmetastasis(myDim_d)

      
#%% Launch GUI

# Overview window: Create ---------------------------------------------------------------------
plt.figure(overview_fig.number)
overview_fig.suptitle('Annotator: '+NameOfAnnotator+'\nGlobal overview for ' + dataset, fontsize=fs[3], fontweight='bold')
overview_ax1 = overview_fig.add_subplot(1,6,1)
overview_ax2 = overview_fig.add_subplot(1,6,2)
overview_ax3 = overview_fig.add_subplot(1,6,3)
overview_ax4 = overview_fig.add_subplot(4,2,4)
overview_ax5 = overview_fig.add_subplot(2,2,4)
overview_next_patch(None)
overview_update_plot()
plt.get_current_fig_manager().window.showMaximized()
plt.draw()
plt.pause(1)
plt.draw()
screensize = overview_fig.get_size_inches()*overview_fig.dpi

# Overview window: Prev patch
overview_wp_prev = [0.55,0.81,0.08,0.05]
overview_wa_prev = plt.axes(overview_wp_prev, facecolor='lightgoldenrodyellow')
overview_w_prev = Button(overview_wa_prev, 'Return to \nprevious patch')
overview_w_prev.on_clicked(overview_prev_patch)

# Overview window: Next patch
overview_wp_next = [0.65,0.81,0.08,0.05]
overview_wa_next = plt.axes(overview_wp_next, facecolor='lightgoldenrodyellow')
overview_w_next = Button(overview_wa_next, 'Select next \nrandom patch')
overview_w_next.on_clicked(overview_next_patch)

# Overview window: Open patch
overview_wp_open = [0.80,0.80,0.10,0.07]
overview_wa_open = plt.axes(overview_wp_open, facecolor='lightgoldenrodyellow')
overview_w_open = Button(overview_wa_open, 'Open current patch')
overview_w_open.on_clicked(overview_open_patch)


# Main window: Create -------------------------------------------------------------------------
plt.figure(main_fig.number)
plt.get_current_fig_manager().window.setGeometry(int(screensize[0]*1.045),int(screensize[0]*0.02),int(screensize[0]*0.49), screensize[1])
main_fig.canvas.mpl_connect('button_press_event', main_plotclick)

# Main window: View
main_wp_view = [0.05, 0.85, 0.15, 0.08]
main_wa_view = plt.axes(main_wp_view, facecolor='lightgoldenrodyellow')
main_w_view = RadioButtons(main_wa_view, ('Z-projection', 'Y-projection', 'X-projection'))
for label in main_w_view.labels: label.set_fontsize(fs[0])
main_w_view.on_clicked(main_view_change)

# Main window: Mode
main_wp_mode = [main_wp_view[0]+main_wp_view[2]+0.05, main_wp_view[1], main_wp_view[2], main_wp_view[3]]
main_wa_mode = plt.axes(main_wp_mode, facecolor='lightgoldenrodyellow')
main_w_mode = RadioButtons(main_wa_mode, ('Inspect', 'DirectAdd', 'DirectRemoval'))
for label in main_w_mode.labels: label.set_fontsize(fs[0])
main_w_mode.on_clicked(main_mode_change)

# Main window: Set flags
main_wp_flags = [main_wp_mode[0]+main_wp_mode[2]+0.05, main_wp_mode[1], main_wp_mode[2], main_wp_mode[3]]
main_wa_flags = plt.axes(main_wp_flags, facecolor='lightgoldenrodyellow')
main_w_flag = CheckButtons(main_wa_flags, ['Completed', 'Interesting', 'Problematic'], [False,False,False])
for label in main_w_flag.labels: label.set_fontsize(fs[0])
main_w_flag.on_clicked(main_flag_change)

# Main window: Signal Cap slider
main_wp_sigcap = [main_wp_flags[0]+main_wp_flags[2]+0.05, main_wp_flags[1]+0.05, 0.13, 0.015]
main_wa_sigcap = plt.axes(main_wp_sigcap)
main_w_sigcap = Slider(main_wa_sigcap, 'Signal intensity cap', 0,3000, valinit=signalcap, dragging=False, valfmt='%.0f')
main_w_sigcap.label.set_position((1.1,1.5))
main_w_sigcap.on_changed(update_sigcap)

# Main window: Delete all
main_wp_delall = [main_wp_flags[0]+main_wp_flags[2]+0.05, main_wp_flags[1], 0.16, 0.035]
main_wa_delall = plt.axes(main_wp_delall, facecolor='lightgoldenrodyellow')
main_w_delall = Button(main_wa_delall, 'Remove all metastases')
main_w_delall.on_clicked(remove_all_metastases)

# Main window: Save&Close
main_wp_snc = [main_wp_delall[0]+main_wp_delall[2]+0.05, main_wp_delall[1], 0.12, 0.08]
main_wa_snc = plt.axes(main_wp_snc, facecolor='lightgoldenrodyellow')
main_w_snc = Button(main_wa_snc, 'Save & Close')
main_w_snc.on_clicked(main_saveandclose)


# Inspection window: Create ---------------------------------------------------------------------------------
plt.figure(num=inspect_fig.number)
plt.get_current_fig_manager().window.setGeometry(int(screensize[0]*1.54),int(screensize[0]*0.02),int(screensize[0]*0.49), screensize[1])

# Inspection window: view
inspect_wp_view = [0.03, 0.85, 0.2, 0.12]
inspect_wa_view = plt.axes(inspect_wp_view, facecolor='lightgoldenrodyellow')
inspect_w_view = RadioButtons(inspect_wa_view, ('Max. Projections', 'Cum. Projections', 'Slices through center', '3D view'))
for label in inspect_w_view.labels: label.set_fontsize(fs[0])
inspect_w_view.on_clicked(inspect_view_change)

# Inspection window: mask size
inspect_wp_masksize = [inspect_wp_view[0]+inspect_wp_view[2]+0.03, inspect_wp_view[1], 0.12, 0.09]
inspect_wa_masksize = plt.axes(inspect_wp_masksize, facecolor='lightgoldenrodyellow')
inspect_w_masksize = RadioButtons(inspect_wa_masksize, ('Shrink outline', 'Standard','Expand outline'),1)
for label in inspect_w_masksize.labels: label.set_fontsize(fs[0])
inspect_w_masksize.on_clicked(inspect_masksize_change)

# Inspection window: mask visualization
inspect_wp_maskvis = [inspect_wp_masksize[0]+inspect_wp_masksize[2]+0.03, inspect_wp_masksize[1], 0.14, 0.06]
inspect_wa_maskvis = plt.axes(inspect_wp_maskvis, facecolor='lightgoldenrodyellow')
inspect_w_maskvis = RadioButtons(inspect_wa_maskvis, ('Standard view', 'Highlight outline'))
for label in inspect_w_maskvis.labels: label.set_fontsize(fs[0])
inspect_w_maskvis.on_clicked(inspect_maskvis_change)

# Inspection window: Set flags
inspect_wp_flag = [inspect_wp_maskvis[0]+inspect_wp_maskvis[2]+0.03, inspect_wp_maskvis[1], 0.10, 0.06]
inspect_wa_flag = plt.axes(inspect_wp_flag, facecolor='lightgoldenrodyellow')
inspect_w_flag = CheckButtons(inspect_wa_flag, ['Dim (<200)', 'Interesting', 'Problematic'], [False, False, False])
for label in inspect_w_flag.labels: label.set_fontsize(fs[0])
inspect_w_flag.on_clicked(inspect_flag_change)

# Inspection window: Add/remove metastasis
inspect_wp_addremove = [inspect_wp_flag[0]+inspect_wp_flag[2]+0.03, inspect_wp_flag[1], 0.12, 0.06]
inspect_wa_addremove = plt.axes(inspect_wp_addremove, facecolor='lightgoldenrodyellow')
inspect_w_addremove = Button(inspect_wa_addremove, 'n/a')
inspect_w_addremove.on_clicked(inspect_addremove)



print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n The tool has launched. Please bring the newly opend windows to the front if they have not popped up automatically.\n\n\n\n\n\n\n\n\n')

