CODEPATH = '/home/olikugel/PMSD_code'
DATAPATH = '/home/olikugel/PMSD_data'
import sys
sys.path.insert(0, CODEPATH + '/helperfunctions')

import filehandling
import dataconversions

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

import datetime
import traceback
import warnings
import copy
from functools import wraps
def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner


#%%

NAME_OF_ANNOTATOR = 'Oliver Kugel'
d = datetime.datetime.today()
DATE_OF_TODAY = d.strftime('%d-%m-%Y')
print()
print("Annotator: ", NAME_OF_ANNOTATOR)
print ("Today's date: ", DATE_OF_TODAY)
print()

mice = ['H2030IC10dn573','IC2dn2','IC6dn1','IC6dn2', 'IC14dn1', 'MCF7IC21dn528']
mouse = mice[0]

#%% Set up

# Global dynamic variables
main_fig = plt.figure(num=101)

prediction = filehandling.pload(DATAPATH + '/mice_metadata/' + mouse + '/reviewed_prediction.pickledump')
metastases = prediction['metastases']
TP_candidates = dataconversions.filter_dicts(metastases,'evaluation-manually_confirmed',True)
number_of_candidates = len(TP_candidates)
candidate_dict = {}

candidate_IDs = []
for TP_candidate in TP_candidates:
    candidate_IDs.append(TP_candidate['global_id'])
   
all_candidate_IDs = copy.deepcopy(candidate_IDs)
print('All candidate IDs: ', all_candidate_IDs)

region = filehandling.pload(DATAPATH + '/mice_metadata/' + mouse + '/region.pickledump')
whole_mouse_thumbnails = region['thumbnails']['MaxProjections_Z']

# for debugging
#for x in range(230):
#    candidate_ID = candidate_IDs.pop(0)
    
#%% functions
   
    
def get_current_metastasis(candidate_ID):
    current_metastasis = dataconversions.filter_dicts(TP_candidates,'global_id',candidate_ID)[0]
    return current_metastasis


def get_whole_mouse_thumbnail(candidate_ID):
    current_metastasis = get_current_metastasis(candidate_ID)
    patch_ID = current_metastasis['patch_id']
    patches = region['patches']
    patch = dataconversions.filter_dicts(patches, 'id', patch_ID)[0]
    patchstep = patch['patchstep']
    z_patchstep = patchstep[2]
    whole_mouse_thumbnail = whole_mouse_thumbnails[:, :, z_patchstep]
    return whole_mouse_thumbnail
    
'''
def get_patch_projection(candidate_ID):
    current_metastasis = get_current_metastasis(candidate_ID)
    patch_ID = current_metastasis['patch_id']
'''    
    

def get_filename(candidate_ID, axis):
    current_metastasis = get_current_metastasis(candidate_ID)
    patch_ID = current_metastasis['patch_id']
    met_ID = current_metastasis['id']
    patch_ID_padded = filehandling.pad_ID(patch_ID)
    met_ID_padded = filehandling.pad_ID(met_ID)
    
    filename_prefix = 'patch' + patch_ID_padded + '_met' + met_ID_padded
    filename = filename_prefix + '_' + axis + '.png'
    return filename
    


def save_decision(decision):
    current_metastasis = get_current_metastasis(candidate_ID)
    current_metastasis['evaluation']['reviewed_via_GUI'] = decision
    current_metastasis['evaluation']['annotator_via_GUI'] = NAME_OF_ANNOTATOR
    current_metastasis['evaluation']['date_of_review_via_GUI'] = DATE_OF_TODAY



def check_if_file_exists(file):
    try:
        image = mpimg.imread(file)
    except Exception as e:
        print('Image files for this metastasis are missing')
        save_decision('could not load image files')
        next_candidate()



@ignore_warnings
def update_plot():
    global main_fig
    plt.figure(num=main_fig.number)
       
    # channel C00
    filepath = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C00/ROI_50/PNG/'
    check_if_file_exists(filepath + get_filename(candidate_ID, 'y'))
    if len(candidate_IDs) == 0: return
    image_C00_x = mpimg.imread(filepath + get_filename(candidate_ID, 'x'))
    image_C00_z = mpimg.imread(filepath + get_filename(candidate_ID, 'z'))

    # channel C01
    filepath = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C01/ROI_50/PNG/'
    check_if_file_exists(filepath + get_filename(candidate_ID, 'y'))
    image_C01_x = mpimg.imread(filepath + get_filename(candidate_ID, 'x'))
    image_C01_z = mpimg.imread(filepath + get_filename(candidate_ID, 'z'))
    
    # channel C02
    filepath = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C02/ROI_50/PNG/'
    check_if_file_exists(filepath + get_filename(candidate_ID, 'y'))
    image_C02_x = mpimg.imread(filepath + get_filename(candidate_ID, 'x'))
    image_C02_z = mpimg.imread(filepath + get_filename(candidate_ID, 'z'))
    

    # whole mouse thumbnail
    ax1 = main_fig.add_subplot(1, 4, 1)
    plt.subplots_adjust(top=0.76)
    whole_mouse_thumbnail = get_whole_mouse_thumbnail(candidate_ID)
    plt.imshow(whole_mouse_thumbnail, vmin=0, vmax=3000)
    
    ax2 = main_fig.add_subplot(1, 4, 2)
    #patch_projection = get_patch_projection(candidate_ID)
    #plt.imshow(patch_projection)
    #plt.imshow(whole_mouse_thumbnail, vmin=0, vmax=3000)
    
    # channel C00 -- z
    ax3 = main_fig.add_subplot(3, 4, 3)
    ax3.set_title('Projection along z-axis', fontsize='x-large', y=1.2)
    ax3.set_ylabel('C00 channel', rotation=0, fontsize='x-large')
    ax3.yaxis.set_label_coords(-0.5, 0.5)
    plt.imshow(image_C00_z) 
    
    # channel C00 -- x
    ax4 = main_fig.add_subplot(3, 4, 4)
    ax4.set_title('Projection along x-axis', fontsize='x-large', y=1.2)
    #ax.set_ylabel('C00 channel', rotation=0, fontsize='x-large')
    ax4.yaxis.set_label_coords(-0.5, 0.5)
    plt.imshow(image_C00_x)
    
    # channel C01 -- z
    ax5 = main_fig.add_subplot(3, 4, 7)
    ax5.set_ylabel('C01 channel', rotation=0, fontsize='x-large')
    ax5.yaxis.set_label_coords(-0.5, 0.5)
    plt.imshow(image_C01_z) 
    
    # channel C01 -- x
    ax6 = main_fig.add_subplot(3, 4, 8)
    #ax.set_ylabel('C01 channel', rotation=0, fontsize='x-large')
    ax6.yaxis.set_label_coords(-0.5, 0.5)
    plt.imshow(image_C01_x)
    
    # channel C02 -- z
    ax7 = main_fig.add_subplot(3, 4, 11)
    ax7.set_ylabel('C02 channel', rotation=0, fontsize='x-large')
    ax7.yaxis.set_label_coords(-0.5, 0.5)
    plt.imshow(image_C02_z)
    
    # channel C02 -- x
    ax8 = main_fig.add_subplot(3, 4, 12)
    #ax.set_ylabel('C02 channel', rotation=0, fontsize='x-large')
    ax8.yaxis.set_label_coords(-0.5, 0.5)
    plt.imshow(image_C02_x)
    
    # Update rest
    displayed_ID = all_candidate_IDs.index(candidate_ID) + 1
    current_mouse_title = 'Mouse ' + mouse + '\n\n' + 'Candidate ' + str(displayed_ID) + ' of ' + str(number_of_candidates) + '\n\n(Actual candidate ID: ' + str(candidate_ID) + ')'
    plt.suptitle(current_mouse_title,  fontsize='x-large', y=0.977)
    
    plt.show()
    
    
    
    
def next_candidate():
    global candidate_IDs, candidate_ID, candidate_dict
    
    if len(candidate_IDs) > 0: 
        candidate_ID = candidate_IDs.pop(0)
        print('\nCurrent candidate: ', candidate_ID)
        update_plot()
    else:
        print('\nNo more candidates to review')



def mark_as_TP(event):
    if len(candidate_IDs) > 0: 
        print('Candidate was marked as true positive')
        save_decision('true positive')
    next_candidate()



def mark_as_FP(event): 
    if len(candidate_IDs) > 0: 
        print('Candidate was marked as false positive')
        save_decision('false positive')
    next_candidate()



def mark_as_UC(event):
    if len(candidate_IDs) > 0: 
        print('Candidate was marked as unclear')
        save_decision('unclear')
    next_candidate()



def save_to_file(event):
    filename = 'reviewed_via_GUI_by_' + NAME_OF_ANNOTATOR.replace(' ', '_') + '_on_the_' + DATE_OF_TODAY + '.pickledump'
    filepath_with_filename = DATAPATH + '/mice_metadata/' + mouse + '/' + filename
    filehandling.psave(filepath_with_filename, TP_candidates)
    print()
    print('----------------------------------------------------')
    print('All decisions saved to file ', filepath_with_filename)
    print('----------------------------------------------------')

#%% Launch GUI

# window: Create
plt.figure(main_fig.number)

plt.get_current_fig_manager().window.showMaximized()
screensize = main_fig.get_size_inches()*main_fig.dpi
buttontext_size = 15

# button: mark as true positive
main_wp_TP = [0.03, 0.9, 0.10, 0.08]
main_wa_TP = plt.axes(main_wp_TP)
main_w_TP = Button(main_wa_TP, 'This is a\nreal metastasis')
main_w_TP.label.set_fontsize(buttontext_size)
main_w_TP.on_clicked(mark_as_TP) 

# button: mark as false positive
main_wp_FP = [0.03+0.115, 0.9, 0.10, 0.08]
main_wa_FP = plt.axes(main_wp_FP)
main_w_FP = Button(main_wa_FP, 'This is NOT a\nreal metastasis')
main_w_FP.label.set_fontsize(buttontext_size)
main_w_FP.on_clicked(mark_as_FP) 

# button: mark as unclear
main_wp_UC = [0.03+0.23, 0.9, 0.10, 0.08]
main_wa_UC = plt.axes(main_wp_UC)
main_w_UC = Button(main_wa_UC, "I'm not sure\nwhat this is")
main_w_UC.label.set_fontsize(buttontext_size)
main_w_UC.on_clicked(mark_as_UC) 

# button: save all
main_wp_SA = [0.03+0.65, 0.9, 0.10, 0.08]
main_wa_SA = plt.axes(main_wp_SA)
main_w_SA = Button(main_wa_SA, 'Save all\ndecisions')
main_w_SA.label.set_fontsize(buttontext_size)
main_w_SA.on_clicked(save_to_file) 

next_candidate()
plt.show(block=True)



# DECISIONS = filehandling.pload('/home/olikugel/PMSD_data/mice_metadata/H2030IC10dn573/reviewed_via_GUI_by_Oliver_Kugel_on_the_20-11-2019.pickledump')
