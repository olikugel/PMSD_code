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

import traceback
import warnings
from functools import wraps
def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

#%% Main parameters 

NAME_OF_ANNOTATOR = 'Oliver Kugel'

mice = ['H2030IC10dn573','IC2dn2','IC6dn1','IC6dn2', 'IC14dn1', 'MCF7IC21dn528']
mouse = mice[0]

#%% Set up

# Global dynamic variables
main_fig = plt.figure(num=101)
main_ax_plt = main_fig.gca()

prediction = filehandling.pload(DATAPATH + '/mice_metadata/' + mouse + '/reviewed_prediction.pickledump')
metastases = prediction['metastases']
TP_candidates = dataconversions.filter_dicts(metastases,'evaluation-manually_confirmed',True)
number_of_candidates = len(TP_candidates)

candidate_IDs = []
for TP_candidate in TP_candidates:
    candidate_IDs.append(TP_candidate['global_id'])
    
print('Candidate IDs: ', candidate_IDs)

#%% functions
    
def get_filename(candidate_ID, axis):
    
    current_metastasis = dataconversions.filter_dicts(TP_candidates,'global_id',candidate_ID)[0]
    patch_ID = current_metastasis['patch_id']
    met_ID = current_metastasis['id']
    patch_ID_padded = filehandling.pad_ID(patch_ID)
    met_ID_padded = filehandling.pad_ID(met_ID)
    
    filename_prefix = 'patch' + patch_ID_padded + '_met' + met_ID_padded
    filename = filename_prefix + '_' + axis + '.png'
    
    return filename
    

@ignore_warnings
def update_plot():
    global main_fig

    plt.figure(num=main_fig.number)

    # Load all 3x3 projection-images for given candidate ID
    
    # channel C00
    filepath = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C00/ROI_50/PNG/'
    image_C00_y = mpimg.imread(filepath + get_filename(candidate_ID, 'y'))
    image_C00_x = mpimg.imread(filepath + get_filename(candidate_ID, 'x'))
    image_C00_z = mpimg.imread(filepath + get_filename(candidate_ID, 'z'))

    # channel C01
    filepath = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C01/ROI_50/PNG/'
    image_C01_y = mpimg.imread(filepath + get_filename(candidate_ID, 'y'))
    image_C01_x = mpimg.imread(filepath + get_filename(candidate_ID, 'x'))
    image_C01_z = mpimg.imread(filepath + get_filename(candidate_ID, 'z'))
    
    # channel C02
    filepath = DATAPATH + '/Potential_TP_Metastases/' + mouse + '/C02/ROI_50/PNG/'
    image_C02_y = mpimg.imread(filepath + get_filename(candidate_ID, 'y'))
    image_C02_x = mpimg.imread(filepath + get_filename(candidate_ID, 'x'))
    image_C02_z = mpimg.imread(filepath + get_filename(candidate_ID, 'z'))
    
    # Update subplots
    
    # channel C00
    ax = plt.subplot(3,2,1)
    ax.cla()
    plt.imshow(image_C00_z) # z
    ax = plt.subplot(3,2,2)
    ax.cla()
    plt.imshow(image_C00_x) # x
    
    # channel C01
    ax = plt.subplot(3,2,3)
    ax.cla()
    plt.imshow(image_C01_z) # z
    ax = plt.subplot(3,2,4)
    ax.cla()
    plt.imshow(image_C01_x) # x
    
    # channel C02
    ax = plt.subplot(3,2,5)
    ax.cla()
    plt.imshow(image_C02_z) # z
    ax = plt.subplot(3,2,6)
    ax.cla()
    plt.imshow(image_C02_x) # x
    
    # Update rest
    plt.suptitle('Mouse ' + mouse + '\nCandidate ' + str(candidate_ID) + ' of ' + str(number_of_candidates))
    plt.draw()
    plt.subplots_adjust(top   = 0.85)
    plt.subplots_adjust(right = 0.98)
    plt.subplots_adjust(left  = 0)
    

def next_candidate(not_needed=False):
    # define optional argument so that function can be called from button
    global candidate_IDs, candidate_ID, candidate_dict
    try:
        candidate_ID = candidate_IDs.pop(0)
        print()
        print('Current candidate: ', candidate_ID)
        print(candidate_IDs)
        # load candidate_dict
        # candidate_dict = {}
        # candidate_dict['evaluation'] = {}
        # candidate_dict['evaluation']['XYAZ'] = None
        update_plot()
    except Exception as e:
        print(e)
        # traceback.print_exc()
        # print("\n\nNo more candidates to review.")


def mark_as_FP(event):
    global candidate_dict
    # candidate_dict['evaluation']['XYAZ'] = False
    save_candidate()
    print("Candidate was marked as FP")
    next_candidate()


def mark_as_TP(event):
    global candidate_dict
    # candidate_dict['evaluation']['XYAZ'] = True
    save_candidate()
    print("Candidate was marked as TP")
    next_candidate()


def mark_as_unclear(event):
    global candidate_dict
    # candidate_dict['evaluation']['XYAZ'] = None
    save_candidate()
    print("Candidate was marked as unclear")
    next_candidate()


def save_candidate():
    #filehandling.psave(BASEP + 'results/' + dataset + '/reviewed_prediction',prediction)
    print('We noted the change somewhere in some variable')


def save_to_file(event):
    #filehandling.psave(BASEP + 'results/' + dataset + '/reviewed_prediction',prediction)
    print(event)
    print('Saved to file')


#%% Launch GUI

# window: Create
plt.figure(main_fig.number)

plt.get_current_fig_manager().window.showMaximized()
screensize = main_fig.get_size_inches()*main_fig.dpi

# window: Save
main_wp_s = [0.03, 0.9, 0.10, 0.08]
main_wa_s = plt.axes(main_wp_s)
main_w_s = Button(main_wa_s, 'Save changes')
main_w_s.on_clicked(save_to_file) 

# window: mark as TP
main_wp_t = [0.03+0.1, 0.9, 0.10, 0.08]
main_wa_t = plt.axes(main_wp_t)
main_w_t = Button(main_wa_t, 'Mark as TP')
main_w_t.on_clicked(mark_as_TP) 

next_candidate()
plt.show(block=True)

