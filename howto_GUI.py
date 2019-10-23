import sys
ROOTP = sys.path[-1].replace('.ipython','').replace('\\','/')
sys.path.insert(0, ROOTP + 'Documents/Synced/_Promotion/scripts/helperfunctions')

import plotting
from plotting import print_dict as pd
import filehandling
import dataconversions
import blobanalysis
import p_leo

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider
import matplotlib.patches as patches
import time

BASEP = ROOTP + 'Documents/Synced/_Promotion/Projects/Leo/'

#%% 
#
# This script iterates through all predicted blobs for manual review. 
# Please refer to the PDF manual for detailed instructions. In short,
# thsi script let's you quickly click through all detected blobs and
# decide whether they are real metastases or false positives.
#
# === Short instructions ============================================
#
# Step 1: Insert name of mouse in line 47
# Step 2: Insert your name in line 48
# Step 3: Press F5 and wait until loaded
#
# Important: Click on the "Save changes" button before closing
#
# ===================================================================
# 
# This script is based on "Refinement_2_ReviewFalsePositives".
# Run this script after postprocessing of prediction.


#%% Main parameters 
# Already finished..: F15, F14, H2030IC10dn573, IC14dn1, IC6dn2, IC2dn2, PC14dn04595, MCF7IC21dn528, Con1, Con2, IC6dn1, 
# Still to do.......: 
# Not needed for now: MCF7IC10dn529, H2030IC31dn520, IC16dn1
# Not yet available.: Con3, Con4, Con5
dataset           = 'Name of mouse'
NAME_OF_ANNOTATOR = 'Oliver Kugel'

# accumulation: "Add & next" but mark as "problematic"
# dim metastasis: "Add & next" but mark as "dim"
# food debris/pgments at glands/etc: "Remove & next"


#%% Set up

# Prepare data & constants
list_of_candidates = [1,2,3,4,5,6,7]

# Global dynamic variables
main_fig = plt.figure(num=101)
main_ax_plt = main_fig.gca()
candidate_ID = None


#%% functions
def update_plot():
    global main_fig

    plt.figure(num=main_fig.number)
    
    # Load all 3x3 projections for given candidate ID
    p_C00_y = np.random.rand(50,50) #filehandling.pload(PATH + 'canidate_' + str(candidate_ID) + '_C00_z')
    p_C00_x = np.zeros((50,50))
    p_C00_z = np.zeros((50,50))
    
    p_C01_y = np.zeros((50,50))
    p_C01_x = np.zeros((50,50))
    p_C01_z = np.zeros((50,50))
    
    p_C02_y = np.zeros((50,50))
    p_C02_x = np.zeros((50,50))
    p_C02_z = np.zeros((50,50))
    
    # Update subplots
    ax = plt.subplot(3,3,1)
    ax.cla()
    plt.imshow(p_C00_y)
    
    # Update rest
    plt.suptitle(dataset + '\nCanidate 13 of 128')
    plt.draw()
    plt.subplots_adjust(top   = 0.85)
    plt.subplots_adjust(right = 0.98)
    plt.subplots_adjust(left  = 0)
    

def next_candidate(not_needed=False):
    # define optional argument so that function can be called from button
    global list_of_candidates, candidate_ID, candidate_dict
    try:
        candidate_ID = list_of_candidates.pop(0)
        # load candidate_dict
        candidate_dict = {}
        candidate_dict['evaluation'] = {}
        candidate_dict['evaluation']['XYAZ'] = None
        update_plot()
    except:
        print("\n\nNo more candidates to review.")


def mark_as_FP(event):
    global candidate_dict
    candidate_dict['evaluation']['XYAZ'] = False
    save_candidate()
    print("Candidate was marked as FP")
    next_candidate()


def mark_as_TP(event):
    global candidate_dict
    candidate_dict['evaluation']['XYAZ'] = True
    save_candidate()
    print("Candidate was marked as TP")
    next_candidate()


def mark_as_unclear(event):
    global candidate_dict
    candidate_dict['evaluation']['XYAZ'] = None
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
plt.draw()
plt.pause(1)
plt.draw()
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

#%% Main

next_candidate()


