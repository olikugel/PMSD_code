import sys

CODEPATH = '/home/okugel/PMSD/PMSD_code'
DATAPATH = '/home/okugel/PMSD/PMSD_data'

sys.path.insert(0, CODEPATH + '/helperfunctions')

import filehandling
import dataconversions

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from build_dataset import MetDataset
from train_neural_net import NeuralNet




def pick_sample():

    samplecards = filehandling.pload(DATAPATH + '/mice_metadata/' + 'list_of_samplecards.pickledump')

    mice = ['H2030IC10dn573','IC2dn2','IC6dn1','IC6dn2', 'IC14dn1', 'MCF7IC21dn528', 'PC14dn04595']
    print()
    print("Please choose a mouse:")
    for m in enumerate(mice):
        print('[', m[0], ']', ' ', m[1])
    chosen_index = input('--> ')
    mouse = mice[int(chosen_index)]

    print()
    print("What's the patch ID?")
    patch_ID = input("--> ")
    patch_ID = int(patch_ID)

    print()
    print("What's the metastasis ID?")
    met_ID = input("--> ")
    met_ID = int(met_ID)

    samplecards_mouse = dataconversions.filter_dicts(samplecards, 'mouse', mouse)
    samplecards_patch = dataconversions.filter_dicts(samplecards_mouse, 'patch_id', patch_ID)
    samplecards_met   = dataconversions.filter_dicts(samplecards_patch, 'met_id', met_ID)
    chosen_samplecard = samplecards_met

    if chosen_samplecard:
        print()
        print('Chosen sample:')
        print(chosen_samplecard)
        print()

        sample_set = MetDataset(chosen_samplecard)

        sample_loader = DataLoader(sample_set)

        for i, (images, label) in enumerate(sample_loader):
            return images

    else:
        return None



def classify():

    print()
    print("Which model? Please provide just the ID")
    model_ID = input("--> ")

    model_filename = 'model_' + model_ID + '.model'
    print()
    print('Using ' + model_filename + ' for prediction')

    MODEL = NeuralNet(num_classes=1)
    MODEL.load_state_dict(torch.load('./' + model_filename))
    MODEL.eval()

    images = pick_sample()

    if images is None:
        print()
        print('Sorry, the chosen sample is not eligible.')
        sys.exit()

    prediction = MODEL(images)
    prediction = prediction.item()
    prediction = round(prediction,3)
    print('Prediction: ' + str(prediction))
    print()

    if prediction >= 0.5:
        classification = 1 # classified as metastasis
        certainty = prediction
    else:
        classification = 0 # classified as not-a-mestastasis
        certainty = 1 - prediction

    certainty = round((certainty * 100),3)

    return classification, certainty





classification, certainty = classify()

print('Classification:', classification)

if classification == 1:
   print('Predicted label: true positive')
   print("With a certainty of " + str(certainty) + "%, this sample was classified as 'real metastasis'.")
else: # classification == 0
   print('Predicted label: false positive')
   print("With a certainty of " + str(certainty) + "%, this sample was classified as 'not a real metastasis'.")
