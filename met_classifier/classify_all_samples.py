import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) 
import basepaths
CODEPATH, DATAPATH = basepaths.get_basepaths()

sys.path.insert(0, CODEPATH + '/helperfunctions')
import filehandling

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from build_dataset import MetDataset
from train_neural_net import NeuralNet

from classify_sample import classify

# -----------------------------------------------------------------------------------


def get_model():

    model_ID = 22

    model_filename = 'model_' + str(model_ID) + '.model'
    #print()
    #print('Using ' + model_filename + ' for prediction')
    MODEL = NeuralNet(num_classes=1)
    
    if torch.cuda.is_available():
        MODEL.load_state_dict(torch.load('./' + model_filename))
    else:
        MODEL.load_state_dict(torch.load('./' + model_filename, map_location=torch.device('cpu')))

    return MODEL




if __name__ == "__main__":

    print('Mouse , Patch ID , Met ID , True Label , Predicted Label , Certainty')

    samplecards = filehandling.pload(DATAPATH + '/mice_metadata/' + 'list_of_samplecards.pickledump')

    for samplecard in samplecards:

        sample_set = MetDataset([samplecard])

        sample_loader = DataLoader(sample_set)

        for i, (images, label) in enumerate(sample_loader):
            MODEL = get_model()
            classification, certainty = classify(images, MODEL)
            if classification == 1: 
                predicted_label = 'true positive'
            else: # classification == 0: 
                predicted_label = 'false positive'        

            print(samplecard['mouse'], ',', samplecard['patch_id'], ',', samplecard['met_id'], ',', samplecard['label'], ',', predicted_label, ',', certainty)
            
            # print falsely predicted samples
            #true_label = samplecard['label']
            #if predicted_label != true_label:
            #    print(samplecard['mouse'], ',', samplecard['patch_id'], ',', samplecard['met_id'], ',', samplecard['label'], ',', predicted_label, ',', certainty)



