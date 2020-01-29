import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) 
import basepaths
CODEPATH, DATAPATH = basepaths.get_basepaths()

sys.path.insert(0, CODEPATH + '/helperfunctions')
import filehandling
from filehandling import pad_ID

import torch
from torch.utils.data import Dataset
import imageio

import numpy as np
from torchvision.transforms import transforms





class MetDataset(Dataset):
    def __init__(self, samplecards):
        self.samplecards = samplecards

    def __getitem__(self, index):
        samplecard = self.samplecards[index]

        images = torch.tensor(np.zeros((6,50,50),np.float32))
        for a, axis in enumerate(['x','z']):
            C00_image = imageio.imread(DATAPATH + '/Potential_TP_Metastases/' + samplecard['mouse'] + '/C00/ROI_50/PNG/patch' + pad_ID(samplecard['patch_id']) + '_met' + pad_ID(samplecard['met_id']) + '_' + axis + '.png', as_gray=True)
            C01_image = imageio.imread(DATAPATH + '/Potential_TP_Metastases/' + samplecard['mouse'] + '/C01/ROI_50/PNG/patch' + pad_ID(samplecard['patch_id']) + '_met' + pad_ID(samplecard['met_id']) + '_' + axis + '.png', as_gray=True)
            C02_image = imageio.imread(DATAPATH + '/Potential_TP_Metastases/' + samplecard['mouse'] + '/C02/ROI_50/PNG/patch' + pad_ID(samplecard['patch_id']) + '_met' + pad_ID(samplecard['met_id']) + '_' + axis + '.png', as_gray=True)

            transformations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

            # data type of images?
            images[a]   = transformations(C00_image) # C00-x and C00-z
            images[a+2] = transformations(C01_image) # C01-x and C01-z
            images[a+4] = transformations(C02_image) # C02-x and C02-z

        label = samplecard['label']

        # label needs to be numeric
        if label == 'true positive':
            label = 1
            label = torch.tensor(label, dtype=torch.float32)
        elif label == 'false positive':
            label = 0
            label = torch.tensor(label, dtype=torch.float32)

        sample = (images, label)
        return sample


    def __len__(self):
        number_of_samples = len(self.samplecards)
        return number_of_samples
