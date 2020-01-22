"""
Building own dataset
"""
import torch
from torch.utils.data import Dataset
import imageio
import sys
import numpy as np
from torchvision.transforms import transforms

CODEPATH = '/home/okugel/PMSD/PMSD_code'
DATAPATH = '/home/okugel/PMSD/PMSD_data'


# helper function to pad ID with zeros
def pad_ID(ID):
    if ID >= 0 and ID < 10:
        ID_padded = '000' + str(ID)
    elif ID >= 10 and ID < 100:
        ID_padded = '00' + str(ID)
    elif ID >= 100 and ID < 1000:
        ID_padded = '0' + str(ID)
    elif ID >= 1000 and ID < 10000:
        ID_padded = str(ID)
    else:
        print('pad_ID helperfunction is not implemented for IDs >= 10000 or < 0')
        return None
    return ID_padded



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
