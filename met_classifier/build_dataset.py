"""
Building own dataset
"""
import torch
from torch.utils.data import Dataset
import imageio
import sys
import numpy as np
from torchvision.transforms import transforms


# helper function to convert array to tensor
def array_to_tensor(array):
    """
    Should be standard in future
    """
    array = array.astype(np.float32)
    tensor = torch.from_numpy(array).type(torch.FloatTensor)
    return tensor

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
    def __init__(self, points, pointIDs, labels, dataFolder):
        """
        Loads & returns tensors of all 3 projections of input (and mask) based on patch IDs.
        --> This is used for evaluation of test set where we want to assess the full power by
        making use of all 3 projections at once.
        --> this is NOT used for training
        Thus, no augmentation or maskweighting is needed/valid
        """
        self.points = points
        self.pointIDs = pointIDs
        self.labels = labels
        self.dataFolder = dataFolder

    def __getitem__(self, index):
        point_as_string = self.points[index]
        pointID = self.pointIDs[index]
        label = self.labels[index]

        images = torch.tensor(np.zeros((6,50,50),np.float32))
        for a, axis in enumerate(['y','x','z']):
            cancer_image   = imageio.imread(self.dataFolder + '/point' + pad_ID(pointID) + '_TB_' + point_as_string + '_' + label + '_' + axis + '.png')
            autofluo_image = imageio.imread(self.dataFolder + '/point' + pad_ID(pointID) + '_AF_' + point_as_string + '_' + label + '_' + axis + '.png')
            '''
            transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5),(0.5))
            ])
            images[a] = transform(cancer_image)
            images[a+3] = transform(autofluo_image)
            '''
            cancer_image   = array_to_tensor(cancer_image)
            autofluo_image = array_to_tensor(autofluo_image)
            normalize_image = transforms.Normalize((0.5),(0.5))
            print('pointID = ' + str(pointID))
            print('a = ' + str(a))
            print('axis = ' + str(axis))
            images[a] = normalize_image(cancer_image) # tumorBoost y, x, z
            images[a+3] = normalize_image(autofluo_image) # autofluo y, x, z

        if(self.labels is None):
            # We are in prediction mode
            return images
        else:
            # We are in validating or testing mode
            return  images, label


    def __len__(self):
        return len(self.pointIDs)
