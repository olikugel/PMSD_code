import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) 
import basepaths
CODEPATH, DATAPATH = basepaths.get_basepaths()

sys.path.insert(0, CODEPATH + '/helperfunctions')
import filehandling

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Subset
from torch.optim import Adam

from build_dataset import MetDataset
import numpy as np
from sklearn.model_selection import KFold

GPU_ID = 7

# -----------------------------------------------------------------------------------



class Conv_Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv_Unit,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class NeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(NeuralNet,self).__init__()

        self.conv1 = Conv_Unit(in_channels=6,out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2) # / 2

        self.conv2 = Conv_Unit(in_channels=32, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2) # / 2

        self.conv3 = Conv_Unit(in_channels=64, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2) # / 2

        self.conv4 = Conv_Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4) # / 4
        # self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.convolutions = nn.Sequential(self.conv1, self.pool1, self.conv2,
                                          self.pool2, self.conv3, self.pool3,
                                          self.conv4, self.avgpool)

        self.fully_connected = nn.Linear(in_features=128,out_features=num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #print()
        #print('Input shape: ', input.size())

        convolution_output = self.convolutions(input)
        #print('Shape after convolutions: ', convolution_output.size())

        flattened_output = convolution_output.view(-1,128)
        #print('Shape after flattening: ', flattened_output.size())

        logits = self.fully_connected(flattened_output)
        #print('Shape after fully-connected layer: ', logits.size())

        probabilities = self.sigmoid(logits).squeeze()
        #print('Shape after sigmoid: ', probabilities.size())

        return probabilities




class WeightedBCELoss2d(torch.nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, probs, targets, classweights):
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        loss = classweights[0] * (targets_flat * torch.log(probs_flat + 0.0001)) +  classweights[1] * ((1 - targets_flat) * torch.log(1 - probs_flat + 0.0001))
        return torch.neg(loss.sum())





#####################################################################################################

# check if GPU support is available
cuda_avail = torch.cuda.is_available()

# create model by instantiating neural net
MODEL = NeuralNet(num_classes=1)

if cuda_avail:
    print('Cuda is available.')
    torch.cuda.init()
    torch.cuda.set_device(GPU_ID)
    print('Using GPU ' + str(GPU_ID))
    torch.cuda.empty_cache()
    MODEL.cuda()
else:
    print('Cuda is not available.')

optimizer = Adam(MODEL.parameters(), lr=0.001,weight_decay=0.0001)

loss_function = WeightedBCELoss2d()

batch_size = 32
print()

#####################################################################################################



#Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr




def save_model(epoch, F1_score):
    print('--> Saving model of epoch ' + str(epoch) + ' with an F1-score of ' + str(round(F1_score,3)))
    model_filename = 'model_' + str(epoch) + '.model'
    torch.save(MODEL.state_dict(), model_filename)
    print('--> Checkpoint saved: ' + model_filename)



def test():
    MODEL.eval()

    global test_loader
    global test_size
    number_of_Ts = 0 # trues
    number_of_Fs = 0 # falses
    number_of_TPs = 0 # true positives
    number_of_FPs = 0 # false positives
    number_of_TNs = 0 # true negatives
    number_of_FNs = 0 # false negatives

    for i, (images, labels) in enumerate(test_loader):
        if cuda_avail:
            device_name = 'cuda:' + str(GPU_ID)
            device = torch.device(device_name)
            images = images.to(device) # CPU --> GPU

        # predict classes using images from the test set
        predictions = MODEL(images)

        # GPU --> CPU
        predictions = predictions.data.cpu()

        # prediction --> 0 or 1
        threshold = 0.5
        predictions[predictions >= threshold] = 1
        predictions[predictions <  threshold] = 0

        number_of_Ts += torch.sum(predictions == labels)
        number_of_Fs += torch.sum(predictions != labels)

        number_of_TPs += np.sum(np.logical_and(predictions.numpy() == 1, labels.numpy() == 1))
        number_of_FPs += np.sum(np.logical_and(predictions.numpy() == 1, labels.numpy() == 0))
        number_of_TNs += np.sum(np.logical_and(predictions.numpy() == 0, labels.numpy() == 0))
        number_of_FNs += np.sum(np.logical_and(predictions.numpy() == 0, labels.numpy() == 1))

    number_of_Ts = number_of_Ts.item()
    number_of_Fs = number_of_Fs.item()

    test_accuracy = number_of_Ts / test_size
    precision = number_of_TPs / (number_of_TPs + number_of_FPs)
    recall = number_of_TPs / (number_of_TPs + number_of_FNs)
    F1_score = 2 * (precision * recall) / (precision + recall)

    print('Correctly classified: ' + str(number_of_Ts))
    print('Incorrectly classified: ' + str(number_of_Fs))
    print('Number of TPs: ' + str(number_of_TPs))
    print('Number of FPs: ' + str(number_of_FPs))
    print('Number of TNs: ' + str(number_of_TNs))
    print('Number of FNs: ' + str(number_of_FNs))
    print('Test Accuracy: ' + str(round(test_accuracy,3)))
    print('Precision: ' + str(round(precision,3)))
    print('Recall: ' + str(round(recall,3)))
    print('F1 score: ' + str(round(F1_score,3))) # optimise this one

    return F1_score



def train(num_epochs):

    global train_loader
    global train_size
    best_F1_test_score = 0.0

    for epoch in range(num_epochs):
        MODEL.train()
        number_of_Ts = 0 # number of correct predictions
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if cuda_avail:
                device_name = 'cuda:' + str(GPU_ID)
                device = torch.device(device_name)
                images = images.to(device) # CPU --> GPU
                labels = labels.to(device) # CPU --> GPU

            # clear all accumulated gradients
            optimizer.zero_grad()

            # predict classes (predictions consist of probabilities)
            predictions = MODEL(images)

            # define weights to counter class imbalance
            class1_weight = 0.5 # weight for TP-samples
            class0_weight = 2.0 # weight for FP-samples
            class_weights = [class1_weight, class0_weight]

            # compute the loss based on the predictions and actual labels (happens on GPU)
            loss = loss_function(predictions, labels, class_weights)

            # backpropagate the loss
            loss.backward()

            # adjust parameters according to the computed gradients
            optimizer.step()

            # GPU --> CPU
            loss = loss.data.cpu()
            predictions = predictions.data.cpu()
            labels = labels.data.cpu()

            train_loss += loss * images.size(0)

            # prediction --> 0 or 1
            threshold = 0.5
            predictions[predictions >= threshold] = 1
            predictions[predictions <  threshold] = 0

            number_of_Ts += torch.sum(predictions == labels)


        adjust_learning_rate(epoch)

        # compute the average acc and loss over all training images
        train_accuracy = number_of_Ts.item() / train_size
        train_loss = train_loss.item() / train_size

        # print metrics
        print()
        print('##### Epoch ' + str(epoch) + ' #####')
        print('Train Accuracy: ' + str(round(train_accuracy,3)))
        print('Train Loss: ' + str(round(train_loss,3)))

        # evaluate on the test set
        F1_test_score = test()

        # save model if test acc is greater than our current best
        if F1_test_score > best_F1_test_score:
            save_model(epoch, F1_test_score)
            best_F1_test_score = F1_test_score



if __name__ == "__main__":

    #load list of samplecards
    samplecards = filehandling.pload(DATAPATH + '/mice_metadata/' + 'list_of_samplecards.pickledump')
    print('Size of complete dataset: ' + str(len(samplecards)))

    # split dataset into train-set and test-set using k-fold cross-validation
    kfold = KFold(5, True, 1) # 5-fold, prior shuffling, 1 as seed
    fold_count = 1
    for train_indices, test_indices in kfold.split(samplecards):
        print('\n-------------------------- FOLD ' + str(fold_count) + ' --------------------------')
        train_samplecards = [samplecards[i] for i in train_indices]
        test_samplecards = [samplecards[i] for i in test_indices]
        train_set = MetDataset(train_samplecards)
        test_set = MetDataset(test_samplecards)

        train_size = len(train_set)
        test_size = len(test_set)
        print('Size of trainset: ' + str(train_size))
        print('Size of testset: ' + str(test_size))

        #Create a loader for the training set
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4)

        #Create a loader for the test set
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,  num_workers=4)

        train(30)
        fold_count += 1
