#Import needed packages
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from build_dataset import MetDataset
import numpy as np
import pickle

GPU_ID = 7

CODEPATH = '/home/okugel/PMSD_code'
DATAPATH = '/home/okugel/PMSD_data'

import sys
sys.path.insert(0, CODEPATH + '/helperfunctions')
import filehandling



class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()


        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=2):
        super(SimpleNet,self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=6,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,128)
        output = self.fc(output)
        return output


#####################################################################################################

batch_size = 32

#load list of samplecards
samplecards = filehandling.pload(DATAPATH + '/mice_metadata/' + 'list_of_samplecards.pickledump')

#load dataset
dataset = MetDataset(samplecards)

#split dataset into train-set and test-set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
print('Size of test_set: ' + str(len(test_set)))
print('Size of train_set: ' + str(len(train_set)))

#Create a loader for the training set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4)

#Create a loader for the test set
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,  num_workers=4)

#####################################################################################################

#Check if gpu support is available
cuda_avail = torch.cuda.is_available()

#Create model, optimizer and loss function
model = SimpleNet(num_classes=2)

if cuda_avail:
    print('Cuda is available.')
    # model.cuda()
    torch.cuda.init()
    torch.cuda.set_device(GPU_ID)
    print('Using GPU ' + str(GPU_ID))
    torch.cuda.empty_cache()
    model.cuda()
else:
    print('Cuda is not available.')

optimizer = Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

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




def save_models(epoch):
    torch.save(model.state_dict(), "met_model_{}.model".format(epoch))
    print("Checkpoint saved --> " + "met_model_{}".format(epoch))


def test():
    model.eval()
    # test_acc = 0.0
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
            images = images.to(device)
            labels = labels.to(device)

        #Predict classes using images from the test set
        outputs = model(images)
        _,prediction = torch.max(outputs.data, 1)
        number_of_Ts += (prediction == labels.data).sum().item()
        number_of_Fs += (prediction != labels.data).sum().item()
        number_of_TPs += np.sum(np.logical_and(prediction.cpu().numpy() == 1, labels.data.cpu().numpy() == 1))
        number_of_FPs += np.sum(np.logical_and(prediction.cpu().numpy() == 1, labels.data.cpu().numpy() == 0))
        number_of_TNs += np.sum(np.logical_and(prediction.cpu().numpy() == 0, labels.data.cpu().numpy() == 0))
        number_of_FNs += np.sum(np.logical_and(prediction.cpu().numpy() == 0, labels.data.cpu().numpy() == 1))


    test_acc = float(number_of_Ts) / test_size
    precision = number_of_TPs / (number_of_TPs + number_of_FPs)
    recall = number_of_TPs / (number_of_TPs + number_of_FNs)
    F1_score = 2 * (precision * recall) / (precision + recall)

    print('Number of true predictions: ' + str(number_of_Ts))
    print('Number of false predictions: ' + str(number_of_Fs))
    print('Number of TPs: ' + str(number_of_TPs))
    print('Number of FPs: ' + str(number_of_FPs))
    print('Number of TNs: ' + str(number_of_TNs))
    print('Number of FNs: ' + str(number_of_FNs))
    print('Test Accuracy: ' + str(round(test_acc,3)))
    print('Precision: ' + str(round(precision,3)))
    print('Recall: ' + str(round(recall,3)))
    print('F1 score: ' + str(round(F1_score,3))) # optimise this one

    return F1_score

def train(num_epochs):
    best_F1_test_score = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            #Move images and labels to gpu if available
            if cuda_avail:
                device_name = 'cuda:' + str(GPU_ID)
                device = torch.device(device_name)
                images = images.to(device)
                labels = labels.to(device)

            #Clear all accumulated gradients
            optimizer.zero_grad()
            #Predict classes using images from the test set
            outputs = model(images)
            #Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs,labels)
            #Backpropagate the loss
            loss.backward()

            #Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += (prediction == labels.data).sum().item()


        #Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        #Compute the average acc and loss over all training images
        train_acc = train_acc / train_size
        train_loss = train_loss / train_size

        #Print the metrics
        print()
        print('##### Epoch ' + str(epoch) + ' #####')
        print('Train Accuracy: ' + str(round(train_acc,3)))
        print('Train Loss: ' + str(round(train_loss.item(),3)))

        #Evaluate on the test set
        F1_test_score = test()

        # Save the model if the test acc is greater than our current best
        if F1_test_score > best_F1_test_score:
            save_models(epoch)
            best_F1_test_score = F1_test_score



if __name__ == "__main__":
    train(100)
