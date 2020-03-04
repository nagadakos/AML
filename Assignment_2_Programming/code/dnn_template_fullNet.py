import torch
import torchvision
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
import argparse
# Python Imaging Library
import PIL
import numpy as np
import sys as sys
from data_loader import get_dataset
import torch.utils.data as data_utils

#  Global Parameters
# Automatically detect if there is a GPU or just use CPU.
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ========================================================================================================================
# Functions and Network Template
# ========================================================================================================================
def load_data(bSize = 32):
    # bundle common args to the Dataloader module as a kewword list.
    # pin_memory reserves memory to act as a buffer for cuda memcopy 
    # operations
    comArgs = {'shuffle': True,'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Data Loading -----------------------
    # ******************
    # load the data set as a colldection of tuples. First element in tuple are the data formated as a numOFLetters / word x 128; second 
    # elements is a one-hot endoding of the label of the letter so: numOfLetter / word x 26. Zero padding has been used to keep word size
    # consistent to 14 letters per word. The labels for the padding rows are all 0s.
    # ******************
    dataset = get_dataset(type='letter-features', convToTensor = True)
    #dataset = get_dataset(type='word-features')
    split = int(0.5 * len(dataset.data)) # train-test split
    train_data, test_data = dataset.data[:split], dataset.data[split:]
    train_target, test_target = dataset.target[:split], dataset.target[split:]

    # Convert dataset into torch tensors
    train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
    test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())
    # Define train and test loaders
    trainLoader = data_utils.DataLoader(train,  # dataset to load from
                                         batch_size=bSize,  # examples per batch (default: 1)
                                         shuffle=True,
                                         sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                         num_workers=5,  # subprocesses to use for sampling
                                         pin_memory=False,  # whether to return an item pinned to GPU
                                         )

    testLoader = data_utils.DataLoader(test,  # dataset to load from
                                        batch_size=bSize,  # examples per batch (default: 1)
                                        shuffle=False,
                                        sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                        num_workers=5,  # subprocesses to use for sampling
                                        pin_memory=False,  # whether to return an item pinned to GPU
                                        )
    print('Loaded dataset... ')
    # End of DataLoading -------------------


    # Sanity Prints---
    #print(len(train))
    #print(type(train[0]))
    #print(train[0][0].shape)
    #print(train[0][1].shape)
    #print(train[0][1].shape)

    return trainLoader, testLoader

# --------------------------------------------------------------------------------------------------------

# Model Definition
class Net(nn.Module):

    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    # Mod init + boiler plate code
    # Skeleton of this network; the blocks to be used.
    # Similar to Fischer prize building blocks!
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        # Output here is of 24x24 dimnesions
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        # output of conv2 is of 20x20
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # ------------------

    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # output here is 10 x 12x12
        x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        # output here is 20 x 4x4 = 320 params
        # Flatten in to 1D to feed to dense Layer.
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    # ------------------

    def forward_no_drop(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), (2,1)))
        print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), (2,1)))
        print(x.shape)
        x = x.view(-1, )
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    # ------------------

    # Call this function to facilitate the traiing process
    # While there are many ways to go on about calling the
    # traing and testing code, define a function within
    # a "Net" class seems quite intuitive. Many examples
    # do not have a class function; rather they set up
    # the training logic as a block script layed out.
    # Perhaps the object oriented approach leads to less 
    # anguish in large projects...
    def train(self, args, device, indata, optim, verbose = True):

        for idx, (img, label) in enumerate(indata):
            data, label = img.to(device), label.to(device)
            # forward pass calculate output of model
            output      = self.forward_no_drop(data)
            # compute loss
            loss        = F.nll_loss(output, label)
            # loss        = F.CrossEntropyLoss(output, target)

            # Backpropagation part
            # 1. Zero out Grads
            optim.zero_grad()
            # 2. Perform the backpropagation based on loss
            loss.backward()            
            # 3. Update weights 
            optim.step()

           # Training Progress report for sanity purposes! 
            if verbose:
                if idx % 20 == 0: 
                    print("Epoch: {}->Batch: {} / {}. Loss = {}".format(args, idx, len(indata), loss.item() ))
        # Log the current train loss
        self.trainLoss = loss   
    # -----------------------

    # Testing and error reports are done here
    def test(self, device, testLoader):
        print("In Testing Function!")        
        loss = 0 
        true = 0
        acc  = 0
        # Inform Pytorch that keeping track of gradients is not required in
        # testing phase.
        with torch.no_grad():
            for data, label in testLoader:
                data, label = data.to(device), label.to(device)
                # output = self.forward(data)
                output = self.forward_no_drop(data)
                # Sum all loss terms and tern then into a numpy number for late use.
                loss  += F.nll_loss(output, label, reduction = 'sum').item()
                # Find the max along a row but maitain the original dimenions.
                # in this case  a 10 -dimensional array.
                pred   = output.max(dim = 1, keepdim = True)
                # Select the indexes of the prediction maxes.
                # Reshape the output vector in the same form of the label one, so they 
                # can be compared directly; from batchsize x 10 to batchsize. Compare
                # predictions with label;  1 indicates equality. Sum the correct ones
                # and turn them to numpy number. In this case the idx of the maximum 
                # prediciton coincides with the label as we are predicting numbers 0-9.
                # So the indx of the max output of the network is essentially the predicted
                # label (number).
                true  += label.eq(pred[1].view_as(label)).sum().item()
        acc = true/len(testLoader.dataset)
        self.accuracy = acc
        self.testLoss = loss 
        # Print accuracy report!
        print("Accuracy: {} ({} / {})".format(acc, true,
                                              len(testLoader.dataset)))

    def report(self):

        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

# --------------------------------------------------------------------------------------------------------

def parse_args():
    ''' Description: This function will create an argument parser. This will accept inputs from the console.
                     But if no inputs are given, the default values listed will be used!
        

    '''
    parser = argparse.ArgumentParser(prog='Fashion MNIST Network building!')
    # Tell parser to accept the following arguments, along with default vals.
    parser.add_argument('--lr',    type = float,metavar = 'lr',   default='0.001',help="Learning rate for the oprimizer.")
    parser.add_argument('--m',     type = float,metavar = 'float',default= 0,     help="Momentum for the optimizer, if any.")
    parser.add_argument('--bSize', type = int,  metavar = 'bSize',default=32,     help="Batch size of data loader, in terms of samples. a size of 32 means 32 images for an optimization step.")
    parser.add_argument('--epochs',type = int,  metavar = 'e',    default=12   ,  help="Number of training epochs. One epoch is to perform an optimization step over every sample, once.")
    # Parse the input from the console. To access a specific arg-> dim = args.dim
    args = parser.parse_args()
    lr, m, bSize, epochs = args.lr, args.m, args.bSize, args.epochs
    # Sanitize input
    m = m if (m>0 and m <1) else 0 
    lr = lr if lr < 1 else 0.1
    # It is standard in larger project to return a dictionary instead of a myriad of args like:
    # return {'lr':lr,'m':m,'bSize':bbSize,'epochs':epochs}
    return lr, m , bSize, epochs

# ================================================================================================================================
# Execution
# ================================================================================================================================
def main():
    # Get keyboard arguments, if any! (Try the dictionary approach in the code aboe for some practice!)
    lr, m , bSize, epochs = parse_args()
    # Load data, initialize model and optimizer!
    trainLoader, testLoader = load_data(bSize=bSize)
    model = Net().to(device)
    optim = optm.SGD(model.parameters(), lr=lr, momentum=m)

    print("######### Initiating Fashion MNIST network training #########\n")
    print("Parameters: lr:{}, momentum:{}, batch Size:{}, epochs:{}".format(lr,m,bSize,epochs))
    for e in range(epochs):
        print("Epoch: {} start ------------\n".format(e))
        # print("Dev {}".format(device))
        args = e
        model.train(args, device, trainLoader, optim)
        model.test(device, testLoader)

    # Final report
    model.report()

# Define behavior if this module is the main executable. Standard code.
if __name__ == '__main__':
    main()

