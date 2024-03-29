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
import my_loader
import torch.utils.data as data_utils
import math
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
    #dataset = get_dataset(dtype='letter-features', convToTensor = False)
    #dataset = get_dataset(type='word-features')
    #split = int(0.5 * len(dataset.data)) # train-test split
    #train_data, test_data = dataset.data[:split], dataset.data[split:]
    #train_target, test_target = dataset.target[:split], dataset.target[split:]

    # Convert dataset into torch tensors
    #train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
    #test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())
    # Define train and test loaders
    #trainLoader = data_utils.DataLoader(train,  # dataset to load from
                                         #batch_size=bSize,  # examples per batch (default: 1)
                                         #shuffle=True,
                                         #sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                         #num_workers=5,  # subprocesses to use for sampling
                                         #pin_memory=False,  # whether to return an item pinned to GPU
                                         #)

    #testLoader = data_utils.DataLoader(test,  # dataset to load from
                                        #batch_size=bSize,  # examples per batch (default: 1)
                                        #shuffle=False,
                                        #sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                        #num_workers=5,  # subprocesses to use for sampling
                                        #pin_memory=False,  # whether to return an item pinned to GPU
                                        #)
    print('Loaded dataset... ')
    # End of DataLoading -------------------


    # Sanity Prints---
    #print(len(train))
    #print(type(train[0]))
    #print(train[0][0].shape)
    #print(train[0][1].shape)
    #print(train[0][1].shape)
    
    trainLoader, testLoader, listReturns= my_loader.load_data(dType = 'letter-features')
    return trainLoader, testLoader, listReturns
#----------------------------------------------------------------------------------------------
def comp_pool_dimensions(layerType,height, width, kSize, depth = 0, padding=0, dilation=1, stride=1,retType = 'list'):
    dims = int(''.join(filter(str.isdigit, str(layerType))))

    if type(padding) is not (list and tuple):
        padding = [padding, padding]
    if type(dilation) is not (list and tuple):
        dilation = [dilation, dilation]
    if type(kSize) is not (list and tuple):
        kSize = [kSize, kSize]
    if type(stride) is not (list and tuple):
        stride = [stride, stride]


   
    heightOut = math.floor(int(((height+ 2*padding[0] - dilation[0] * (kSize[0]-1)-1) / stride[0]) +1))
    widthOut  = math.floor(int(((width + 2*padding[1] - dilation[1] * (kSize[1]-1)-1) / stride[1]) +1))
    if dims == 3:
        depthOut  = int(((depth + 2*padding[0] - dilation[2] * (kSize[0]-1)-1) / stride[0]) +1)
        heightOut = int(((height+ 2*padding[1] - dilation[1] * (kSize[1]-1)-1) / stride[1]) +1)
        widthOut  = int(((width + 2*padding[2] - dilation[2] * (kSize[2]-1)-1) / stride[2]) +1)

    if retType == 'list':
        return [heightOut, widthOut] if dims != 3 else [depth,height, width]
    else:
        return  dict(height=heightOut, width = widthOut) if dims != 3 else dict(depth=depthOut, height=heightOut, width = widthOut)
# --------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def comp_conv_dimensions(layerType, height, width, kSize, depth = 0, padding=0, dilation=1, stride=1, outputPadding = 0,retType = 'list'):
    ''' DESCRIPTION: This function computes the out dimensions of any convolutional or transpose convolutional
                     pytorch layer. It returns a list or dict of the computed output dimensions of size 1 for
                     1D conv, size 2 for 2D and 3 for 3D.
        ARGUMENTS: layerType-> (type) type of this layer.
                   Rest of args: (int or list,tuple): Input to this layer in this order: height,width, kernel
                   size of layer. If the given inputs are not list, tuple the scalars are repeated to form the
                   required holder.
                   Rest of keword args: Similarly to args. These are usually set to the default values, hence
                   the keyword format.
    '''
    dims = int(''.join(filter(str.isdigit, str(layerType))))

    if type(padding) is not (list and tuple):
        padding = [padding, padding]
    if type(dilation) is not (list and tuple):
        dilation = [dilation, dilation]
    if type(kSize) is not (list and tuple):
        kSize = [kSize, kSize]
    if type(stride) is not (list and tuple):
        stride = [stride, stride]
    if type(outputPadding) is not (list and tuple):
        outputPadding = [outputPadding, outputPadding]


    if 'Transpose' in str(layerType):
        heightOut = math.floor(int( (height-1) * stride[0]  - 2*padding[0] + dilation[0] * (kSize[0]-1) + outputPadding[0] +1))
        widthOut  = math.floor(int( (width -1) * stride[1]  - 2*padding[1] + dilation[1] * (kSize[1]-1) + outputPadding[1] +1))
        if dims == 3:
            depthOut  = math.floor(int( (depth -1) * stride[0] - 2*padding[0] + dilation[0] * (kSize[0]-1) + outputPadding[0] +1))
            heightOut = math.floor(int( (height-1) * stride[1] - 2*padding[1] + dilation[1] * (kSize[1]-1) + outputPadding[1] +1))
            widthOut  = math.floor(int( (width -1) * stride[2] - 2*padding[2] + dilation[2] * (kSize[2]-1) + outputPadding[2] +1))
    else:
        heightOut = math.floor(int(((height+ 2*padding[0] - dilation[0] * (kSize[0]-1)-1) / stride[0]) +1))
        widthOut  = math.floor(int(((width + 2*padding[1] - dilation[1] * (kSize[1]-1)-1) / stride[1]) +1))
        if dims == 3:
            depthOut  = math.floor(int(((depth + 2*padding[0] - dilation[2] * (kSize[0]-1)-1) / stride[0]) +1))
            heightOut = math.floor(int(((height+ 2*padding[1] - dilation[1] * (kSize[1]-1)-1) / stride[1]) +1))
            widthOut  = math.floor(int(((width + 2*padding[2] - dilation[2] * (kSize[2]-1)-1) / stride[2]) +1))

    if retType == 'list':
        return [heightOut, widthOut] if dims != 3 else [depth,height, width]
    else:
        return  dict(height=heightOut, width = widthOut) if dims != 3 else dict(depth=depthOut, height=heightOut, width = widthOut)
# --------------------------------------------------------------------------------------------------------

# Model Definition
class Net(nn.Module):

    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    history = [[],[],[]]
    
    
    # Mod init + boiler plate code
    # Skeleton of this network; the blocks to be used.
    # Similar to Fischer prize building blocks!
    def __init__(self, device = device, dims = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1 }, 
                                                'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)} }):
        super(Net, self).__init__()
        
        # Layer Dimensions computation
        # Compute dims after 1nd conv layer
        h1, w1 = comp_conv_dimensions('2d', 16,8, dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        print("Dims conv1 for linear layaer: {} {}".format(h1,w1))
        # Compute dims after 1nd max layer
        h2, w2 = comp_pool_dimensions('2d', h1, w1, dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        print("Dims pool1 for linear layaer: {} {}".format(h2,w2))
        # Compute dims after 2nd  layer
        h3, w3 = comp_conv_dimensions('2d', h2,w2, dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        print("Dims conv2 for linear layaer: {} {}".format(h3,w3))
        # Compute dims after 2nd max layer
        h4, w4 = comp_pool_dimensions('2d', h3,w3, dims['pool2']['kSize'], stride = dims['pool2']['stride'])
        print("Dims pool2 for linear layaer: {} {}".format(h4,w4))
        self.linearSize = dims['conv2']['nodes'] * h4*w4
        print("Dims for linear layaer: " + str(self.linearSize))
        # ---|
        
        # Layers Declaration
        self.conv1 = nn.Conv2d(1, dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.drop = nn.Dropout2d()
        self.mPool1 = torch.nn.MaxPool2d(dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        self.mPool2 = torch.nn.MaxPool2d(dims['pool2']['kSize'], stride = dims['pool2']['stride']) 
        self.fc1 = nn.Linear(self.linearSize, 50)
        self.fc2 = nn.Linear(50, 26)
        
        # Device handling CPU or GPU 
        self.device = device

    # ------------------

    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.mPool1(x))
        x = self.conv2(x)
        x = F.relu(self.mPool2(x))
        x = F.dropout(x, training=self.training)
        #print(x.shape)
        x = x.view(-1, self.linearSize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x

    # ------------------

    def forward_no_drop(self, x):
        x = self.conv1(x)
        x = F.relu(self.mPool1(x))
        x = self.conv2(x)
        x = F.relu(self.mPool2(x))
        #print(x.shape)
        x = x.view(-1, self.linearSize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    # ------------------
    def get_predictions(self, x):
        x = self.forward_no_drop(x)
        x = torch.max(x, dim = 1)[1]
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
            #loss        = F.cross_entropy(output, label.squeeze())
            if label.shape[0] > 1:
                label = label.squeeze()
            loss = F.nll_loss(output, label)

            # Backpropagation part
            # 1. Zero out Grads
            optim.zero_grad()
            # 2. Perform the backpropagation based on loss
            loss.backward()            
            # 3. Update weights 
            optim.step()

           # Training Progress report for sanity purposes! 
            if verbose:
                if idx % 200 == 0: 
                    print("Epoch: {}->Batch: {} / {}. Loss = {}".format(args, idx, len(indata), loss.item() ))
        # Log the current train loss
        self.trainLoss = loss   
        self.history[0].append(loss)
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
                #loss  += F.nll_loss(output, label, reduction = 'sum').item()
                loss  += F.cross_entropy(output, label.squeeze(), reduction = 'sum').item()
                # Find the max along a row but maitain the original dimenions.
                # in this case  a 10 -dimensional array.
                pred   = output.max(dim = 1, keepdim = True)
                # Select the indexes of the prediction maxes(max[1]).
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
        self.history[1].append(loss/len(testLoader.dataset))
        self.history[2].append(acc)
        # Print accuracy report!
        print("Accuracy: {} ({} / {})".format(acc, true,
                                              len(testLoader.dataset)))

        # Testing and error reports are done here
    def predict(self, testLoader):
        print("In  Function!")        
        loss = 0 
        true = 0
        acc  = 0
        dev = self.device
        # Inform Pytorch that keeping track of gradients is not required in
        # testing phase.
        with torch.no_grad():
            for data, label in testLoader:
                data, label = data.to(dev), label.to(dev)
                output = self.get_redicitons(data)
                
        
    def report(self):

        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    def plot(self, inData = None, figType = 'acc', saveFile = None):
        if figType == 'acc':
            fig = plt.figure()
            plt.title('Test Accuracy vs Epoch')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            xAxis = np.arange(len(self.history[2]))+1
            plt.plot(xAxis, self.history[2], marker = 'o', label = 'Letterwize Acc')
            if inData is not None:
                plt.plot(xAxis, inData, marker = 'o', label = 'Wordwise Acc')
            if saveFile is not None:
                plt.savefig(saveFile)
            return fig

        if figType == 'lrCurve':
            fig = plt.figure()
            plt.title('Learning Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            xAxis = np.arange(len(self.history[2]))+1
            plt.plot(xAxis, self.history[0], marker = 'o', label="Training Loss")
            plt.plot(xAxis, self.history[1], marker = 'o', label = 'Test Loss')
            plt.legend()
            if saveFile is not None:
                plt.savefig(saveFile)
            return fig
    
    def evaluate(self, dataLoaders, kwArgs = {'lr':0.001, 'm':0.9, 'bSize':32, 'epochs':12, 'opt': 'adam'}):
        """ DESCRIPTION: THis function will fit and test on hte given dataloader and the given kwargs.
                         It stores training and testing information in the model class variables, along
                         with learned weights.
            ARGUMENTS: dataLoaders (list of dataloaders) : Tensor dataloader objects for train and test
                       kwArgs (dict): dicitionary holding the training and testing parameters
        """
        
        # Get args
        lr, m , bSize, epochs = kwArgs['lr'], kwArgs['m'], kwArgs['bSize'], kwArgs['epochs']
        dev = self.device
        # Load data, initialize model and optimizer!
        trainLoader, testLoader = dataLoaders[0], dataLoaders[1]
        if kwArgs['opt'] == 'adam':
            optim = optm.Adam(self.parameters(), lr=lr)
        else:
            optim = optm.SGD(self.parameters(), lr=lr, momentum=m)

        print("######### Initiating Network Training and Testing #########\n")
        print("Parameters: lr:{}, momentum:{}, batch Size:{}, epochs:{}".format(lr,m,bSize,epochs))
        print("Procedure carried on device: {}\n".format(dev))
        for e in range(epochs):
            print("Epoch: {} start ------------\n".format(e))
            args = e
            self.train(args, dev, trainLoader, optim)
            self.test(dev, testLoader)

        # Final report
        self.report()
        
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
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args() 
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
    trainLoader, testLoader, _ = load_data(bSize=bSize)
    dims = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1 }, 
            'pool1':{'stride':(1,1), 'kSize':(3,3)}, 'pool2':{'stride':(1,1), 'kSize':(2,2)} }
    model = Net(dims=dims).to(device)
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
    return model

# Define behavior if this module is the main executable. Standard code.
if __name__ == '__main__':
    main()

