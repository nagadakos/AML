import torch
import torchvision
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
# Python Imaging Library
import PIL
import numpy as np
import sys as sys
import utils

#



# --------------------------------------------------------------------------------------------------------

# Model Definition
class MNISTNet(nn.Module):
    """ DESCRIPTION : This is a classic MNIST classificationj net based on LeNet. It is used for demonstration purposes
                      on how models should be declared (not of the hierarchical type such as siamese or VAE, or autoencoder).  
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self):
        super(MNISTNET, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'MNIST_NET'
        self.classMethod = 'standard'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Output here is of 24x24 dimnesions
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # output of conv2 is of 20x20
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
   
    # SECTION A.2
    # ***********
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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        # forward return format should be: predictions and then any other misc stuff the used loss might need such
        # as temperature etc.
        return x

    # ------------------
    def predict(self, x):
        return self.forward(x)
    # ------------------

    def report(self):

        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================

# =========================================================================================================================================================================================
# FRUITS NETWORKS
# =========================================================================================================================================================================================

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AGADAKOS NET    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ANET(nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100],
                  dims = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':50}, 'linear2':{'out':120}},
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(ANET, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'A_NET'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h1, w1 = utils.comp_conv_dimensions(dataChannels1, dataSize[1],dataSize[2], dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        print("Dims conv1 for linear layaer: {} {}".format(h1,w1))
        # Compute dims after 1nd max layer
        h2, w2 = utils.comp_pool_dimensions(dataChannels1, h1, w1, dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        print("Dims pool1 for linear layaer: {} {}".format(h2,w2))
        # Compute dims after 2nd  layer
        h3, w3 = utils.comp_conv_dimensions(dataChannels1, h2,w2, dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        print("Dims conv2 for linear layaer: {} {}".format(h3,w3))
        # Compute dims after 2nd max layer
        h4, w4 = utils.comp_pool_dimensions(dataChannels1, h3,w3, dims['pool2']['kSize'], stride = dims['pool2']['stride'])
        print("Dims pool2 for linear layaer: {} {}".format(h4,w4))
        self.linearSize = dims['conv2']['nodes'] * h4*w4
        print("Dims for linear layaer: " + str(self.linearSize))
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.conv1 = nn.Conv2d(dataSize[0], dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.drop = nn.Dropout2d()
        self.mPool1 = torch.nn.MaxPool2d(dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        self.mPool2 = torch.nn.MaxPool2d(dims['pool2']['kSize'], stride = dims['pool2']['stride']) 
        self.fc1 = nn.Linear(self.linearSize, dims['linear1']['out'])
        self.fc2 = nn.Linear(dims['linear1']['out'], dims['linear2']['out'])
        
        # Device handling CPU or GPU 
        self.device = device
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
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
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------

    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MIRZA NET    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
class MNET(nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100],
                  dims = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':50}, 'linear2':{'out':120}},
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(ANET, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'A_NET'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h1, w1 = utils.comp_conv_dimensions(dataChannels1, dataSize[1],dataSize[2], dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        print("Dims conv1 for linear layaer: {} {}".format(h1,w1))
        # Compute dims after 1nd max layer
        h2, w2 = utils.comp_pool_dimensions(dataChannels1, h1, w1, dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        print("Dims pool1 for linear layaer: {} {}".format(h2,w2))
        # Compute dims after 2nd  layer
        h3, w3 = utils.comp_conv_dimensions(dataChannels1, h2,w2, dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        print("Dims conv2 for linear layaer: {} {}".format(h3,w3))
        # Compute dims after 2nd max layer
        h4, w4 = utils.comp_pool_dimensions(dataChannels1, h3,w3, dims['pool2']['kSize'], stride = dims['pool2']['stride'])
        print("Dims pool2 for linear layaer: {} {}".format(h4,w4))
        self.linearSize = dims['conv2']['nodes'] * h4*w4
        print("Dims for linear layaer: " + str(self.linearSize))
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.conv1 = nn.Conv2d(dataSize[0], dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.drop = nn.Dropout2d()
        self.mPool1 = torch.nn.MaxPool2d(dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        self.mPool2 = torch.nn.MaxPool2d(dims['pool2']['kSize'], stride = dims['pool2']['stride']) 
        self.fc1 = nn.Linear(self.linearSize, dims['linear1']['out'])
        self.fc2 = nn.Linear(dims['linear1']['out'], dims['linear2']['out'])
        
        # Device handling CPU or GPU 
        self.device = device
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
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
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------

    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RYSBEK NET    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RNET(nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100],
                  dims = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':50}, 'linear2':{'out':120}},
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(ANET, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'A_NET'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h1, w1 = utils.comp_conv_dimensions(dataChannels1, dataSize[1],dataSize[2], dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        print("Dims conv1 for linear layaer: {} {}".format(h1,w1))
        # Compute dims after 1nd max layer
        h2, w2 = utils.comp_pool_dimensions(dataChannels1, h1, w1, dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        print("Dims pool1 for linear layaer: {} {}".format(h2,w2))
        # Compute dims after 2nd  layer
        h3, w3 = utils.comp_conv_dimensions(dataChannels1, h2,w2, dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        print("Dims conv2 for linear layaer: {} {}".format(h3,w3))
        # Compute dims after 2nd max layer
        h4, w4 = utils.comp_pool_dimensions(dataChannels1, h3,w3, dims['pool2']['kSize'], stride = dims['pool2']['stride'])
        print("Dims pool2 for linear layaer: {} {}".format(h4,w4))
        self.linearSize = dims['conv2']['nodes'] * h4*w4
        print("Dims for linear layaer: " + str(self.linearSize))
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.conv1 = nn.Conv2d(dataSize[0], dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.drop = nn.Dropout2d()
        self.mPool1 = torch.nn.MaxPool2d(dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        self.mPool2 = torch.nn.MaxPool2d(dims['pool2']['kSize'], stride = dims['pool2']['stride']) 
        self.fc1 = nn.Linear(self.linearSize, dims['linear1']['out'])
        self.fc2 = nn.Linear(dims['linear1']['out'], dims['linear2']['out'])
        
        # Device handling CPU or GPU 
        self.device = device
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
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
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------

    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
