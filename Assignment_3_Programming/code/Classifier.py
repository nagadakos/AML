
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import ProxLSTM as pro
import numpy as np



class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, input_size):
        super(LSTMClassifier, self).__init__()
        # Class variables for measures.
        self.accuracy = 0
        self.trainLoss= 0
        self.testLoss = 0
        self.history = [[],[]] # trainACC, testACC
        self.output_size = output_size	# should be 9
        self.hidden_size = hidden_size  #the dimension of the LSTM output layer
        self.input_size = input_size	  # should be 12
        self.kSize  = 3
        self.stride = 3
        self.oChannels = 64
        self.conv = nn.Conv1d(in_channels= self.input_size, out_channels= self.oChannels, kernel_size= self.kSize, stride= self.stride) # feel free to change out_channels, kernel_size, stride
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.oChannels, hidden_size)
        #self.lstm = nn.LSTMCell(self.oChannels, hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, inX, r, batch_size, mode='plain'):
        # do the forward pass
        # pay attention to the order of input dimension.
        # input now is of dimension: batch_size * sequence_length * input_size
        if mode == 'plain':
            inX = self.relu(self.conv(inX))
            # LSTM expects (seqLen, bSize, inputSize)
            seqLen = inX.shape[2]
            inX = torch.reshape(inX, (seqLen, batch_size, inX.shape[1]))
            lstmOut, _ = self.lstm(inX)
            #toTargetSpace = self.relu(self.linear(lstmOut.view(seqLen*batch_size,-1))) # lstm has the hidden layers for all time time, take thhe last ones as output
            toTargetSpace = self.relu(self.linear(lstmOut[-1,:,:]))
            out = toTargetSpace

        if mode == 'AdvLSTM':
            # different from mode='plain', you need to add r to the forward pass
            # also make sure that the chain allows computing the gradient with respect to the input of LSTM
            inX = self.relu(self.conv(inX))
            self.inX = inX # store the output features of the conv layer, to compute the grad later on
            if isinstance(r, type(inX)): # check in r is a tensor
                inX = inX + r
            # LSTM expects (seqLen, bSize, inputSize)
            seqLen = inX.shape[2]
            inX = torch.reshape(inX, (seqLen, batch_size, inX.shape[1]))
            lstmOut, _ = self.lstm(inX)
            #toTargetSpace = self.relu(self.linear(lstmOut.view(seqLen*batch_size,-1))) # lstm has the hidden layers for all time time, take thhe last ones as output
            toTargetSpace = self.relu(self.linear(lstmOut[-1,:,:]))
            out = toTargetSpace
             

        if mode == 'ProxLSTM':
            pass
                # chain up layers, but use ProximalLSTMCell here
        return out
    
    def save(self,path='./models/LSTM_Plain'):
        torch.save(self.state_dict(), path)
        
    def load(self, path='./models/LSTM_Plain'):
        self.load_state_dict(torch.load(path))
    
    def plot(self,  figType = 'acc', saveFile = None, label = 'plain LSTM ACC'):
        if figType == 'acc':
            fig = plt.figure()
            plt.title('Test Accuracy vs Epoch')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            xAxis = np.arange(len(self.history[1]))+1
            plt.plot(xAxis, self.history[1], marker = 'o', label = label)
            plt.legend()
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
    