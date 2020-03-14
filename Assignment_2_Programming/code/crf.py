import torch
import torch.nn as nn
import numpy as np
from conv import Conv
from crf_layer import CRF_Layer
from dnn_template_fullNet import comp_conv_dimensions
import time
class CRF_NET(nn.Module):

    def __init__(self, input_dim, embed_dim = 18, kernel_size=2, num_labels=27, batch_size=256, m=14, stride = (1,1), convNodes= 5, padding = True):
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF_NET, self).__init__()
        
        # crf param
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.convNodes = convNodes
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.m = m
        
        # conv layer params
        self.out_channels = 1 # output channel of conv layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cout_shape = self.get_cout_dim() # output shape of conv layer
        self.cout_numel = self.cout_shape[0]*self.cout_shape[1]
        h1, w1 = comp_conv_dimensions('2d', 16,8, kernel_size, stride = stride)
        self.embed_dim = h1*w1 * self.convNodes
        print("Cout dims: " + str(self.embed_dim))
        
        self.init_params()
        self.batchLoss = 0
        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    # ------------------------------------------------------------------------------------------
    
    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """
        #self.conv = Conv(self.kernel_size, self.convNodes, stride=self.stride)
        self.conv = nn.Conv2d(1, self.convNodes, kernel_size=self.kernel_size, stride = self.stride)
        self.W = torch.randn(self.num_labels, self.embed_dim, requires_grad=True)
        self.T = torch.randn(self.num_labels, self.num_labels, requires_grad=True)
        self.crf = CRF_Layer(self.W, self.T, inSize = self.embed_dim, labelSize = self.num_labels)
        
    # ------------------------------------------------------------------------------------------
    
    def get_cout_dim(self):
        if self.padding:
            return (int(np.ceil(self.input_dim[0]/self.stride[0])), int(np.ceil(int(self.input_dim[1]/self.stride[1]))))
        return None
    
    
    # ------------------------------------------------------------------------------------------
    
    # X: (batch_size, 14, 16, 8) dimensional tensor
    # iterates over all words in a batch, and decodes them one by one
    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """        
        
        b = time.time()
        batchLoss = 0
        bSize = X[0].shape[0]
        words = [X[0][i].reshape(self.m, 1, self.input_dim[0],self.input_dim[1]) for i in range(bSize)]
        
        features = [[X[1][i], self.conv(words[i]).squeeze().reshape(words[i].shape[0],-1)] for i in range(bSize)]
        # now decode the sequence using conv features
        grads = self.crf.forward(features)
        # print("{} crf_layer.forward".format(time.time()-b))
        batchLoss += self.crf.get_loss()

        self.batchLoss = batchLoss
        return grads
    

    def forward_multi(self,):
        # Reshape the word to (14,1,16,8)


        word = X[0][i].reshape(self.m, 1, self.input_dim[0],self.input_dim[1])
        
        print("{} reshape: {}".format(i,time.time()-b))
        # conv operation performed for one word independently to every letter
        #print(word.shape)
        #features = self.get_conv_features(word)
        features = self.conv(word)
        
        print("{} conv: {}".format(i,time.time()-b))
        features = [[X[1][i], features.squeeze().reshape(word.shape[0],-1)]]
        # now decode the sequence using conv features
        print("{} squeeze: {}".format(i,time.time()-b))
        
        grads = self.crf.forward(features)
        
        print("{} forward: {}".format(i,time.time()-b))

        
        batchLoss += self.crf.get_loss()
        
        print("{} batchLoss: {}".format(i,time.time()-b))
        return

    # ------------------------------------------------------------------------------------------
    
    # input dim: x = [(batch_size, m, 16, 8), (batch_size,labels)] m = 14 world length
    # output: letter_indices: (m, 1), letter labels of a word
    def predict(self, x):
        decods = torch.zeros(self.batch_size, self.m, dtype=torch.int)
        for i in range(self.batch_size):
            # Reshape the word to (14,1,16,8)
            #import pdb; pdb.set_trace()
            word = x[0][i].reshape(self.m,1, self.input_dim[0],self.input_dim[1])
            # conv operation performed for one word independently to every letter
            #features = self.get_conv_features(word)

            features = self.conv(word)
            # now decode the sequence using conv features
            #decods[i] = self.dp_infer(features)
            decods[i] = self.crf.predict([features.reshape(self.m,-1)])

        return decods
    
    # ------------------------------------------------------------------------------------------
    
    def loss(self, X, labels):
        
        
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        #features = self.get_conv_features(X)
        loss = self.batchLoss
        return loss

    
    # ------------------------------------------------------------------------------------------
    
    # performs conv operation to every (16,8) image in the word. m = 14 (default) - word length
    # returns flattened vector of new conv features
    def get_conv_features(self, word):
        """
        Generate convolution features for a given word
        """
        cout = self.conv.forward(word)
        print(cout.shape)
        cout = cout.reshape(cout.shape[0], self.cout_numel)
        return cout

# ========================================================================================
# MAIN
# ========================================================================================

def main():

    model = CRF_NET((16,8), padding = False)
    
if __name__ == "__main__":
    main()




