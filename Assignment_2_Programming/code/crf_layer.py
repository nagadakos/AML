"""
Author: Yeshu Li
The Python program has been tested under macOS Mojava Version 10.14.3 and Ubuntu 18.04.

The file paths are hard-coded in the code for my convenience. There are 4 features in crf.py file.

1. p2a function computes the required log-likelihood and stores the required gradients in gradients.txt.
2. p2b function computes the optimal parameter by using L-BFGS-B optimization method, outputs the final objective function value and stores the optimal parameter in solution.txt.
3. checkGrad function checks the gradients against finite differences.


"""

import time
import math
import numpy as np
from scipy.optimize import check_grad, minimize
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

K = 26
imgSize = 128
paraNum = K * K + K * imgSize

def readDataset(filePath):

	words = []

	with open(filePath, 'r') as f:
		label = []
		data = []
		for line in f.readlines():
			tokens = line.split()
			label.append(ord(tokens[1]) - ord('a'))
			data.append([int(x) for x in tokens[5:]])
			if tokens[2] == '-1':
				words.append([torch.tensor(label), torch.tensor(data,dtype=torch.float32)])
				label = []
				data = []

	return words

def readParameter(filePath):

	w = torch.zeros((K, imgSize))
	T = torch.zeros((K, K))

	with open(filePath, 'r') as f:
		lines = [float(line) for line in f.readlines()]
		for i in range(K):
			w[i] = torch.tensor(lines[i * imgSize : (i + 1) * imgSize])
		offset = K * imgSize
		for i in range(K):
			for j in range(K):
				T[j, i] = lines[offset + i * K + j]

	return w, T

class CRF_Layer(nn.Module):
    
    def __init__(self, W, T, inSize = 128, labelSize= 26, C =1, m = 14):
        super(CRF_Layer, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.w = Parameter(W)
        self.t = Parameter(T)
        self.imgSize = inSize
        self.objValue = torch.tensor([0])
        self.gradients = 0
        self.K = labelSize
        self.C = C
        self.m = m 
    # -------------------------------------------------------------
      
    def computeAllDotProduct(self,w, word):
        #print(len(word), word[0], word[1][0].shape)
        label, data = word
        #dots = np.dot(w, data.transpose())
        dots = torch.mm(w, data.transpose(0,1))

        return dots
 
    # -------------------------------------------------------------
    
    def logTrick(self, numbers):

        if len(numbers.shape) == 1:
            M = torch.max(numbers)
            return M + torch.log(torch.sum(torch.exp(numbers - M)))
        else:
            M = torch.max(numbers, 1)[0]
            return M + torch.log(torch.sum(torch.exp((numbers.transpose(0,1) - M).transpose(0,1)), 1))
 
    # -------------------------------------------------------------
    
    def logPYX(self, word, w, T, alpha, dots):

        label, data = word
        m = len(label)
        res = sum([dots[label[i], i] for i in range(m)]) + sum([T[label[i], label[i + 1]] for i in range(m - 1)])
        logZ = self.logTrick(dots[:, m - 1] + alpha[m - 1, :])
        res -= logZ

        return res
 
    # -------------------------------------------------------------
    
    def computeDP(self, word, w, T, dots):

        label, data = word
        m = len(label)
        alpha = torch.zeros((m, self.K))
        for i in range(1, m):
            #alpha[i] = self.logTrick(torch.repeat(dots[:, i - 1] + alpha[i - 1, :], (K, 1)) + T.transpose())
            alpha[i] = self.logTrick( (dots[:, i - 1] + alpha[i - 1, :]).expand_as(T.transpose(0,1)) + T.transpose(0,1))
        beta = torch.zeros((m, self.K))
        for i in range(m - 2, -1, -1):
            beta[i] = self.logTrick( (dots[:, i + 1] + beta[i + 1, :]).expand_as(T) + T)

        return alpha, beta
 
    # -------------------------------------------------------------
    
    def computeMarginal(self, word, w, T, alpha, beta, dots):

        label, data = word
        m = len(label)
        p1 = torch.zeros((m, self.K))
        for i in range(m):
            p1[i] = alpha[i, :] + beta[i, :] + dots[:, i]
            p1[i] = torch.exp(p1[i] - self.logTrick(p1[i]))
        p2 = torch.zeros((m - 1, self.K, self.K))
        for i in range(m - 1):
            #print((alpha[i, :] + dots[:, i]).expand_as(T).transpose(0,1).shape, (alpha[i, :] + dots[:, i]).shape)
            #print((alpha[i, :] + dots[:, i]).expand_as(T).transpose(0,1), (alpha[i, :] + dots[:, i]))
            p2[i] = (alpha[i, :] + dots[:, i]).expand_as(T).transpose(0,1) + (beta[i + 1, :] + dots[:, i + 1]).expand_as(T) + T
            p2[i] = torch.exp(p2[i] - self.logTrick(p2[i].flatten()))

        return p1, p2
 
    # -------------------------------------------------------------
    
    def computeGradientWy(self,word, p1):

        label, data = word
        m = len(label)
        cof = torch.zeros((self.K, m))
        for i in range(m):
            cof[label[i], i] = 1
        cof -= p1.transpose(0,1)
        res = torch.mm(cof, data)

        return res
 
    # -------------------------------------------------------------
    
    def computeGradientTij(self,word, p2):

        label, data = word
        m = len(label)
        res = torch.zeros(p2.shape)
        for i in range(m - 1):
            res[i, label[i], label[i + 1]] = 1
        res -= p2
        res = torch.sum(res, 0)

        return res
    
    # -------------------------------------------------------------
    
    def compute_pdfs(self, words):
        dots = self.computeAllDotProduct(w, words[0])
        alpha, beta = self.computeDP(words[0], w, T, dots)
        p1, p2 = self.computeMarginal(words[0], w, T, alpha, beta, dots)
        logPYX = self.logPYX(word, w, T, alpha, dots)
                                         
        self.p1 = torch.zeros(len(words), p1.shape)
        self.p2 = torch.zeros(len(words), p2.shape)
        self.logPYX = torch.zeros(len(words), logPYX.shape)
                                         
        for i, word in enumerate(words):
            dots = self.computeAllDotProduct(self.w, word)
            alpha, beta = self.computeDP(word, self.w, self.T, dots)
            self.p1[i], self.p2[i] = self.computeMarginal(word, self.w, self.T, alpha, beta, dots)
            self.logPYX[i] = self.logPYX(word, self.w, self.T, alpha, dots)
        
    # -------------------------------------------------------------
    
    def forward(self, dataset):
        """ DESCRIPTION: (old backward). THis function will compute all the marginals and gradients for each
                         barch. The grads are stored in the class.
        """
        
        w = self.w
        T = self.t
        
        meandw = torch.zeros((self.K, self.imgSize))
        meandT = torch.zeros((self.K, self.K))
        meanLogPYX = 0

        for word in dataset:

            dots = self.computeAllDotProduct(w, word)
            alpha, beta = self.computeDP(word, w, T, dots)
            p1, p2 = self.computeMarginal(word, w, T, alpha, beta, dots)

            dw = self.computeGradientWy(word, p1)
            dT = self.computeGradientTij(word, p2)

            meanLogPYX += self.logPYX(word, w, T, alpha, dots)
            meandw += dw
            meandT += dT
            
        meanLogPYX /= len(dataset)
        meandw /= len(dataset)
        meandT /= len(dataset)

        #meandw *= (-C)
        #meandT *= (-C)

        #meandw += w
        #meandT += T

        self.gradients = torch.cat((meandw.flatten(), meandT.flatten()))

        self.objValue = -self.C * meanLogPYX + 0.5 * torch.sum(w ** 2) + 0.5 * torch.sum(T ** 2)

        return self.gradients
    
    # -------------------------------------------------------------
    
    def compute_obj():
        self.objValue = -C * self.logPYX + 0.5 * torch.sum(self.w ** 2) + 0.5 * torch.sum(self.T ** 2)
        return self.objValue
    
    # -------------------------------------------------------------
    
    def backward(self):

        gradients = self.gradients

        return gradients
    
    # -------------------------------------------------------------
    def predict(self, inX):
        """ DESCRIPTION: Using max-sum algorithm to decode. THis function will use the forward pass to compute
                         the most probable last letter, at place m for a seq of length m. Then using that knowledge
                         it will go backwards at m-1 extracting the character that was most likely to lead to the most probable 
                         character at place m. It will recurcively continue until we have a candidate at place m=0 and return
                         the labels that we are most confident they match the input.
            ARGUMENTS: seq(ndarray): a sequqnce that contains aoiur input data.
        """
        # Init
        w = self.w
        t = self.t
        seq = inX[0]
        seqLen  = seq.shape[0]
        dataLen = w.shape[0]
        cMat    = torch.zeros((seqLen, dataLen))
        bMat    = torch.matmul(seq, w.transpose(0,1)) # this gives the probability of observing each i standalone
        maxIdx  = torch.zeros((seqLen, dataLen), dtype = np.int)
        decSeq  = []
        maxObj  = 0
        ci = np.zeros((26,26))

        # Get step 0 estimates
        step0    = torch.matmul(seq[0,:], w.transpose(0,1))
        c1 = step0+t      # the previous node potential x edje potenital for all i->j pairs
        maxIdx[0,:] = torch.argmax(c1, axis = 0) #1x26 the max elem indexes
        c1 = c1[maxIdx[0,:], torch.arange(c1.shape[1])] #1x26 select the max element, as indexed by maxIdx, for every column
        cMat[0,:] = 0
        #print("Input shapes: sequence: {}, weights: {}, T matrix:{}".format(seq.shape,w.shape,t.shape))

        # Perform a pass over steps 1-99 or 2-100 in 1-index notation!
        for i in range(1,seqLen):
            fi  = bMat[i-1,:].reshape(-1,1)              # get likelihood of i observation being any letter as standalone 
            cYi = cMat[i-1,:].reshape(-1,1)            # i-1 step's potential msg to this node i
            cYj = fi + t + cYi     # mul the likelihood of each letter with all transition probs.
            #if i ==1:
                #print(cYi.shape,cYj.shape)
            maxIdx[i,:] = torch.argmax(cYj, axis = 0) #1x26 the max elem indexes
            cYj = cYj[maxIdx[i,:], torch.arange(cYj.shape[1])] #1x26 select the max element, as indexed by maxIdx, for every column
            #cYj = np.multiply(cYj, cYi)  # multiply current cYj = g_yj->yi* fj with pervious node's(i-1) msgs to yield the msg node i will wend to i+1
            cMat[i,:] = cYj              # store this node's msg, in order to send it to i+1 on next step!
            #if i == 1:
            #    print(cYj, cYj.shape, fi.shape)
        maxObj = torch.max(cYj).item()             # max objective value is the maximum of the lasp step's messages or potentials Y_end 
        # Backward pass.
        # At each step, the element e at slot i is the label that most likely lead to the lable represented as i.
        # So at i.e if 95,5 = 11, then that means that the most probable letter that leads to f(idx 5) if letter 11 or l (label 11). All indexing is 0 based.
        decSeq     = torch.zeros((seqLen), dtype = np.int)
        lastMax    = torch.argmax(maxIdx[-1,:]) # get the most probable last element. use this to recurse and find the most prob sequence
        decSeq[-1] = lastMax
        for i in range(seqLen-2,-1,-1):
            #print("i is ", i, "decSeq of ", i+1, "is", decSeq[i+1])
            curMax = maxIdx[i,decSeq[i+1]]
            decSeq[i] = curMax
        # Remember that python indexes from 0. Labels are 1-26 so we must add 1 to compensate
        return decSeq+1, maxObj 
    # ---------------------------------------------------------------------------
    def predict_old(self, inX):
        # decode a sequence of letters for one word
        w = self.w
        T = self.t
        #m = self.m
        ret = torch.zeros(len(inX), len(inX[0])) # return tensor should be batch size x word size
        for j, word in enumerate(inX):
            x = inX[j]
            m = word.shape[0]
            
            pos_letter_value_table = torch.zeros((m, self.K), dtype=torch.float64)
            pos_best_prevletter_table = torch.zeros((m, self.K), dtype=torch.int)
            # for the position 1 (1st letter), special handling
            # because only w and x dot product is covered and transition is not considered.
            for i in range(self.K):
            # print(w)
            # print(x)
                pos_letter_value_table[0, i] = torch.dot(w[i, :], x[0, :])

            # pos_best_prevletter_table first row is all zero as there is no previous letter for the first letter

            # start from 2nd position
            for pos in range(1, m):
            # go over all possible letters
                for letter_ind in range(self.K):
                    # get the previous letter scores
                    prev_letter_scores = pos_letter_value_table[pos-1, :].clone()
                    # we need to calculate scores of combining the current letter and all previous letters
                    # no need to calculate the dot product because dot product only covers current letter and position
                        # which means it is independent of all previous letters
                    for prev_letter_ind in range(self.K):
                        prev_letter_scores[prev_letter_ind] += T[prev_letter_ind, letter_ind]

                    # find out which previous letter achieved the largest score by now
                    best_letter_ind = torch.argmax(prev_letter_scores)
                    # update the score of current positive torch.dot(w[letter_ind,:], x[pos, :])with current letter
                    term = torch.dot(w[letter_ind,:], x[pos, :])
                    pos_letter_value_table[pos, letter_ind] = prev_letter_scores[best_letter_ind] + term
                    # save the best previous letter for following tracking to generate most possible word
                    pos_best_prevletter_table[pos, letter_ind] = best_letter_ind
            letter_indicies = torch.zeros((m, 1), dtype=torch.int)
            letter_indicies[m-1, 0] = torch.argmax(pos_letter_value_table[m-1, :])
            max_obj_val = pos_letter_value_table[m-1, letter_indicies[m-1, 0]]
            # print(max_obj_val)
            for pos in range(m-2, -1, -1):
                letter_indicies[pos, 0] = pos_best_prevletter_table[pos+1, letter_indicies[pos+1, 0]]
            ret[j,:] = letter_indicies.flatten() 
            
        return ret
    
    # -------------------------------------------------------------
    
    def get_loss(self):
        
        return self.objValue

    # -------------------------------------------------------------
    
    def checkGradient(dataset, w, T):

        lossFunc = lambda x, *args: crfFuncGrad(x, *args)[0]
        gradFunc = lambda x, *args: crfFuncGrad(x, *args)[1]
        print(check_grad(lossFunc, gradFunc, x0, dataset, 1))
 
    # -------------------------------------------------------------
 
# =====================================================================================================
#
# =====================================================================================================

def main():

    start = time.time()

    trainPath = '../data/train.txt'
    testPath = '../data/test.txt'
    modelParameterPath = '../data/model.txt'

    trainSet = readDataset(trainPath)
    testSet = readDataset(testPath)
    w, T = readParameter(modelParameterPath)
    meandw = torch.zeros((K, imgSize))
    meandT = torch.zeros((K, K))
    meanLogPYX = 0
    crfL = CRF_Layer(w,T, labelSize = K, inSize= imgSize)
    i = 0
    for word in trainSet:

        C = 1
        pred = crfL.forward([word])
        grad = crfL.backward([word])
        dw = grad[:K*imgSize].reshape(K,imgSize)
        dT = grad[K*imgSize:].reshape(K,K)

        meandw += dw
        meandT += dT
        i +=1
        if i % 500 == 0:
            print("Finished up to word "+str(i))
    meandw /= len(trainSet)
    meandT /= len(trainSet)

    grad= torch.cat((meandw.flatten(), meandT.flatten()))

    outputPath = 'grads.txt'
    np.savetxt( outputPath, grad.numpy())
    print('Time elapsed: %lf' % (time.time() - start))

if __name__ == '__main__':

	main()
