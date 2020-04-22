
import torch
import torch.nn as nn
import torch.autograd as ag
from scipy import optimize
from scipy.optimize import check_grad
import numpy

class ProximalLSTMCell(ag.Function):
    def __init__(self,lstm, epsilon=1.0, batch_size=27, p = 0.2):	# feel free to add more input arguments as needed
        super(ProximalLSTMCell, self).__init__()
        self.lstm = lstm   # use LSTMCell as blackbox
        self.epsilon = epsilon
        self.dropout = nn.Dropout(p=p)
        self.batch_size = batch_size
        self.hidden_size = self.lstm.hidden_size
        self.input_size = self.lstm.input_size
        self.gg= 0
    
    def forward(self, vt, pre_h, pre_c):
        # vt [batch, input_size]
        '''need to be implemented'''
        Gt = torch.zeros(self.batch_size, self.hidden_size, self.input_size)
        with torch.enable_grad():
            vt = ag.Variable(vt, requires_grad=True)
            vt = self.dropout(vt)
            ht, st = self.lstm(vt, (pre_h,pre_c))
            for i in range(st.size(-1)):
                gt = ag.grad(st[:,i], vt, grad_outputs=torch.ones_like(st[:,0]), retain_graph=True)[0]
                Gt[:,i,:] = gt
        
        GtT = Gt.transpose(1,2)
        gg = torch.matmul(Gt,GtT)
        self.G = Gt
        self.gg = ag.Variable(gg)
        st= st.reshape((self.batch_size, gg.size(1),1))
        
        ct = self.comp_ct(gg,st) 
        
        return (ht, ct)
    
    def comp_ct(self, gg, st):
        
        I = torch.eye(gg.size(1))
        I = I.reshape((1, gg.size(1),gg.size(1)))
        I = I.repeat(self.batch_size, 1, 1)
        #print(I.shape, gg.shape)
        self.storedI = I
        ct = torch.matmul(torch.inverse(I + self.epsilon*gg), st)
        ct = ct.reshape((self.batch_size, gg.size(1)))
        
        return ct

    def backward(self, grad_h, grad_c):
        #
        fc = grad_h
        # Equation 12 of appendix
        dl_ds = torch.matmul(fc, (I + torch.inverse(self.gg)))
        # Terms for eq 21 of appendix: a, ac
        a = torch.matmul((I + torch.inverse(self.gg)), fc.transpose(1,0))
        ac = torch.matmul(a, c.transpose(1,0))
        # Equation 21
        dl_dg = - torch.matmul((ac + ac.tanspose(1,0)), self.G)
        
        return dl_ds, dl_dg
    
    #def backward(self, grad_h, grad_c):
        #dL/dc
     #   return 1
        #return d ht/ d input, d ct/d input
    
        #return dL/dst, dL/dgt



