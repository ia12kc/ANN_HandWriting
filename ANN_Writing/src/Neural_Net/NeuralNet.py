
import numpy as np
import scipy as sc
from digit_Recognition import program

class NeuralNet(object):

    rWeight = 0
    syn0 = []
    syn1 = []
    batch_size = 0
    
    ''' Nueral network receives arguments: number of nodes in middle layer, initial weight limiter,
      and batch size(small sample size from dataset to train network)'''
    def __init__(self, middle_layer, rweight, batch_size):
        self.m_layer = middle_layer
        self.rWeight = rweight
        self.batch_size = batch_size
        self.initialize_Layers()
        
    #set up random weights in the network    
    def initialize_Layers(self):
        self.syn0 = np.random.random((64,self.m_layer)) - self.rWeight
        self.syn1 = np.random.random((self.m_layer,10)) - self.rWeight 
    
    #passing data into network, returns final output
    def feed_Forward(self,nums):
        layer1 = program.nonlin(np.dot(nums,self.syn0))
        output = program.nonlin(np.dot(layer1,self.syn1))
        return output

    '''backpropagation algorithm, feeds forward, calculates total error, 
    then calculates error from each node and distributes changes according to that'''
    def back_Propagation(self, X, y,bias0,bias1,b):
        l1 = program.nonlin(np.dot(X,self.syn0)+bias0)
        l2 = program.nonlin(np.dot(l1,self.syn1)+bias1)

        l2_error = y - l2
    
        l2_delta = l2_error*program.nonlin(l2,deriv=True)
        
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(self.syn1.T)
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * program.nonlin(l1,deriv=True)
        if (b==True):
            bias1=bias1+l2_delta
            bias0=bias0+l1_delta
        
        self.syn0 += X.T.dot(l1_delta)
        self.syn1 += l1.T.dot(l2_delta) 
        return l2_error