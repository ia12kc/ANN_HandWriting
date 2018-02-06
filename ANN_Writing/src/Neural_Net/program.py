'''
@author: wamjam
'''
import numpy as np
import scipy as sc

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

#return each training example in file     
def training_Data(file):
    train_x1=[]
    #file_X1
    with open(file, "r") as f:
    # Read in all the lines of your file into a list of lines
        for line in f:
            lines = list(map(float,line.split(",")))
            l=np.array(lines)
            train_x1.append(l)
    x=np.array(train_x1)
    return x

#returns testing data in file
def testing_Data(file):#file_Y1, file_Y2)
    train_x1=[]
    #file_X1
    with open(file, "r") as f:
    # Read in all the lines your file into a list of lines
        for line in f:
            lines = list(map(float,line.split(",")))
            l=np.array(lines)
            train_x1.append(l)
    x=np.array(train_x1)
    return x

#simply creates an array of y hats
def set_Expected(batch_size, arr):
    expected_val=[]
    for i in range(batch_size):
        expected_val.append(arr)
    y=np.array(expected_val)
    return y

#Initialize training value
def get_training_Data():
    EX=[]
    EX.append(training_Data("digit_train_0.txt"))
    EX.append(training_Data("digit_train_1.txt"))
    EX.append(training_Data("digit_train_2.txt"))
    EX.append(training_Data("digit_train_3.txt"))
    EX.append(training_Data("digit_train_4.txt"))
    EX.append(training_Data("digit_train_5.txt"))
    EX.append(training_Data("digit_train_6.txt"))
    EX.append(training_Data("digit_train_7.txt"))
    EX.append(training_Data("digit_train_8.txt"))
    EX.append(training_Data("digit_train_9.txt"))
    return EX
#Initialize expected_data    
def get_expected_Data( batch_size):
    y=[]
    y.append(set_Expected(batch_size,[0,0,0,0,0,0,0,0,0,1]))
    y.append(set_Expected(batch_size,[0,0,0,0,0,0,0,0,1,0]))
    y.append(set_Expected(batch_size,[0,0,0,0,0,0,0,1,0,0]))
    y.append(set_Expected(batch_size,[0,0,0,0,0,0,1,0,0,0]))
    y.append(set_Expected(batch_size,[0,0,0,0,0,1,0,0,0,0]))
    y.append(set_Expected(batch_size,[0,0,0,0,1,0,0,0,0,0]))
    y.append(set_Expected(batch_size,[0,0,0,1,0,0,0,0,0,0]))
    y.append(set_Expected(batch_size,[0,0,1,0,0,0,0,0,0,0]))
    y.append(set_Expected(batch_size,[0,1,0,0,0,0,0,0,0,0]))
    y.append(set_Expected(batch_size,[1,0,0,0,0,0,0,0,0,0]))
    return y
