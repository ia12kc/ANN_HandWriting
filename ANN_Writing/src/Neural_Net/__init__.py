import numpy as np
import scipy as sc

        
def training_Data(file):#data_X1, data_X2):
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

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
    
def forwardPropagate(nums,syn0,syn1):
    layer1 = nonlin(np.dot(nums,syn0))
    output = nonlin(np.dot(layer1,syn1))
    return output

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
 
def set_Expected(batch_size,arr):
    expected_val=[]
    for i in range(batch_size):
        expected_val.append(arr)
    y=np.array(expected_val)
    return y

def begin(m_layer,rWeight,epoch, b,batch_size):
    print ("Begin: ")
    #Initialize training_data
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
    
    
    
    #Initialize input values
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
    
    # randomly initialize our weights with mean 0
    syn0 = np.random.random((64,m_layer)) - rWeight
    syn1 = np.random.random((m_layer,10)) - rWeight
    

    if (b==True):
        bias0=np.random.rand(1,m_layer)
        bias1=np.random.rand(1,10)
    else:
        bias0=np.zeros(m_layer)
        bias1=np.zeros(10)
    start=0
    end=batch_size
        
    def back_Prop(X, y,syn0,syn1,bias0,bias1):
        l1 = nonlin(np.dot(X,syn0)+bias0)
        l2 = nonlin(np.dot(l1,syn1)+bias1)

        l2_error = y - l2
    
        l2_delta = l2_error*nonlin(l2,deriv=True)
        
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)
        if (b==True):
            bias1=bias1+l2_delta
            bias0=bias0+l1_delta
        
        syn0 += X.T.dot(l1_delta)
        syn1 += l1.T.dot(l2_delta) 
        return l2_error
        
        
    for j in range(epoch):
        if (j%(700/batch_size)==0):
            for i in range(0,10):
                np.random.shuffle(EX[i])
            start=0
            end=batch_size
        
        err=[]    
        for i in range(0,10):
            X=EX[i][start:end,]
            err.append(back_Prop(X,y[i],syn0,syn1,bias0,bias1))
            
        start=start+batch_size
        end=end+batch_size
        # Feed forward through layers 0, 1, and 2    
        
        #**********************************************P2*************************************
    
        
        if (j% 99) == 0:
            for x in range(0,10):
                print ("Error ",j,":" + str(np.mean(np.abs(err[x]))))
            
            
    #Testing several random#       
    test1=testing_Data("digit_test_0.txt")  
    test2=testing_Data("digit_test_1.txt")
    test3=testing_Data("digit_test_2.txt")  
    test4=testing_Data("digit_test_3.txt")
    test5=testing_Data("digit_test_4.txt")  
    test6=testing_Data("digit_test_5.txt")
    test7=testing_Data("digit_test_6.txt")  
    test8=testing_Data("digit_test_7.txt")
    test9=testing_Data("digit_test_8.txt")  
    test10=testing_Data("digit_test_9.txt")     
    #2, then 6
    print (np.around(forwardPropagate(test1[100],syn0,syn1),5))
    print (np.around(forwardPropagate(test2[188],syn0,syn1),5))
    print (np.around(forwardPropagate(test3[14],syn0,syn1),5))
    print (np.around(forwardPropagate(test4[10],syn0,syn1),5))
    print (np.around(forwardPropagate(test5[10],syn0,syn1),5))
    print (np.around(forwardPropagate(test6[188],syn0,syn1),5))
    print (np.around(forwardPropagate(test7[14],syn0,syn1),5))
    print (np.around(forwardPropagate(test8[10],syn0,syn1),5))
            
begin(40,1,5000,True,28)
#nodes in layer, weight initial range, epochs, bias ,batch size #


