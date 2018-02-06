import numpy as np
import scipy as sc
from digit_Recognition.NeuralNet import NeuralNet
from digit_Recognition import program

def begin(m_layer,rWeight,epoch, isBias, batch_size):
    print ("Begin: ")
    
    # randomly initialize our weights with mean 0
    #begin neural net here
    #initialize its weights, all values
    
    EX = program.get_training_Data()
    y = program.get_expected_Data(batch_size)
    
    #Initialize NeuralNet
    nn = NeuralNet(m_layer, rWeight, batch_size)
    
    #If isBias is true then bias is initialized to random
    #otherwise it is zero
    if (isBias==True):
        bias0=np.random.rand(1,m_layer)
        bias1=np.random.rand(1,10)
    else:
        bias0=np.zeros(m_layer)
        bias1=np.zeros(10)
        
    #train using batch size of random examples from example list
    start=0
    end=batch_size
        
    for j in range(epoch):
        if (j%(700/batch_size)==0):
            #get random batch from data set
            for i in range(0,10):
                np.random.shuffle(EX[i])
            start=0
            end=batch_size
        #once data set is retrieved feed forward and then use backpropagation
        err = []    
        for i in range(0,10):
            X = EX[i][start:end,]
            err.append(nn.back_Propagation(X, y[i], bias0, bias1, isBias))
            
        start=start+batch_size
        end=end+batch_size
 
        #print error after every 100 iterations
        if (j% 100) == 0:
            for x in range(0,10):
                print ("Error ",j,":" + str(np.mean(np.abs(err[x]))))
             
             
             
    #Testing
    #**********************************************P2*************************************************************************
                  
    test1=program.testing_Data("digit_test_0.txt")  
    test2=program.testing_Data("digit_test_1.txt")
    test3=program.testing_Data("digit_test_2.txt")  
    test4=program.testing_Data("digit_test_3.txt")
    test5=program.testing_Data("digit_test_4.txt")  
    test6=program.testing_Data("digit_test_5.txt")
    test7=program.testing_Data("digit_test_6.txt")  
    test8=program.testing_Data("digit_test_7.txt")
    test9=program.testing_Data("digit_test_8.txt")  
    test10=program.testing_Data("digit_test_9.txt")     
    #2, then 6
    print (np.around(nn.feed_Forward(test1[100]),5))
    print (np.around(nn.feed_Forward(test2[188]),5))
    print (np.around(nn.feed_Forward(test3[14]),5))
    print (np.around(nn.feed_Forward(test4[10]),5))
    print (np.around(nn.feed_Forward(test5[10]),5))
    print (np.around(nn.feed_Forward(test6[188]),5))
    print (np.around(nn.feed_Forward(test7[14]),5))
    print (np.around(nn.feed_Forward(test8[10]),5))
    
    
#nodes in layer, weight initial range, epochs, bias ,batch size            
begin(40,1,5000,True,28)



