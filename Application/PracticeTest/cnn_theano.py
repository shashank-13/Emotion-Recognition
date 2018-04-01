#import all the stuffs
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import random
import glob
import os
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime
import cv2

from theano import function

#Loading Functions

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise","neutral"] #Define emotions

dir1='F:\Chaos\EmotionRecognition\Application\sorted_set'

def shuffle_files(emotion):

    files = glob.glob(dir1+"\\"+emotion+"\*")
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

#Function to rearrange data
def rearrange(X):

    N,H,W,C=X.shape
    out = np.zeros((N,C,H,W), dtype=np.float32)
    for i in range(N):
        for j in range(C):
            out[:,j,:,:] = X[:,:,:,j]

    return out/255.0


def get_Data():                     # Get Data of the network for training and prediction

    training_data=[]
    training_labels=[]
    prediction_data=[]
    prediction_labels=[]

    for emotion in emotions:

        training,prediction = shuffle_files(emotion)

        for item in training:
            image = cv2.imread(item)
            training_data.append(image)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            prediction_data.append(image)
            prediction_labels.append(emotions.index(emotion))


    return training_data,training_labels,prediction_data,prediction_labels


def error_rate(p,t):
    return np.mean(p!=t)

def relu(a):
    return a*(a>0)

def init_filter(shape,poolsz):
    W=np.random.randn(*shape) * np.sqrt(2.0/np.prod(shape[1:]))
    return W.astype(np.float32)

#Convolution and Pooling
def convpool(X,W,b,poolsize=(2,2)):

    conv_out = conv2d(input=X,filters=W)
    pooled_out = pool.pool_2d(
        input = conv_out,
        ds=poolsize,
        ignore_border=True
    )

    return T.tanh(pooled_out)

def fit_cNN():

    X_train=[]
    Y_train=[]
    prediction_Data=[]
    prediction_labels=[]

    X_train,Y_train,prediction_Data,prediction_labels = get_Data()

    X_train = rearrange(np.array(X_train))
    Y_train = np.array(Y_train)


    prediction_Data = rearrange(np.array(prediction_Data))
    prediction_labels = np.array(prediction_labels)

    N,_,_,_ = X_train.shape
    M=100
    K=7
    constant = 4050

    #Setting Hyper-Parameters
    learning_rate = np.float32(5*10e-7)
    mu = np.float32(0.9)
    decay_rate = np.float32(0.999)
    reg = np.float32(0.01)
    pool_size=(2,2)

    W1_shape =(20,3,5,5)
    W1_init = init_filter(W1_shape,pool_size)
    b1_init = np.zeros(W1_shape[0],dtype=np.float32)

    W2_shape = (50,20,5,5)
    W2_init = init_filter(W2_shape,pool_size)
    b2_init = np.zeros(W2_shape[0],dtype=np.float32)

    W3_init = np.random.randn(W2_shape[0]*5*5,M)/np.sqrt(W2_shape[0]*5*5 +M)
    b3_init = np.zeros(M,np.float32)

    W4_init = np.random.randn(M,K) / np.sqrt(M+K)
    b4_init = np.zeros(K,np.float32)

    X=T.tensor4('X',dtype='float32')
    Y=T.ivector('T')

    print (X.shape.eval({X: X_train}))

    W1=theano.shared(W1_init,'W1')
    b1=theano.shared(b1_init,'b1')

    W2=theano.shared(W2_init,'W2')
    b2=theano.shared(b2_init,'b2')

    W3=theano.shared(W3_init.astype(np.float32),'W3')
    b3=theano.shared(b3_init,'b3')

    W4=theano.shared(W4_init.astype(np.float32),'W4')
    b4=theano.shared(b4_init,'b4')

    dW1 = theano.shared(np.zeros(W1_init.shape,dtype=np.float32),'dW1')
    db1 = theano.shared(np.zeros(b1_init.shape,dtype=np.float32),'db1')

    dW2 = theano.shared(np.zeros(W2_init.shape,dtype=np.float32),'dW2')
    db2 = theano.shared(np.zeros(b2_init.shape,dtype=np.float32),'db2')

    dW3 = theano.shared(np.zeros(W3_init.shape,dtype=np.float32),'dW3')
    db3 = theano.shared(np.zeros(b3_init.shape,dtype=np.float32),'db3')

    dW4 = theano.shared(np.zeros(W4_init.shape,dtype=np.float32),'dW4')
    db4 = theano.shared(np.zeros(b4_init.shape,dtype=np.float32),'db4')

    print ('\n\n\n')

    print ('Train Input ={0} ,Train Output ={1} , Test Input ={2} ,Test Output ={3}'.format(X_train.shape,Y_train.shape,prediction_Data.shape,prediction_labels.shape))

    print ('W1 ={0} ,W2 ={1} , W3 ={2}, W4 ={3}'.format(W1_init.shape,W2_init.shape,W3_init.shape,W4_init.shape))

    print ('b1 ={0} ,b2 ={1} , b3={2} , b4 ={3}'.format(b1_init.shape,b2_init.shape,b3_init.shape,b4_init.shape))

    print ('\n\n\n')
    Z1=convpool(X_train,W1_init,b1_init)
    Z2=convpool(Z1,W2_init,b2_init)


    #print ('{0} {1} {2}'.format(Z2.shape,W3_init.shape,b3_init))
    Z3=relu(Z2.flatten(ndim=2).dot(W3)+b3)
    pY=T.nnet.softmax(Z3.dot(W4)+b4)

    '''params = (W1,b1,W2,b2,W3,b3,W4,b4)
    reg_cost = reg*np.sum((param*param).sum() for param in params)
    cost = -(Y * T.log(pY)).sum() + reg_cost
    prediction = T.argmax(pY,axis=1)

    update_W1 = W1 + (mu*dW1) -(learning_rate*T.grad(cost,W1))
    update_b1 = b1 + (mu*db1) -(learning_rate*T.grad(cost,b1))

    update_W2 = W2+ (mu*dW2) -(learning_rate*T.grad(cost,W2))
    update_b2 = b2 + (mu*db2) -(learning_rate*T.grad(cost,b2))

    update_W3 = W3 + (mu*dW3) -(learning_rate*T.grad(cost,W3))
    update_b3 = b3+ (mu*db3) -(learning_rate*T.grad(cost,b3))

    update_W4 = W4 + (mu*dW4) -(learning_rate*T.grad(cost,W4))
    update_b4 = b4 + (mu*db4) -(learning_rate*T.grad(cost,b4))

    update_dW1 = mu*dW1 - (learning_rate*T.grad(cost,W1))
    update_db1 = mu*db1 - (learning_rate*T.grad(cost,b1))

    update_dW2 = mu*dW2 - (learning_rate*T.grad(cost,W2))
    update_db2 = mu*db2 - (learning_rate*T.grad(cost,b2))

    update_dW3 = mu*dW3 - (learning_rate*T.grad(cost,W3))
    update_db3 = mu*db3 - (learning_rate*T.grad(cost,b3))

    update_dW4 = mu*dW4 - (learning_rate*T.grad(cost,W4))
    update_db4 = mu*db4 - (learning_rate*T.grad(cost,b4))

    #For training and Prediction Function
    train = theano.function(
        inputs = [X,Y],
        updates = [
            (W1,update_W1),
            (b1,update_b1),
            (W2,update_W2),
            (b2,update_b2),
            (W3,update_W3),
            (b3,update_b3),
            (W4,update_W4),
            (b4,update_b4),
            (dW1,update_dW1),
            (db1,update_db1),
            (dW2,update_dW2),
            (db2,update_db2),
            (dW3,update_dW3),
            (db3,update_db3),
            (dW4,update_dW4),
            (db4,update_db4),
        ]
    )

    get_prediction = theano.function(
        inputs =[X,Y],
        outputs = [cost,prediction],
    )


    #Run function
    costs=[]

    for i in range(500):

        train(X_train,Y_train)

        if i%50==0:

            #Testing on usual train Set
            c, prediction_val = get_prediction(X_train,Y_train)
            e = error_rate(prediction_val,Y_train)
            if e < best_validation_error:
                best_validation_error = e
            print ('Cost = {0} , Training Score = {1} ,Error ={2}'.format(c,1-e,e))
            costs.append(c)

    print("best_validation_error:", best_validation_error)

    # Test npar_train
    plt.plot(costs)
    plt.show()
    #Testing on test set
    c, prediction_val = get_prediction(prediction_data,prediction_labels)
    e = error_rate(prediction_val, prediction_labels)
    print ('Cost = {0} , Test Score = {1} ,Error ={2}'.format(c,1-e,e))'''


#Call CNN Train Method
fit_cNN()
