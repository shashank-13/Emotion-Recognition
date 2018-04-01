import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import dlib
import math
import random
from sklearn.utils import shuffle

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise","neutral"] #Define emotions

dir1='F:\Chaos\EmotionRecognition\Application\\trainData'


def shuffle_files(emotion):

    files = glob.glob(dir1+"\\"+emotion+"\*")
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def forward(X,W1,b1,W2,b2):   #For forward propagation

    Z= 1/(1+np.exp(-X.dot(W1)-b1))
    A=Z.dot(W2)+b2
    expA= np.exp(A)
    Y=expA/expA.sum(axis=1,keepdims=True)
    return Y,Z                  # Return output and hidden units


def score(Y,P):         #Classification score

    n_correct = 0
    n_total = 0

    for i in range(len(Y)):
        if(Y[i] == P[i]):
            n_correct+=1
        n_total+=1

    return n_correct/n_total


def derivative_w2(Z,T,Y):

    return Z.T.dot(T-Y)

def derivative_w1(X,W2,Z,T,Y):

    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    ret2 = X.T.dot(dZ)
    return ret2

def derivative_b2(T,Y):

    return ((T-Y)).sum(axis =0)

def derivative_b1(T,Y,W2,Z):

    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis = 0)

def cost(T,Y):

    return (T*np.log(Y)).sum()



def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def get_Data():                     # Get Data of the network for training and prediction

    training_data=[]
    training_labels=[]
    prediction_data=[]
    prediction_labels=[]

    for emotion in emotions:

        training,prediction = shuffle_files(emotion)

        for item in training:
            image = cv2.imread(item,0)
            image = np.array(image).flatten()
            training_data.append(image)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item,0)
            image = np.array(image).flatten()
            prediction_data.append(image)
            prediction_labels.append(emotions.index(emotion))


    return training_data,training_labels,prediction_data,prediction_labels



def fit_Full():

    X=[]
    Y=[]
    prediction_data=[]
    prediction_labels = []

    X,Y,prediction_data,prediction_labels = get_Data()

    print ('Train data length ={0} , Test data length ={1}'.format(len(X),len(prediction_data)))
    #print(X)
    X=np.array(X)/255.0
    Y=np.array(Y)

    prediction_data = np.array(prediction_data)/255.0
    prediction_labels = np.array(prediction_labels)





    N,D = X.shape

    #print ('Dshape {0}'.format(D))
    M=100
    K=7

    T=np.zeros((N,K))

    for i in range(N):
        T[i,Y[i]] =1



    # randomly initialize weights
    W1 = np.random.randn(D, M) / np.sqrt(D+M)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M+K)
    b2 = np.zeros(K)

    learning_rate = 5*10e-7
    costs=[]
    best_validation_error = 1

    #Use of Momentum
    mu = 0.9
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0
    reg=0.01

    #rms propagation
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1
    decay_rate = 0.999
    eps = 1e-10

    for i in range(1000):

        output,hidden = forward(X,W1,b1,W2,b2)


        gW2= derivative_w2(hidden,T,output) +reg*W2
        gb1= derivative_b1(T,output,W2,hidden) + reg*b1
        gW1 = derivative_w1(X,W2,hidden,T,output) + reg*W1
        gb2 = derivative_b2(T,output) + reg*b2

        dW1 = dW1*mu + learning_rate*gW1
        dW2= dW2*mu + learning_rate*gW2
        db1 = db1*mu + learning_rate*gb1
        db2 = db2*mu +learning_rate*gb2

        #cache calculation
        cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
        cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
        cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
        cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1




        W2 += (dW2/(np.sqrt(cache_W2)+eps))
        b1 += (db1/(np.sqrt(cache_b1)+eps))
        W1 += (dW1/(np.sqrt(cache_W1)+eps))
        b2 += (db2/(np.sqrt(cache_b2)+eps))




        if i%50==0:
            c=cost(T,output)
            P=np.argmax(output,axis=1)
            r=score(Y,P)
            e = error_rate(Y,P)
            if e < best_validation_error:
                best_validation_error = e
            print ('Cost = {0} , Training Score = {1} ,Error ={2}'.format(c,r,e))
            costs.append(c)

    print("best_validation_error:", best_validation_error)

    plt.plot(costs)
    plt.show()

    # Test npar_train

    output , hidden = forward(prediction_data,W1,b1,W2,b2)
    P=np.argmax(output,axis =1)
    print ('Test Score {0}'.format(score(prediction_labels,P)))  # Final output




    #np.savetxt(os.path.join(temp_Dir,'weight1'),)

def fit_batch():

    X=[]
    Y=[]
    prediction_data=[]
    prediction_labels = []

    X,Y,prediction_data,prediction_labels = get_Data()

    print ('Train data length ={0} , Test data length ={1}'.format(len(X),len(prediction_data)))
    #print(X)
    X=np.array(X)/255.0
    Y=np.array(Y)

    prediction_data = np.array(prediction_data)/255.0
    prediction_labels = np.array(prediction_labels)

    N,D = X.shape


    #print ('Dshape {0}'.format(D))
    M=100
    K=7

    # randomly initialize weights
    W1 = np.random.randn(D, M) / np.sqrt(D+M)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M+K)
    b2 = np.zeros(K)

    learning_rate = 5*10e-7
    costs=[]
    best_validation_error = 1

    batch_sz = 220
    n_batches = int(N/batch_sz)

    #Use of Momentum
    mu = 0.9
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0
    reg=0.01

    #rms propagation
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1
    decay_rate = 0.999
    eps = 1e-10

    #print ('{0} {1} {2}'.format(N,n_batches,D))

    for m in range(1000):
        tmpX,tmpY=shuffle(X,Y)
        for j in range(n_batches):

            x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
            y =tmpY[j*batch_sz:(j*batch_sz + batch_sz)]

            n=len(x) # Getting new length

            T=np.zeros((n,K))
            for i in range(n):
                T[i,y[i]] =1

            output,hidden = forward(x,W1,b1,W2,b2)

            gW2= derivative_w2(hidden,T,output) +reg*W2
            gb1= derivative_b1(T,output,W2,hidden) + reg*b1
            gW1 = derivative_w1(x,W2,hidden,T,output) + reg*W1
            gb2 = derivative_b2(T,output) + reg*b2

            dW1 = dW1*mu + learning_rate*gW1
            dW2= dW2*mu + learning_rate*gW2
            db1 = db1*mu + learning_rate*gb1
            db2 = db2*mu +learning_rate*gb2

            #cache calculation
            cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
            cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
            cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
            cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1




            W2 += (dW2/(np.sqrt(cache_W2)+eps))
            b1 += (db1/(np.sqrt(cache_b1)+eps))
            W1 += (dW1/(np.sqrt(cache_W1)+eps))
            b2 += (db2/(np.sqrt(cache_b2)+eps))


            #print('i value ={0}'.format(i))

            if m%50==0:
                c=cost(T,output)
                P=np.argmax(output,axis=1)
                r=score(y,P)
                e = error_rate(y,P)
                if e < best_validation_error:
                    best_validation_error = e
                print ('Cost = {0} , Training Score = {1} ,Error ={2}'.format(c,r,e))
                costs.append(c)

    print("best_validation_error:", best_validation_error)

    plt.plot(costs)
    plt.show()

    # Test npar_train

    output , hidden = forward(prediction_data,W1,b1,W2,b2)
    P=np.argmax(output,axis =1)
    print ('Test Score {0}'.format(score(prediction_labels,P)))  # Final output

if __name__ == '__main__':
    fit_Full()      #Test using full gradient descent
    #fit_batch()     #Test using batch gradient descent
