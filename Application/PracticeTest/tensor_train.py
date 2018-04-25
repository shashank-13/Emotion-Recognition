#Test Neural Network using Theano
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import cv2
import dlib
import math
import random
from sklearn.utils import shuffle

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise","neutral"] #Define emotions
new_emotions = ["anger","happy", "neutral", "sadness"] #Define emotions
dir1='F:\Chaos\EmotionRecognition\Application\\sorted_set'

def error_rate(p,t):
    return np.mean(p!=t)

def shuffle_files(emotion):

    files = glob.glob(dir1+"\\"+emotion+"\*")
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def get_Data():                     # Get Data of the network for training and prediction

    training_data=[]
    training_labels=[]
    prediction_data=[]
    prediction_labels=[]

    for emotion in new_emotions:

        training,prediction = shuffle_files(emotion)

        for item in training:
            image = cv2.imread(item,0)
            image = np.array(image).flatten()
            training_data.append(image)
            training_labels.append(new_emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item,0)
            image = np.array(image).flatten()
            prediction_data.append(image)
            prediction_labels.append(new_emotions.index(emotion))


    return training_data,training_labels,prediction_data,prediction_labels

def fit_Tensor():
    #General Set Up
    X_train=[]
    Y_train=[]
    prediction_data=[]
    prediction_labels = []

    X_train,Y_train,prediction_data,prediction_labels = get_Data()

    print ('Train data length ={0} , Test data length ={1}'.format(len(X_train),len(prediction_data)))
    #print(X)
    X_train=np.array(X_train)/255.0
    Y_train=np.array(Y_train)

    prediction_data = np.array(prediction_data)/255.0
    prediction_labels = np.array(prediction_labels)

    N,D = X_train.shape

    #print ('Dshape {0}'.format(D))
    M=100
    K=7

    T_y=np.zeros((N,K))

    for i in range(N):
        T_y[i,Y_train[i]] =1

    n,d = prediction_data.shape
    T_x= np.zeros((n,K))

    for i in range(n):
        T_x[i,prediction_labels[i]]=1


    learning_rate = 5*10e-7
    mu = 0.9
    decay_rate = 0.999

    # randomly initialize weights
    W1_init = np.random.randn(D, M) / np.sqrt(D+M)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M+K)
    b2_init = np.zeros(K)

    # define variables and expressions
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')

    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))


    tZ= tf.nn.relu(tf.matmul(X,W1)+b1)
    tY=tf.matmul(tZ,W2)+b2

    #Cost function
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tY, labels=T))

    #Operation
    train_op = tf.train.RMSPropOptimizer(learning_rate,decay_rate,mu).minimize(cost)

    predict_op = tf.argmax(tY, 1)

    costs = []
    init = tf.global_variables_initializer()
    max_iter=2000
    print_period=50

    #For saving purpose
    saver = tf.train.Saver()

    with tf.Session() as session:

        saver.restore(session,'/PracticeTest/new_tensor_model.ckpt')
        '''session.run(init)

        for i in range(max_iter):
            session.run(train_op, feed_dict={X: X_train, T: T_y})
            if i % print_period == 0:
                test_cost = session.run(cost, feed_dict={X: X_train, T: T_y})
                prediction = session.run(predict_op, feed_dict={X: X_train})
                err = error_rate(prediction, Y_train) #Testing on Train test
                print("Iteration ={0}, Cost ={1}, Score ={2}".format(i,test_cost, 1-err))
                costs.append(test_cost)'''

        #Prediction on test Set
        test_cost = session.run(cost, feed_dict={X: prediction_data, T: T_x})
        prediction = session.run(predict_op, feed_dict={X: prediction_data})
        err = error_rate(prediction, prediction_labels)

        print("Test Results : , Cost ={0}, Score ={1}".format(test_cost, 1-err))
        '''save_path = saver.save(session,'/PracticeTest/new_tensor_model.ckpt')
        print ('Model saved in path {0}'.format(save_path))'''

    '''plt.plot(costs)
    plt.show()'''

if __name__ == '__main__':
    fit_Tensor()
