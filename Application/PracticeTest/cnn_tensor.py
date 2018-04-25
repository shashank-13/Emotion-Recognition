#import all the stuffs
import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d
from sklearn.utils import shuffle
import random
import glob
import os
import cv2
import matplotlib.pyplot as plt

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise","neutral"] #Define emotions
new_emotions = ["anger","happy", "neutral", "sadness"] #Define emotions

dir1='F:\Chaos\EmotionRecognition\Application\\sorted_set'
dest='F:\Chaos\EmotionRecognition\Application\Results'

def shuffle_files(emotion):

    files = glob.glob(dir1+"\\"+emotion+"\*")
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction


def error_rate(p,t):
    return np.mean(p!=t)

def convpool(X,W,b):
    conv_out = tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')
    conv_out = tf.nn.bias_add(conv_out,b)
    pool_out = tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return tf.nn.relu(pool_out)

def init_filter(shape,poolsz):
    w=np.random.randn(*shape) * np.sqrt(2.0/np.prod(shape[:-1]))
    return w.astype(np.float32)

def get_Data():                     # Get Data of the network for training and prediction

    training_data=[]
    training_labels=[]
    prediction_data=[]
    prediction_labels=[]

    for emotion in new_emotions:

        training,prediction = shuffle_files(emotion)

        for item in training:
            image = cv2.imread(item)
            training_data.append(image)
            training_labels.append(new_emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            prediction_data.append(image)
            prediction_labels.append(new_emotions.index(emotion))


    return training_data,training_labels,prediction_data,prediction_labels

def fit_tensor():

    X_train=[]
    Y_train=[]
    prediction_data=[]
    prediction_labels = []

    X_train,Y_train,prediction_data,prediction_labels = get_Data()

    print ('Train-Data Length {0} , Test-Data Length {1}'.format(len(X_train),len(prediction_data)))

    X_train=np.array(X_train)/255.0
    Y_train=np.array(Y_train)

    prediction_data = np.array(prediction_data)/255.0
    prediction_labels = np.array(prediction_labels)


    N,_,_,_ = X_train.shape

    #print ('Dshape {0}'.format(D))
    M=100
    K=7

    T_y=np.zeros((N,K))

    for i in range(N):
        T_y[i,Y_train[i]] =1

    n,_,_,_ = prediction_data.shape
    T_x= np.zeros((n,K))

    for i in range(n):
        T_x[i,prediction_labels[i]]=1

    poolsz=(2,2)

    constant = 7200

    W1_shape = (5,5,3,20)
    W1_init = init_filter(W1_shape,poolsz)
    b1_init = np.zeros(W1_shape[-1],dtype=np.float32)

    W2_shape=(5,5,20,50)
    W2_init = init_filter(W2_shape,poolsz)
    b2_init = np.zeros(W2_shape[-1],dtype=np.float32)

    W3_init = np.random.randn(constant,M) / np.sqrt(constant + M)
    b3_init = np.zeros(M,dtype=np.float32)

    W4_init = np.random.randn(M,K) / np.sqrt(M+K)
    b4_init = np.zeros(K,dtype=np.float32)

    X = tf.placeholder(tf.float32,shape=(n,48,48,3),name='X')
    T = tf.placeholder(tf.float32,shape=(n,K),name='T')

    W1=tf.Variable(W1_init.astype(np.float32))
    b1=tf.Variable(b1_init.astype(np.float32))

    W2=tf.Variable(W2_init.astype(np.float32))
    b2=tf.Variable(b2_init.astype(np.float32))

    W3=tf.Variable(W3_init.astype(np.float32))
    b3=tf.Variable(b3_init.astype(np.float32))

    W4=tf.Variable(W4_init.astype(np.float32))
    b4=tf.Variable(b4_init.astype(np.float32))

    Z1 = convpool(X,W1,b1)
    Z2 = convpool(Z1,W2,b2)
    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2,[Z2_shape[0],np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu(tf.matmul(Z2r,W3)+b3)
    Yish = tf.matmul(Z3,W4) +b4

    cost = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=Yish,
            labels=T
        )
    )

    train_op = tf.train.RMSPropOptimizer(5*10e-6,decay=0.99,momentum=0.9).minimize(cost)

    predict_op = tf.argmax(Yish,1)

    costs = []
    init = tf.global_variables_initializer()
    max_iter=151
    print_period=50

    batch_sz=n
    n_batches = N/batch_sz
    count=0


    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            j=0
            session.run(train_op, feed_dict={X:X_train[j*batch_sz:(j*batch_sz +batch_sz),] , T:T_y[j*batch_sz:(j*batch_sz +batch_sz),]})
            j=j+1
            session.run(train_op, feed_dict={X:X_train[j*batch_sz:(j*batch_sz +batch_sz),] , T:T_y[j*batch_sz:(j*batch_sz +batch_sz),]})
            j=j+1
            session.run(train_op, feed_dict={X:X_train[j*batch_sz:(j*batch_sz +batch_sz),] , T:T_y[j*batch_sz:(j*batch_sz +batch_sz),]})
            j=j+1
            session.run(train_op, feed_dict={X:X_train[j*batch_sz:(j*batch_sz +batch_sz),] , T:T_y[j*batch_sz:(j*batch_sz +batch_sz),]})
            if (i>0 and (i % print_period == 0)):

                test_cost = session.run(cost, feed_dict={X:X_train[count*batch_sz:(count*batch_sz +batch_sz),] , T:T_y[count*batch_sz:(count*batch_sz +batch_sz),]})
                prediction = session.run(predict_op, feed_dict={X:X_train[count*batch_sz:(count*batch_sz +batch_sz),]})
                err = error_rate(prediction,Y_train[count*batch_sz:(count*batch_sz +batch_sz),]) #Testing on Train test
                print("Iteration ={0}, Cost ={1}, Score ={2}".format(i,test_cost, 1-err))
                costs.append(test_cost)
                count = count+1

        #Prediction on test Set
        test_cost = session.run(cost, feed_dict={X: prediction_data, T: T_x})
        prediction = session.run(predict_op, feed_dict={X: prediction_data})
        err = error_rate(prediction, prediction_labels)

        for i in range(len(prediction)):
            cv2.imwrite(os.path.join(dest,new_emotions[prediction_labels[i]],new_emotions[prediction[i]]+str(i)+'.jpg'),prediction_data[i]*255.0)

        print("Test Results : , Cost ={0}, Score ={1}".format(test_cost, 1-err))

    plt.plot(costs)
    plt.show()

#Caal fit_tensor
fit_tensor()
