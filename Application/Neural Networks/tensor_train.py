#Test Neural Network using Theano
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from main import get_Data

def error_rate(p,t):
    return np.mean(p!=t)


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
    max_iter=500
    print_period=50

    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            session.run(train_op, feed_dict={X: X_train, T: T_y})
            if i % print_period == 0:
                '''test_cost = session.run(cost, feed_dict={X: prediction_data, T: T_x})
                prediction = session.run(predict_op, feed_dict={X: prediction_data})
                err = error_rate(prediction, prediction_labels)#Testing on Test test'''
                test_cost = session.run(cost, feed_dict={X: X_train, T: T_y})
                prediction = session.run(predict_op, feed_dict={X: X_train})
                err = error_rate(prediction, Y_train) #Testing on Train test
                print("Iteration ={0}, Cost ={1}, Score ={2}".format(i,test_cost, 1-err))
                costs.append(test_cost)

        #Prediction on test Set
        test_cost = session.run(cost, feed_dict={X: prediction_data, T: T_x})
        prediction = session.run(predict_op, feed_dict={X: prediction_data})
        err = error_rate(prediction, prediction_labels)

        print("Test Results : , Cost ={0}, Score ={1}".format(test_cost, 1-err))

    plt.plot(costs)
    plt.show()









if __name__ == '__main__':
    fit_Tensor()
