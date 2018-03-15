#Test Neural Network using Theano
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from main import get_Data

def error_rate(p,t):
    return np.mean(p!=t)

def relu(a):
    return a*(a>0)

def fit_theano():


    #General Set Up
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

    T_y=np.zeros((N,K))

    for i in range(N):
        T_y[i,Y[i]] =1

    n,d = prediction_data.shape
    T_x= np.zeros((n,K))

    for i in range(n):
        T_x[i,prediction_labels[i]]=1



    # randomly initialize weights
    W1_init = np.random.randn(D, M) / np.sqrt(D+M)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M+K)
    b2_init = np.zeros(K)

    learning_rate = 5*10e-7
    costs=[]
    best_validation_error = 1
    reg=0.01

    #Theano SetUp
    thX=T.matrix('X')
    thT=T.matrix('T')
    W1=theano.shared(W1_init,'W1')
    b1=theano.shared(b1_init,'b1')
    W2=theano.shared(W2_init,'W2')
    b2=theano.shared(b2_init,'b2')

    thZ = relu( thX.dot(W1) + b1 )
    thY = T.nnet.softmax( thZ.dot(W2) + b2 )

    # define the cost function and prediction
    cost = -(thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())
    prediction = T.argmax(thY, axis=1)

    update_W1 = W1 - learning_rate*T.grad(cost, W1)
    update_b1 = b1 - learning_rate*T.grad(cost, b1)
    update_W2 = W2 - learning_rate*T.grad(cost, W2)
    update_b2 = b2 - learning_rate*T.grad(cost, b2)

    train = theano.function(
        inputs=[thX, thT],
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)],
    )

    # create another function for this because we want it over the whole dataset
    get_prediction = theano.function(
        inputs=[thX, thT],
        outputs=[cost, prediction],
    )

    #Run function
    costs=[]
    for i in range(500):

        train(X,T_y)

        if i%50==0:

            #Testing on test set instead of usual train set
            '''c, prediction_val = get_prediction(prediction_data,T_x)
            e = error_rate(prediction_val, prediction_labels)'''
            #Testing on usual train Set
            c, prediction_val = get_prediction(X,T_y)
            e = error_rate(prediction_val, Y)
            if e < best_validation_error:
                best_validation_error = e
            print ('Cost = {0} , Training Score = {1} ,Error ={2}'.format(c,1-e,e))
            costs.append(c)

    #print("best_validation_error:", best_validation_error)

    plt.plot(costs)
    plt.show()
    #Testing on test set
    c, prediction_val = get_prediction(prediction_data,T_x)
    e = error_rate(prediction_val, prediction_labels)
    print ('Cost = {0} , Test Score = {1} ,Error ={2}'.format(c,1-e,e))
# Test npar_train

if __name__ == '__main__':
    fit_theano()  #Training using theano
