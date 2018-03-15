import sys
import numpy as np
sys.path.insert(0,r'F:\\Chaos\\EmotionRecognition\\Application\\Neural Networks')  #Adding to the path
from main import fit,get_Data,forward,error_rate,cost,derivative_b1,derivative_b2,derivative_w1,derivative_w2,score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def batch_fit():

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

    batch_sz = 500
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

            if j%(n_batches/2)==0:
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
    batch_fit()    #Using batch gradient descent
    #fit()          #Using full gradient descent
