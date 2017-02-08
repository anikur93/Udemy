import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

def y2indicator(y):
    N = len(y)
    n = np.zeros((N,10))
    for i in range(N):
        n[i,y[i]] = 1
    return n

def error_rate(p,t):
    return np.mean(p != t)
    
def flatten(x):
    N = x.shape[-1]
    #L = x.shape[1]*x.shape[2]*x.shape[3]
    flat = np.zeros((N, 3072))
    for i in range(N):
        flat[i] = x[:,:,:,i].reshape(3072)
    return flat
    
def main():
    train = loadmat('train_32x32.mat')
    test = loadmat('test_32x32.mat')
    
    xtrain = flatten(train['X'].astype(np.float32)/255)
    ytrain = train['y'].flatten() - 1
    xtrain, ytrain = shuffle(xtrain, ytrain)
    ytrain_ind = y2indicator(ytrain)

    xtest  = flatten(test['X'].astype(np.float32) / 255)
    ytest  = test['y'].flatten() - 1
    ytest_ind  = y2indicator(ytest) 

    max_iter = 20
    print_period = 10
    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N / batch_sz

    # initial weights
    M1 = 100 # hidden layer size
    M2 = 50
    K = 10
    W1_init = np.random.randn(D, M1) / np.sqrt(D + M1)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2 + K)
    b3_init = np.zeros(K)

    # define variables and expressions
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
    Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
    Yish = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Yish, T))

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    # we'll use this to calculate the error rate
    predict_op = tf.argmax(Yish, 1)

    t0 = datetime.now()
    LL = []
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: xtest, T: ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: xtest})
                    err = error_rate(prediction, ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(LL)
    plt.show()       
if __name__ == '__main__':
    main()