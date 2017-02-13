import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def get_normalized_data():
    print('Reading and loading data')
    df = pd.read_csv('train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    x = data[:,1:]
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    x = (x-mu)/sd
    y = data[:,0]
    return x,y
    
def y2indicator(y):
    N = len(y)
    n = np.zeros((N,10))
    for i in range(N):
        n[i,y[i]] = 1
    return n
    
def error_rate(p,t):
    return np.mean(p != t)
    
def main():
    x,y = get_normalized_data()
    max_iter = 20
    print_period = 10
    
    lr = 0.00004
    reg = 0.01
    
    xtrain = x[:-1000,:]
    ytrain = y[:-1000]
    xtest = x[-1000:,:]
    ytest = y[-1000:,]

    ytrain_ind = y2indicator(ytrain)
    ytest_ind = y2indicator(ytest)
    
    N, D = xtrain.shape
    batch_sz = 500
    n_batches = int(N / batch_sz)
    
    M = 300
    K = 10
    w1_init = np.random.randn(D, M)
    b1_init = np.zeros(M)
    w2_init = np.random.randn(M,K)
    b2_init = np.zeros(K)
    
    x = tf.placeholder(tf.float32, shape=(None,D), name='x')
    T = tf.placeholder(tf.float32, shape=(None,K), name='T')
    w1 = tf.Variable(w1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    w2 = tf.Variable(w2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    
    Z = tf.nn.relu(tf.matmul(x,w1)+ b1)
    Yish = tf.matmul(Z,w2)+b2

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Yish, T))

    train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)

    predict_op = tf.argmax(Yish, 1)

    LL =[]
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        
        for i in range(max_iter):
            for j in range(n_batches):
                xbatch = xtrain[j*batch_sz:((j+1)*batch_sz), :]
                ybatch = ytrain_ind[j*batch_sz:((j+1)*batch_sz),]

                session.run(train_op, feed_dict={x:xbatch, T:ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={x:xtest, T:ytest_ind})
                    prediction = session.run(predict_op, feed_dict={x:xtest})
                    err = error_rate(prediction, ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)

    plt.plot(LL)
    plt.show()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    