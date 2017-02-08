import numpy as np
import theano
import theano.tensor as T
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
    
def relu(a):
    return a*(a>0)
    
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
    
    thX = T.matrix('X')
    thT = T.matrix('T')
    w1 = theano.shared(w1_init, 'w1')
    b1 = theano.shared(b1_init, 'b1')
    w2 = theano.shared(w2_init, 'w2')
    b2 = theano.shared(b2_init, 'b2')
    
    thZ = T.nnet.relu(thX.dot(w1)+b1)
    thY = T.nnet.softmax(thZ.dot(w2)+b2)
    
    cost = -(thT*T.log(thY)).sum() + reg*((w1*w1).sum()+(b1*b1).sum()+(w2*w2).sum()+(b2*b2).sum())
    prediction = T.argmax(thY, axis=1)
    
    update_w1 = w1 - lr*T.grad(cost,w1)
    update_b1 = b1 - lr*T.grad(cost,b1)
    update_w2 = w2 - lr*T.grad(cost,w2)
    update_b2 = b2 - lr*T.grad(cost,b2)
    
    train = theano.function(inputs=[thX,thT], updates=[(w1, update_w1),(b1, update_b1),
                            (w2,update_w2), (b2,update_b2)])
    
    get_prediction = theano.function(inputs=[thX,thT], outputs=[cost,prediction])
    
    LL = []
    for i in range(max_iter):
        for j in range(n_batches):
            xbatch = xtrain[j*batch_sz:((j+1)*batch_sz), :]
            ybatch = ytrain_ind[j*batch_sz:((j+1)*batch_sz),]

            train(xbatch,ybatch)
            if j% print_period == 0:
                cost_val, pred_val = get_prediction(xtest, ytest_ind)
                err = error_rate(pred_val, ytest)
                print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, err))
                LL.append(cost_val)

    plt.plot(LL)
    plt.show()
    

if __name__== '__main__':
    main()


    