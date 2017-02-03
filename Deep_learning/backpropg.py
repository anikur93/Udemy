import numpy as np
import matplotlib.pyplot as plt

def forward (x,w1,b1,w2,b2):
    z = 1/(1+np.exp(-x.dot(w1)-b1))
    a = z.dot(w2) + b2
    aexp = np.exp(a)
    y = aexp/aexp.sum(axis=1, keepdims=True)
    return y, z
    
def classification_error(y,p):
    n_crct = 0 
    n_total = 0
    for i in range(len(y)):
        n_total += 1
        if y[i] == p[i]:
            n_crct += 1
    return float(n_crct)/ n_total

def cost(T, output):
    t = T* np.log(output)
    return t.sum()
    
def derivative_w2(hidden, T, output):
    re = hidden.T.dot(T - output)
    return re
    
def derivative_w1(x, hidden, T, output, w2):
    dZ = (T - output).dot(w2.T) * hidden * (1 - hidden)
    ret2 = x.T.dot(dZ)
    return ret2
    
def derivative_b2(T, output):
    re = (T-output).sum(axis=0)
    return re
    
def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)
    
    

def main():
    N = 500
    x1 = np.random.randn(N,2) + np.array([0,-2])
    x2 = np.random.randn(N,2) + np.array([2,2])
    x3 = np.random.randn(N,2) + np.array([-2,2])
    x = np.vstack([x1, x2, x3])
    
    
    D = 2#DIMENSIONS
    M = 3#NO OF NEURONS IN HIDDEN LAYER
    K = 3#NO OF CLASSES
    
    y = np.array([0]*N +[1]*N + [2]*N)
    N = len(y)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, y[i]] = 1

    plt.scatter(x[:,0], x[:,1], c=y, alpha=0.5)
    plt.show()
    

    w1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    w2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 10e-7
    costs = []
    for epoch in range(1000):
        output, hidden = forward(x,w1,b1,w2,b2)
        if epoch%500 == 0 :
            c = cost(T, output)
            P = np.argmax(output, axis =1)
            r = classification_error(y, P)
            print("cost:", c, "classification_rate:", r)
            costs.append(c)
            print(w2.shape,derivative_w2(hidden, T, output).shape,T.shape,hidden.shape)
            
        w2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        w1 += learning_rate * derivative_w1(x, hidden, T, output, w2)
        b1 += learning_rate * derivative_b1(T, output, w2, hidden)

    plt.plot(costs)
    plt.show()            
            
if __name__=='__main__':
    main()