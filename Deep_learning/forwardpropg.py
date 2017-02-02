import numpy as np
import matplotlib.pyplot as plt

N = 500
x1 = np.random.randn(N,2) + np.array([0,-2])
x2 = np.random.randn(N,2) + np.array([2,2])
x3 = np.random.randn(N,2) + np.array([-2,2])
x = np.vstack([x1, x2, x3])

y = np.array([0]*N +[1]*N + [N]*N)

plt.scatter(x[:,0], x[:,1], c=y, alpha=0.5)
plt.show()

D = 2#DIMENSIONS
M = 3#NO OF NEURONS IN HIDDEN LAYER
K = 3#NO OF CLASSES

w1 = np.random.randn(D, M)
b1 = np.random.randn(M)
w2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward (x,w1,b1,w2,b2):
    z = 1/(1+np.exp(-x.dot(w1)-b1))
    a = z.dot(w2) + b2
    aexp = np.exp(a)
    y = aexp/aexp.sum(axis=1, keepdims=True)
    return y
    
def classification_error(y,p):
    n_crct = 0 
    n_total = 0
    for i in range(len(y)):
        n_total += 1
        if y[i] == p[i]:
            n_crct += 1
    return float(n_crct)/ n_total


P_Y_given_X = forward(x, w1, b1, w2, b2)

P = np.argmax(P_Y_given_X, axis=1)

assert(len(P) == len(y))
             

print("Classification rate for randomly chosen weights:", classification_error(y, P))



