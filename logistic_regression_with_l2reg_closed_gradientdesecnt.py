import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2
x = np.random.randn(N,D)

x[:50,:] = x[:50,:] - 2*np.ones((50,D))
x[50:,:] = x[50:,:] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)

ones = np.ones((N,1))

x = np.concatenate((ones, x), axis=1)

w = np.random.randn(D+1)

z = x.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
y = sigmoid(z)

def cross_entropy(T, y):
    e = 0
    for i in range(N):
        if T[i] == 1:
            e -= np.log(y[i])
        else:
            e -= np.log(1 - y[i])
    return e
print(cross_entropy(T,y))

#closed form solution
w1 = np.array([0,4,4])
z1 = x.dot(w1)
y1 = sigmoid(z1)
print(cross_entropy(T,y1))

#gradient descent
lrate = 0.1
for i in range(1000):
    if i % 100 == 0 :
        print(cross_entropy(T,y))
    w += lrate*(x.T.dot(y-T) - 0.1*w)
    z = x.dot(w)
    y = sigmoid(z)
print('final w:',w, cross_entropy(T,y))    

plt.scatter(x[:,0], x[:,1], c=T, s=100, alpha=0.5)

x_axis = np.linspace(-6, 6, 100)

y_axis = -(w[0] + x_axis*w[1]) / w[2]

plt.plot(x_axis, y_axis)

plt.show()

























        