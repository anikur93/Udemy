import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x = []
y = []
for line in open('data_2d.csv'):
    a,b,c = line.split(',')
    x.append([float(a), float(b), 1])
    y.append(float(c))
    
x = np.array(x)
y = np.array(y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y)
plt.show()

w = np.linalg.solve( np.dot(x.T, x), np.dot(x.T, y))
yhat = np.dot(x, w)

d1 = y - yhat
d2 = y - y.mean()

r2 = 1 - (d1.dot(d1)/d2.dot(d2))
print('r2 is',r2)

    
